"""从完整R3原始窗口重算统计；不把profile轮或不匹配发布对照混入正式结论。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from serving_lab.benchmark import summarize, StreamObservation, distribution
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile
from serving_lab.execution import write_once


def report(directory, output, diagnosis=None):
    root = Path(directory).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    initial = read(root / 'input.json')
    config = initial['config']
    final = read(root / 'summary.json')
    for name, expected in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == expected
    ledgers = {
        which: {
            r['client_key']: r for r in read(root / (which + '-arrivals.json')) if r['client_key']
        }
        for which in ('control', 'treatment')
    }
    groups = defaultdict(list)
    strata = defaultdict(list)
    release_strata = defaultdict(list)
    all_ids = set()
    formal_count = 0
    windows = []
    verified_gateway = 0
    for recorded in final['windows']:
        folder = root / 'windows' / recorded['label']
        data = read(folder / 'input.json')
        result = read(folder / 'result.json')
        rows = result['rows']
        assert len(rows) == len(data['requests']) == len(data['offsets'])
        for item, row, offset in zip(data['requests'], rows, data['offsets']):
            assert item['id'] == row['id'] and offset == row['scheduled_offset']
            assert abs(row['send_lag'] - (row['send_offset'] - offset)) < 1e-8
            assert row['id'] not in all_ids
            all_ids.add(row['id'])
            assert item['input_tokens'] == len(item['input_token_ids'])
            observation = StreamObservation(
                item['expected_version'],
                item['input_tokens'],
                item['output_budget'],
                item.get('allowed_versions'),
            )
            for event in row.get('events', []):
                observation.feed(event['data'], event['elapsed'])
            assert row['protocol_valid'] == bool(
                row['sent'] and observation.valid and not row.get('error')
            )
            if row['protocol_valid']:
                assert (
                    observation.usage == row['usage']
                    and observation.content == row['content']
                    and observation.actual_model == row['actual_version']
                )
                assert (
                    abs(row['ttft_scheduled'] - row['send_lag'] - observation.first_content) < 1e-8
                )
                assert (
                    row['e2e_sent'] == observation.done
                    and abs(row['e2e_scheduled'] - row['send_lag'] - observation.done) < 1e-8
                )
            if item['path'] == 'gateway' and row['sent']:
                stored = ledgers[data['gateway']].get(item['id'])
                if stored is None:
                    assert row['status'] is None
                    continue
                if row['status'] == 200:
                    assert stored['id'] == row['response_headers']['x-arrival-id']
                    route = json.loads(stored['route'])
                    model = row['actual_version']
                    assert route['profile_digest'] == digest(data['profiles'][model])
                    wire = read(
                        root
                        / (data['gateway'] + '-runtime')
                        / 'wire-requests'
                        / (stored['id'] + '.json')
                    )
                    profile = ServingProfile(data['profiles'][model], model)
                    assert wire['payload'] == profile.payload({**item['body'], 'model': model})
                    assert wire['input_token_ids'] == item['input_token_ids']
                    verified_gateway += 1
                    row['server_pre_dispatch_seconds'] = json.loads(stored['observation']).get(
                        'dispatch_seconds'
                    )
        computed = summarize(
            rows,
            result['elapsed_seconds'],
            config['slo'],
            interval=1 / data['rate'],
            lag_fraction=config['valid_schedule_p95_interval_fraction'],
        )
        for key, value in computed.items():
            assert value == recorded[key], (recorded['label'], key)
        recorded['gateway_pre_dispatch'] = distribution(
            [
                r['server_pre_dispatch_seconds']
                for r in rows
                if r.get('server_pre_dispatch_seconds') is not None and r['protocol_valid']
            ]
        )
        windows.append(recorded)
        if recorded['diagnostic']:
            continue
        formal_count += len(rows)
        if recorded['label'].startswith('steady-'):
            key = (recorded['versions'][0], recorded['rate'], recorded['path'])
            groups[key].append(recorded)
            for row in rows:
                strata[
                    (row['expected_version'], recorded['rate'], recorded['path'], row['shape'])
                ].append(row)
        if recorded['label'].startswith('release-'):
            for index, row in enumerate(rows):
                release_strata[
                    (
                        recorded['label'],
                        'before' if index < len(rows) // 2 else 'after',
                        row.get('actual_version') or row['expected_version'],
                        row['shape'],
                    )
                ].append(row)
    assert (
        formal_count == config['formal_requests'] == 1152
        and len([w for w in windows if not w['diagnostic']]) == 30
    )
    cells = []
    for key, values in sorted(groups.items()):
        assert len(values) == 3
        cells.append(
            {
                'model': key[0],
                'rate': key[1],
                'path': key[2],
                'planned': sum(v['planned'] for v in values),
                'protocol_valid': sum(v['protocol_valid'] for v in values),
                'valid_repeats': sum(v['schedule_valid'] for v in values),
                'ttft_repeat_medians': [v['ttft_scheduled']['median'] for v in values],
                'e2e_repeat_medians': [v['e2e_scheduled']['median'] for v in values],
                'slo_fraction': [v['slo_fraction'] for v in values],
                'output_token_medians': [v['output_tokens']['median'] for v in values],
                'gateway_pre_dispatch_medians': [
                    v['gateway_pre_dispatch']['median'] for v in values
                ],
                'engine_queue_mean': [
                    v['engine_metrics'][key[0]]['vllm:request_queue_time_seconds_sum']
                    / v['engine_metrics'][key[0]]['vllm:request_queue_time_seconds_count']
                    if v['engine_metrics'][key[0]]['vllm:request_queue_time_seconds_count']
                    and v['engine_metrics'][key[0]]['vllm:request_queue_time_seconds_sum']
                    is not None
                    else None
                    for v in values
                ],
            }
        )
    detail = []
    for key, rows in sorted(strata.items()):
        good = [r for r in rows if r['protocol_valid']]
        detail.append(
            {
                'model': key[0],
                'rate': key[1],
                'path': key[2],
                'shape': key[3],
                'planned': len(rows),
                'valid': len(good),
                'ttft': distribution([r['ttft_scheduled'] for r in good]),
                'e2e': distribution([r['e2e_scheduled'] for r in good]),
                'output_tokens': distribution([r['usage']['completion_tokens'] for r in good]),
            }
        )
    release_detail = []
    for key, rows in sorted(release_strata.items()):
        good = [r for r in rows if r['protocol_valid']]
        release_detail.append(
            {
                'window': key[0],
                'phase': key[1],
                'actual_model': key[2],
                'shape': key[3],
                'planned': len(rows),
                'valid': len(good),
                'ttft': distribution([r['ttft_scheduled'] for r in good]),
                'e2e': distribution([r['e2e_scheduled'] for r in good]),
                'output_tokens': distribution([r['usage']['completion_tokens'] for r in good]),
            }
        )
    pairs = []
    for model in sorted({key[0] for key in groups}):
        for rate in config['rates']:
            deltas = []
            length_matches = 0
            both_valid = 0
            for rep in range(config['repeats']):
                direct = root / 'windows' / f'steady-r{rep}-{model}-rate{rate}-direct'
                gateway = root / 'windows' / f'steady-r{rep}-{model}-rate{rate}-gateway'
                di = read(direct / 'input.json')
                gi = read(gateway / 'input.json')
                dr = read(direct / 'result.json')['rows']
                gr = read(gateway / 'result.json')['rows']
                profile = ServingProfile(gi['profiles'][model], model)
                for a, b, x, y in zip(di['requests'], gi['requests'], dr, gr):
                    assert (
                        a['body'] == profile.payload(b['body'])
                        and a['input_token_ids'] == b['input_token_ids']
                    )
                    if x['protocol_valid'] and y['protocol_valid']:
                        both_valid += 1
                        if x['usage']['completion_tokens'] == y['usage']['completion_tokens']:
                            length_matches += 1
                            deltas.append(y['ttft_sent'] - x['ttft_sent'])
            pairs.append(
                {
                    'model': model,
                    'rate': rate,
                    'both_protocol_valid': both_valid,
                    'actual_output_length_matched': length_matches,
                    'gateway_minus_direct_ttft_sent': distribution(deltas),
                    'scope': 'paired observations; high rate includes admission/queue policy, not pure gateway cost',
                }
            )
    result = {
        'integrity_verified': True,
        'formal_requests': formal_count,
        'gateway_wire_records_verified': verified_gateway,
        'cells': cells,
        'strata': detail,
        'paired_requests': pairs,
        'release_strata': release_detail,
        'release_windows': [w for w in windows if w['label'].startswith('release-')],
        'quarantined_cprofile': {
            'present': (root / 'control.pstats').exists(),
            'accepted_for_attribution': False,
            'reason': 'CPython3.12 interpreter-wide monitoring mixed thread events; impossible caller edges observed',
        },
        'diagnostic_selection': read(root / 'diagnostic-selection.json'),
        'telemetry': read(root / 'telemetry-summary.json'),
        'scope': config['scope'],
    }
    if (root / 'storage-diagnostic.json').exists():
        result['storage_diagnostic'] = read(root / 'storage-diagnostic.json')
        result['storage_selection'] = (
            'B exploratory CPU observations retained; original cProfile trigger invalidated, not a new C experiment'
        )
    result['scoped_timing_verified'] = False
    if diagnosis:
        diag = Path(diagnosis).resolve()
        source = read(diag / 'input.json')
        timing = read(diag / 'timing.json')
        observed = read(diag / 'result.json')
        selected = root / 'windows' / source['selected_window'] / 'input.json'
        assert source['selected_input_sha256'] == file_hash(selected)
        for name, expected in source['source_hashes'].items():
            assert file_hash(diag / 'source' / name) == expected
        old = read(selected)['requests']
        calls = source['requests']
        assert len(calls) == len(observed['rows']) == 32
        arrivals = {r['id']: r for r in read(diag / 'arrivals.json')}
        for original, item, row in zip(old, calls, observed['rows']):
            assert (
                original['body'] == item['body']
                and original['input_token_ids'] == item['input_token_ids']
            )
            observer = StreamObservation(
                item['expected_version'], item['input_tokens'], item['output_budget']
            )
            for event in row['events']:
                observer.feed(event['data'], event['elapsed'])
            assert observer.valid and row['protocol_valid']
            stored = arrivals[row['response_headers']['x-arrival-id']]
            assert (
                stored['state'] == 'completed'
                and stored['cleanup'] == 'confirmed'
                and stored['lease_released'] == 1
            )
        records = timing['records']
        assert records and timing['owner_native'] == timing['pid']
        assert all(
            r['thread'] == timing['owner_thread']
            and r['native_thread'] == timing['owner_native']
            and timing['started'] <= r['start'] <= r['end'] <= timing['ended']
            for r in records
        )
        methods = sorted(
            [r for r in records if not r['operation'].startswith('sqlite_')],
            key=lambda r: r['start'],
        )
        exits = sorted(
            [r for r in records if r['operation'] == 'sqlite_transaction_exit'],
            key=lambda r: r['start'],
        )
        assert len(methods) == 320 and len(exits) == 256
        assert all(a['end'] <= b['start'] for a, b in zip(methods, methods[1:]))
        assert all(a['end'] <= b['start'] for a, b in zip(exits, exits[1:]))
        for child in exits:
            assert (
                len(
                    [
                        parent
                        for parent in methods
                        if parent['arrival_id'] == child['arrival_id']
                        and parent['start'] <= child['start']
                        and child['end'] <= parent['end']
                    ]
                )
                == 1
            )
        result['scoped_timing_verified'] = True
        result['scoped_timing'] = {
            'directory': str(diag),
            'requests': 32,
            'trace_seconds': timing['ended'] - timing['started'],
            'method_spans': len(methods),
            'transaction_exits': len(exits),
            'commits': sum(r['outcome'] == 'commit' for r in exits),
            'method_union_seconds': sum(r['seconds'] for r in methods),
            'transaction_exit_seconds': sum(r['seconds'] for r in exits),
            'nested_not_additive': True,
            'scope': 'main-thread synchronous wall time; not CPU percentage or speedup',
        }
        if (diag / 'storage-diagnostic.json').exists():
            result['storage_diagnostic'] = read(diag / 'storage-diagnostic.json')
            result['storage_selection'] = (
                'conditional CPU observation from verified scoped timing; not an end-to-end experiment'
            )
    write_once(output / 'report.json', result)
    (output / 'source_report_benchmark.py').write_bytes(Path(__file__).read_bytes())

    def fmt(values):
        return '/'.join(f'{v:.4f}' if v is not None else 'NA' for v in values)

    lines = [
        '# 固定矩阵原始数据重算报告',
        '',
        '所有时间为秒；每格显示3个重复窗口的中位数，不是3次独立硬件复现。',
        '正式请求分母包括失败/拒绝/未发送；profile轮排除。旧应用资格不由这些工程指标升级。',
        '',
        '| 模型 | 到达率 | 路径 | 协议合格/计划 | 有效调度重复 | TTFT中位数(3轮) | E2E中位数(3轮) | SLO比例(3轮) |',
        '|---|---:|---|---:|---:|---|---|---|',
    ]
    for cell in cells:
        lines.append(
            f"|{cell['model']}|{cell['rate']}|{cell['path']}|{cell['protocol_valid']}/{cell['planned']}|{cell['valid_repeats']}/3|{fmt(cell['ttft_repeat_medians'])}|{fmt(cell['e2e_repeat_medians'])}|{fmt(cell['slo_fraction'])}|"
        )
    lines += [
        '',
        '## 诊断证据',
        '',
        (
            '原cProfile出现跨线程调用关系混合，已隔离，不展示或解释其精确归因表。原始文件保留。'
            if result['quarantined_cprofile']['present']
            else '本轮没有cProfile数据；若提供诊断，仅使用独立线程标记计时。'
        ),
    ]
    if result['scoped_timing_verified']:
        t = result['scoped_timing']
        lines += [
            f"独立32请求诊断通过线程、区间及账本核验：{t['method_spans']}个同步方法区间共{t['method_union_seconds']:.5f}秒，其中{t['transaction_exits']}个事务退出区间共{t['transaction_exit_seconds']:.5f}秒；追踪跨度{t['trace_seconds']:.5f}秒。",
            '这些区间嵌套，不能相加，也不是CPU百分比或端到端优化收益。',
        ]
    lines += [
        '',
        '## 解释限制',
        '',
        '高负载差异包含准入/排队政策；发布窗口有共同1秒空档且只涉及已预热版本。',
        '按真实版本和长度分层的详细结果见report.json；不匹配发布计划或调度无效的窗口不能用于因果比较。',
        '若执行存储诊断，它仅有一组固定顺序的同初态CPU对照，不能估计稳定或端到端加速比。',
    ]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {
        k: result[k]
        for k in (
            'integrity_verified',
            'formal_requests',
            'gateway_wire_records_verified',
            'scoped_timing_verified',
        )
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnosis')
    args = parser.parse_args()
    print(json.dumps(report(args.directory, args.output, args.diagnosis), indent=2))
