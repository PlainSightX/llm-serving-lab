"""从网关A/B原件重算：配对首token额外等待、窗口SLO、发布窗口一致性与网关内部阶段耗时。

只读取运行目录（窗口原件、账本导出与runtime快照），不触碰模型或服务。
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from serving_lab.benchmark import StreamObservation, percentile, summarize
from serving_lab.execution import write_once
from serving_lab.lab import file_hash

PHASES = [
    'arrival_recorded',
    'body_read',
    'validated',
    'pinned',
    'ready_checked',
    'tokenized',
    'reserved',
    'payload_saved',
    'slot_acquired',
    'ready_rechecked',
    'dispatch_recorded',
    'upstream_headers',
    'first_content',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        'n': len(values),
        'median': statistics.median(values),
        'p95': percentile(values, 0.95),
        'mean': statistics.fmean(values),
    }


def segments(phase_ms):
    """相邻阶段时刻之差（毫秒）；缺失的阶段跳过，下一段从上一个存在的时刻算起。"""
    result = {}
    previous_name, previous = 'received', 0.0
    for name in PHASES:
        if name in phase_ms:
            result[f'{previous_name}->{name}'] = phase_ms[name] - previous
            previous_name, previous = name, phase_ms[name]
    return result


def report(directory, output):
    root = Path(directory).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = read(root / 'input.json')
    config = initial['config']
    final = read(root / 'summary.json')
    if not final.get('integrity_completed'):
        raise ValueError('运行未完成完整性检查')
    gateways = list(initial['gateways'])
    for name, item in initial['gateways'].items():
        frozen = root / 'gateway-sources' / name
        for rel, expected in item['files'].items():
            if file_hash(frozen / rel) != expected:
                raise ValueError(f'{name}冻结源码与运行记录不符：{rel}')
    ledgers = {
        name: {
            r['client_key']: r for r in read(root / (name + '-arrivals.json')) if r['client_key']
        }
        for name in gateways
    }
    workloads = read(root / 'workloads.json')
    checked = defaultdict(int)
    windows = []
    by_key = {}
    for recorded in final['windows']:
        folder = root / 'windows' / recorded['label']
        data = read(folder / 'input.json')
        result = read(folder / 'result.json')
        rows = result['rows']
        if len(rows) != len(data['requests']):
            raise ValueError('窗口请求数不符')
        for item, row in zip(data['requests'], rows):
            if item['id'] != row['id']:
                raise ValueError('请求顺序不符')
            observation = StreamObservation(
                item['expected_version'],
                item['input_tokens'],
                item['output_budget'],
                item.get('allowed_versions'),
            )
            for event in row.get('events', []):
                observation.feed(event['data'], event['elapsed'])
            if row['protocol_valid'] != bool(
                row['sent'] and observation.valid and not row.get('error')
            ):
                raise ValueError('协议有效性无法从原始事件重算')
            if item['path'] == 'gateway' and row['status'] == 200:
                gateway = item['gateway']
                stored = ledgers[gateway][item['id']]
                if stored['id'] != row['response_headers']['x-arrival-id']:
                    raise ValueError('账本到达与响应头不符')
                if (stored['state'], stored['cleanup'], stored['lease_released']) != (
                    'completed',
                    'confirmed',
                    1,
                ):
                    raise ValueError(f'账本终态/清理/租约异常：{item["id"]}')
                wire = read(
                    root / (gateway + '-runtime') / 'wire-requests' / (stored['id'] + '.json')
                )
                if wire['input_token_ids'] != item['input_token_ids']:
                    raise ValueError('wire输入token与冻结输入不符')
                row['phase_ms'] = json.loads(stored['observation']).get('phase_ms', {})
                checked[gateway] += 1
        computed = summarize(
            rows,
            result['elapsed_seconds'],
            config['slo'],
            interval=1 / data['rate'],
            lag_fraction=config['valid_schedule_p95_interval_fraction'],
        )
        for key, value in computed.items():
            if value != recorded[key]:
                raise ValueError(f'{recorded["label"]}的{key}无法从原件重算')
        windows.append({'recorded': recorded, 'rows': rows, 'data': data})
        by_key[recorded['label']] = windows[-1]

    steady = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(lambda: defaultdict(list))
    phases = defaultdict(lambda: defaultdict(list))
    slo = defaultdict(lambda: defaultdict(list))
    for window in windows:
        label = window['recorded']['label']
        if not label.startswith('steady-'):
            continue
        _, rep, rest = label.split('-', 2)
        model, rate_part, path = rest.rsplit('-', 2)
        stratum = f'{model} {rate_part.replace("rate", "")}req/s'
        rows = window['rows']
        slo[stratum][path].append(window['recorded']['slo_fraction'])
        steady[stratum][path].extend(r['ttft_sent'] for r in rows if r['protocol_valid'])
        if path != 'direct':
            direct = by_key[f'steady-{rep}-{model}-{rate_part}-direct']['rows']
            for gw_row, direct_row in zip(rows, direct):
                if gw_row['protocol_valid'] and direct_row['protocol_valid']:
                    paired[stratum][path].append(gw_row['ttft_sent'] - direct_row['ttft_sent'])
                for name, value in segments(gw_row.get('phase_ms', {})).items():
                    phases[f'{stratum} {path}'][name].append(value)

    def ms(stats):
        return (
            None if stats is None else {k: (v * 1000 if k != 'n' else v) for k, v in stats.items()}
        )

    summary = {
        'run': str(root),
        'gateways': {
            n: {'note': initial['gateways'][n].get('note'), 'verified_ledger_rows': checked[n]}
            for n in gateways
        },
        'formal_requests': final['formal_requests'],
        'protocol_valid': sum(w['recorded']['protocol_valid'] for w in windows),
        'schedule_valid_windows': final['schedule_valid_windows'],
        'windows': len(windows),
        'ttft_ms': {
            s: {p: ms(describe(v)) for p, v in paths.items()} for s, paths in steady.items()
        },
        'paired_extra_ttft_ms': {
            s: {p: ms(describe(v)) for p, v in paths.items()} for s, paths in paired.items()
        },
        'slo_fraction_by_window': {s: dict(paths) for s, paths in slo.items()},
        'phase_ms_median': {
            key: {name: statistics.median(values) for name, values in segs.items()}
            for key, segs in phases.items()
        },
        'release': [
            {
                'label': w['recorded']['label'],
                'protocol_valid': w['recorded']['protocol_valid'],
                'planned': w['recorded']['planned'],
                'planned_version_matches': w['recorded'].get('planned_version_matches'),
                'switch_completed_in_gap': w['recorded'].get('switch_completed_in_gap'),
                'ttft_ms': ms(describe([r['ttft_sent'] for r in w['rows'] if r['protocol_valid']])),
            }
            for w in windows
            if w['recorded']['label'].startswith('release-')
        ],
        'workload_seed': config['seed'],
        'steady_cases_per_repeat': {rep: len(cases) for rep, cases in workloads['steady'].items()},
    }
    write_once(out / 'report.json', summary)
    lines = [f'# 网关A/B结果：{root.name}', '']
    lines.append(
        f'正式请求 {summary["formal_requests"]}，协议有效 {summary["protocol_valid"]}，'
        f'窗口 {summary["windows"]}（调度有效 {summary["schedule_valid_windows"]}）。'
    )
    lines += ['', '## 配对首token额外等待（网关 − 直连，毫秒，按发送时刻计）', '']
    lines.append('| 负载 | 网关 | 配对数 | 中位数 | P95 | 均值 |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for stratum in sorted(paired):
        for path in sorted(paired[stratum]):
            stats = ms(describe(paired[stratum][path]))
            lines.append(
                f'| {stratum} | {path} | {stats["n"]} | {stats["median"]:.1f} | {stats["p95"]:.1f} | {stats["mean"]:.1f} |'
            )
    lines += [
        '',
        '## 首token延迟（毫秒）',
        '',
        '| 负载 | 路径 | 中位数 | P95 | SLO达标比例（各窗口） |',
        '|---|---|---:|---:|---|',
    ]
    for stratum in sorted(steady):
        for path in sorted(steady[stratum]):
            stats = ms(describe(steady[stratum][path]))
            fractions = ', '.join(f'{v:.3f}' for v in slo[stratum][path])
            lines.append(
                f'| {stratum} | {path} | {stats["median"]:.1f} | {stats["p95"]:.1f} | {fractions} |'
            )
    lines += ['', '## 网关内部阶段耗时中位数（毫秒）', '']
    for key in sorted(summary['phase_ms_median']):
        lines.append(
            f'**{key}**：'
            + '；'.join(f'{n} {v:.2f}' for n, v in summary['phase_ms_median'][key].items())
        )
        lines.append('')
    lines += [
        '## 发布窗口',
        '',
        '| 窗口 | 有效/计划 | 版本符合计划 | 切换在空档内完成 | 首token中位数/P95（毫秒） |',
        '|---|---|---|---|---|',
    ]
    for item in summary['release']:
        t = item['ttft_ms']
        lines.append(
            f'| {item["label"]} | {item["protocol_valid"]}/{item["planned"]} | {item["planned_version_matches"]} | '
            f'{item["switch_completed_in_gap"]} | {t["median"]:.1f} / {t["p95"]:.1f} |'
        )
    (out / 'report.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.directory, args.output)
    print(json.dumps(result['paired_extra_ttft_ms'], indent=2, ensure_ascii=False))
