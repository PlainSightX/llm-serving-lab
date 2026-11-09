"""核验原生位置基线并应用事前判别；历史变化只作描述，不是因果收益。"""

import argparse
import json
import math
import sqlite3
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.lab import digest
from serving_lab.benchmark import parse_metrics, metric_delta


def verified_timing(report, summary, diagnosis):
    assert report['scoped_timing_verified']
    raw = report['scoped_timing']
    assert Path(raw['directory']).resolve() == Path(diagnosis).resolve()
    seconds = sum(v for k, v in summary['operation_seconds'].items() if not k.startswith('sqlite_'))
    assert math.isclose(seconds, raw['method_union_seconds'], abs_tol=1e-9)
    assert summary['trace_seconds'] == raw['trace_seconds']
    return raw['method_union_seconds'], raw['trace_seconds']


@contextmanager
def copied_database(path):
    """只在临时副本打开SQLite，保护原始WAL/SHM和checkpoint前字节。"""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix='lsl-r4-sqlite-') as temp:
        target = Path(temp) / 'service.sqlite'
        for suffix in ('', '-wal', '-shm'):
            source = Path(str(path) + suffix)
            if source.exists():
                shutil.copyfile(source, Path(str(target) + suffix))
        db = sqlite3.connect(target.as_uri() + '?mode=ro', uri=True)
        try:
            yield db
        finally:
            db.close()


def database_logical_digest(path):
    with copied_database(path) as db:
        schema = db.execute(
            'SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name'
        ).fetchall()
        tables = {}
        for kind, name, *_ in schema:
            if kind == 'table':
                tables[name] = sorted(
                    db.execute('SELECT * FROM "' + name.replace('"', '""') + '"').fetchall(),
                    key=lambda r: json.dumps(r, ensure_ascii=False),
                )
        return digest({'schema': schema, 'tables': tables})


def verify_snapshot(snapshot):
    for relative, h in snapshot['files'].items():
        assert file_hash(Path(snapshot['snapshot']) / relative) == h
        if relative not in {'service.sqlite', 'service.sqlite-wal', 'service.sqlite-shm'}:
            assert file_hash(Path(snapshot['source']) / relative) == h
    before = database_logical_digest(Path(snapshot['snapshot']) / 'service.sqlite')
    after = database_logical_digest(Path(snapshot['source']) / 'service.sqlite')
    assert before == after
    return before


def verify_native_diagnosis(diagnosis):
    diagnosis = Path(diagnosis)
    read = lambda p: json.loads(Path(p).read_text())
    placement = read(diagnosis / 'runtime-placement.json')
    settings = read(diagnosis / 'runtime-settings.json')
    cfg = read(diagnosis / 'gateway.json')
    assert (
        placement['filesystem']['fstype'] == 'ext4'
        and settings['journal_mode'] == 'wal'
        and settings['synchronous'] == 2
    )
    assert (
        settings['runtime']
        == cfg['runtime']
        == str(Path(placement['native_root']) / 'control-runtime')
    )
    snapshot = read(diagnosis / 'runtime-snapshot.json')['runtimes']['control']
    assert snapshot['source'] == cfg['runtime']
    verify_snapshot(snapshot)


def compare(current, report, diagnosis, output, historical=None):
    current = Path(current)
    report = Path(report)
    diagnosis = Path(diagnosis)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    result = read(report / 'report.json')
    timing = read(diagnosis / 'timing-summary.json')
    initial = read(current / 'input.json')
    assert initial['config']['formal_requests'] == 1152
    placement = read(current / 'runtime-placement.json')
    assert placement['filesystem']['fstype'] == 'ext4'
    snapshots = read(current / 'runtime-snapshot.json')['runtimes']
    counts = {}
    for name, expected in [('control', 592), ('treatment', 208)]:
        cfg = read(current / (name + '-config.json'))
        settings = read(current / (name + '-runtime-settings.json'))
        assert settings['runtime'] == cfg['runtime'] == snapshots[name]['source']
        assert settings['journal_mode'] == 'wal' and settings['synchronous'] == 2
        verify_snapshot(snapshots[name])
        with copied_database(Path(cfg['runtime']) / 'service.sqlite') as db:
            db.row_factory = sqlite3.Row
            rows = [dict(r) for r in db.execute('SELECT * FROM arrivals')]
        assert len(rows) == expected and rows == read(current / (name + '-arrivals.json'))
        assert all(
            r['state'] == 'completed' and r['cleanup'] == 'confirmed' and r['lease_released'] == 1
            for r in rows
        )
        counts[name] = len(rows)
    assert result['formal_requests'] == 1152 and result['gateway_wire_records_verified'] == 768
    summary = read(current / 'summary.json')
    assert summary['formal_windows'] == summary['schedule_valid_windows'] == 30
    assert all(w['protocol_valid'] == w['planned'] for w in summary['windows'])
    assert all(w['matched_plan_valid'] for w in result['release_windows'])
    assert all(
        p['both_protocol_valid'] == p['actual_output_length_matched'] == 96
        for p in result['paired_requests']
    )
    release_pairs = 0
    for rep in range(3):
        left = current / 'windows' / f'release-r{rep}-control'
        right = current / 'windows' / f'release-r{rep}-treatment'
        li = read(left / 'input.json')
        ri = read(right / 'input.json')
        lr = read(left / 'result.json')['rows']
        rr = read(right / 'result.json')['rows']
        assert li['offsets'] == ri['offsets']
        for a, b, x, y in zip(li['requests'], ri['requests'], lr, rr):
            assert a['input_token_ids'] == b['input_token_ids'] and a['messages'] == b['messages']
            assert (
                x['actual_version'] == y['actual_version']
                and x['usage']['completion_tokens'] == y['usage']['completion_tokens']
                and x['content'] == y['content']
            )
            release_pairs += 1
    assert release_pairs == 192
    # 所有540个引擎原生计数差分逐项重算；缺失计数不能假定为0。
    metric_fields = 0
    for w in summary['windows']:
        for version, values in w['engine_metrics'].items():
            before = parse_metrics(
                (current / (w['label'] + '-before-' + version + '-metrics.txt')).read_text()
            )
            after = parse_metrics(
                (current / (w['label'] + '-after-' + version + '-metrics.txt')).read_text()
            )
            for key, value in values.items():
                assert metric_delta(before, after, key) == value
                metric_fields += 1
    low = [p for p in result['paired_requests'] if p['rate'] == 1]
    extra_ok = len(low) == 2 and all(
        p['gateway_minus_direct_ttft_sent']['median'] <= 0.020 for p in low
    )
    low_slo = all(all(x == 1 for x in c['slo_fraction']) for c in result['cells'] if c['rate'] == 1)
    methods, trace_seconds = verified_timing(result, timing, diagnosis)
    verify_native_diagnosis(diagnosis)
    fraction = methods / trace_seconds
    timing_ok = fraction <= 0.05
    conclusion = {
        'formal_requests': 1152,
        'windows': 30,
        'gateway_wire': 768,
        'ledger_rows': counts,
        'native_metric_fields_recomputed': metric_fields,
        'low_load_pairs': low,
        'low_load_slo_all': low_slo,
        'diagnostic_method_seconds': methods,
        'diagnostic_trace_seconds': timing['trace_seconds'],
        'diagnostic_method_fraction': fraction,
        'predeclared_checks': {
            'extra_ttft_max_20ms': extra_ok,
            'low_load_slo_all': low_slo,
            'method_fraction_max_5percent': timing_ok,
        },
        'close_current_writer_candidate': extra_ok and low_slo and timing_ok,
        'meaning': 'native reasonable baseline/decision only; not source-level gain, no power-loss or production-capacity claim',
    }
    conclusion['matched_release_pairs'] = release_pairs
    if historical:
        old = Path(historical)
        assert read(old / 'input.json')['config'] == initial['config']
        for size in ('small', 'large'):
            a = read(old / (size + '-preparation.json'))['profile']
            b = read(current / (size + '-preparation.json'))['profile']
            assert {k: v for k, v in a.items() if k != 'implementation'} == {
                k: v for k, v in b.items() if k != 'implementation'
            }
            assert (
                read(old / (size + '-worker/launch.json'))['base_identity']
                == read(current / (size + '-worker/launch.json'))['base_identity']
            )
        assert read(old / 'workloads.json') == read(current / 'workloads.json')
        conclusion['historical_effective_profiles_weights_workloads_equal'] = True
        conclusion['historical_timing_comparison'] = (
            'descriptive only; different time and implementation source identity'
        )
    write_once(out / 'decision.json', conclusion)
    (out / 'source_compare_runtime_baseline.py').write_bytes(Path(__file__).read_bytes())
    return conclusion


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--current', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--diagnosis', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--historical')
    a = p.parse_args()
    print(
        json.dumps(
            compare(a.current, a.report, a.diagnosis, a.output, a.historical),
            ensure_ascii=False,
            indent=2,
        )
    )
