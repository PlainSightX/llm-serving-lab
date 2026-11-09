"""离线核对本包真实请求、worker代际、释放/终止证明与残留资源。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from serving_lab.engine import process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import canonical, file_hash
from serving_lab.profile import ServingProfile
from compare_runtime_baseline import copied_database


def audit(root, output):
    root = Path(root).resolve()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    initial = read(root / 'input.json')
    summary = read(root / 'summary.json')
    assert summary['passed'] and summary['reserved_generation_calls'] == 23
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    manifests = [
        read(root / name)
        for name in ('worker-1-manifest.json', 'worker-2-manifest.json', 'rebound-manifest.json')
    ]
    by_identity = {canonical(m['identity']): m for m in manifests}
    assert (
        len(by_identity) == 3
        and len({m['parameter_digest'] for m in manifests})
        == len({m['profile_digest'] for m in manifests})
        == 1
    )
    events = {}
    bootstrap = []
    bindings = []
    for p in root.glob('worker-*/binding.json'):
        b = read(p)
        bindings.append(b)
        for raw in Path(b['events_path']).read_bytes().splitlines(keepends=True):
            if not raw.endswith(b'\n'):
                continue
            event = json.loads(raw)
            if event.get('event') == 'request_released':
                events[event['request_id']] = event
    for folder in (root / 'direct-1', root / 'direct-2', root / 'rebound-direct'):
        for name in ('plain', 'stream'):
            r = read(folder / (name + '-response.json'))
            assert r['release'] == events[r['release']['request_id']]
            bootstrap.append(r['request_id'])
    assert len(bootstrap) == 6 and len(events) == 22
    with copied_database(root / 'runtime/service.sqlite') as db:
        db.row_factory = __import__('sqlite3').Row
        rows = [dict(r) for r in db.execute('SELECT * FROM arrivals')]
        assert rows == read(root / 'arrivals.json')
        histories = [dict(r) for r in db.execute('SELECT * FROM pool_history')]
        states = [dict(r) for r in db.execute('SELECT * FROM pool_replicas')]
    assert len(rows) == 21 and len(histories) == 3
    counts = Counter(r['state'] for r in rows)
    assert counts == {'completed': 16, 'failed': 1, 'rejected': 4}
    assert all(r['lease_released'] == 1 for r in rows)
    fault = read(root / 'engine-kill-stream.json')
    kill = read(root / 'engine-kill-target.json')
    termination = 0
    wire_count = 0
    for row in rows:
        if not row['dispatched']:
            continue
        assert row['worker'] in by_identity
        m = by_identity[row['worker']]
        evidence = json.loads(row['cleanup_evidence'])
        obs = json.loads(row['observation'])
        assert row['cleanup'] == 'confirmed'
        payload = read(root / 'runtime/requests' / (row['id'] + '.json'))
        wire = read(root / 'runtime/wire-requests' / (row['id'] + '.json'))
        assert wire['profile_digest'] == m['profile_digest'] and wire['payload'] == ServingProfile(
            m['profile'], m['version_id']
        ).payload(payload)
        wire_count += 1
        if evidence['status'] == 'WORKER_GENERATION_TERMINATED':
            assert (
                row['id'] == fault['arrival']
                and row['state'] == 'failed'
                and evidence['worker'] == m['identity'] == kill['manifest']['identity']
            )
            assert (
                evidence['supervisor'] == m['supervisor']
                and evidence['processes'] == m['identity']['processes']
            )
            termination += 1
            assert 'data: [DONE]' not in fault['events'] and any(
                'incomplete_stream' in x for x in fault['events']
            )
        else:
            assert evidence['request_id'] in {
                'chatcmpl-' + row['attempt_id'],
                'chatcmpl-' + row['attempt_id'] + '-0',
            }
            assert (
                events[evidence['request_id']] == evidence
                and evidence['worker_instance_id'] == m['identity']['worker_instance_id']
                and evidence['startup_generation'] == m['identity']['startup_generation']
            )
            assert obs['usage']['prompt_tokens'] == len(wire['input_token_ids'])
            if obs.get('response'):
                assert (
                    obs['response']['model'] == m['version_id']
                    and obs['response']['id'] == 'chatcmpl-' + row['attempt_id']
                )
            for event in obs.get('events', []):
                if event == '[DONE]':
                    continue
                value = json.loads(event)
                assert (
                    value['model'] == m['version_id']
                    and value['id'] == 'chatcmpl-' + row['attempt_id']
                )
    assert termination == 1 and wire_count == 17
    assert sum(r['dispatched'] for r in rows) + len(bootstrap) == 23
    cleanup = read(root / 'cleanup.json')
    identities = [read(p) for p in root.glob('*-process.json')]
    for item in cleanup['workers']:
        assert (
            'error' not in item
            and item['result']['stopped']
            and not item['result'].get('remaining')
        )
        identities.extend(item['result']['processes'])
    assert all('error' not in item for item in cleanup['processes'])
    assert all(process_identity(p['pid']) != p for p in identities)
    for port in (19080, 19082, 19180, 19182, 19183):
        check_port_available('127.0.0.1', port)
    value = {
        'verified': True,
        'source_files': len(initial['source_hashes']),
        'arrivals': len(rows),
        'states': dict(counts),
        'dispatched_wire_records': wire_count,
        'actual_generations': 23,
        'bootstrap_generations': 6,
        'exact_request_release_events': 22,
        'owned_old_generation_termination_proofs': termination,
        'worker_generations': len(by_identity),
        'same_parameter_identity': True,
        'all_leases_released': True,
        'owned_recorded_processes_exited': len(identities),
        'ports_free': [19080, 19082, 19180, 19182, 19183],
        'scope': 'single-GPU two-replica plus rebind, not two-node deployment or performance claim',
    }
    write_once(output, value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.run, a.output), indent=2))
