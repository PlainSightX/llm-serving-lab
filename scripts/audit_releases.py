"""复核已停止的v2验收工件；不补跑、不改变原输出或旧应用判定。"""

import argparse
import hashlib
import json
from pathlib import Path

from serving_lab.lab import digest
from serving_lab.profile import ServingProfile, TRANSPORT_SCOPE

REQUIRED = {
    'both_real_versions_qualified',
    'entry_fidelity',
    'second_entry_fidelity',
    'cancel',
    'overload',
    'stream_queue_version_pinning_and_global_dedup',
    'canary',
    'retire_stop_reload_requalify_rollback',
    'controller_crash_recovery',
    'deadline',
    'dead_candidate_keeps_active',
    'active_worker_exit',
    'active_worker_rebind_recovery',
}


def verify_fault(kill, arrival):
    """失败终态不等于故障注入，必须定位同一原worker的明确SIGKILL记录。"""
    worker = json.loads(arrival['worker'])
    if kill.get('signal') != 'SIGKILL' or kill.get('process') not in worker['processes']:
        raise ValueError('故障注入与在途worker身份不匹配')
    if arrival['state'] != 'failed' or arrival['cleanup'] != 'confirmed':
        raise ValueError('故障终态或恢复释放未确认')


def audit(directory):
    root = Path(directory).resolve()

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    summary = read(root / 'summary.json')
    initial = read(root / 'input.json')
    assert (
        summary['passed'] is True
        and summary['qualification_scope'] == 'r2_engineering_acceptance_v2'
    )
    phases = summary['phases']
    assert len(phases) == len(REQUIRED) and {p['name'] for p in phases} == REQUIRED
    by_phase = {p['name']: p for p in phases}
    for name, expected in initial['source_hashes'].items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == expected
    rows = read(root / 'arrivals.json')
    by_id = {r['id']: r for r in rows}
    assert len(rows) == len(by_id) == summary['arrivals'] and all(
        r['lease_released'] == 1 for r in rows
    )
    assert all(r['cleanup'] == 'confirmed' for r in rows if r['dispatched'])
    profiles = {}
    qualified = 0
    for receipt_path in sorted((root / 'runtime/qualifications').glob('*/receipt.json')):
        folder = receipt_path.parent
        receipt = read(receipt_path)
        source = read(folder / 'input.json')
        panel = source['panel']
        assert receipt['qualification_scope'] == TRANSPORT_SCOPE and receipt['passed'] is True
        assert (
            receipt['application_history']['status'] == 'failed'
            and receipt['application_history']['passed_cases'] == 3
        )
        assert digest(source['profile']) == receipt['profile_digest'] == source['profile_digest']
        assert (
            digest(panel) == receipt['panel_digest']
            and receipt['cases'] == receipt['passed_cases'] == len(panel['cases']) == 4
        )
        profile = ServingProfile(source['profile'], receipt['version_id'])
        profiles[profile.digest] = profile
        for name, expected in source['source_hashes'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
        for case in panel['cases']:
            observed = read(folder / (case['id'] + '.json'))
            assert observed['passed'] and observed['finish_reason'] == case['expected_finish']
            assert observed['input_token_ids'] == observed['engine_token_ids']
            body = {
                'model': receipt['version_id'],
                'messages': case['messages'],
                'max_tokens': case['max_tokens'],
                'stream': case['stream'],
            }
            assert observed['wire_payload'] == profile.payload(body)
            assert (
                observed['release']['worker_instance_id'] == receipt['worker']['worker_instance_id']
            )
            assert (
                observed['release']['startup_generation'] == receipt['worker']['startup_generation']
            )
            assert json.loads(by_id[observed['arrival_id']]['observation']) == observed
        qualified += 1
    assert qualified == 4
    wire_count = 0
    for row in rows:
        path = root / 'runtime/wire-requests' / (row['id'] + '.json')
        if row['dispatched']:
            assert path.exists()
        if not path.exists():
            continue
        wire = read(path)
        route = json.loads(row['route'])
        profile = profiles[wire['profile_digest']]
        assert wire['profile_digest'] == route['profile_digest']
        body = read(root / 'runtime/requests' / (row['id'] + '.json'))
        assert wire['payload'] == profile.payload(body)
        wire_count += 1
    for name in ('baseline', 'new-after-switch'):
        response = read(root / (name + '.json'))
        arrival = by_id[response['headers']['x-arrival-id']]
        assert json.loads(arrival['observation'])['response'] == response['response']
        assert (
            response['headers']['x-serving-profile']
            == json.loads(arrival['route'])['profile_digest']
        )
    stream = read(root / 'old-stream.json')
    arrival = by_id[stream['headers']['x-arrival-id']]
    assert [e[5:].strip() for e in stream['events'] if e.startswith('data:')] == json.loads(
        arrival['observation']
    )['events']
    verify_fault(
        read(root / 'active-worker-kill.json'), by_id[by_phase['active_worker_exit']['arrival']]
    )
    assert (
        json.loads(by_id[by_phase['cancel']['arrival']]['cleanup_evidence'])['status']
        == 'FINISHED_ABORTED'
    )
    assert by_id[by_phase['deadline']['arrival']]['state'] == 'timed_out'
    assert any(r['state'] == 'unknown' and r['client_key'] == 'crash-inflight' for r in rows)
    for key in ('cross-version-dedup', 'crash-inflight', 'worker-fault'):
        assert sum(bool(r['dispatched']) for r in rows if r['client_key'] == key) == 1
    cleanup = read(root / 'cleanup.json')
    assert all(c.get('result', {}).get('stopped') for c in cleanup)
    return {
        'verified': True,
        'phases': len(phases),
        'arrivals': len(rows),
        'dispatched': sum(r['dispatched'] for r in rows),
        'prequalification_runs': qualified,
        'wire_requests': wire_count,
        'historical_application_results': 'both3/8 failed unchanged',
        'source_hashes': len(initial['source_hashes']),
        'scope': 'r2_engineering_acceptance_v2',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
