"""离线核C0发布原件：质量门、请求绑定、实际wire、释放和网关应用路径。"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from serving_lab.application_release import load_policy, POOL_SCOPE, WORKER_SCOPE
from serving_lab.execution import write_once
from serving_lab.lab import canonical, digest, file_hash
from serving_lab.profile import ServingProfile
from serving_lab.standard_actor import StandardActor
from serving_lab.task_protocol import validate_manifest, verify_files
from compare_runtime_baseline import copied_database
from q2_teacher import IMAGE, read
from probe_application_release import prior_failures, reservation_count, same_pair_output


def verify_pairs(pairs, direct_receipts, direct_reservations, client_rows, gateway_wires):
    if len(pairs) != 3 or {p['pair'] for p in pairs} != {1, 2, 3}:
        raise ValueError('必须为三组独立配对')
    direct_ids = set()
    gateway_ids = set()
    for pair in pairs:
        direct, gateway = pair['direct'], pair['gateway']
        aid = direct['arrival_id']
        gid = gateway['headers']['x-arrival-id']
        if (
            aid in direct_ids
            or gid in gateway_ids
            or direct != direct_receipts.get(aid)
            or gid not in client_rows
            or gateway != client_rows[gid][1]
        ):
            raise ValueError('配对回执未绑定独立原始调用')
        direct_ids.add(aid)
        gateway_ids.add(gid)
        original = direct_reservations[aid]
        wire = gateway_wires[gid]
        if (
            original['metadata'].get('pair') != pair['pair']
            or original['attempt_id'] != direct['attempt_id']
            or original['arrival_id'] != aid
            or original['profile_digest'] != wire['profile_digest']
            or original['input_token_ids'] != wire['input_token_ids']
        ):
            raise ValueError('配对身份/profile/token不符')
        left, right = original['wire'], wire['payload']
        without_stream = lambda b: {
            k: v for k, v in b.items() if k not in {'stream', 'stream_options'}
        }
        if (
            without_stream(left) != without_stream(right)
            or left.get('stream') is not False
            or right.get('stream') is not (pair['pair'] == 3)
            or not same_pair_output(direct, gateway)
        ):
            raise ValueError('直连与入口的实际输入/约束或输出不一致')


def check_support_source(record, environment, launch, task, data, image):
    if (
        record['task_id'] != task['task_id']
        or environment['task_id'] != task['task_id']
        or record['role'] != task['role']
        or task['role'] != 'support'
        or launch['image'] != image
        or launch['command'][-2:] != [image, '/runner.py']
    ):
        raise ValueError('集成回合未绑定冻结任务或镜像')
    verify_files(task, data)


def audit(run, output):
    root = Path(run).resolve()
    initial = read(root / 'input.json')
    summary = read(root / 'summary.json')
    previous = prior_failures(
        [r['path'] for r in initial['prior_failed_runs']], initial['evidence_sha256']
    )
    assert previous == initial['prior_failed_runs']
    prior_count = sum(r['reservations'] for r in previous)
    assert (
        prior_count
        == initial['prior_generation_reservations']
        == summary['prior_generation_reservations']
    )
    support_data = Path(initial['support_data'])
    assert (
        file_hash(root / 'support-manifest.json')
        == initial['support_manifest_sha256']
        == file_hash(support_data / 'manifest.json')
        == read(support_data / 'manifest-anchor.json')['sha256']
    )
    support_tasks = validate_manifest(read(root / 'support-manifest.json'))
    assert read(root / 'transport-panel.json') == initial['transport_panel']
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    policy = load_policy(initial['application_evidence'], initial['evidence_sha256'])
    imported = Path(initial['application_evidence']).parent / 'import-receipt.json'
    assert (
        file_hash(imported) == initial['import_receipt_sha256']
        and read(imported)['sha256'] == initial['evidence_sha256']
    )
    assert policy.bundle['bundle_digest'] == initial['bundle_digest']
    manifests = {
        role: read(root / name)
        for role, name in [
            ('base', 'base-manifest.json'),
            ('c0', 'c0-manifest.json'),
            ('rebound', 'c0-rebound-manifest.json'),
        ]
    }
    by_worker = {}
    engines = {}
    parameters = {}
    release_events = {}
    usage = Counter()
    seen_release = set()
    for role, manifest in manifests.items():
        assert (
            digest({k: v for k, v in manifest.items() if k != 'manifest_digest'})
            == manifest['manifest_digest']
        )
        engine = SimpleNamespace(
            version_id=manifest['version_id'],
            identity=manifest['identity'],
            profile=ServingProfile(manifest['profile'], manifest['version_id']),
            binding=SimpleNamespace(artifact_digest=manifest['artifact_digest']),
        )
        engines[role] = engine
        parameters[engine.version_id] = {'digest': manifest['parameter_digest']}
        by_worker[canonical(engine.identity)] = manifest
        worker_folder = root / ('c0-worker-rebound' if role == 'rebound' else role + '-worker')
        binding, launch = read(worker_folder / 'binding.json'), read(worker_folder / 'launch.json')
        assert (
            binding['artifact_digest']
            == manifest['artifact_digest']
            == digest(launch['base_identity']['files'])
        )
        assert binding['startup_generation'] == manifest['identity']['startup_generation']
        for line in Path(binding['events_path']).read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'request_released':
                key = (
                    event['worker_instance_id'],
                    event['startup_generation'],
                    event['request_id'],
                )
                assert key not in release_events
                release_events[key] = event
    applications = policy.bind(engines, parameters)
    c0_version = manifests['c0']['version_id']
    base_version = manifests['base']['version_id']
    assert applications[c0_version]['passed'] and not applications[base_version]['passed']
    assert (
        manifests['c0']['parameter_digest']
        == manifests['rebound']['parameter_digest']
        != manifests['base']['parameter_digest']
    )
    assert manifests['c0']['identity'] != manifests['rebound']['identity']

    def released(evidence):
        key = (
            evidence['worker_instance_id'],
            evidence['startup_generation'],
            evidence['request_id'],
        )
        assert key not in seen_release and release_events[key] == evidence
        seen_release.add(key)

    ledgers = {}
    events = []
    for runtime in ('runtime', 'direct-runtime'):
        with copied_database(root / runtime / 'service.sqlite') as db:
            db.row_factory = sqlite3.Row
            ledgers[runtime] = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
            if runtime == 'runtime':
                contract = db.execute('SELECT scope,document FROM release_contract').fetchone()
                assert (
                    contract['scope'] == POOL_SCOPE
                    and json.loads(contract['document'])['application_policy'] == policy.document
                )
                events = [dict(r) for r in db.execute('SELECT * FROM events ORDER BY id')]
    from transformers import AutoTokenizer

    tokenizers = {}
    for role, folder in [('base', 'base-worker'), ('c0', 'c0-worker')]:
        binding = read(root / folder / 'binding.json')
        tokenizers[manifests[role]['artifact_digest']] = AutoTokenizer.from_pretrained(
            binding['model_path'], local_files_only=True, trust_remote_code=False
        )
    tokenize = lambda tokenizer, messages: tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    reservations = {read(p)['arrival_id']: p.parent for p in root.glob('**/reservation.json')}
    row_kinds = Counter()
    dispatched = set()
    qualification_counts = Counter()
    for runtime, ledger in ledgers.items():
        for aid, row in ledger.items():
            assert row['lease_released'] == 1
            if not row['dispatched']:
                assert row['state'] == 'rejected'
                continue
            assert (
                row['state'] == 'completed' and row['cleanup'] == 'confirmed' and row['attempt_id']
            )
            assert row['attempt_id'] not in dispatched
            dispatched.add(row['attempt_id'])
            manifest = by_worker[row['worker']]
            profile = ServingProfile(manifest['profile'], manifest['version_id'])
            body = read(root / runtime / 'requests' / (aid + '.json'))
            wire = read(root / runtime / 'wire-requests' / (aid + '.json'))
            assert (
                wire['payload'] == profile.payload(body)
                and wire['profile_digest'] == manifest['profile_digest']
            )
            assert wire['input_token_ids'] == tokenize(
                tokenizers[manifest['artifact_digest']], body['messages']
            )
            receipt = json.loads(row['observation'])
            evidence = json.loads(row['cleanup_evidence'])
            released(evidence)
            expected = 'chatcmpl-' + row['attempt_id']
            assert evidence['request_id'] in {expected, expected + '-0'}
            assert (
                evidence['worker_instance_id'] == manifest['identity']['worker_instance_id']
                and evidence['startup_generation'] == manifest['identity']['startup_generation']
            )
            u = receipt['usage']
            assert (
                u['prompt_tokens'] == len(wire['input_token_ids'])
                and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens']
            )
            usage.update({k: u[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
            route = json.loads(row['route'])
            kind = route.get('kind', route.get('purpose', 'application'))
            row_kinds[kind] += 1
            finish = receipt.get('finish_reason')
            if finish is None and receipt.get('response'):
                finish = receipt['response']['choices'][0]['finish_reason']
            assert finish in {'stop', 'length'}
            assert (
                evidence['status']
                == {'stop': 'FINISHED_STOPPED', 'length': 'FINISHED_LENGTH_CAPPED'}[finish]
            )
            if kind in {'qualification', 'application_interface_qualification'}:
                qualification_counts[row['worker'], kind] += 1
            if kind == 'application_interface_qualification':
                case = next(
                    c for c in policy.bundle['interface_panel']['cases'] if c['id'] == route['case']
                )
                assert (
                    body['messages'] == case['messages']
                    and body['action_choices'] == case['choices']
                )
                assert finish == 'stop' and receipt['content'] in case['choices']
            if kind == 'application':
                assert (
                    route['qualification_scope'] == POOL_SCOPE
                    and row['version'] == c0_version
                    and route['replica_id'] == 'c0'
                )
                assert route['application_history'] == applications[c0_version]
                policy.validate_request(body)
            if aid in reservations:
                folder = reservations[aid]
                reservation = read(folder / 'reservation.json')
                raw = read(folder / 'response.raw')
                exact = read(folder / 'receipt.json')
                assert raw['id'] == expected and raw['model'] == row['version'] and exact == receipt
                assert reservation['wire'] == wire['payload'] and exact['release'] == evidence
            elif receipt.get('response'):
                assert (
                    receipt['response']['id'] == expected
                    and receipt['response']['model'] == row['version']
                )
            else:
                stream = receipt['events']
                assert stream[-1] == '[DONE]'
                for text in stream[:-1]:
                    event_data = json.loads(text)
                    assert event_data['id'] == expected and event_data['model'] == row['version']
    assert (
        row_kinds['qualification'] == 12 and row_kinds['application_interface_qualification'] == 12
    )
    assert row_kinds['publication_interface_pair'] == 3
    for worker in by_worker:
        assert (
            qualification_counts[worker, 'qualification']
            == qualification_counts[worker, 'application_interface_qualification']
            == 4
        )
    qualifications = list((root / 'runtime/qualifications').glob('*/receipt.json'))
    assert len(qualifications) == 3
    for path in qualifications:
        q = read(path)
        original = read(path.parent / 'transport/receipt.json')
        source = read(path.parent / 'transport/input.json')
        assert (
            q['qualification_scope'] == WORKER_SCOPE
            and q['passed']
            and q['transport_passed']
            and q['interface_passed']
        )
        assert (
            q['transport_receipt'] == original
            and original['passed']
            and original['cases'] == original['passed_cases'] == 4
        )
        assert (
            source['panel'] == initial['transport_panel']
            and len(q['interface_results']) == 4
            and all(r['passed'] for r in q['interface_results'])
        )
    for folder in ('base-bootstrap', 'c0-bootstrap', 'c0-rebound-bootstrap'):
        for label in ('plain', 'stream'):
            response = read(root / folder / (label + '-response.json'))
            assert response['token_counts_match'] and response['content']
            released(response['release'])
            usage.update(
                {
                    k: response['usage'][k]
                    for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                }
            )
    assert set(release_events) == seen_release
    client_rows = {}
    for path in root.glob('calls/*/receipt.json'):
        result = read(path)
        request = read(path.parent / 'request.json')
        aid = result['headers']['x-arrival-id']
        row = ledgers['runtime'][aid]
        assert aid not in client_rows
        client_rows[aid] = (request, result)
        expected_status = request['expected_status']
        expected_status = [expected_status] if isinstance(expected_status, int) else expected_status
        assert (
            result['status'] in expected_status
            and result['client_key'] == request['client_key'] == row['client_key']
        )
        assert json.loads(row['observation'])['http_status'] == result['status']
        if result['status'] == 200:
            route = json.loads(row['route'])
            headers = result['headers']
            obs = json.loads(row['observation'])
            assert headers['x-model-version'] == row['version'] == c0_version
            assert headers['x-replica-id'] == route['replica_id'] == 'c0'
            assert headers['x-worker-generation'] == route['worker_generation']
            assert int(headers['x-route-generation']) == route['route_generation']
            assert headers['x-application-evidence'] == applications[c0_version]['evidence_digest']
            assert (
                headers['x-serving-profile'] == route['profile_digest']
                and headers['x-qualification-scope'] == POOL_SCOPE
            )
            assert read(root / 'runtime/requests' / (aid + '.json')) == {
                **request['body'],
                'model': c0_version,
            }
            assert (
                request['input_token_ids']
                == read(root / 'runtime/wire-requests' / (aid + '.json'))['input_token_ids']
            )
            assert (
                result['content'] == obs['content']
                and result['content'] in request['body']['action_choices']
                and result['finish_reason'] == 'stop'
            )
            assert result['usage'] == obs['usage']
            raw = (path.parent / 'response.raw').read_bytes()
            if request['body']['stream']:
                assert (
                    result['value']['events']
                    == obs['events']
                    == [
                        line[5:].strip()
                        for line in raw.decode().splitlines()
                        if line.startswith('data:')
                    ]
                )
            else:
                assert json.loads(raw) == result['value'] == obs['response']
        else:
            assert row['state'] == 'rejected' and not row['dispatched']
            assert result['value']['error']['code'] == json.loads(row['observation'])['code']
    assert len(ledgers['runtime']) == len(client_rows) + 24
    assert (
        read(root / 'controller-recovery.json')['observed']['route']
        == read(root / 'controller-recovery.json')['expected_route']
    )
    assert read(root / 'outage.json')['base_still_ready'] == {
        'manifest': manifests['base'],
        'ready': True,
    }
    stages = [json.loads(line) for line in (root / 'stages.jsonl').read_text().splitlines()]
    failed_routes = [s for s in stages if s['path'] == '/admin/route' and s['status'] == 409]
    assert {s['body']['operation'] for s in failed_routes} == {'promote', 'canary', 'rollback'}
    assert all(
        s['body']['target'] == base_version
        and s['response']['error']['code'] == 'application_quality_rejected'
        for s in failed_routes
    )
    route_events = [json.loads(e['body']) for e in events if e['kind'] == 'pool_route_changed']
    assert all(e['operation'] in {'promote', 'withdraw'} for e in route_events)
    observed = [e for e in route_events if e['operation'] == 'withdraw' and e['bound_arrivals']]
    assert len(observed) == 1
    for aid in observed[0]['bound_arrivals']:
        request, result = client_rows[aid]
        assert (
            result['status'] == 200
            and int(result['headers']['x-route-generation'])
            == observed[0]['previous']['generation']
        )
        assert result['headers']['x-model-version'] == c0_version
    for version in ('base', 'c0'):
        assert applications[manifests[version]['version_id']]['passed'] is (version == 'c0')
    integration = []
    for path in sorted(root.glob('support/*/record.json')):
        r, env, exit_state = (
            read(path),
            read(path.parent / 'environment.json'),
            read(path.parent / 'exit.json'),
        )
        check_support_source(
            r,
            env,
            read(path.parent / 'launch.json'),
            support_tasks[r['task_id']],
            support_data,
            initial['environment_image'],
        )
        assert initial['environment_image'] == IMAGE
        assert r['record_hash'] == digest({k: v for k, v in r.items() if k != 'record_hash'})
        assert (
            r['purpose'] == 'publication_integration'
            and not r['eligible_for_training']
            and not r['eligible_for_primary_history']
        )
        assert r['task_id'] == env['task_id'] == path.parent.name and not env['reference_enabled']
        assert (
            r['initial'] == env['initial']
            and r['result'] == env['result']
            and len(r['trace']) == len(env['trace']) == r['result']['steps'] <= 100
        )
        assert exit_state['ExitCode'] == 0 and not exit_state['Running']
        for i, (step, raw_step) in enumerate(zip(r['trace'], env['trace'])):
            assert (
                all(step[k] == raw_step[k] for k in ('step', 'before', 'after', 'action'))
                and raw_step['admissible']
            )
            request, response = client_rows[step['receipt']['headers']['x-arrival-id']]
            assert response == step['receipt'] and response['content'] == step['action']
            prepared = StandardActor().prepare(
                r['initial'],
                step['before'],
                r['trace'][:i],
                lambda ms: tokenize(tokenizers[manifests['c0']['artifact_digest']], ms),
            )
            assert (
                request['body']['messages'] == prepared['messages']
                and request['body']['action_choices'] == prepared['choices']
            )
            assert step['context'] == prepared['context']
        integration.append({'task_id': r['task_id'], 'result': r['result']})
    assert len(integration) == 6 and {r['task_id'] for r in integration} == set(initial['supports'])
    pairs = read(root / 'pairs.json')
    direct_receipts = {
        aid: read(folder / 'receipt.json')
        for aid, folder in reservations.items()
        if aid in ledgers['direct-runtime']
    }
    direct_reservations = {
        aid: read(reservations[aid] / 'reservation.json') for aid in direct_receipts
    }
    gateway_wires = {
        aid: read(root / 'runtime/wire-requests' / (aid + '.json'))
        for aid, row in ledgers['runtime'].items()
        if row['dispatched']
    }
    verify_pairs(pairs, direct_receipts, direct_reservations, client_rows, gateway_wires)
    assert file_hash(initial['parity_data']) == initial['parity_data_sha256']
    source_rows = {
        (r['task_id'], r['step']): r
        for line in Path(initial['parity_data']).read_text().splitlines()
        if (r := json.loads(line))
    }
    selected = read(root / 'probe-inputs.json')
    assert len(selected) == 3 and len({r['task_id'] for r in selected}) == 3
    for pair in pairs:
        original = selected[pair['pair'] - 1]
        assert original == source_rows[original['task_id'], original['step']]
        actual = direct_reservations[pair['direct']['arrival_id']]['wire']
        assert actual['messages'] == original['messages'][:-1]
        assert (
            actual['structured_outputs']['choice']
            == json.loads(original['messages'][-2]['content'])['CURRENT_COMMANDS']
        )
    cleanup = read(root / 'cleanup.json')
    assert not cleanup['errors'] and all(
        w['result']['stopped'] and not w['result'].get('remaining') for w in cleanup['workers']
    )
    reservations_total = reservation_count(root)
    assert len(release_events) <= reservations_total == summary['generation_reservations']
    assert (
        reservations_total + prior_count
        == summary['package_generation_reservations']
        <= initial['generation_cap']
        == 680
    )
    result = {
        'verified': True,
        'scope': POOL_SCOPE,
        'generation_calls': len(release_events),
        'usage': dict(usage),
        'auditor_sha256': file_hash(__file__),
        'gateway_arrivals': len(ledgers['runtime']),
        'gateway_states': dict(Counter(r['state'] for r in ledgers['runtime'].values())),
        'row_kinds': dict(row_kinds),
        'client_calls': len(client_rows),
        'source_files': len(initial['source_hashes']),
        'quality_rejections': len(failed_routes),
        'inflight_bindings_preserved': len(observed[0]['bound_arrivals']),
        'generation_reservations': reservations_total,
        'prior_failed_reservations': prior_count,
        'package_generation_reservations': reservations_total + prior_count,
        'integration': integration,
        'integration_successes': sum(r['result']['success'] for r in integration),
        'support_actions': sum(r['result']['steps'] for r in integration),
        'cross_parameter_application_rollback': False,
        'candidate_artifact_digest': manifests['c0']['artifact_digest'],
        'different_parameter_base_blocked': True,
        'new_C0_generation': manifests['rebound']['identity'],
        'publication_state': 'local runtime verified; all owned services stopped',
        'evidence_scope': 'source quality retained23/24; integration results are not new research samples or cluster evidence',
    }
    assert summary['support_integration'] == {
        'episodes': 6,
        'successes': result['integration_successes'],
        'actions': result['support_actions'],
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.run, a.output), ensure_ascii=False, indent=2))
