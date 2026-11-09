"""以一致数据库、客户端原始响应和原生释放独立重算N1跨参数发布。"""

import argparse
from collections import Counter
import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from serving_lab.application_release import load_policy
from serving_lab.lab import digest, file_hash, inspect_base
from serving_lab.parameters import parameter_identity
from serving_lab.profile import ServingProfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def same_output(left, right):
    return (
        left['content'] == right['content']
        and left['finish_reason'] == right['finish_reason']
        and all(
            left['usage'][key] == right['usage'][key]
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
        )
    )


def parse_response(raw, stream, expected_id, model):
    if stream:
        frames = [
            line[5:].strip() for line in raw.decode().splitlines() if line.startswith('data:')
        ]
        if not frames or frames[-1] != '[DONE]':
            raise ValueError('流缺少结束标记')
        values = [json.loads(s) for s in frames[:-1]]
        content = ''
        finish = None
        usage = None
        for value in values:
            if value.get('id') != expected_id or value.get('model') != model or 'error' in value:
                raise ValueError('流请求或模型混淆')
            if value.get('usage'):
                usage = value['usage']
            for choice in value.get('choices', []):
                content += choice.get('delta', {}).get('content') or ''
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
        return content, finish, usage
    value = json.loads(raw)
    if value.get('id') != expected_id or value.get('model') != model or len(value['choices']) != 1:
        raise ValueError('响应请求或模型混淆')
    choice = value['choices'][0]
    return choice['message']['content'], choice['finish_reason'], value['usage']


def audit(root, output):
    from transformers import AutoTokenizer

    root = Path(root).resolve()
    control = root / 'control'
    pub = control / 'gateway-state/publication-b'
    initial, summary = read(pub / 'input.json'), read(pub / 'summary.json')
    source = read(control / 'publication-input.json')
    assert initial['source'] == source and initial['input_sha256'] == file_hash(
        control / 'publication-input.json'
    )
    script = control / 'probe_parameter_publication_b.py'
    assert initial['script_sha256'] == file_hash(script)
    spec = importlib.util.spec_from_file_location('frozen_publication_probe', script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    policy = load_policy(control / 'application-evidence.json', source['bundle_sha256'])
    assert read(control / 'gateway-state/publication-a/failure.json')['reserved'] == 0
    manifests = {}
    engines = {}
    parameters = {}
    native = {}
    native_order = {}
    tokenizers = {}
    repo = Path(__file__).resolve().parents[1]
    models = {
        'a': repo / 'artifacts/q2-coldstart-training-20261103-b/merged',
        'b': repo / 'artifacts/n1-20261105-a/cloud-training/training-b/merged',
    }
    for rid in ('a', 'b'):
        manifest = read(control / f'worker-{rid}-manifest.json')
        folder = root / f'worker-{rid}'
        assert manifest == read(folder / 'manifest.json')
        assert manifest['manifest_digest'] == digest(
            {k: v for k, v in manifest.items() if k != 'manifest_digest'}
        )
        assert manifest['identity']['binding_sha256'] == file_hash(folder / 'native/binding.json')
        binding, launch = read(folder / 'native/binding.json'), read(folder / 'native/launch.json')
        assert (
            launch['versions']['vllm'] == '0.11.2' and launch['versions']['torch'] == '2.9.0+cu128'
        )
        assert (
            binding['artifact_digest']
            == manifest['artifact_digest']
            == digest(inspect_base(models[rid])['files'])
        )
        assert manifest['parameter_digest'] == parameter_identity(models[rid])['digest']
        p = ServingProfile(manifest['profile'], manifest['version_id'])
        assert p.digest == manifest['profile_digest']
        engine = SimpleNamespace(
            version_id=manifest['version_id'],
            identity=manifest['identity'],
            profile=p,
            binding=SimpleNamespace(artifact_digest=manifest['artifact_digest']),
        )
        engines[rid] = engine
        parameters[engine.version_id] = {'digest': manifest['parameter_digest']}
        manifests[digest(engine.identity)] = manifest
        tokenizers[engine.version_id] = AutoTokenizer.from_pretrained(
            models[rid], local_files_only=True, trust_remote_code=False
        )
        ordered = []
        for line in (
            (folder / 'native/release-events.jsonl').read_text(encoding='utf-8').splitlines()
        ):
            event = json.loads(line)
            if event.get('event') == 'request_released':
                key = (
                    event['worker_instance_id'],
                    event['startup_generation'],
                    event['request_id'],
                )
                assert key not in native
                native[key] = event
                ordered.append(key)
        native_order[rid] = ordered
    assert len({e.identity['processes'][0]['boot_id'] for e in engines.values()}) == 2
    applications = policy.bind(engines, parameters)
    assert all(x['passed'] for x in applications.values())
    assert len({m['parameter_digest'] for m in manifests.values()}) == 2
    rows_by_db = {}
    seen = set()
    usage = Counter()
    kinds = Counter()
    for dbname, dbfile, runtime in [
        (
            'controller',
            control / 'controller-consistent.sqlite',
            control / 'gateway-state/controller',
        ),
        (
            'direct',
            control / 'publication-b-direct-runtime-consistent.sqlite',
            pub / 'direct-runtime',
        ),
    ]:
        with sqlite3.connect(dbfile) as db:
            db.row_factory = sqlite3.Row
            rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
        rows_by_db[dbname] = rows
        for row in rows.values():
            obs = json.loads(row['observation'])
            route = json.loads(row['route'])
            assert row['lease_released'] == 1
            if not row['dispatched']:
                assert (
                    row['state'] == 'rejected'
                    and obs['code'] == 'application_version_not_published'
                )
                continue
            assert row['state'] == 'completed' and row['cleanup'] == 'confirmed'
            manifest = manifests[digest(json.loads(row['worker']))]
            profile = ServingProfile(manifest['profile'], manifest['version_id'])
            body, wire = (
                read(runtime / 'requests' / (row['id'] + '.json')),
                read(runtime / 'wire-requests' / (row['id'] + '.json')),
            )
            assert (
                wire['payload'] == profile.payload(body)
                and wire['profile_digest'] == profile.digest
            )
            ids = tokenizers[row['version']].apply_chat_template(
                body['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            assert wire['input_token_ids'] == ids and obs['usage']['prompt_tokens'] == len(ids)
            assert row['input_hash'] == digest(body)
            assert 0 < obs['usage']['completion_tokens'] <= body['max_tokens']
            assert obs['usage']['total_tokens'] == len(ids) + obs['usage']['completion_tokens']
            event = json.loads(row['cleanup_evidence'])
            key = (event['worker_instance_id'], event['startup_generation'], event['request_id'])
            assert (
                key not in seen
                and native[key] == event
                and key[:2]
                == (
                    manifest['identity']['worker_instance_id'],
                    manifest['identity']['startup_generation'],
                )
            )
            assert key[2] in {
                'chatcmpl-' + row['attempt_id'],
                'chatcmpl-' + row['attempt_id'] + '-0',
            }
            assert event['request_absent'] and not any(event['after_blocks'])
            if obs.get('release'):
                assert obs['release'] == event
            if obs.get('response'):
                response = obs['response']
                if 'choices' in response:
                    assert (
                        response['id'] == 'chatcmpl-' + row['attempt_id']
                        and response['model'] == row['version']
                    )
            if obs.get('raw_response'):
                content, finish, tokens = parse_response(
                    obs['raw_response'].encode(),
                    body['stream'],
                    'chatcmpl-' + row['attempt_id'],
                    row['version'],
                )
                assert same_output(
                    {'content': content, 'finish_reason': finish, 'usage': tokens}, obs
                )
            finish = obs.get('finish_reason', 'stop')
            assert (
                event['status']
                == {'stop': 'FINISHED_STOPPED', 'length': 'FINISHED_LENGTH_CAPPED'}[finish]
            )
            seen.add(key)
            usage.update(
                {k: obs['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            )
            kind = route.get('kind', 'direct' if dbname == 'direct' else 'application')
            kinds[kind] += 1
            if kind == 'application':
                policy.validate_request(body)
                assert obs['content'] in body['action_choices']
                assert route['application_history'] == applications[row['version']]
    # 同一worker本轮首个匹配释放之后，不允许多出未记账的生成。
    for rid, ordered in native_order.items():
        positions = [i for i, key in enumerate(ordered) if key in seen]
        assert positions and all(key in seen for key in ordered[min(positions) :])
    controller = rows_by_db['controller']
    by_key = {r['client_key']: r for r in controller.values() if r['client_key']}
    raw_paths = [
        (p, 'controller')
        for p in (control / 'gateway-state/controller/qualifications').rglob('response.raw')
    ]
    raw_paths += [(p, 'direct') for p in (pub / 'direct').rglob('response.raw')]
    assert len(raw_paths) == 16
    for path, scope in raw_paths:
        reservation = read(path.with_name('reservation.json'))
        receipt = read(path.with_name('receipt.json'))
        row = rows_by_db[scope][reservation['arrival_id']]
        assert json.loads(row['observation']) == receipt
        content, finish, tokens = parse_response(
            path.read_bytes(), False, 'chatcmpl-' + row['attempt_id'], row['version']
        )
        assert same_output({'content': content, 'finish_reason': finish, 'usage': tokens}, receipt)
    requests = list((pub / 'requests').glob('*/request.json'))
    assert len(requests) == len(by_key) == 47
    receipts = {}
    specs = {
        e.version_id: {
            'profile_digest': e.profile.digest,
            'evidence_digest': applications[e.version_id]['evidence_digest'],
        }
        for e in engines.values()
    }
    for path in requests:
        request, receipt = read(path), read(path.with_name('receipt.json'))
        row = by_key[request['client_key']]
        assert (
            receipt['client_key'] == request['client_key']
            and receipt['status'] == request['expected_status']
        )
        receipts[receipt['client_key']] = receipt
        if receipt['status'] != 200:
            assert not row['dispatched']
            continue
        body = request['body']
        actual = (path.parent / 'response.raw').read_bytes()
        content, finish, tokens = parse_response(
            actual, body['stream'], 'chatcmpl-' + row['attempt_id'], row['version']
        )
        assert (content, finish, tokens) == (
            receipt['content'],
            receipt['finish_reason'],
            receipt['usage'],
        )
        assert (
            receipt['version'] == row['version']
            and int(receipt['route_generation']) == json.loads(row['route'])['route_generation']
        )
        assert receipt['headers']['x-replica-id'] == json.loads(row['route'])['replica_id']
        assert (
            receipt['headers']['x-worker-generation']
            == json.loads(row['worker'])['startup_generation']
        )
        probe.validate_response(
            receipt,
            {**body, 'input_token_ids': request['input_token_ids']},
            specs,
            request['expected_version'],
        )
    direct_by_attempt = {r['attempt_id']: r for r in rows_by_db['direct'].values()}
    for pair in summary['pairs']:
        reference = json.loads(direct_by_attempt[pair['direct_attempt']]['observation'])
        client = receipts[pair['client_key']]
        assert client['version'] == pair['version'] and same_output(reference, client)
    with sqlite3.connect(control / 'controller-consistent.sqlite') as db:
        db.row_factory = sqlite3.Row
        events = [
            dict(r) for r in db.execute('SELECT id,arrival_id,kind,body FROM events ORDER BY id')
        ]
        final = dict(db.execute('SELECT * FROM pool_route WHERE id=1').fetchone())
    changes = [json.loads(e['body']) for e in events if e['kind'] == 'pool_route_changed']
    assert [x['operation'] for x in changes] == [
        'promote',
        'canary',
        'promote',
        'promote',
        'rollback',
    ]
    assert [x['current']['generation'] for x in changes] == [1, 2, 3, 4, 5]
    batch = {
        r['id']
        for r in controller.values()
        if r['client_key']
        in {q['client_key'] for q in receipts.values() if q['label'].startswith('queued-')}
    }
    proof = probe.queued_cutover_proof(
        events, batch, source['versions']['c0'], source['versions']['c1'], 4
    )
    assert (
        len(proof) >= 4
        and proof == read(pub / 'queued-at-cutover-proof.json')
        and len(proof) == summary['queued_bindings_preserved']
    )
    canaries = Counter(r['version'] for r in receipts.values() if r['label'].startswith('canary-'))
    assert (
        dict(canaries) == summary['canary_counts']
        and sum(canaries.values()) == 24
        and set(canaries) == set(source['versions'].values())
    )
    assert (
        changes[1]['current']['percentage'] == 50
        and changes[1]['current']['canary'] == source['versions']['c1']
    )
    for receipt in receipts.values():
        if receipt['label'].startswith('canary-'):
            row = by_key[receipt['client_key']]
            bucket = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'big') % 100
            assert (
                row['version'] == source['versions']['c1' if bucket < 50 else 'c0']
                and int(receipt['route_generation']) == 2
            )
    assert (
        final == summary['final_route']
        and final['active'] == source['versions']['c0']
        and final['generation'] == 5
    )
    management = [read(p) for p in sorted((pub / 'management').glob('*.json'))]
    denied = [x for x in management if x['status'] == 503]
    assert len(denied) == 1 and denied[0]['body']['operation'] == 'rollback'
    qualifications = [
        read(p)
        for p in (control / 'gateway-state/controller/qualifications').glob('*/receipt.json')
    ]
    assert len(qualifications) == 3 and all(
        q['passed'] and q['transport_cases'] == q['interface_cases'] == 4 for q in qualifications
    )
    assert kinds == Counter(
        {
            'qualification': 12,
            'application_interface_qualification': 12,
            'application': 46,
            'direct': 4,
        }
    )
    for item in read(pub / 'unqualified-parameter-rejection.json'):
        assert (
            item['decision']
            == policy.summary(policy.bundle['certificates'][item['artifact_digest']])
            and not item['decision']['passed']
        )
    reservations = [json.loads(s) for s in (pub / 'reservations.jsonl').read_text().splitlines()]
    assert sum(r['count'] for r in reservations) == summary['generation_reservations'] == 75
    assert source['prior_n1_generations'] + 75 <= 11000 and len(seen) == 74
    result = {
        'raw_verified': True,
        'controller_arrivals': len(controller),
        'direct_calls': len(direct_by_attempt),
        'actual_generations': len(seen),
        'rejected_before_dispatch': 1,
        'unreleased_leases': 0,
        'hot_rollback': True,
        'cold_reload': False,
        'qualified_parameter_versions': 2,
        'canary_counts': dict(canaries),
        'queued_after_cutover': len(proof),
        'final_route': final,
        'usage': dict(usage),
        'canary_hash_rule_verified': True,
        'script_sha256': file_hash(__file__),
    }
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.root, a.output)
