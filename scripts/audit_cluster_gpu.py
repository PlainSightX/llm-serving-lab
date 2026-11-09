"""离线复核交付镜像的真实GPU回执、应用绑定和最终资源释放。"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from serving_lab.application_release import load_policy
from serving_lab.execution import write_once
from serving_lab.lab import canonical, digest, file_hash
from serving_lab.profile import ServingProfile
from compare_runtime_baseline import copied_database
from prepare_cluster import read_bundle


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(run, bundle_path, output):
    root, package = Path(run).resolve(), Path(bundle_path).resolve()
    bundle, _ = read_bundle(package)
    initial, summary = read(root / 'input.json'), read(root / 'summary.json')
    assert initial['bundle_sha256'] == file_hash(package / 'bundle.json')
    assert read(root / 'orchestrator-source.json')['sha256'] == file_hash(
        root / 'source-probe_cluster_gpu.py'
    )
    policy = load_policy(package / 'config/application-evidence.json', bundle['evidence_sha256'])
    engines, parameters, manifests, releases = {}, {}, {}, {}
    bootstrap = []
    for slot in ('a', 'b'):
        folders = list((root / ('worker-' + slot)).glob('generation-*'))
        assert len(folders) == 1
        generation = folders[0]
        manifest = read(generation / 'manifest.json')
        assert (
            digest({key: value for key, value in manifest.items() if key != 'manifest_digest'})
            == manifest['manifest_digest']
        )
        binding = read(generation / 'native/binding.json')
        launch = read(generation / 'native/launch.json')
        assert (
            binding['artifact_digest']
            == manifest['artifact_digest']
            == bundle['model_artifact_digest']
        )
        assert manifest['identity']['binding_sha256'] == file_hash(
            generation / 'native/binding.json'
        )
        assert (
            launch['versions']['vllm'] == '0.11.2' and launch['versions']['torch'] == '2.9.0+cu128'
        )
        assert launch['versions']['transformers'] == '4.57.1'
        assert read(generation / 'cleanup.json')['engine']['stopped']
        engine = SimpleNamespace(
            version_id=manifest['version_id'],
            identity=manifest['identity'],
            binding=SimpleNamespace(artifact_digest=manifest['artifact_digest']),
            profile=ServingProfile(manifest['profile'], manifest['version_id']),
        )
        engines[slot] = engine
        parameters[engine.version_id] = {'digest': manifest['parameter_digest']}
        manifests[canonical(engine.identity)] = manifest
        for line in (
            (generation / 'native/release-events.jsonl').read_text(encoding='utf-8').splitlines()
        ):
            event = json.loads(line)
            if event.get('event') == 'request_released':
                key = (
                    event['worker_instance_id'],
                    event['startup_generation'],
                    event['request_id'],
                )
                assert key not in releases
                releases[key] = event
        for label in ('plain', 'stream'):
            bootstrap.append(read(generation / 'bootstrap' / (label + '-response.json')))
    assert len({engine.identity['worker_instance_id'] for engine in engines.values()}) == 2
    applications = policy.bind(engines, parameters)
    assert all(row['passed'] for row in applications.values())
    seen = set()

    def released(event):
        key = (event['worker_instance_id'], event['startup_generation'], event['request_id'])
        assert key not in seen and releases[key] == event
        assert event['request_absent'] and not any(event['after_blocks'])
        seen.add(key)

    usage = Counter()
    for response in bootstrap:
        assert response['done'] and response['token_counts_match']
        released(response['release'])
        usage.update(
            {
                key: response['usage'][key]
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            }
        )
    with copied_database(root / 'runtime/controller/service.sqlite') as database:
        database.row_factory = sqlite3.Row
        rows = {row['id']: dict(row) for row in database.execute('SELECT * FROM arrivals')}
    kinds = Counter()
    for arrival, row in rows.items():
        assert (
            row['state'] == 'completed'
            and row['cleanup'] == 'confirmed'
            and row['lease_released'] == 1
        )
        manifest = manifests[row['worker']]
        profile = ServingProfile(manifest['profile'], manifest['version_id'])
        body = read(root / 'runtime/controller/requests' / (arrival + '.json'))
        wire = read(root / 'runtime/controller/wire-requests' / (arrival + '.json'))
        assert wire['profile_digest'] == profile.digest and wire['payload'] == profile.payload(body)
        observation = json.loads(row['observation'])
        event = json.loads(row['cleanup_evidence'])
        released(event)
        assert event['request_id'] in {
            'chatcmpl-' + row['attempt_id'],
            'chatcmpl-' + row['attempt_id'] + '-0',
        }
        assert observation['usage']['prompt_tokens'] == len(wire['input_token_ids'])
        route = json.loads(row['route'])
        kind = route.get('kind', 'application')
        kinds[kind] += 1
        if kind == 'application':
            policy.validate_request(body)
            assert observation['content'] in body['action_choices']
            assert route['application_history'] == applications[row['version']]
        usage.update(
            {
                key: observation['usage'][key]
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            }
        )
    assert kinds == {'qualification': 8, 'application_interface_qualification': 8, 'application': 4}
    assert set(releases) == seen and len(seen) == 24
    qualifications = list((root / 'runtime/controller/qualifications').glob('*/receipt.json'))
    assert len(qualifications) == 2
    qualified_workers = set()
    for path in qualifications:
        qualification = read(path)
        assert (
            qualification['passed']
            and qualification['transport_passed']
            and qualification['interface_passed']
        )
        assert len(qualification['interface_results']) == 4
        qualified_workers.add(canonical(qualification['worker']))
    assert qualified_workers == set(manifests)
    first, second = read(root / 'first-request.json'), read(root / 'second-request.json')
    assert first['headers']['x-replica-id'] != second['headers']['x-replica-id']
    assert all(item['pending'] == 0 for item in read(root / 'final-state.json')['replicas'])
    assert read(root / 'load/summary.json')['valid'] == 2
    cleanup = read(root / 'cleanup.json')
    assert cleanup['network_removed'] and all(
        item['state']['ExitCode'] == 0 and not item['state']['Running']
        for item in cleanup['containers']
    )
    assert (
        summary['package_reservations']
        == summary['reservations'] + summary['prior_reservations']
        <= 64
    )
    result = {
        'verified': True,
        'generation_calls': len(seen),
        'gateway_arrivals': len(rows),
        'usage': dict(usage),
        'row_kinds': dict(kinds),
        'all_leases_released': True,
        'container_exits': 3,
        'package_reservations': summary['package_reservations'],
        'scope': 'local shared GPU container and mTLS integration',
        'auditor_sha256': file_hash(__file__),
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.bundle, args.output), ensure_ascii=False, indent=2))
