"""部署产物必须保持独立节点、持久状态和有界身份交接。"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_real_manifests_bind_gpu_slots_and_keep_native_port_private():
    from prepare_cluster import workloads

    result = workloads('lsl-r5', 'control:fixed', 'worker:fixed', '192.0.2.1/32')
    deployments = {
        item['metadata']['name']: item for item in result if item['kind'] == 'Deployment'
    }
    assert set(deployments) == {'worker-a', 'worker-b', 'gateway'}
    for slot in ('a', 'b'):
        spec = deployments['worker-' + slot]['spec']
        pod = spec['template']['spec']
        assert spec['replicas'] == 1 and spec['strategy']['type'] == 'Recreate'
        assert pod['nodeSelector']['serving-lab/slot'] == slot
        assert pod['runtimeClassName'] == 'nvidia'
        assert pod['containers'][0]['resources']['limits']['nvidia.com/gpu'] == 1
        assert [port['containerPort'] for port in pod['containers'][0]['ports']] == [8443]
        assert not pod['automountServiceAccountToken']
    gateway = deployments['gateway']['spec']['template']['spec']
    assert gateway['nodeSelector']['serving-lab/role'] == 'control'
    assert gateway['initContainers'][0]['command'][1].endswith('collect_worker_manifests.py')
    assert any(
        volume.get('hostPath', {}).get('path') == '/var/lib/llm-serving-lab/gateway'
        for volume in gateway['volumes']
    )
    policy = next(
        item
        for item in result
        if item['kind'] == 'NetworkPolicy' and item['metadata']['name'] == 'gateway'
    )
    assert any(
        source.get('ipBlock', {}).get('cidr') == '192.0.2.1/32'
        for source in policy['spec']['ingress'][0]['from']
    )


def test_worker_supervision_remains_discoverable_when_native_is_unready():
    from prepare_cluster import workloads

    documents = workloads('lsl-r5', 'control:fixed', 'worker:fixed', '192.0.2.1/32')
    services = {
        item['metadata']['name']: item['spec'] for item in documents if item['kind'] == 'Service'
    }
    assert services['worker-a']['publishNotReadyAddresses'] is True
    assert services['worker-b']['publishNotReadyAddresses'] is True
    assert not services['gateway'].get('publishNotReadyAddresses', False)


def test_bundle_consumer_reads_utf8_and_checks_original_digest(tmp_path):
    import pytest
    from serving_lab.lab import file_hash
    from prepare_cluster import read_bundle

    manifest = tmp_path / 'manifests.json'
    manifest.write_text(
        json.dumps({'items': [{'说明': '中文配置'}]}, ensure_ascii=False), encoding='utf-8'
    )
    (tmp_path / 'bundle.json').write_text(
        json.dumps({'manifests_sha256': file_hash(manifest)}), encoding='utf-8'
    )
    bundle, documents = read_bundle(tmp_path)
    assert documents[0]['说明'] == '中文配置'
    (tmp_path / 'retired.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='退役'):
        read_bundle(tmp_path)
    (tmp_path / 'retired.json').unlink()
    manifest.write_text('{changed}', encoding='utf-8')
    with pytest.raises(ValueError, match='摘要'):
        read_bundle(tmp_path)


def test_bundle_consumer_rejects_stale_secret_without_echoing_contents(tmp_path):
    import base64
    import pytest
    from serving_lab.lab import file_hash
    from prepare_cluster import read_bundle

    secret = tmp_path / 'secrets/worker-a/worker-token'
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b'test-only-original')
    manifest = tmp_path / 'manifests.json'
    manifest.write_text(
        json.dumps(
            {
                'items': [
                    {
                        'kind': 'Secret',
                        'metadata': {'name': 'worker-a'},
                        'data': {'worker-token': base64.b64encode(secret.read_bytes()).decode()},
                    }
                ]
            }
        ),
        encoding='utf-8',
    )
    (tmp_path / 'bundle.json').write_text(
        json.dumps({'manifests_sha256': file_hash(manifest)}), encoding='utf-8'
    )
    read_bundle(tmp_path)
    secret.write_bytes(b'test-only-changed')
    with pytest.raises(ValueError) as error:
        read_bundle(tmp_path)
    assert 'test-only' not in str(error.value)
