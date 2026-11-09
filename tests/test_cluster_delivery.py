"""交付入口核验节点网络边界，并通过真实mTLS消费管理接口。"""

import asyncio
import json
from pathlib import Path
import shutil
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deployment'))


def bundle(tmp_path):
    from serving_lab.lab import file_hash

    root = tmp_path / 'bundle'
    (root / 'images').mkdir(parents=True)
    manifest = {
        'items': [
            {
                'kind': 'NetworkPolicy',
                'metadata': {'name': 'gateway'},
                'spec': {'ingress': [{'from': [{'ipBlock': {'cidr': '192.0.2.1/32'}}]}]},
            }
        ]
    }
    (root / 'manifests.json').write_text(json.dumps(manifest), encoding='utf-8')
    (root / 'bundle.json').write_text(
        json.dumps({'manifests_sha256': file_hash(root / 'manifests.json')}), encoding='utf-8'
    )
    (root / 'images/images.json').write_text(
        json.dumps({'control': {'tag': 'fixture:fixed'}}), encoding='utf-8'
    )
    return root


def test_node_plan_rejects_overlap_and_shell_metacharacters(tmp_path):
    from render_node_plan import render

    root = bundle(tmp_path)
    with pytest.raises(ValueError, match='网段'):
        render(
            root,
            'control',
            '10.72.0.2',
            None,
            '/opt/lsl-package',
            tmp_path / 'bad',
            '10.72.0.3',
            '10.72.0.0/16',
        )
    with pytest.raises(ValueError, match='路径'):
        render(root, 'control', '10.10.0.2', None, '/opt/a;command', tmp_path / 'bad', '10.10.0.3')
    output = tmp_path / 'valid'
    result = render(root, 'control', '10.10.0.2', None, '/opt/lsl-package', output, '10.10.0.3')
    assert result['mode'] == 'rendered_only'
    assert json.loads((output / 'k3s-config.yaml').read_text())['secrets-encryption'] is True
    assert (
        json.loads((output / 'bound-manifests.json').read_text())['items'][0]['spec']['ingress'][0][
            'from'
        ][0]['ipBlock']['cidr']
        == '10.10.0.3/32'
    )


def test_load_plan_maps_gateway_inside_the_container(tmp_path):
    from render_node_plan import render

    root = bundle(tmp_path)
    output = tmp_path / 'loadgen'
    render(root, 'loadgen', '10.10.0.13', '10.10.0.10', '/opt/lsl-package', output)
    command = (output / 'load-command.sh').read_text(encoding='utf-8')
    assert '--add-host gateway.lsl.internal:10.10.0.10' in command
    assert b'\r\n' not in (output / 'load-command.sh').read_bytes()
    assert b'\r\n' not in (output / 'setup.sh').read_bytes()


@pytest.mark.parametrize('role', ['gpu-a', 'gpu-b'])
@pytest.mark.parametrize('tag', [':fixed', ''])
def test_gpu_node_plan_imports_anonymous_digest_image_with_repository_name(tmp_path, role, tag):
    from render_node_plan import render

    root = bundle(tmp_path)
    reference = 'fixture.registry:5000/nvidia/device-plugin' + tag + '@sha256:' + 'e' * 64
    images_path = root / 'images/images.json'
    images = json.loads(images_path.read_text(encoding='utf-8'))
    images['device-plugin'] = {'tag': reference}
    images_path.write_text(json.dumps(images), encoding='utf-8')
    output = tmp_path / role
    render(root, role, '10.10.0.11', '10.10.0.10', '/opt/lsl-package', output)
    commands = (output / 'setup.sh').read_text(encoding='utf-8')
    # 摘要导出的OCI归档可以没有名称，真实CRI需要可按仓库摘要查找的镜像。
    imported = (
        '/usr/local/bin/k3s ctr -n k8s.io images import --platform linux/amd64 '
        '--digests --base-name fixture.registry:5000/nvidia/device-plugin images/device-plugin.tar.gz'
    )
    inspected = (
        f'/usr/local/bin/k3s crictl inspecti {reference} > node-plan/device-plugin-image.json'
    )
    assert imported in commands
    assert inspected in commands
    assert (
        commands.index('sh assets/install-k3s.sh agent')
        < commands.index(imported)
        < commands.index(inspected)
    )


def test_loadgen_preflight_requires_local_docker_daemon(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import cluster_node_preflight as preflight

    monkeypatch.setattr(preflight.shutil, 'which', lambda name: None)
    monkeypatch.setattr(preflight, 'command', lambda args: None)
    result = preflight.collect('loadgen', tmp_path)
    assert result['checks']['docker_cli'] is False and result['checks']['docker_daemon'] is False
    calls = []
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: '/usr/bin/docker')

    def execute(args):
        calls.append(args)
        return '29.8.0' if args[0] == 'docker' else None

    monkeypatch.setattr(preflight, 'command', execute)
    result = preflight.collect('loadgen', tmp_path)
    assert result['checks']['docker_daemon'] is True
    assert any(args[:3] == ['docker', '--host', 'unix:///var/run/docker.sock'] for args in calls)


@pytest.mark.skipif(not shutil.which('openssl'), reason='真实TLS握手需要临时证书')
async def test_standalone_admin_client_reads_credentials_without_command_line_values(tmp_path):
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from test_cluster_tls import certificates
    from serving_lab.tls import listener_options
    import cluster_admin

    tls = certificates(tmp_path / 'tls')
    root = tmp_path / 'credentials'
    root.mkdir()
    for source, name in (('ca', 'ca.pem'), ('cert', 'client.pem'), ('key', 'client.key')):
        shutil.copyfile(tls[source], root / name)
    (root / 'admin-token').write_text('test-only-admin-token')
    app = FastAPI()

    @app.get('/admin/releases')
    def status(request: Request):
        if request.headers.get('authorization') != 'Bearer test-only-admin-token':
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        return {'route': {'generation': 0}}

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, **listener_options('127.0.0.1', port, **tls), log_level='error')
    )
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        result = await asyncio.to_thread(
            cluster_admin.request, f'https://localhost:{port}', root, '/admin/releases'
        )
        assert result == {'route': {'generation': 0}}
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
