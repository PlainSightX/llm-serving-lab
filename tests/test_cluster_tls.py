"""远端监听需要双向TLS，客户端必须校验服务端身份。"""

import ssl
import asyncio
import shutil
import socket
import subprocess

import pytest


def test_remote_listener_requires_complete_mtls_material(tmp_path):
    from serving_lab.tls import listener_options

    assert listener_options('127.0.0.1', 19382) == {'host': '127.0.0.1', 'port': 19382}
    for options in ({}, {'cert': 'missing', 'key': 'missing'}, {'ca': 'missing'}):
        with pytest.raises(ValueError):
            listener_options('0.0.0.0', 19382, **options)


def test_client_tls_rejects_missing_or_partial_identity(tmp_path):
    from serving_lab.tls import client_context

    with pytest.raises(ValueError):
        client_context({'ca': str(tmp_path / 'absent')})
    with pytest.raises(ValueError):
        client_context({'ca': 'a', 'cert': 'b'})
    with pytest.raises(ValueError):
        client_context({'verify': False})


def certificates(directory):
    directory.mkdir()

    def openssl(*arguments):
        subprocess.run(['openssl', *arguments], cwd=directory, check=True, capture_output=True)

    openssl(
        'req',
        '-x509',
        '-newkey',
        'ec',
        '-pkeyopt',
        'ec_paramgen_curve:P-256',
        '-nodes',
        '-keyout',
        'ca.key',
        '-out',
        'ca.pem',
        '-days',
        '1',
        '-subj',
        '/CN=local-test-ca',
        '-addext',
        'basicConstraints=critical,CA:TRUE',
    )
    openssl(
        'req',
        '-new',
        '-newkey',
        'ec',
        '-pkeyopt',
        'ec_paramgen_curve:P-256',
        '-nodes',
        '-keyout',
        'leaf.key',
        '-out',
        'leaf.csr',
        '-subj',
        '/CN=localhost',
    )
    (directory / 'extensions').write_text(
        'subjectAltName=DNS:localhost\nextendedKeyUsage=serverAuth,clientAuth\nbasicConstraints=CA:FALSE\n'
    )
    openssl(
        'x509',
        '-req',
        '-in',
        'leaf.csr',
        '-CA',
        'ca.pem',
        '-CAkey',
        'ca.key',
        '-CAcreateserial',
        '-out',
        'leaf.pem',
        '-days',
        '1',
        '-extfile',
        'extensions',
    )
    return {
        'ca': str(directory / 'ca.pem'),
        'cert': str(directory / 'leaf.pem'),
        'key': str(directory / 'leaf.key'),
    }


@pytest.mark.skipif(not shutil.which('openssl'), reason='真实TLS握手需要openssl生成临时证书')
async def test_remote_engine_consumes_mtls_and_rejects_wrong_ca_name_and_token(tmp_path):
    import uvicorn
    import httpx
    from serving_lab.tls import listener_options
    from serving_lab.worker_protocol import create_worker_app, worker_manifest, RemoteEngine
    from test_replicas import workers

    tls = certificates(tmp_path / 'trusted')
    untrusted = certificates(tmp_path / 'untrusted')
    engine = workers()['a1']
    manifest = worker_manifest(engine, 'test-parameters')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_worker_app(engine, 'test-parameters', 'worker-token-long'),
            **listener_options('127.0.0.1', port, **tls),
            log_level='error',
            lifespan='on',
        )
    )
    task = asyncio.create_task(server.serve())
    remotes = []
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        remote = RemoteEngine(f'https://localhost:{port}', 'worker-token-long', manifest, tls=tls)
        remotes.append(remote)
        assert await remote.ready()
        no_identity = httpx.AsyncClient(
            base_url=f'https://localhost:{port}',
            verify=ssl.create_default_context(cafile=tls['ca']),
            trust_env=False,
        )
        missing_certificate = RemoteEngine(
            f'https://localhost:{port}', 'worker-token-long', manifest, client=no_identity
        )
        remotes.append(missing_certificate)
        assert not await missing_certificate.ready()
        for endpoint, token, material in (
            (f'https://localhost:{port}', 'worker-token-long', untrusted),
            (f'https://127.0.0.1:{port}', 'worker-token-long', tls),
            (f'https://localhost:{port}', 'wrong-worker-token', tls),
        ):
            invalid = RemoteEngine(endpoint, token, manifest, tls=material)
            remotes.append(invalid)
            assert not await invalid.ready()
    finally:
        for remote in remotes:
            await remote.close()
        server.should_exit = True
        await asyncio.wait_for(task, 5)


@pytest.mark.skipif(not shutil.which('openssl'), reason='真实TLS握手需要openssl生成临时证书')
async def test_existing_load_window_consumes_tls_configuration(tmp_path):
    import json
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from serving_lab.benchmark import window
    from serving_lab.tls import listener_options

    tls = certificates(tmp_path / 'load')
    app = FastAPI()

    @app.post('/v1/chat/completions')
    def stream():
        event = {
            'model': 'test-model',
            'choices': [{'delta': {'content': 'OK'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
        }
        return StreamingResponse(
            iter(['data: ' + json.dumps(event) + '\n\ndata: [DONE]\n\n']),
            media_type='text/event-stream',
        )

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
        requests = [
            {
                'id': 'test',
                'shape': 'fixture',
                'expected_version': 'test-model',
                'input_tokens': 1,
                'output_budget': 16,
                'body': {},
                'headers': {},
            }
        ]
        requests.append({**requests[0], 'id': 'test-second'})
        result = await window(
            f'https://localhost:{port}',
            requests,
            [0, 0.05],
            {'client_timeout_seconds': 5, 'late_drop_seconds': 1, 'tls': tls},
        )
        assert all(row['protocol_valid'] and row['content'] == 'OK' for row in result['rows'])
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
