"""真实loopback socket上的合同测试；backend明确是协议替身，不是模型证据。"""

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace
import httpx
import pytest
import uvicorn
from serving_lab.profile import ServingProfile
from serving_lab.worker_protocol import RemoteEngine, worker_manifest, create_worker_app
from serving_lab.replica_api import create_replica_app
from serving_lab.online import ServiceLimits

TOKEN = 'test-worker-token-only'
ADMIN = 'test-gateway-token-only'


class Server:
    def __init__(self, app, port=0):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('127.0.0.1', port))
        self.port = self.socket.getsockname()[1]
        self.server = None
        self.error = None

        def serve():
            try:
                selected = app() if callable(app) and not hasattr(app, 'router') else app
                self.server = uvicorn.Server(
                    uvicorn.Config(selected, log_level='error', timeout_graceful_shutdown=2)
                )
                self.server.run(sockets=[self.socket])
            except BaseException as exc:
                self.error = exc

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        until = time.monotonic() + 5
        while self.server is None or not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > until:
                self.close()
                raise RuntimeError('test server failed to start') from self.error
            time.sleep(0.01)
        self.url = f'http://127.0.0.1:{self.port}'

    def close(self):
        if self.server:
            self.server.should_exit = True
        self.thread.join(5)
        self.socket.close()
        if self.thread.is_alive():
            raise RuntimeError('owned test server did not stop')


class ProtocolEngine:
    def __init__(self, rid, version, generation='g1'):
        self.version_id = version
        self.identity = {'worker_instance_id': rid, 'startup_generation': generation}
        self.profile = ServingProfile(
            {
                'name': 'contract-fixture',
                'max_tokens': 128,
                'artifact_digest': version + '-artifact',
            },
            version,
        )
        self.binding = SimpleNamespace(artifact_digest=version + '-artifact')
        self.gate = threading.Event()
        self.started = threading.Event()
        self.calls = []
        self.events = {}

    async def ready(self):
        return True

    async def token_ids(self, messages):
        return [1] * max(1, sum(len(m['content']) for m in messages) // 4)

    async def engine_token_ids(self, messages):
        return await self.token_ids(messages)

    async def close(self):
        pass

    async def open(self, attempt, body):
        self.calls.append(attempt)
        self.started.set()
        return ProtocolResponse(self, attempt, body)

    async def wait_release(self, attempt, timeout):
        deadline = time.monotonic() + timeout
        while attempt not in self.events and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        return self.events.get(attempt)


class ProtocolResponse:
    status_code = 200

    def __init__(self, engine, attempt, body):
        self.engine = engine
        self.attempt = attempt
        self.body = body

    async def wait(self):
        if self.body['messages'][0]['content'] == 'hold':
            while not self.engine.gate.is_set():
                await asyncio.sleep(0.005)

    def released(self, status):
        self.engine.events.setdefault(
            self.attempt,
            {
                'status': status,
                'request_id': 'chatcmpl-' + self.attempt,
                **self.engine.identity,
                'request_absent': True,
                'after_blocks': [[]],
            },
        )

    async def data(self):
        n = len(await self.engine.token_ids(self.body['messages']))
        budget = self.body['max_tokens']
        finish = 'length' if budget == 1 else 'stop'
        return {
            'id': 'chatcmpl-' + self.attempt,
            'model': self.engine.version_id,
            'choices': [
                {'message': {'content': '1' if budget == 1 else 'OK'}, 'finish_reason': finish}
            ],
            'usage': {'prompt_tokens': n, 'completion_tokens': 1, 'total_tokens': n + 1},
        }

    async def aread(self):
        await self.wait()
        data = await self.data()
        self.released(
            'FINISHED_LENGTH_CAPPED' if self.body['max_tokens'] == 1 else 'FINISHED_STOPPED'
        )
        return json.dumps(data).encode()

    async def aiter_bytes(self):
        data = await self.data()
        first = {
            'id': data['id'],
            'model': data['model'],
            'choices': [{'delta': {'content': 'OK'}, 'finish_reason': None}],
        }
        yield ('data: ' + json.dumps(first) + '\n\n').encode()
        await self.wait()
        self.released('FINISHED_STOPPED')
        yield (
            'data: '
            + json.dumps({**data, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
            + '\n\ndata: [DONE]\n\n'
        ).encode()

    async def aclose(self):
        self.released('FINISHED_ABORTED')


def setup_pool(tmp_path):
    from pathlib import Path

    panel = json.loads(
        (Path(__file__).resolve().parents[1] / 'fixtures/serving/r2-transport-v2.json').read_text(
            encoding='utf-8'
        )
    )
    backends = {}
    servers = []
    remotes = {}
    for rid, version in [('a1', 'a'), ('a2', 'a'), ('b1', 'b')]:
        e = ProtocolEngine(rid, version)
        backends[rid] = e
        s = Server(create_worker_app(e, version + '-params', TOKEN))
        servers.append(s)
        remotes[rid] = RemoteEngine(s.url, TOKEN, worker_manifest(e, version + '-params'))

    def factory():
        return create_replica_app(
            remotes,
            tmp_path,
            {v: {'digest': v + '-params'} for v in ('a', 'b')},
            panel,
            ADMIN,
            {r: {'max_inflight': 1, 'max_queue': 1} for r in remotes},
            {v: {'status': 'fixture-only'} for v in ('a', 'b')},
            limits=ServiceLimits(max_inflight=3, max_queue=3),
        )

    try:
        gateway = Server(factory)
    except BaseException:
        for server in reversed(servers):
            server.close()
        raise
    servers.append(gateway)
    return backends, servers, gateway


async def prepare(client):
    headers = {'Authorization': 'Bearer ' + ADMIN}
    for rid in ('a1', 'a2', 'b1'):
        r = await client.post('/admin/replicas/' + rid + '/qualify', headers=headers, json={})
        assert r.status_code == 200, r.text
    r = await client.post(
        '/admin/route',
        headers=headers,
        json={'operation': 'promote', 'target': 'a', 'expected_generation': 0},
    )
    assert r.status_code == 200


async def test_real_tcp_stream_stays_on_draining_replica_through_publication(tmp_path):
    engines, servers, gateway = setup_pool(tmp_path)
    try:
        async with httpx.AsyncClient(base_url=gateway.url, timeout=10, trust_env=False) as client:
            await prepare(client)
            headers = {'Authorization': 'Bearer ' + ADMIN}
            body = {
                'model': 'active',
                'messages': [{'role': 'user', 'content': 'hold'}],
                'max_tokens': 128,
                'stream': True,
            }
            async with client.stream(
                'POST', '/v1/chat/completions', json=body, headers={'Idempotency-Key': 'old-stream'}
            ) as stream:
                assert stream.status_code == 200 and stream.headers['x-replica-id'] == 'a1'
                iterator = stream.aiter_lines()
                assert (await anext(iterator)).startswith('data:')
                assert (
                    await client.post('/admin/replicas/a1/drain', headers=headers, json={})
                ).status_code == 409
                other = await client.post(
                    '/v1/chat/completions',
                    json={**body, 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False},
                )
                assert other.status_code == 200 and other.headers['x-replica-id'] == 'a2'
                assert (
                    await client.post(
                        '/admin/route',
                        headers=headers,
                        json={'operation': 'promote', 'target': 'b', 'expected_generation': 1},
                    )
                ).status_code == 200
                engines['a1'].gate.set()
                remaining = [line async for line in iterator]
                assert 'data: [DONE]' in remaining
            for _ in range(100):
                state = (await client.get('/admin/releases', headers=headers)).json()
                if not next(r['pending'] for r in state['replicas'] if r['id'] == 'a1'):
                    break
                await asyncio.sleep(0.01)
            assert (
                await client.post('/admin/replicas/a1/retire', headers=headers, json={})
            ).status_code == 200
            duplicate = await client.post(
                '/v1/chat/completions',
                json={**body, 'stream': False},
                headers={'Idempotency-Key': 'old-stream'},
            )
            assert duplicate.status_code == 409 and len(engines['b1'].calls) == 4
    finally:
        for e in engines.values():
            e.gate.set()
        for server in reversed(servers):
            server.close()


async def test_real_tcp_endpoint_replacement_is_fenced():
    old = ProtocolEngine('same-slot', 'a')
    server = Server(create_worker_app(old, 'a-params', TOKEN))
    port = server.port
    remote = RemoteEngine(server.url, TOKEN, worker_manifest(old, 'a-params'))
    try:
        assert await remote.ready()
        server.close()
        replacement = ProtocolEngine('same-slot', 'a', 'g2')
        server = Server(create_worker_app(replacement, 'a-params', TOKEN), port)
        response = await remote.open(
            'stale-attempt',
            {
                'model': 'a',
                'messages': [{'role': 'user', 'content': 'x'}],
                'max_tokens': 4,
                'stream': False,
            },
        )
        assert response.status_code == 409 and replacement.calls == []
        assert (await remote.wait_release('stale-attempt', 1))['status'] == 'NOT_DISPATCHED'
        await response.aclose()
    finally:
        await remote.close()
        server.close()


@pytest.mark.parametrize('streaming', [False, True])
async def test_real_tcp_disconnect_releases_worker_attempt(streaming):
    engine = ProtocolEngine('r', 'a')
    server = Server(create_worker_app(engine, 'a-params', TOKEN))
    manifest = worker_manifest(engine, 'a-params')
    headers = {
        'Authorization': 'Bearer ' + TOKEN,
        'X-Expected-Worker': manifest['manifest_digest'],
        'X-Request-Id': 'cancel-me',
    }
    try:
        async with httpx.AsyncClient(base_url=server.url, timeout=5, trust_env=False) as client:
            body = {
                'model': 'a',
                'messages': [{'role': 'user', 'content': 'hold'}],
                'max_tokens': 64,
                'stream': streaming,
            }
            if streaming:
                async with client.stream('POST', '/chat', json=body, headers=headers) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith('data:'):
                            break
            else:
                task = asyncio.create_task(client.post('/chat', json=body, headers=headers))
                for _ in range(100):
                    if engine.started.is_set():
                        break
                    await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            for _ in range(150):
                if 'cancel-me' in engine.events:
                    break
                await asyncio.sleep(0.01)
            assert engine.events['cancel-me']['status'] == 'FINISHED_ABORTED'
    finally:
        engine.gate.set()
        server.close()
