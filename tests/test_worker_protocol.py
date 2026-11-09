"""网络资格按实际调用代际检查；探测成功不允许继承替代实例。"""

import httpx
import asyncio
import pytest
from test_replicas import workers


async def test_generation_fence_rejects_replacement_before_dispatch():
    from serving_lab.worker_protocol import worker_manifest, create_worker_app, RemoteEngine

    old = workers()['a1']
    manifest = worker_manifest(old, 'a-parameters')
    replacement = workers()['a1']
    replacement.identity['startup_generation'] = 'g2'
    app = create_worker_app(replacement, 'a-parameters', 'worker-token-long')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://127.0.0.1')
    remote = RemoteEngine('http://127.0.0.1', 'worker-token-long', manifest, client=client)
    assert not await remote.ready()
    response = await remote.open(
        'attempt1',
        {
            'model': 'a',
            'messages': [{'role': 'user', 'content': 'x'}],
            'max_tokens': 4,
            'stream': False,
        },
    )
    assert response.status_code == 409 and not replacement.calls
    assert (await remote.wait_release('attempt1', 0.1))['status'] == 'NOT_DISPATCHED'
    assert (
        await remote.reconcile(
            'unknown-old', {'worker_instance_id': 'old', 'startup_generation': 'old'}, 0.01
        )
        is None
    )
    await response.aclose()
    await remote.close()


async def test_wrong_token_cannot_read_identity_or_dispatch():
    from serving_lab.worker_protocol import create_worker_app

    e = workers()['a1']
    app = create_worker_app(e, 'a-parameters', 'worker-token-long')
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url='http://127.0.0.1'
    ) as c:
        assert (await c.get('/identity')).status_code == 401
        assert (await c.post('/chat', json={})).status_code == 401
    assert not e.calls


def test_remote_manifest_rejects_changed_profile_and_plaintext_external_endpoint():
    from serving_lab.worker_protocol import worker_manifest, RemoteEngine

    m = worker_manifest(workers()['a1'], 'a-parameters')
    with pytest.raises(ValueError):
        RemoteEngine('http://example.com:9000', 'worker-token-long', m)
    m['profile']['max_tokens'] = 1
    with pytest.raises(ValueError):
        RemoteEngine('http://127.0.0.1:9000', 'worker-token-long', m)


async def test_concurrent_attempt_duplicates_cannot_cross_ready_await():
    from serving_lab.worker_protocol import worker_manifest, create_worker_app
    from test_replica_tcp import ProtocolEngine

    e = ProtocolEngine('same', 'a')
    ready_gate = asyncio.Event()
    ready_calls = 0
    finish_gate = asyncio.Event()
    original = e.open

    async def ready():
        nonlocal ready_calls
        ready_calls += 1
        if ready_calls >= 2:
            ready_gate.set()
        await ready_gate.wait()
        return True

    async def opening(attempt, body):
        response = await original(attempt, body)
        await finish_gate.wait()
        return response

    e.ready = ready
    e.open = opening
    manifest = worker_manifest(e, 'p')
    app = create_worker_app(e, 'p', 'worker-token-long')
    headers = {
        'Authorization': 'Bearer worker-token-long',
        'X-Expected-Worker': manifest['manifest_digest'],
        'X-Request-Id': 'same-attempt',
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url='http://127.0.0.1'
    ) as c:
        body = {
            'model': 'a',
            'messages': [{'role': 'user', 'content': 'x'}],
            'max_tokens': 4,
            'stream': False,
        }
        tasks = [asyncio.create_task(c.post('/chat', headers=headers, json=body)) for _ in range(2)]
        await asyncio.wait_for(ready_gate.wait(), 1)
        await asyncio.sleep(0.03)
        finish_gate.set()
        responses = await asyncio.gather(*tasks)
        assert sorted(r.status_code for r in responses) == [200, 409] and len(e.calls) == 1


async def test_owned_termination_evidence_is_checked_on_worker_not_controller(monkeypatch):
    import serving_lab.worker_protocol as w
    from test_replica_tcp import ProtocolEngine

    assert hasattr(w, 'owned_termination')
    e = ProtocolEngine('owned', 'a')
    e.identity['processes'] = [{'pid': 12, 'boot_id': 'boot', 'start_ticks': 'old'}]
    supervisor = {'boot_id': 'boot', 'pid_namespace': 'pid:[same]'}
    monkeypatch.setattr(w, 'supervisor_identity', lambda: supervisor)
    monkeypatch.setattr(w, 'process_identity', lambda pid: None)
    manifest = w.worker_manifest(e, 'p')
    evidence = w.owned_termination(e, manifest)
    assert evidence['status'] == 'WORKER_GENERATION_TERMINATED' and evidence['worker'] == e.identity
    monkeypatch.setattr(w, 'supervisor_identity', lambda: {**supervisor, 'pid_namespace': 'other'})
    assert w.owned_termination(e, manifest) is None
    monkeypatch.setattr(w, 'supervisor_identity', lambda: supervisor)
    monkeypatch.setattr(w, 'process_identity', lambda pid: e.identity['processes'][0])
    assert w.owned_termination(e, manifest) is None


@pytest.mark.parametrize('process_alive', [False, True])
async def test_release_wait_consumes_original_supervisor_proof_only_after_process_exit(
    monkeypatch, process_alive
):
    import serving_lab.worker_protocol as protocol
    from test_replica_tcp import ProtocolEngine

    engine = ProtocolEngine('original', 'a')
    process = {'pid': 12, 'boot_id': 'boot', 'start_ticks': 'original'}
    engine.identity['processes'] = [process]
    supervisor = {'boot_id': 'boot', 'pid_namespace': 'pid:[original]'}
    monkeypatch.setattr(protocol, 'supervisor_identity', lambda: supervisor)
    monkeypatch.setattr(
        protocol, 'process_identity', lambda pid: process if process_alive else None
    )
    manifest = protocol.worker_manifest(engine, 'parameters')
    app = protocol.create_worker_app(engine, 'parameters', 'worker-token-long')
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://127.0.0.1')
    remote = protocol.RemoteEngine('http://127.0.0.1', 'worker-token-long', manifest, client=client)
    try:
        evidence = await remote.wait_release('interrupted-attempt', 0.05)
        if process_alive:
            assert evidence is None
        else:
            assert evidence is not None
            assert evidence['status'] == 'WORKER_GENERATION_TERMINATED'
            assert evidence['worker'] == engine.identity
            assert evidence['supervisor'] == supervisor
    finally:
        await remote.close()


@pytest.mark.parametrize('field', ['worker', 'processes', 'supervisor'])
async def test_release_wait_rejects_mismatched_termination_proof(monkeypatch, field):
    import serving_lab.worker_protocol as protocol
    from test_replica_tcp import ProtocolEngine

    engine = ProtocolEngine('original', 'a')
    engine.identity['processes'] = [{'pid': 12, 'boot_id': 'boot', 'start_ticks': 'original'}]
    supervisor = {'boot_id': 'boot', 'pid_namespace': 'pid:[original]'}
    monkeypatch.setattr(protocol, 'supervisor_identity', lambda: supervisor)
    manifest = protocol.worker_manifest(engine, 'parameters')
    evidence = {
        'status': 'WORKER_GENERATION_TERMINATED',
        'worker': engine.identity,
        'processes': engine.identity['processes'],
        'supervisor': supervisor,
    }
    evidence[field] = {'wrong': 'identity'}

    def response(request):
        return httpx.Response(
            200,
            json={
                'manifest_digest': manifest['manifest_digest'],
                'evidence': evidence,
                'release': None,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(response), base_url='http://127.0.0.1')
    remote = protocol.RemoteEngine('http://127.0.0.1', 'worker-token-long', manifest, client=client)
    try:
        assert await remote.wait_release('interrupted-attempt', 0.03) is None
    finally:
        await remote.close()
