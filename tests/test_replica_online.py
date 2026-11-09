"""复用真实入口生命周期，证明副本分配、去重与隔离是消费行为。"""

import asyncio
import httpx
import pytest
from test_replicas import workers
from test_online import body


def app_for(tmp_path, ws, initialize=True):
    from serving_lab.replica_api import create_replica_app
    from serving_lab.online import ServiceLimits

    app = create_replica_app(
        ws,
        tmp_path,
        {v: {'digest': v + '-parameters'} for v in ('a', 'b')},
        {'qualification_scope': 'version_transport_ready_v2'},
        'long-contract-token',
        {r: {'max_inflight': 1, 'max_queue': 1} for r in ws},
        {v: {'status': 'historical-failed'} for v in ('a', 'b')},
        limits=ServiceLimits(max_inflight=3, max_queue=3, cancel_grace_seconds=0.01),
    )
    pool = app.state.service.pool
    for rid, e in ws.items():

        async def ids(messages):
            return [1]

        e.token_ids = ids
        e.wire_payload = lambda b, e=e: e.profile.payload(b)
        receipt = {
            'passed': True,
            'qualification_scope': 'version_transport_ready_v2',
            'profile_digest': e.profile.digest,
            'version_id': e.version_id,
            'worker': e.identity,
            'artifact_digest': e.binding.artifact_digest,
            'panel_digest': pool.panel_digest,
            'application_history': pool.application_results[e.version_id],
        }
        if initialize:
            pool.qualified(rid, receipt)
            pool.observed_ready[rid] = True
    if initialize:
        pool.change('promote', 'a', 0)
    return app


async def test_parallel_same_version_requests_use_two_replicas_and_stay_pinned(tmp_path):
    ws = workers()
    ws['a1'].release.clear()
    ws['a2'].release.clear()
    app = app_for(tmp_path, ws)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url='http://test'
        ) as client:
            one = asyncio.create_task(
                client.post('/v1/chat/completions', json=body(model='active'))
            )
            await asyncio.wait_for(ws['a1'].started.wait(), 1)
            two = asyncio.create_task(
                client.post('/v1/chat/completions', json=body(model='active'))
            )
            await asyncio.wait_for(ws['a2'].started.wait(), 1)
            queued = asyncio.create_task(
                client.post('/v1/chat/completions', json=body(model='active'))
            )
            async with asyncio.timeout(1):
                while not any(
                    r['version'] == 'a' and r['state'] == 'queued'
                    for r in app.state.service.store.rows()
                ):
                    await asyncio.sleep(0)
            app.state.service.pool.change('promote', 'b', 1)
            ws['a1'].release.set()
            ws['a2'].release.set()
            responses = await asyncio.gather(one, two, queued)
            assert {r.headers['x-replica-id'] for r in responses} == {'a1', 'a2'}
            assert all(
                r.status_code == 200 and r.headers['x-model-version'] == 'a' for r in responses
            )
            assert not ws['b1'].calls


async def test_unconfirmed_replica_is_quarantined_without_blocking_healthy_peer(tmp_path):
    ws = workers()
    ws['a1'].cleanup_ok = False
    app = app_for(tmp_path, ws)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url='http://test'
        ) as client:
            first = await client.post(
                '/v1/chat/completions', json=body(model='a'), headers={'Idempotency-Key': 'same'}
            )
            assert first.headers['x-replica-id'] == 'a1'
            assert app.state.service.pool.replica('a1')['state'] == 'quarantined'
            assert (await client.get('/health/ready')).status_code == 200
            assert (
                await client.post(
                    '/v1/chat/completions',
                    json=body(model='a'),
                    headers={'Idempotency-Key': 'same'},
                )
            ).status_code == 409
            result = await client.post('/v1/chat/completions', json=body(model='a'))
            assert result.status_code == 200 and result.headers['x-replica-id'] == 'a2'
            assert len(ws['a1'].calls) == len(ws['a2'].calls) == 1


def test_new_entry_refuses_old_runtime_before_recovery(tmp_path):
    from serving_lab.runtime import RuntimeStore

    with RuntimeStore(tmp_path) as s:
        s.record_arrival('untouched', None)
    before = (tmp_path / 'service.sqlite').read_bytes()
    with pytest.raises(ValueError):
        app_for(tmp_path, workers())
    assert (tmp_path / 'service.sqlite').read_bytes() == before


async def test_restart_reconciles_exact_replica_and_never_replays_unknown_on_peer(tmp_path):
    ws = workers()
    app = app_for(tmp_path, ws)
    s = app.state.service
    s.store.record_arrival('orphan', 'old-key')
    lease = s.pool.pin('a', 'orphan')
    assert lease['replica_id'] == 'a1'
    s.store.reserve_attempt('orphan', 'old-attempt', 'a', 'hash')
    s.store.dispatched('orphan', ws['a1'].identity)
    s.store.close()
    reconciled = []

    async def original(attempt, identity, timeout):
        reconciled.append((attempt, identity))
        return None

    async def wrong(*args):
        raise AssertionError('same version peer must not reconcile the orphan')

    ws['a1'].reconcile = original
    ws['a2'].reconcile = wrong
    restored = app_for(tmp_path, ws, initialize=False)
    async with restored.router.lifespan_context(restored):
        assert reconciled == [('old-attempt', ws['a1'].identity)]
        assert restored.state.service.store.row('orphan')['state'] == 'unknown'
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(restored), base_url='http://test'
        ) as client:
            response = await client.post('/v1/chat/completions', json=body(model='a'))
            assert response.status_code == 200 and response.headers['x-replica-id'] == 'a2'
            duplicate = await client.post(
                '/v1/chat/completions', json=body(model='a'), headers={'Idempotency-Key': 'old-key'}
            )
            assert duplicate.status_code == 409
        assert ws['a1'].calls == [] and len(ws['a2'].calls) == 1
        assert restored.state.service.pool.pending_replica('a1') == 1
