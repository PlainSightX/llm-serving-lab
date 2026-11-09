"""切换发生于真实入口异步边界时，请求仍保留原实例和全局去重。"""

import asyncio
import json

import httpx
import pytest

from test_online import body
from test_releases import engines


def app_for(tmp_path, workers, **limits):
    from serving_lab.online import create_release_app, ServiceLimits

    app = create_release_app(
        workers,
        tmp_path,
        {v: {'digest': v + '-parameters'} for v in workers},
        {'name': 'test-panel'},
        'secret-for-contract-test',
        limits=ServiceLimits(**limits),
    )
    pool = app.state.service.pool
    for v in workers:
        mark_ready_with_panel(pool, v)
    pool.change('promote', 'small', 0)
    return app


def mark_ready_with_panel(pool, v):
    pool.qualified(
        v,
        {
            'passed': True,
            'version_id': v,
            'panel_digest': pool.panel_digest,
            'artifact_digest': v + '-bundle',
            'worker': pool.engines[v].identity,
        },
    )


@pytest.mark.parametrize('phase', ['count', 'queue', 'stream'])
async def test_switch_pins_counter_queued_and_streaming_requests(tmp_path, phase):
    workers = engines()
    old = workers['small']
    gate = asyncio.Event()
    count_started = asyncio.Event()
    if phase == 'count':

        async def count(messages):
            count_started.set()
            await gate.wait()
            return 1

        old.count_tokens = count
    else:
        old.release.clear()
    app = app_for(tmp_path, workers, max_inflight=1, max_queue=3)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            request = asyncio.create_task(
                client.post(
                    '/v1/chat/completions', json=body(model='active', stream=phase == 'stream')
                )
            )
            await asyncio.wait_for(
                count_started.wait() if phase == 'count' else old.started.wait(), 1
            )
            queued = None
            if phase == 'queue':
                queued = asyncio.create_task(
                    client.post('/v1/chat/completions', json=body(model='active'))
                )
                async with asyncio.timeout(1):
                    while (
                        len([r for r in app.state.service.store.rows() if r['version'] == 'small'])
                        < 2
                    ):
                        await asyncio.sleep(0)
            app.state.service.pool.change('promote', 'large', 1)
            with pytest.raises(ValueError):
                app.state.service.pool.retire('small')
            gate.set()
            old.release.set()
            response = await request
            assert response.status_code == 200 and response.headers['X-Model-Version'] == 'small'
            assert response.headers['X-Route-Generation'] == '1'
            if queued:
                assert (await queued).headers['X-Model-Version'] == 'small'
            rows = app.state.service.store.rows()
            assert all(
                r['version'] == 'small'
                and json.loads(r['worker'])['worker_instance_id'] == 'small-worker'
                for r in rows
            )
            assert all(
                json.loads(r['observation'])['version_id'] == 'small' and r['lease_released'] == 1
                for r in rows
            )
            assert not workers['large'].calls
            new = await client.post('/v1/chat/completions', json=body(model='active'))
            assert new.headers['X-Model-Version'] == 'large'
            app.state.service.pool.retire('small')


async def test_admin_auth_dead_worker_and_unconfirmed_cleanup_cannot_publish_or_retire(tmp_path):
    workers = engines()
    app = app_for(tmp_path, workers, cancel_grace_seconds=0.01)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            change = {'operation': 'promote', 'target': 'large', 'expected_generation': 1}
            assert (await client.post('/admin/route', json=change)).status_code == 401
            headers = {'Authorization': 'Bearer secret-for-contract-test'}
            workers['large'].alive = False
            assert (
                await client.post('/admin/route', json=change, headers=headers)
            ).status_code == 503
            assert app.state.service.pool.route()['active'] == 'small'
            workers['large'].alive = True
            workers['small'].cleanup_ok = False
            assert (
                await client.post('/v1/chat/completions', json=body(model='active'))
            ).status_code == 200
            app.state.service.pool.change('promote', 'large', 1)
            response = await client.post('/admin/releases/small/retire', json={}, headers=headers)
            assert response.status_code == 409 and app.state.service.pool.pending('small') == 1


async def test_live_dedup_survives_publication(tmp_path):
    workers = engines()
    app = app_for(tmp_path, workers)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            headers = {'Idempotency-Key': 'same'}
            assert (
                await client.post(
                    '/v1/chat/completions', json=body(model='active'), headers=headers
                )
            ).status_code == 200
            app.state.service.pool.change('promote', 'large', 1)
            assert (
                await client.post(
                    '/v1/chat/completions', json=body(model='active'), headers=headers
                )
            ).status_code == 409
            assert not workers['large'].calls and len(app.state.service.store.rows()) == 2
