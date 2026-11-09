import asyncio

import httpx
import pytest


class FakeEngine:
    """CPU合同替身；不用于项目真实服务验收。"""

    version_id = 'base'
    identity = {'worker_instance_id': 'test', 'startup_generation': 'test-generation'}
    alive = True

    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.released = {}
        self.cleanup_ok = True

    async def ready(self):
        return self.alive

    async def count_tokens(self, messages):
        return sum(len(m['content']) for m in messages)

    async def open(self, attempt, body):
        self.calls.append(attempt)
        self.started.set()
        return FakeResponse(self, attempt, body)

    async def wait_release(self, attempt, timeout):
        return self.released.get(attempt) if self.cleanup_ok else None

    async def close(self):
        pass


class FakeResponse:
    status_code = 200

    def __init__(self, engine, attempt, body):
        self.engine = engine
        self.attempt = attempt
        self.body = body

    async def aread(self):
        import json

        await self.engine.release.wait()
        self.engine.released[self.attempt] = {'status': 'FINISHED_STOPPED'}
        return json.dumps(
            {
                'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            }
        ).encode()

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}'
        await self.engine.release.wait()
        self.engine.released[self.attempt] = {'status': 'FINISHED_STOPPED'}
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield 'data: [DONE]'

    async def aclose(self):
        self.engine.released.setdefault(self.attempt, {'status': 'FINISHED_ABORTED'})


def body(**kwargs):
    return {
        'model': 'base',
        'messages': [{'role': 'user', 'content': 'x'}],
        'max_tokens': 8,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_limits_dedup_and_all_arrivals_are_recorded(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    engine = FakeEngine()
    app = create_app(
        engine, tmp_path, ServiceLimits(max_request_bytes=512, max_input_tokens=10, max_messages=2)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            assert (
                await client.post('/v1/chat/completions', content=b'x' * 513)
            ).status_code == 413
            assert (
                await client.post(
                    '/v1/chat/completions',
                    json=body(messages=[{'role': 'user', 'content': 'x' * 11}]),
                )
            ).status_code == 422
            assert (
                await client.post('/v1/chat/completions', json=body(model='other'))
            ).status_code == 404
            assert (
                await client.post(
                    '/v1/chat/completions', json=body(), headers={'Idempotency-Key': 'same'}
                )
            ).status_code == 200
            assert (
                await client.post(
                    '/v1/chat/completions', json=body(), headers={'Idempotency-Key': 'same'}
                )
            ).status_code == 409
        rows = app.state.service.store.rows()
        assert len(rows) == 5 and len(engine.calls) == 1
        assert len({r['id'] for r in rows}) == 5


@pytest.mark.asyncio
async def test_queue_deadline_and_capacity(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    class HeldCleanup(FakeEngine):
        def __init__(self):
            super().__init__()
            self.cleanup_gate = asyncio.Event()

        async def wait_release(self, attempt, timeout):
            await self.cleanup_gate.wait()
            return self.released.get(attempt)

    engine = HeldCleanup()
    engine.release.clear()
    app = create_app(
        engine, tmp_path, ServiceLimits(max_inflight=1, max_queue=1, deadline_seconds=0.15)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            a = asyncio.create_task(client.post('/v1/chat/completions', json=body()))
            await engine.started.wait()
            b = asyncio.create_task(client.post('/v1/chat/completions', json=body()))
            while app.state.service.capacity_used < 2:
                await asyncio.sleep(0)
            assert (await client.post('/v1/chat/completions', json=body())).status_code == 429
            assert (await b).status_code == 504
            assert len(engine.calls) == 1
            engine.cleanup_gate.set()
            assert (await a).status_code == 504
        assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_unconfirmed_release_closes_admission_and_restart_is_not_ready(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    engine = FakeEngine()
    engine.cleanup_ok = False
    app = create_app(engine, tmp_path, ServiceLimits(cancel_grace_seconds=0.01))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            await client.post('/v1/chat/completions', json=body())
            assert (await client.get('/health/ready')).status_code == 503
            assert (await client.post('/v1/chat/completions', json=body())).status_code == 503
    next_app = create_app(FakeEngine(), tmp_path, ServiceLimits())
    async with next_app.router.lifespan_context(next_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=next_app), base_url='http://test'
        ) as client:
            assert (await client.get('/health/ready')).status_code == 503


@pytest.mark.asyncio
async def test_stream_usage_missing_is_not_invented(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    app = create_app(FakeEngine(), tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            r = await client.post('/v1/chat/completions', json=body(stream=True))
            assert '[DONE]' in r.text and r.status_code == 200
        import json

        row = app.state.service.store.rows()[0]
        observation = json.loads(row['observation'])
        assert (
            row['state'] == 'completed'
            and observation['usage'] is None
            and observation['ttft_seconds'] is not None
        )


@pytest.mark.asyncio
async def test_timed_out_token_count_keeps_capacity_until_it_really_stops(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    class SlowCounter(FakeEngine):
        def __init__(self):
            super().__init__()
            self.counting = asyncio.Event()
            self.finish_count = asyncio.Event()

        async def count_tokens(self, messages):
            self.counting.set()
            await self.finish_count.wait()
            return 1

    engine = SlowCounter()
    app = create_app(
        engine, tmp_path, ServiceLimits(max_inflight=1, max_queue=0, deadline_seconds=0.03)
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            assert (await client.post('/v1/chat/completions', json=body())).status_code == 504
            assert app.state.service.capacity_used == 1 and not engine.calls
            assert (await client.post('/v1/chat/completions', json=body())).status_code == 429
            engine.finish_count.set()
            await asyncio.gather(*app.state.service.cleanups)
            assert app.state.service.capacity_used == 0


@pytest.mark.asyncio
async def test_stream_eof_is_failure_not_success(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    class IncompleteResponse(FakeResponse):
        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'

    class IncompleteEngine(FakeEngine):
        async def open(self, attempt, body):
            return IncompleteResponse(self, attempt, body)

    app = create_app(IncompleteEngine(), tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            r = await client.post('/v1/chat/completions', json=body(stream=True))
            assert 'incomplete_stream' in r.text and '[DONE]' not in r.text
        assert app.state.service.store.rows()[0]['state'] == 'failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['slot', 'open'])
async def test_disconnect_same_turn_keeps_resource_ownership(tmp_path, operation):
    import time
    from starlette.requests import ClientDisconnect
    from serving_lab.online import Call, Service, ServiceLimits

    class Disconnected:
        async def is_disconnected(self):
            return True

    engine = FakeEngine()
    service = Service(engine, tmp_path, ServiceLimits(max_inflight=1))
    call = Call('arrival', time.monotonic(), time.monotonic() + 1, attempt='attempt')
    service.store.record_arrival(call.arrival, None)
    service.store.reserve_attempt(call.arrival, call.attempt, 'base', 'h')
    if operation == 'slot':
        task = asyncio.create_task(service.acquire_slot(call))
    else:
        service.store.dispatched(call.arrival, engine.identity)
        call.dispatched = True
        task = asyncio.create_task(service.open_upstream(call, body()))
    with pytest.raises(ClientDisconnect):
        await service.wait(task, Disconnected(), call.deadline)
    if operation == 'slot':
        assert call.slot
    else:
        assert call.upstream is not None
    await service.settle(call)
    assert service.slots._value == 1
    if operation == 'open':
        assert service.store.row(call.arrival)['cleanup'] == 'confirmed'
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('spec_version', ['2.0', '2.4'])
async def test_slow_downstream_send_obeys_absolute_deadline(tmp_path, spec_version):
    import time
    from serving_lab.online import Call, ManagedStream, Service, ServiceLimits

    engine = FakeEngine()
    service = Service(engine, tmp_path, ServiceLimits())
    call = Call('slow', time.monotonic(), time.monotonic() + 0.02, attempt='attempt', capacity=True)
    service.capacity_used = 1
    await service.acquire_slot(call)
    service.store.record_arrival(call.arrival, None)
    service.store.reserve_attempt(call.arrival, call.attempt, 'base', 'h')
    service.store.dispatched(call.arrival, engine.identity)
    call.dispatched = True
    call.upstream = FakeResponse(engine, call.attempt, body(stream=True))

    async def send(message):
        if message['type'] == 'http.response.body' and b'"content"' in message.get('body', b''):
            await asyncio.sleep(1)

    async def receive():
        await asyncio.Event().wait()

    started = time.monotonic()
    await ManagedStream(service, call, {})(
        {'type': 'http', 'asgi': {'spec_version': spec_version}}, receive, send
    )
    assert service.store.row(call.arrival)['state'] == 'timed_out'
    import json

    assert json.loads(service.store.row(call.arrival)['observation'])['events']
    assert time.monotonic() - started < 0.5 and service.capacity_used == 0
    await service.shutdown()


@pytest.mark.asyncio
async def test_slow_readiness_is_inside_capacity_budget(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    class SlowReady(FakeEngine):
        def __init__(self):
            super().__init__()
            self.ready_started = asyncio.Event()
            self.allow_ready = asyncio.Event()
            self.ready_calls = 0

        async def ready(self):
            self.ready_calls += 1
            self.ready_started.set()
            await self.allow_ready.wait()
            return True

    engine = SlowReady()
    app = create_app(engine, tmp_path, ServiceLimits(max_inflight=1, max_queue=0))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            jobs = [asyncio.create_task(client.post('/v1/chat/completions', json=body()))]
            await engine.ready_started.wait()
            jobs.extend(
                asyncio.create_task(client.post('/v1/chat/completions', json=body()))
                for _ in range(5)
            )
            await asyncio.sleep(0.01)
            observed = engine.ready_calls
            engine.allow_ready.set()
            results = await asyncio.gather(*jobs)
            assert observed == 1 and sum(r.status_code == 429 for r in results) == 5


@pytest.mark.asyncio
async def test_cleanup_transport_close_has_bounded_grace(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    class HangingClose(FakeResponse):
        async def aclose(self):
            await asyncio.Event().wait()

    class Engine(FakeEngine):
        async def open(self, attempt, body):
            return HangingClose(self, attempt, body)

    app = create_app(Engine(), tmp_path, ServiceLimits(cancel_grace_seconds=0.02))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            await asyncio.wait_for(client.post('/v1/chat/completions', json=body()), 0.2)
            assert (await client.get('/health/ready')).status_code == 503
        assert app.state.service.store.rows()[0]['cleanup'] == 'unconfirmed'


@pytest.mark.asyncio
async def test_completed_stream_records_ordered_phase_marks(tmp_path):
    """逐请求阶段时刻按处理顺序单调递增，用于拆解网关在首 token 前的额外耗时。"""
    import json

    from serving_lab.online import create_app, ServiceLimits

    app = create_app(FakeEngine(), tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            response = await client.post('/v1/chat/completions', json=body(stream=True))
            assert response.status_code == 200 and '[DONE]' in response.text
        phases = json.loads(app.state.service.store.rows()[0]['observation'])['phase_ms']
    expected = [
        'arrival_recorded',
        'body_read',
        'validated',
        'tokenized',
        'reserved',
        'payload_saved',
        'slot_acquired',
        'ready_rechecked',
        'dispatch_recorded',
        'upstream_headers',
        'first_content',
    ]
    # 账本用canonical JSON（键排序）持久化，因此按预期处理顺序取值后检查单调性。
    assert set(phases) == set(expected)
    values = [phases[name] for name in expected]
    assert values == sorted(values) and values[0] >= 0
