"""审查补测：失败探测不排队、有效期从探测开始算、副本池刷新不等挂起副本、
流的最后一帧之前终态已落盘、账本失败后仍归还容量并对新请求返回 503、启动失败也执行关闭。"""

import asyncio
import time

import httpx
import pytest

from serving_lab.engine import ReadinessCache
from serving_lab.runtime import RuntimeStore
from test_runtime_group_commit import FakeEngine, committed_rows, request


async def test_failing_probe_is_shared_not_queued():
    cache = ReadinessCache(10)
    calls = 0

    async def failing():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.2)
        return False

    started = time.monotonic()
    results = await asyncio.gather(*(cache.check(failing) for _ in range(5)))
    elapsed = time.monotonic() - started
    # 5 个等待者共享一次在途探测：总耗时约一次探测，而不是 5 次串行。
    assert results == [False] * 5 and calls == 1 and elapsed < 0.5
    # 失败不缓存：下一次调用重新探测。
    assert not await cache.check(failing) and calls == 2


async def test_validity_counts_from_probe_start():
    cache = ReadinessCache(0.5)

    async def slow_pass():
        await asyncio.sleep(0.3)
        return True

    before = time.monotonic()
    assert await cache.check(slow_pass)
    # 有效期从探测开始算：结果在完成后最多再保持约 0.2 秒。
    assert cache.passed_at < before + 0.05
    await asyncio.sleep(0.25)
    assert not cache.fresh()


async def test_cancelled_waiter_does_not_cancel_shared_probe():
    cache = ReadinessCache(10)
    gate = asyncio.Event()

    async def probe():
        await gate.wait()
        return True

    first = asyncio.create_task(cache.check(probe))
    second = asyncio.create_task(cache.check(probe))
    await asyncio.sleep(0.01)
    first.cancel()
    gate.set()
    assert await second
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_replica_refresh_reads_monitor_results_without_waiting(tmp_path):
    from test_replicas import make_pool

    hung = asyncio.Event()

    class Monitored:
        def __init__(self, engine, state):
            self.engine = engine
            self.state = state

        def ready_nowait(self):
            return self.state

        async def ready(self):
            await hung.wait()
            return True

    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store)
        engines = pool.engines
        engines['a1'].ready_nowait = Monitored(engines['a1'], True).ready_nowait
        engines['a2'].ready_nowait = Monitored(engines['a2'], False).ready_nowait
        engines['a2'].ready = Monitored(engines['a2'], False).ready
        engines['b1'].ready_nowait = Monitored(engines['b1'], True).ready_nowait
        # a2 的同步探测会一直挂起；有后台结果时刷新不应等待它。
        await asyncio.wait_for(pool.refresh(), 0.5)
        assert pool.observed_ready == {'a1': True, 'a2': False, 'b1': True}


async def test_terminal_state_is_durable_before_done_frame(tmp_path):
    from serving_lab.online import Call, ServiceLimits, create_app

    engine = FakeEngine(tmp_path)
    app = create_app(engine, tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        service = app.state.service
        store = service.store
        now = time.monotonic()
        call = Call('arrival-1', now, now + 30)
        store.record_arrival(call.arrival, None, durable=False)
        call.attempt = 'attempt-1'
        assert store.reserve_attempt(call.arrival, call.attempt, 'base', 'h', durable=False)
        store.dispatched(call.arrival, engine.identity)
        call.dispatched = True
        call.engine = engine
        call.upstream = await engine.open(call.attempt, {})
        seen_at_done = None
        generator = service.stream(call)
        async for chunk in generator:
            if chunk.startswith(b'data: [DONE]'):
                seen_at_done = committed_rows(tmp_path)[call.arrival]['state']
        assert seen_at_done == 'completed'


async def test_failed_ledger_still_releases_capacity_and_refuses_new_requests(
    tmp_path, monkeypatch
):
    from serving_lab.online import ServiceLimits, create_app

    app = create_app(FakeEngine(tmp_path), tmp_path, ServiceLimits(max_inflight=2))
    async with app.router.lifespan_context(app):
        service = app.state.service

        def broken_cleanup(*args, **kwargs):
            service.store.failed = RuntimeError('simulated disk failure')
            raise RuntimeError('simulated disk failure')

        monkeypatch.setattr(service.store, 'cleanup', broken_cleanup)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            await client.post('/v1/chat/completions', json=request())
            # 账本写入失败也必须归还容量与执行槽。
            assert service.capacity_used == 0 and service.slots._value == 2
            refused = await client.post('/v1/chat/completions', json=request())
            assert refused.status_code == 503
            assert refused.json()['error']['code'] == 'ledger_unavailable'
        service.store.failed = None


async def test_startup_failure_still_closes_engines_and_ledger(tmp_path, monkeypatch):
    from serving_lab.online import ServiceLimits, create_app

    engine = FakeEngine(tmp_path)
    closed = []

    async def close():
        closed.append(True)

    engine.close = close
    app = create_app(engine, tmp_path, ServiceLimits())

    def broken_unresolved():
        raise RuntimeError('simulated reconciliation failure')

    monkeypatch.setattr(app.state.service.store, 'unresolved', broken_unresolved)
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            pass
    assert closed == [True]
    # 账本已关闭：控制者锁已释放，新的控制者可以打开同一目录。
    RuntimeStore(tmp_path).close()
