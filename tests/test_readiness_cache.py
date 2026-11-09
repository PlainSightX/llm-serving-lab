"""就绪检查分层：进程身份每次核对；慢检查（清单哈希、HTTP健康探测）短期复用且并发共享。"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from serving_lab.engine import HttpEngine, ReadinessCache


async def test_concurrent_checks_share_one_probe_and_reuse_until_ttl():
    cache = ReadinessCache(0.2)
    calls = 0
    gate = asyncio.Event()

    async def probe():
        nonlocal calls
        calls += 1
        await gate.wait()
        return True

    tasks = [asyncio.create_task(cache.check(probe)) for _ in range(5)]
    await asyncio.sleep(0.01)
    gate.set()
    assert all(await asyncio.gather(*tasks)) and calls == 1
    assert await cache.check(probe) and calls == 1
    await asyncio.sleep(0.25)
    assert await cache.check(probe) and calls == 2


async def test_failures_and_errors_are_never_cached():
    cache = ReadinessCache(10)
    results = iter([False, True])

    async def probe():
        return next(results)

    assert not await cache.check(probe)
    assert await cache.check(probe)

    async def broken():
        raise OSError('engine down')

    cache.invalidate()
    assert not await cache.check(broken)
    assert cache.passed_at is None


def fake_http_engine(monkeypatch, alive):
    """不启动vLLM：只替换进程身份读取与HTTP客户端，检查HttpEngine.ready的分层顺序。"""
    engine = HttpEngine.__new__(HttpEngine)
    process = {'pid': 1, 'start_ticks': '10', 'boot_id': 'b'}
    scheduler = {'pid': 2, 'start_ticks': '20', 'boot_id': 'b'}
    manifest = {'unchanged': True}
    engine.binding = SimpleNamespace(
        process=process, manifest_unchanged=lambda: manifest['unchanged']
    )
    engine.identity = {'processes': [process, scheduler]}
    engine.readiness = ReadinessCache(10)
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200)

    engine.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url='http://engine'
    )
    monkeypatch.setattr(
        'serving_lab.engine.process_identity',
        lambda pid: {1: process, 2: scheduler}.get(pid) if alive[pid] else None,
    )
    return engine, requests, manifest


async def test_http_engine_checks_identity_every_time_but_health_once_per_ttl(monkeypatch):
    alive = {1: True, 2: True}
    engine, requests, manifest = fake_http_engine(monkeypatch, alive)
    assert await engine.ready() and await engine.ready() and await engine.ready()
    assert requests == ['/health']
    # Scheduler 进程退出：即使健康结果仍在有效期内，也立即判为未就绪并作废缓存。
    alive[2] = False
    assert not await engine.ready()
    alive[2] = True
    assert await engine.ready() and requests == ['/health', '/health']
    # 清单被改动属于慢检查：缓存作废后的下一次探测发现并拒绝。
    engine.readiness.invalidate()
    manifest['unchanged'] = False
    assert not await engine.ready()
    await engine.client.aclose()


async def test_remote_engine_reuses_passing_identity_probe(monkeypatch):
    from test_replicas import workers
    from serving_lab.worker_protocol import RemoteEngine, create_worker_app, worker_manifest

    worker = workers()['a1']
    manifest = worker_manifest(worker, 'a-parameters')
    app = create_worker_app(worker, 'a-parameters', 'worker-token-long')
    probes = []
    inner = httpx.ASGITransport(app)

    class Counting(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path == '/identity':
                probes.append(request.url.path)
            return await inner.handle_async_request(request)

    client = httpx.AsyncClient(transport=Counting(), base_url='http://127.0.0.1')
    remote = RemoteEngine('http://127.0.0.1', 'worker-token-long', manifest, client=client)
    monkeypatch.setattr(remote.ready_cache, 'ttl_seconds', 0.2)
    assert await remote.ready() and await remote.ready()
    assert len(probes) == 1
    await asyncio.sleep(0.25)
    assert await remote.ready() and len(probes) == 2
    await remote.close()


async def test_background_monitor_keeps_requests_off_the_probe_path():
    cache = ReadinessCache(0.3)
    healthy = {'value': True}
    background = 0
    on_request = 0

    async def probe():
        nonlocal background
        background += 1
        return healthy['value']

    async def request_probe():
        nonlocal on_request
        on_request += 1
        return True

    cache.start(probe, 0.05)
    await asyncio.sleep(0.12)
    assert background >= 2
    # 后台探测保持结果新鲜时，请求路径只读结果，从不自己发探测。
    assert all([await cache.check(request_probe) for _ in range(20)])
    assert on_request == 0
    # 后台探测发现故障后，请求立即得到未就绪。
    healthy['value'] = False
    await asyncio.sleep(0.12)
    assert not cache.fresh()
    await cache.stop()
    assert cache.monitor is None
    with pytest.raises(ValueError):
        cache.start(probe, 0.3)
