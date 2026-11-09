"""受控worker网络协议；实例身份、token计数和释放判断留在worker本机。"""

import asyncio
import copy
import hmac
import json
import os
from pathlib import Path
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from contextlib import suppress
from types import SimpleNamespace
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect
from .lab import digest
from .profile import ServingProfile
from .engine import ReadinessCache, process_identity


def supervisor_identity():
    try:
        return {
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid_namespace': os.readlink('/proc/self/ns/pid'),
        }
    except OSError:
        return None


def owned_termination(engine, manifest):
    """只接受原worker同kernel、同PID namespace内的完整原代际消失证据。"""
    supervisor = manifest.get('supervisor')
    processes = engine.identity.get('processes', [])
    if (
        not supervisor
        or supervisor_identity() != supervisor
        or engine.identity != manifest['identity']
        or not processes
    ):
        return None
    if any(p['boot_id'] != supervisor['boot_id'] for p in processes):
        return None
    if any(process_identity(p['pid']) == p for p in processes):
        return None
    return {
        'status': 'WORKER_GENERATION_TERMINATED',
        'worker': engine.identity,
        'processes': processes,
        'supervisor': supervisor,
    }


def worker_manifest(engine, parameter_digest):
    value = {
        'schema': 1,
        'version_id': engine.version_id,
        'identity': copy.deepcopy(engine.identity),
        'parameter_digest': parameter_digest,
        'artifact_digest': engine.binding.artifact_digest,
        'profile_digest': engine.profile.digest,
        'profile': engine.profile.document,
        'supervisor': supervisor_identity() if engine.identity.get('processes') else None,
    }
    return {**value, 'manifest_digest': digest(value)}


def create_worker_app(engine, parameter_digest, token, *, max_active=16):
    if len(token) < 16:
        raise ValueError('worker密钥过短')
    manifest = worker_manifest(engine, parameter_digest)
    active = set()

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(lifespan=lifespan)

    def authorized(request):
        return hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token)

    def matches(request):
        return (
            request.headers.get('x-expected-worker') == manifest['manifest_digest']
            and worker_manifest(engine, parameter_digest) == manifest
        )

    def reject(request):
        if not authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        if not matches(request):
            return JSONResponse(
                {
                    'error': 'generation_mismatch',
                    'attempt': request.headers.get('x-request-id'),
                    'expected': request.headers.get('x-expected-worker'),
                    'dispatched': False,
                },
                status_code=409,
            )

    async def payload(request):
        raw = bytearray()
        async with asyncio.timeout(10):
            async for part in request.stream():
                raw.extend(part)
                if len(raw) > 65536:
                    raise ValueError('worker请求过大')
        return json.loads(raw)

    async def close_request(response, attempt):
        if response is not None:
            await response.aclose()
        try:
            released = await engine.wait_release(attempt, 10)
        except Exception:
            released = None
        if released:
            active.discard(attempt)

    async def while_connected(awaitable, request):
        task = asyncio.create_task(awaitable)

        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.02)

        watcher = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                {task, watcher}, timeout=90, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                return await task
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if watcher in done:
                raise ClientDisconnect()
            raise TimeoutError('worker response deadline')
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    @app.get('/identity')
    async def identity(request: Request):
        if not authorized(request):
            return JSONResponse({'error': 'unauthorized'}, status_code=401)
        ready = (
            worker_manifest(engine, parameter_digest) == manifest
            and await engine.ready()
            and len(active) < max_active
        )
        return {'manifest': manifest, 'ready': ready}

    @app.post('/tokenize')
    async def tokenize(request: Request):
        error = reject(request)
        if error is not None:
            return error
        body = await payload(request)
        ids = await (
            engine.engine_token_ids(body['messages'])
            if body.get('engine')
            else engine.token_ids(body['messages'])
        )
        return {'tokens': ids, 'manifest_digest': manifest['manifest_digest']}

    @app.get('/release/{attempt}')
    async def release(attempt: str, request: Request):
        error = reject(request)
        if error is not None:
            return error
        event = await engine.wait_release(attempt, 0.01)
        if event:
            active.discard(attempt)
        return {'release': event, 'manifest_digest': manifest['manifest_digest']}

    @app.post('/reconcile/{attempt}')
    async def reconcile(attempt: str, request: Request):
        error = reject(request)
        if error is not None:
            return error
        event = await engine.wait_release(attempt, 0.01)
        if not event:
            event = owned_termination(engine, manifest)
        if event:
            active.discard(attempt)
        return {'evidence': event, 'manifest_digest': manifest['manifest_digest']}

    @app.post('/chat')
    async def chat(request: Request):
        error = reject(request)
        if error is not None:
            return error
        attempt = request.headers.get('x-request-id', '')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', attempt):
            return JSONResponse({'error': 'invalid_attempt'}, status_code=422)
        body = await payload(request)
        try:
            if body.get('model') != engine.version_id:
                raise ValueError('wrong_model')
            engine.profile.payload(body)
        except (ValueError, AttributeError):
            return JSONResponse({'error': 'invalid_body'}, status_code=422)
        if attempt in active:
            return JSONResponse({'error': 'duplicate_active_attempt'}, status_code=409)
        if len(active) >= max_active or not await engine.ready():
            return JSONResponse({'error': 'not_ready'}, status_code=503)
        # 检查后无await地保留worker并发名额；生成请求不自动重试。
        if not matches(request):
            return reject(request)
        if attempt in active:
            return JSONResponse({'error': 'duplicate_active_attempt'}, status_code=409)
        if len(active) >= max_active:
            return JSONResponse({'error': 'worker_capacity_full'}, status_code=429)
        active.add(attempt)
        response = None
        transferred = False
        try:

            async def opening():
                nonlocal response
                response = await engine.open(attempt, body)
                return response

            await while_connected(opening(), request)
            if not await engine.ready():
                raise RuntimeError('local_worker_generation_changed')
            if body.get('stream') and response.status_code == 200:
                upstream = response

                class Relay(StreamingResponse):
                    async def __call__(self, scope, receive, send):
                        try:
                            await super().__call__(scope, receive, send)
                        finally:
                            await close_request(upstream, attempt)

                relay = Relay(
                    upstream.aiter_bytes(),
                    headers={'X-Actual-Worker': manifest['manifest_digest']},
                    media_type='text/event-stream',
                )
                transferred = True
                return relay
            data = await while_connected(response.aread(), request)
            return JSONResponse(
                json.loads(data),
                status_code=response.status_code,
                headers={'X-Actual-Worker': manifest['manifest_digest']},
            )
        finally:
            if not transferred:
                await close_request(response, attempt)

    return app


class RemoteEngine:
    """配置给定的实例代际不可变；GET ready不能替代生成端的expected校验。"""

    READY_TTL_SECONDS = 0.5
    READY_INTERVAL_SECONDS = 0.2

    def __init__(self, endpoint, token, manifest, *, client=None, tls=None):
        url = urlsplit(endpoint)
        if (
            url.scheme not in {'http', 'https'}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {'', '/'}
            or (url.scheme == 'http' and url.hostname not in {'localhost', '127.0.0.1', '::1'})
        ):
            raise ValueError('远端worker要求受控HTTPS endpoint；本机HTTP仅用于本地资格')
        if len(token) < 16 or digest(
            {k: v for k, v in manifest.items() if k != 'manifest_digest'}
        ) != manifest.get('manifest_digest'):
            raise ValueError('无效worker清单')
        self.manifest = copy.deepcopy(manifest)
        self.identity = copy.deepcopy(manifest['identity'])
        self.version_id = manifest['version_id']
        self.profile = ServingProfile(manifest['profile'], self.version_id)
        if self.profile.digest != manifest['profile_digest']:
            raise ValueError('worker profile不匹配')
        self.binding = SimpleNamespace(
            artifact_digest=manifest['artifact_digest'], manifest_sha256=manifest['manifest_digest']
        )
        self.parameter_digest = manifest['parameter_digest']
        self.endpoint = endpoint.rstrip('/')
        self.headers = {
            'Authorization': 'Bearer ' + token,
            'X-Expected-Worker': manifest['manifest_digest'],
        }
        verify = True
        if tls is not None:
            if url.scheme != 'https' or client is not None:
                raise ValueError('TLS配置要求HTTPS且由RemoteEngine创建客户端')
            from .tls import client_context

            verify = client_context(tls)
        self.client = client or httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=None,
            trust_env=False,
            verify=verify,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
        self.refusals = OrderedDict()
        # 远端代际由生成端 X-Expected-Worker 栅栏逐请求核对；身份探测的通过结果只短期复用。
        self.ready_cache = ReadinessCache(self.READY_TTL_SECONDS)

    async def readiness(self):
        probe = {'ready': False}
        started = asyncio.get_running_loop().time()
        try:
            response = await self.client.get('/identity', headers=self.headers, timeout=2)
            probe['http_status'] = response.status_code
            response.raise_for_status()
            value = response.json()
            probe.update(
                worker_ready=value.get('ready') is True,
                same_manifest=value.get('manifest') == self.manifest,
            )
            probe['ready'] = probe['worker_ready'] and probe['same_manifest']
        except Exception as error:
            probe['error'] = {'type': type(error).__name__, 'message': str(error)}
        probe['elapsed_seconds'] = asyncio.get_running_loop().time() - started
        return probe

    async def ready(self):
        return await self.ready_cache.check(self._probe_ready)

    async def _probe_ready(self):
        return (await self.readiness())['ready']

    def start_monitoring(self):
        """在服务事件循环内启动后台身份探测（由网关启动时调用）。"""
        self.ready_cache.start(self._probe_ready, self.READY_INTERVAL_SECONDS)

    def ready_nowait(self):
        """后台探测运行时不等待，直接返回最近一次身份探测结果；未运行时返回 None。"""
        if not self.ready_cache.monitoring():
            return None
        return self.ready_cache.fresh()

    async def _tokens(self, messages, engine):
        r = await self.client.post(
            '/tokenize',
            headers=self.headers,
            json={'messages': messages, 'engine': engine},
            timeout=10,
        )
        r.raise_for_status()
        v = r.json()
        if v['manifest_digest'] != self.manifest['manifest_digest']:
            raise RuntimeError('worker_changed')
        return v['tokens']

    async def token_ids(self, messages):
        return await self._tokens(messages, False)

    async def engine_token_ids(self, messages):
        return await self._tokens(messages, True)

    async def count_tokens(self, messages):
        return len(await self.token_ids(messages))

    def wire_payload(self, body):
        return self.profile.payload(body)

    async def open(self, attempt, body):
        self.profile.payload(body)
        req = self.client.build_request(
            'POST', '/chat', headers={**self.headers, 'X-Request-Id': attempt}, json=body
        )
        response = await self.client.send(req, stream=True)
        if response.status_code == 409:
            data = json.loads(await response.aread())
            if (
                data.get('error') == 'generation_mismatch'
                and data.get('attempt') == attempt
                and data.get('expected') == self.manifest['manifest_digest']
                and data.get('dispatched') is False
            ):
                self.refusals[attempt] = {
                    'status': 'NOT_DISPATCHED',
                    'request_id': attempt,
                    'worker': self.identity,
                    'reason': 'authenticated_generation_fence_rejection',
                }
                if len(self.refusals) > 128:
                    self.refusals.popitem(last=False)
        if (
            response.status_code == 200
            and response.headers.get('x-actual-worker') != self.manifest['manifest_digest']
        ):
            await response.aclose()
            raise RuntimeError('actual_worker_mismatch')
        return response

    async def wait_release(self, attempt, timeout):
        if attempt in self.refusals:
            return self.refusals[attempt]
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                # 正常清理也消费原监督者证明，避免只轮询请求日志直到旧桥退出。
                response = await self.client.post(
                    '/reconcile/' + attempt,
                    headers=self.headers,
                    json={},
                    timeout=min(2, max(0.01, deadline - asyncio.get_running_loop().time())),
                )
                response.raise_for_status()
                value = response.json()
                event = value['evidence']
                if value['manifest_digest'] != self.manifest['manifest_digest']:
                    return None
                if self._termination_matches(event):
                    return event
                if (
                    event
                    and event.get('worker_instance_id') == self.identity['worker_instance_id']
                    and event.get('startup_generation') == self.identity['startup_generation']
                    and event.get('request_id')
                    in {'chatcmpl-' + attempt, 'chatcmpl-' + attempt + '-0'}
                    and event.get('request_absent')
                    and not any(event.get('after_blocks', [['unknown']]))
                ):
                    return event
            except Exception:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.02)

    def _termination_matches(self, evidence):
        return (
            isinstance(evidence, dict)
            and evidence.get('status') == 'WORKER_GENERATION_TERMINATED'
            and evidence.get('worker') == self.identity
            and evidence.get('processes') == self.identity.get('processes')
            and evidence.get('supervisor') == self.manifest.get('supervisor')
            and evidence.get('supervisor') is not None
        )

    async def reconcile(self, attempt, old_identity, timeout):
        if old_identity != self.identity:
            return None
        try:
            r = await self.client.post(
                '/reconcile/' + attempt,
                headers=self.headers,
                json={},
                timeout=max(0.01, min(timeout, 3)),
            )
            r.raise_for_status()
            value = r.json()
            e = value['evidence']
            if value['manifest_digest'] != self.manifest['manifest_digest']:
                return None
            if self._termination_matches(e):
                return e
        except Exception:
            return None
        return await self.wait_release(attempt, timeout)

    async def close(self):
        await self.ready_cache.stop()
        await self.client.aclose()
