"""R1薄在线入口：固定版本、有限资源、绝对期限与可核验的引擎清理。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from .lab import digest
from .runtime import RuntimeStore
from .releases import ReleaseController, ReleaseError


@dataclass(frozen=True)
class ServiceLimits:
    max_inflight: int = 2
    max_queue: int = 16
    max_request_bytes: int = 65536
    max_messages: int = 32
    max_input_tokens: int = 2048
    max_tokens: int = 128
    context_tokens: int = 4096
    deadline_seconds: float = 60
    cancel_grace_seconds: float = 10

    def __post_init__(self):
        for name in (
            'max_inflight',
            'max_request_bytes',
            'max_messages',
            'max_input_tokens',
            'max_tokens',
            'context_tokens',
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('无效资源上限')
        if type(self.max_queue) is not int or self.max_queue < 0:
            raise ValueError('无效队列上限')
        for value in (self.deadline_seconds, self.cancel_grace_seconds):
            if type(value) not in {float, int} or not math.isfinite(value) or value <= 0:
                raise ValueError('无效期限')


class Rejected(Exception):
    def __init__(self, status, code):
        self.status = status
        self.code = code


class DownstreamDeadline(Exception):
    """不继承OSError，避免ASGI层把超时误转换成客户端断开。"""


@dataclass
class Call:
    arrival: str
    received: float
    deadline: float
    capacity: bool = False
    slot: bool = False
    attempt: str | None = None
    dispatched: bool = False
    count_task: asyncio.Task | None = None
    upstream: object = None
    cleanup_task: asyncio.Task | None = None
    engine: object = None
    lease: bool = False
    route_generation: int | None = None
    profile_digest: str | None = None
    replica_id: str | None = None
    replica_slot: bool = False
    slot_semaphore: object = None
    observation: dict = field(
        default_factory=lambda: {
            'usage': None,
            'ttft_seconds': None,
            'content': '',
            'partial': False,
            'phase_ms': {},
        }
    )

    def mark(self, phase):
        """记录从到达起算的阶段时刻（毫秒），用于拆解网关在派发前后的额外耗时。"""
        self.observation['phase_ms'][phase] = round((time.monotonic() - self.received) * 1000, 3)


class ManagedStream(StreamingResponse):
    """即使连接在生成器第一步前断开，响应所有者也会完成终态与清理。"""

    def __init__(self, service, call, headers):
        self.service = service
        self.call = call
        super().__init__(service.stream(call), media_type='text/event-stream', headers=headers)

    async def __call__(self, scope, receive, send):
        async def bounded_send(message):
            remaining = self.call.deadline - time.monotonic()
            if remaining <= 0:
                raise DownstreamDeadline()
            try:
                await asyncio.wait_for(send(message), remaining)
            except TimeoutError:
                raise DownstreamDeadline() from None

        try:
            await super().__call__(scope, receive, bounded_send)
        except DownstreamDeadline:
            # 已无法在deadline内向慢客户端发送错误帧；关闭不完整流，不伪造DONE。
            self.service.finish(self.call, 'timed_out', 504, 'downstream_deadline')
        except ClientDisconnect:
            self.service.finish(self.call, 'cancelled', 499, 'client_disconnected')
        finally:
            self.service.finish(self.call, 'cancelled', 499, 'stream_closed')
            if hasattr(self.body_iterator, 'aclose'):
                with contextlib.suppress(Exception):
                    await self.body_iterator.aclose()
            await self.service.settle(self.call)


class Service:
    def __init__(self, engine, root, limits):
        self.engine = engine
        self.limits = limits
        self.store = RuntimeStore(root)
        self.capacity_used = 0
        self.capacity_lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(limits.max_inflight)
        self.healthy = True
        self.cleanups = set()
        self.closed = False
        self.pool = None

    async def startup(self):
        for row in self.store.unresolved():
            evidence = None
            engine = (
                (
                    self.pool.recovery_engine(row)
                    if getattr(self.pool, 'replicated', False)
                    else self.pool.engines[row['version']]
                )
                if self.pool
                else self.engine
            )
            if hasattr(engine, 'reconcile'):
                evidence = await engine.reconcile(
                    row['attempt_id'], json.loads(row['worker']), self.limits.cancel_grace_seconds
                )
            if evidence:
                self.store.cleanup(row['id'], 'confirmed', evidence)
                self.store.release_lease(row['id'])
        unresolved = self.store.unresolved()
        self.healthy = not unresolved
        if getattr(self.pool, 'replicated', False):
            self.healthy = True
            for row in unresolved:
                engine = self.pool.recovery_engine(row)
                if engine is None:
                    self.healthy = False
                else:
                    self.pool.isolate(engine)
        # 对账完成后再启动后台主动健康探测：请求路径只核对进程身份并读取探测结果，不再逐请求发探测。
        for engine in self.pool.engines.values() if self.pool else [self.engine]:
            if hasattr(engine, 'start_monitoring'):
                engine.start_monitoring()

    async def ready(self):
        if self.store.failed is not None:
            return False
        if self.pool:
            if getattr(self.pool, 'replicated', False):
                await self.pool.refresh()
                route = self.pool.route()
                versions = {route['active'], route['canary']} - {None}
                return (
                    bool(versions)
                    and self.healthy
                    and not self.closed
                    and all(
                        self.pool.row(v)['state'] in {'active', 'canary'} and self.pool.eligible(v)
                        for v in versions
                    )
                )
            route = self.pool.route()
            versions = {route['active'], route['canary']} - {None}
            if not versions or not self.healthy or self.closed:
                return False
            for version in versions:
                if (
                    self.pool.row(version)['state'] not in {'active', 'canary'}
                    or not await self.pool.engines[version].ready()
                ):
                    return False
            return True
        return self.healthy and not self.closed and await self.engine.ready()

    def call_engine(self, call):
        return call.engine or self.engine

    async def call_ready(self, call):
        if call.replica_id and self.pool.replica(call.replica_id)['state'] not in {
            'ready',
            'draining',
        }:
            return False
        if (
            call.replica_id
            and self.pool.replica(call.replica_id)['identity'] != self.call_engine(call).identity
        ):
            return False
        if call.profile_digest and self.call_engine(call).profile.digest != call.profile_digest:
            return False
        return (
            self.healthy
            and not self.closed
            and self.store.failed is None
            and await self.call_engine(call).ready()
        )

    async def acquire_slot(self, call):
        if call.slot_semaphore is not None:
            await call.slot_semaphore.acquire()
            call.replica_slot = True
        await self.slots.acquire()
        call.slot = True

    async def open_upstream(self, call, body):
        # 在子任务完成前转移所有权，避免同一轮disconnect与完成竞争时遗失资源。
        call.upstream = await self.call_engine(call).open(call.attempt, body)
        return call.upstream

    def finish(self, call, state, status, code=None):
        observation = {
            **call.observation,
            'http_status': status,
            'code': code,
            'elapsed_seconds': time.monotonic() - call.received,
            'version_id': call.engine.version_id
            if call.engine
            else (None if self.pool else self.engine.version_id),
            'route_generation': call.route_generation,
        }
        if call.profile_digest:
            observation['profile_digest'] = call.profile_digest
        if self.store.failed is not None:
            # 账本已关闭写入：网关已停止准入，这里只保证后续上游关闭与资源归还照常进行。
            return
        # 终态先执行、不单独提交；由调用方在对外可见之前或清理结束时组提交。
        self.store.finish(call.arrival, state, observation, durable=False)

    def commit_visible(self):
        """对外可见的结果（流的最后一帧）发出之前，把已执行的账本操作组提交。"""
        if self.store.failed is None:
            self.store.durable()

    async def cleanup(self, call):
        # count_task被shield保护；超时请求返回后仍占容量，直到后台计数实际退出。
        if call.count_task is not None and not call.count_task.done():
            with contextlib.suppress(Exception):
                await call.count_task
        if call.dispatched:
            try:
                async with asyncio.timeout(self.limits.cancel_grace_seconds):
                    if call.upstream is not None:
                        await call.upstream.aclose()
                    evidence = await self.call_engine(call).wait_release(
                        call.attempt, self.limits.cancel_grace_seconds
                    )
            except Exception as exc:
                evidence = None
                call.observation['cleanup_error'] = type(exc).__name__
        try:
            if call.dispatched:
                self.store.cleanup(
                    call.arrival,
                    'confirmed' if evidence else 'unconfirmed',
                    evidence or {'reason': 'release_not_confirmed'},
                    durable=False,
                )
                if not evidence:
                    if getattr(self.pool, 'replicated', False):
                        self.pool.isolate(self.call_engine(call))
                    else:
                        self.healthy = False
            if call.lease and (
                not call.dispatched or self.store.row(call.arrival)['cleanup'] == 'confirmed'
            ):
                self.store.release_lease(call.arrival, durable=False)
                call.lease = False
            # 清理边界：终态、清理证据与租约释放一次提交，然后才归还执行槽和容量。
            self.store.durable()
        except Exception:
            if self.store.failed is None:
                raise
            # 账本已失败：网关已停止准入（ready 与 handle 都会拒绝），这里不再上抛，只保证归还资源。
            self.healthy = False
        finally:
            self.release_capacity(call)

    def release_capacity(self, call):
        """归还执行槽与容量；即使账本已失败也必须执行，避免容量被永久占住。"""
        if call.slot:
            self.slots.release()
            call.slot = False
        if call.replica_slot:
            call.slot_semaphore.release()
            call.replica_slot = False
        if call.capacity:
            # 单线程事件循环里计数器的读改写不跨 await，无需再拿锁。
            self.capacity_used -= 1
            call.capacity = False

    async def settle(self, call, *, wait=True):
        if call.cleanup_task is None:
            call.cleanup_task = asyncio.create_task(self.cleanup(call))
            self.cleanups.add(call.cleanup_task)
            call.cleanup_task.add_done_callback(self.cleanups.discard)
        if wait:
            await asyncio.shield(call.cleanup_task)

    async def shutdown(self):
        self.closed = True
        if self.cleanups:
            done, pending = await asyncio.wait(
                self.cleanups, timeout=self.limits.cancel_grace_seconds + 1
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        try:
            for engine in self.pool.engines.values() if self.pool else [self.engine]:
                await engine.close()
        finally:
            self.store.close()

    async def wait(self, task, request, deadline, *, cancel=True):
        """正文已读完后才监听断开，避免竞争消费ASGI请求体。"""

        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.02)

        watcher = asyncio.create_task(disconnected())
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            done, _ = await asyncio.wait(
                {task, watcher}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                raise ClientDisconnect()
            if task not in done:
                raise TimeoutError()
            return task.result()
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            if cancel and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def validate(self, body):
        allowed = {'model', 'messages', 'max_tokens', 'stream'}
        application = getattr(self.pool, 'application_policy', None)
        if application is not None:
            allowed.add('action_choices')
        engines = (
            (self.pool.engines.values() if self.pool else [self.engine])
            if application is None
            else []
        )
        length_control = (
            application is None
            and bool(engines)
            and all(
                getattr(e, 'profile', None) is not None
                and e.profile.document.get('name') == 'n2_length_controlled_v1'
                for e in engines
            )
        )
        if length_control:
            allowed.add('ignore_eos')
        if not isinstance(body, dict) or set(body) - allowed:
            raise Rejected(422, 'unsupported_body')
        if not isinstance(body.get('model'), str):
            raise Rejected(422, 'model_schema')
        if not self.pool and body['model'] != self.engine.version_id:
            raise Rejected(404, 'unknown_model')
        messages = body.get('messages')
        maximum = body.get('max_tokens', self.limits.max_tokens)
        stream = body.get('stream', False)
        if not isinstance(messages, list) or not 1 <= len(messages) <= self.limits.max_messages:
            raise Rejected(422, 'message_count')
        for message in messages:
            if (
                not isinstance(message, dict)
                or set(message) != {'role', 'content'}
                or not isinstance(message['role'], str)
                or message['role'] not in {'system', 'user', 'assistant'}
                or not isinstance(message['content'], str)
            ):
                raise Rejected(422, 'message_schema')
        if (
            type(maximum) is not int
            or not 1 <= maximum <= self.limits.max_tokens
            or type(stream) is not bool
        ):
            raise Rejected(422, 'generation_budget')
        result = {
            'model': body['model'],
            'messages': messages,
            'max_tokens': maximum,
            'stream': stream,
        }
        if length_control:
            if type(body.get('ignore_eos', False)) is not bool:
                raise Rejected(422, 'generation_budget')
            result['ignore_eos'] = body.get('ignore_eos', False)
        if application is not None:
            result['action_choices'] = body.get('action_choices')
            try:
                application.validate_request(result)
            except (ValueError, TypeError, KeyError, RecursionError):
                raise Rejected(422, 'application_request_contract') from None
        return result

    async def handle(self, request):
        if self.store.failed is not None:
            return JSONResponse({'error': {'code': 'ledger_unavailable'}}, status_code=503)
        now = time.monotonic()
        call = Call(uuid.uuid4().hex, now, now + self.limits.deadline_seconds)
        key = request.headers.get('idempotency-key')
        valid_key = key is None or bool(re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', key))
        # 到达记录不单独提交：任何对外可见动作（派发或响应）之前都有一次组提交覆盖它。
        self.store.record_arrival(call.arrival, key if valid_key else None, durable=False)
        call.mark('arrival_recorded')
        headers = {'X-Arrival-Id': call.arrival}
        if not self.pool:
            headers['X-Model-Version'] = self.engine.version_id
        streaming = False
        try:
            if not valid_key:
                raise Rejected(422, 'invalid_idempotency_key')
            if not self.healthy or self.closed:
                raise Rejected(503, 'worker_not_ready')
            async with self.capacity_lock:
                if self.capacity_used >= self.limits.max_inflight + self.limits.max_queue:
                    raise Rejected(429, 'capacity_full')
                self.capacity_used += 1
                call.capacity = True
            if not self.pool and not await asyncio.wait_for(
                self.ready(), max(0.001, call.deadline - time.monotonic())
            ):
                raise Rejected(503, 'worker_not_ready')
            raw = bytearray()
            async with asyncio.timeout(max(0, call.deadline - time.monotonic())):
                async for part in request.stream():
                    if len(raw) + len(part) > self.limits.max_request_bytes:
                        raise Rejected(413, 'request_too_large')
                    raw.extend(part)
            call.mark('body_read')
            try:
                body = self.validate(json.loads(raw))
            except (ValueError, UnicodeError):
                raise Rejected(422, 'invalid_json')
            call.mark('validated')
            if self.pool:
                if getattr(self.pool, 'replicated', False):
                    await self.wait(
                        asyncio.create_task(self.pool.refresh()), request, call.deadline
                    )
                    call.mark('pool_refreshed')
                lease = self.pool.pin(body['model'], call.arrival, durable=False)
                call.mark('pinned')
                call.engine = lease['engine']
                call.lease = True
                call.route_generation = lease['route_generation']
                call.profile_digest = lease.get('profile_digest')
                call.replica_id = lease.get('replica_id')
                call.slot_semaphore = lease.get('slot_semaphore')
                if call.replica_id:
                    headers.update(
                        {
                            'X-Replica-Id': call.replica_id,
                            'X-Worker-Generation': lease['worker_generation'],
                        }
                    )
                body['model'] = call.engine.version_id
                headers.update(
                    {
                        'X-Model-Version': call.engine.version_id,
                        'X-Route-Generation': str(call.route_generation),
                        'X-Worker-Instance': call.engine.identity['worker_instance_id'],
                    }
                )
                if call.profile_digest:
                    headers.update(
                        {
                            'X-Serving-Profile': call.profile_digest,
                            'X-Qualification-Scope': self.pool.scope,
                        }
                    )
                if getattr(self.pool, 'application_policy', None) is not None:
                    headers.update(
                        {
                            'X-Application-Qualification': lease['application_history'][
                                'qualification_scope'
                            ],
                            'X-Application-Evidence': lease['application_history'][
                                'evidence_digest'
                            ],
                        }
                    )
                if not await self.wait(
                    asyncio.create_task(self.call_ready(call)), request, call.deadline
                ):
                    raise Rejected(503, 'worker_not_ready')
                call.mark('ready_checked')
            else:
                call.engine = self.engine
            call.count_task = asyncio.create_task(
                self.call_engine(call).token_ids(body['messages'])
                if call.profile_digest
                else self.call_engine(call).count_tokens(body['messages'])
            )
            counted = await self.wait(call.count_task, request, call.deadline, cancel=False)
            call.mark('tokenized')
            count = len(counted) if call.profile_digest else counted
            if type(count) is not int or count < 0:
                raise RuntimeError('invalid tokenizer count')
            if (
                count > self.limits.max_input_tokens
                or count + body['max_tokens'] > self.limits.context_tokens
            ):
                raise Rejected(422, 'context_budget')
            call.observation['input_tokens_counted'] = count
            call.attempt = uuid.uuid4().hex
            if not self.store.reserve_attempt(
                call.arrival,
                call.attempt,
                self.call_engine(call).version_id,
                digest(body),
                durable=False,
            ):
                raise Rejected(409, 'duplicate_key')
            call.mark('reserved')
            self.store.save_payload(call.arrival, body)
            if call.profile_digest:
                self.store.save_wire(
                    call.arrival,
                    self.call_engine(call).wire_payload(body),
                    call.profile_digest,
                    counted,
                )
            call.mark('payload_saved')
            await self.wait(asyncio.create_task(self.acquire_slot(call)), request, call.deadline)
            call.mark('slot_acquired')
            if time.monotonic() >= call.deadline:
                raise TimeoutError()
            if not await self.wait(
                asyncio.create_task(self.call_ready(call)), request, call.deadline
            ):
                raise Rejected(503, 'worker_generation_changed')
            call.mark('ready_rechecked')
            call.observation['dispatch_seconds'] = time.monotonic() - call.received
            self.store.dispatched(call.arrival, self.call_engine(call).identity)
            call.dispatched = True
            call.mark('dispatch_recorded')
            await self.wait(
                asyncio.create_task(self.open_upstream(call, body)), request, call.deadline
            )
            call.mark('upstream_headers')
            if call.upstream.status_code != 200:
                raise Rejected(502, 'engine_rejected')
            if body['stream']:
                streaming = True
                return ManagedStream(self, call, headers)
            data = await self.wait(
                asyncio.create_task(call.upstream.aread()), request, call.deadline
            )
            result = json.loads(data)
            if not result.get('choices') or not result['choices'][0].get('finish_reason'):
                raise RuntimeError('incomplete engine response')
            call.observation.update(
                usage=result.get('usage'),
                content=result['choices'][0]['message'].get('content', ''),
                response=result,
            )
            self.finish(call, 'completed', 200)
            return JSONResponse(result, headers=headers)
        except (Rejected, ReleaseError) as exc:
            self.finish(call, 'failed' if call.dispatched else 'rejected', exc.status, exc.code)
            return JSONResponse(
                {'error': {'code': exc.code}}, status_code=exc.status, headers=headers
            )
        except TimeoutError:
            self.finish(call, 'timed_out', 504, 'deadline_exceeded')
            return JSONResponse(
                {'error': {'code': 'deadline_exceeded'}}, status_code=504, headers=headers
            )
        except (ClientDisconnect, asyncio.CancelledError) as exc:
            self.finish(call, 'cancelled', 499, 'client_disconnected')
            if isinstance(exc, asyncio.CancelledError):
                raise
            return JSONResponse(
                {'error': {'code': 'client_disconnected'}}, status_code=499, headers=headers
            )
        except Exception as exc:
            self.finish(call, 'failed', 502, type(exc).__name__)
            return JSONResponse(
                {'error': {'code': 'engine_failure', 'type': type(exc).__name__}},
                status_code=502,
                headers=headers,
            )
        finally:
            if not streaming:
                # 响应可以超时返回，但未完成前处理的容量不提前归还。
                wait = not (call.count_task is not None and not call.count_task.done())
                await self.settle(call, wait=wait)
                # 非流式响应在handle返回后才发送；未等待清理时也先提交本请求已执行的记录。
                self.store.durable()

    async def stream(self, call):
        complete = False
        finish = None
        raw_events = []
        call.observation['events'] = raw_events
        try:
            iterator = call.upstream.aiter_lines().__aiter__()
            while True:
                remaining = call.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                try:
                    line = await asyncio.wait_for(anext(iterator), remaining)
                except StopAsyncIteration:
                    break
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                raw_events.append(data)
                if data == '[DONE]':
                    if finish is None:
                        raise RuntimeError('DONE without finish_reason')
                    complete = True
                    call.observation.update(partial=False, finish_reason=finish, events=raw_events)
                    # 终态先落盘再把 [DONE] 交给客户端：客户端看到完成时，账本里一定是 completed。
                    self.finish(call, 'completed', 200)
                    self.commit_visible()
                    yield b'data: [DONE]\n\n'
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError('upstream stream error')
                if event.get('usage'):
                    call.observation['usage'] = event['usage']
                for choice in event.get('choices', []):
                    content = choice.get('delta', {}).get('content') or ''
                    if content and call.observation['ttft_seconds'] is None:
                        call.observation['ttft_seconds'] = time.monotonic() - call.received
                        call.mark('first_content')
                    call.observation['content'] += content
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
                call.observation['partial'] = bool(call.observation['content'])
                yield ('data: ' + data + '\n\n').encode()
            if not complete:
                raise RuntimeError('stream ended without DONE')
        except asyncio.CancelledError:
            call.observation['events'] = raw_events
            self.finish(call, 'cancelled', 499, 'client_disconnected')
            raise
        except TimeoutError:
            call.observation['events'] = raw_events
            self.finish(call, 'timed_out', 504, 'deadline_exceeded')
            self.commit_visible()
            yield b'data: {"error":{"code":"deadline_exceeded"}}\n\n'
        except Exception as exc:
            call.observation['events'] = raw_events
            self.finish(call, 'failed', 502, type(exc).__name__)
            self.commit_visible()
            yield b'data: {"error":{"code":"incomplete_stream"}}\n\n'
        finally:
            await self.settle(call)


def _http_app(service):
    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            await service.startup()
        except BaseException:
            await service.shutdown()
            raise
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.service = service

    @app.get('/health/ready')
    async def health():
        ready = await service.ready()
        return JSONResponse(
            {
                'ready': ready,
                'version_id': service.engine.version_id if not service.pool else None,
                'route': service.pool.route() if service.pool else None,
                'capacity_used': service.capacity_used,
            },
            status_code=200 if ready else 503,
        )

    @app.post('/v1/chat/completions')
    async def completions(request: Request):
        return await service.handle(request)

    return app


def create_app(engine, runtime_root, limits=None):
    from .profile import check_runtime_contract, LEGACY_SCOPE

    check_runtime_contract(runtime_root, LEGACY_SCOPE)
    return _http_app(Service(engine, runtime_root, limits or ServiceLimits()))


def create_release_app(
    engines,
    runtime_root,
    parameters,
    panel,
    admin_token,
    *,
    limits=None,
    allow_rebind=(),
    application_results=None,
):
    """管理入口只操作启动时登记的实例；不会接收客户端提供的模型路径。"""
    if not isinstance(admin_token, str) or len(admin_token) < 16:
        raise ValueError('管理密钥过短')
    from .profile import TRANSPORT_SCOPE, LEGACY_SCOPE, check_runtime_contract

    scope = panel.get('qualification_scope', LEGACY_SCOPE)
    expected = (
        {
            'panel_digest': digest(panel),
            'profiles': {v: e.profile.digest for v, e in engines.items()},
            'application_results': application_results,
        }
        if scope == TRANSPORT_SCOPE
        else None
    )
    check_runtime_contract(runtime_root, scope, expected)
    service = Service(next(iter(engines.values())), runtime_root, limits or ServiceLimits())
    try:
        service.pool = ReleaseController(
            service.store,
            engines,
            parameters,
            digest(panel),
            allow_rebind=allow_rebind,
            scope=scope,
            application_results=application_results,
        )
    except Exception:
        service.store.close()
        raise
    app = _http_app(service)
    pool = service.pool
    qualifying = asyncio.Lock()

    def authorize(request):
        if not hmac.compare_digest(
            request.headers.get('authorization', ''), 'Bearer ' + admin_token
        ):
            raise ReleaseError('unauthorized', 401)

    async def payload(request):
        raw = bytearray()
        async with asyncio.timeout(5):
            async for part in request.stream():
                raw.extend(part)
                if len(raw) > 4096:
                    raise ReleaseError('admin_request_too_large', 413)
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ReleaseError('invalid_admin_json', 422)
        if not isinstance(result, dict):
            raise ReleaseError('invalid_admin_body', 422)
        return result

    @app.exception_handler(ReleaseError)
    async def management_error(request, exc):
        return JSONResponse({'error': {'code': exc.code}}, status_code=exc.status)

    @app.get('/admin/releases')
    async def releases(request: Request):
        authorize(request)
        return pool.snapshot()

    @app.post('/admin/route')
    async def route(request: Request):
        authorize(request)
        body = await payload(request)
        if set(body) - {
            'operation',
            'target',
            'expected_generation',
            'percentage',
        } or not isinstance(body.get('target'), str):
            raise ReleaseError('invalid_route_body', 422)
        target = body['target']
        pool.row(target)
        if not service.healthy or not await asyncio.wait_for(engines[target].ready(), 2):
            raise ReleaseError('target_worker_not_ready', 503)
        return pool.change(
            body.get('operation'),
            target,
            body.get('expected_generation'),
            percentage=body.get('percentage', 0),
        )

    @app.post('/admin/releases/{version}/qualify')
    async def qualify(version: str, request: Request):
        authorize(request)
        if pool.row(version)['state'] != 'loading':
            raise ReleaseError('qualification_not_allowed')
        if qualifying.locked():
            raise ReleaseError('qualification_in_progress')
        if not service.healthy or pool.pending(version):
            raise ReleaseError('orphan_leases_not_reconciled', 503)
        async with qualifying:
            from .qualification import qualify_engine

            output = service.store.root / 'qualifications' / (version + '-' + uuid.uuid4().hex)
            receipt = await qualify_engine(
                engines[version],
                panel,
                output,
                store=service.store,
                cleanups=service.cleanups,
                application_result=pool.application_results.get(version),
            )
            service.healthy = not service.store.unresolved()
            pool.qualified(version, receipt)
            return JSONResponse(receipt, status_code=200 if receipt['passed'] else 422)

    @app.post('/admin/releases/{version}/drain')
    async def drain(version: str, request: Request):
        authorize(request)
        body = await payload(request)
        seconds = body.get('timeout_seconds', 0)
        if (
            set(body) - {'timeout_seconds'}
            or type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or not 0 <= seconds <= 10
        ):
            raise ReleaseError('invalid_drain_deadline', 422)
        pool.drain(version)
        until = time.monotonic() + seconds
        while pool.pending(version) and time.monotonic() < until:
            await asyncio.sleep(0.02)
        pending = pool.pending(version)
        return JSONResponse(
            {'drained': pending == 0, 'pending': pending, 'version_id': version},
            status_code=200 if not pending else 409,
        )

    @app.post('/admin/releases/{version}/retire')
    async def retire(version: str, request: Request):
        authorize(request)
        return pool.retire(version)

    return app


def main():
    from .engine import EngineBinding, HttpEngine
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--port', type=int, default=19081)
    parser.add_argument('--config', default='configs/serving/local-smoke.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    engine = HttpEngine(EngineBinding.load(args.binding))
    app = create_app(engine, args.runtime, ServiceLimits(**config['limits']))
    uvicorn.run(app, host='127.0.0.1', port=args.port, timeout_graceful_shutdown=15)


if __name__ == '__main__':
    main()
