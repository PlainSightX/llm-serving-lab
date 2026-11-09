"""多副本管理入口；复用原请求执行器，不给客户端开放后端配置。"""

import asyncio
import hmac
import json
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from .online import Service, ServiceLimits, _http_app
from .replicas import ReplicaController, REPLICA_SCOPE, replica_contract
from .releases import ReleaseError
from .profile import check_runtime_contract, TRANSPORT_SCOPE
from .lab import digest, canonical


def create_replica_app(
    engines,
    root,
    parameters,
    panel,
    admin_token,
    capacities,
    applications,
    *,
    limits=None,
    allow_rebind=(),
    application_policy=None,
):
    if not isinstance(admin_token, str) or len(admin_token) < 16:
        raise ValueError('管理密钥过短')
    if panel.get('qualification_scope') != TRANSPORT_SCOPE:
        raise ValueError('多副本要求明确工程预资格')
    scope = REPLICA_SCOPE
    effective_limits = limits or ServiceLimits()
    if application_policy is not None:
        from .application_release import POOL_SCOPE, validate_transport_panel

        validate_transport_panel(panel)
        scope = POOL_SCOPE
        if (
            effective_limits.max_input_tokens != 3500
            or effective_limits.max_tokens != 64
            or effective_limits.context_tokens != 4096
            or effective_limits.max_messages < 2
            or effective_limits.max_request_bytes != 65536
        ):
            raise ValueError('应用入口预算与评价合同不符')
    contract = replica_contract(
        engines, parameters, digest(panel), capacities, applications, application_policy
    )
    check_runtime_contract(root, scope, contract)
    service = Service(next(iter(engines.values())), root, effective_limits)
    try:
        service.pool = ReplicaController(
            service.store,
            engines,
            parameters,
            digest(panel),
            capacities,
            applications,
            allow_rebind=allow_rebind,
            application_policy=application_policy,
        )
    except Exception:
        service.store.close()
        raise
    pool = service.pool
    app = _http_app(service)
    qualifying = asyncio.Lock()

    def authorize(request):
        if not hmac.compare_digest(
            request.headers.get('authorization', ''), 'Bearer ' + admin_token
        ):
            raise ReleaseError('unauthorized', 401)

    async def body(request):
        raw = bytearray()
        async with asyncio.timeout(5):
            async for part in request.stream():
                raw.extend(part)
                if len(raw) > 4096:
                    raise ReleaseError('admin_request_too_large', 413)
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ReleaseError('invalid_admin_body', 422)
        if not isinstance(value, dict):
            raise ReleaseError('invalid_admin_body', 422)
        return value

    @app.exception_handler(ReleaseError)
    async def error(request, exc):
        return JSONResponse({'error': {'code': exc.code}}, status_code=exc.status)

    @app.get('/admin/releases')
    async def state(request: Request):
        authorize(request)
        return pool.snapshot()

    @app.post('/admin/route')
    async def route(request: Request):
        authorize(request)
        value = await body(request)
        if set(value) - {
            'operation',
            'target',
            'expected_generation',
            'percentage',
        } or not isinstance(value.get('target'), str):
            raise ReleaseError('invalid_route_body', 422)
        # 撤回只关闭新路由，不依赖故障worker响应，也不解除其未清lease。
        withdrawing = application_policy is not None and value.get('operation') == 'withdraw'
        if not withdrawing:
            await asyncio.wait_for(pool.refresh(), 3)
        if not service.healthy and not withdrawing:
            raise ReleaseError('controller_not_ready', 503)
        return pool.change(
            value.get('operation'),
            value['target'],
            value.get('expected_generation'),
            percentage=value.get('percentage', 0),
        )

    @app.post('/admin/replicas/{rid}/qualify')
    async def qualify(rid: str, request: Request):
        authorize(request)
        row = pool.replica(rid)
        if row['state'] != 'loading' or pool.pending_replica(rid):
            raise ReleaseError('replica_not_qualifiable')
        if qualifying.locked():
            raise ReleaseError('qualification_in_progress')
        async with qualifying:
            from .qualification import qualify_engine

            output = service.store.root / 'qualifications' / (rid + '-' + uuid.uuid4().hex)
            if application_policy is None:
                receipt = await qualify_engine(
                    engines[rid],
                    panel,
                    output,
                    store=service.store,
                    cleanups=service.cleanups,
                    application_result=applications[engines[rid].version_id],
                )
            else:
                from .application_release import qualify_application_worker

                receipt = await qualify_application_worker(
                    engines[rid],
                    panel,
                    output,
                    service.store,
                    application_policy,
                    applications[engines[rid].version_id],
                    service.cleanups,
                )
            pool.qualified(rid, receipt)
            await pool.refresh()
            return JSONResponse(receipt, status_code=200 if receipt['passed'] else 422)

    @app.post('/admin/replicas/{rid}/drain')
    async def drain(rid: str, request: Request):
        authorize(request)
        pool.drain_replica(rid)
        pending = pool.pending_replica(rid)
        return JSONResponse(
            {'replica_id': rid, 'drained': pending == 0, 'pending': pending},
            status_code=200 if pending == 0 else 409,
        )

    @app.post('/admin/replicas/{rid}/retire')
    async def retire(rid: str, request: Request):
        authorize(request)
        return pool.retire_replica(rid)

    @app.post('/admin/replicas/{rid}/reload')
    async def reload_replica(rid: str, request: Request):
        authorize(request)
        pool.reload_replica(rid)
        return pool.replica(rid)

    @app.post('/admin/releases/{version}/drain')
    async def drain_version(version: str, request: Request):
        authorize(request)
        pool.drain(version)
        pending = pool.pending(version)
        return JSONResponse(
            {'version_id': version, 'drained': pending == 0, 'pending': pending},
            status_code=409 if pending else 200,
        )

    @app.post('/admin/releases/{version}/retire')
    async def retire_version(version: str, request: Request):
        authorize(request)
        return pool.retire(version)

    @app.post('/admin/replicas/{rid}/reconcile')
    async def reconcile(rid: str, request: Request):
        authorize(request)
        pool.replica(rid)
        engine = engines[rid]
        for row in service.store.unresolved():
            if row['worker'] != canonical(engine.identity):
                continue
            evidence = await engine.reconcile(
                row['attempt_id'], json.loads(row['worker']), service.limits.cancel_grace_seconds
            )
            if evidence:
                service.store.cleanup(row['id'], 'confirmed', evidence)
                service.store.release_lease(row['id'])
        return {'replica_id': rid, 'pending': pool.pending_replica(rid)}

    return app
