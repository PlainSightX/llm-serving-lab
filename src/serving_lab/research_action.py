"""独立研究动作的事前预约、实际wire与释放事实；没有自动重试入口。"""

import asyncio
import json
import time
import uuid
from pathlib import Path
from .execution import write_once
from .lab import digest


async def check_readiness(engine, attempts=1):
    """只复查无生成副作用的就绪读取，代际或认证拒绝立即停止。"""
    if type(attempts) is not int or attempts not in (1, 2):
        raise ValueError('就绪复查次数只能为1或2')
    checks = []
    for index in range(attempts):
        probe_method = getattr(engine, 'readiness', None)
        probe = await probe_method() if callable(probe_method) else {'ready': await engine.ready()}
        checks.append(probe)
        if probe['ready']:
            return True, checks
        if probe.get('same_manifest') is False or probe.get('http_status') in (401, 403):
            break
        if index + 1 < attempts:
            await asyncio.sleep(0.1)
    return False, checks


async def generate_action(
    engine,
    store,
    output,
    messages,
    ids,
    metadata,
    *,
    choices=None,
    readiness_attempts=1,
    max_tokens=64,
    ignore_eos=None,
):
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('输出预算必须为正整数')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrival = uuid.uuid4().hex
    attempt = uuid.uuid4().hex
    body = {
        'model': engine.version_id,
        'messages': messages,
        'max_tokens': max_tokens,
        'stream': False,
    }
    if choices is not None:
        body['action_choices'] = choices
    if ignore_eos is not None:
        body['ignore_eos'] = ignore_eos
    wire = engine.wire_payload(body)
    started = time.monotonic()
    reservation = {
        'arrival_id': arrival,
        'attempt_id': attempt,
        'metadata': metadata,
        'profile_digest': engine.profile.digest,
        'worker': engine.identity,
        'wire': wire,
        'input_token_ids': ids,
        'started_monotonic': started,
    }
    write_once(output / 'reservation.json', reservation)
    store.record_arrival(arrival, None)
    store.bind_arrival(arrival, engine.version_id, engine.identity, metadata)
    store.reserve_attempt(arrival, attempt, engine.version_id, digest(body))
    store.save_payload(arrival, body)
    store.save_wire(arrival, wire, engine.profile.digest, ids)
    response = None
    dispatched = False
    released = None
    receipt = {'protocol_valid': False, 'arrival_id': arrival, 'attempt_id': attempt}
    try:
        async with asyncio.timeout(90):
            ready, checks = await check_readiness(engine, readiness_attempts)
            receipt['readiness_checks'] = checks
            if not ready:
                raise RuntimeError('worker_not_ready')
            remote = await engine.engine_token_ids(messages)
            if remote != ids:
                raise RuntimeError('input_token_mismatch')
            store.dispatched(arrival, engine.identity)
            dispatched = True
            response = await engine.open(attempt, body)
            raw = await response.aread()
            (output / 'response.raw').write_bytes(raw)
            if response.status_code != 200:
                raise RuntimeError('engine_http_' + str(response.status_code))
            value = json.loads(raw)
            choice = value['choices'][0]
            usage = value['usage']
            content = choice['message']['content']
            finish = choice['finish_reason']
            valid = (
                value.get('id') == 'chatcmpl-' + attempt
                and len(value['choices']) == 1
                and value['model'] == engine.version_id
                and isinstance(content, str)
                and finish in {'stop', 'length'}
                and usage['prompt_tokens'] == len(ids)
                and 0 < usage['completion_tokens'] <= max_tokens
                and usage['total_tokens'] == len(ids) + usage['completion_tokens']
                and (finish != 'length' or usage['completion_tokens'] == max_tokens)
            )
            receipt.update(
                protocol_valid=valid,
                content=content,
                finish_reason=finish,
                usage=usage,
                model=value['model'],
            )
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if response:
            await response.aclose()
        if dispatched:
            released = await engine.wait_release(attempt, 10)
            store.cleanup(arrival, 'confirmed' if released else 'unconfirmed', released or {})
        receipt.update(release=released, elapsed_seconds=time.monotonic() - started)
        expected = {'stop': 'FINISHED_STOPPED', 'length': 'FINISHED_LENGTH_CAPPED'}.get(
            receipt.get('finish_reason')
        )
        receipt['protocol_valid'] = bool(
            receipt['protocol_valid'] and released and released.get('status') == expected
        )
        store.finish(arrival, 'completed' if receipt['protocol_valid'] else 'failed', receipt)
        if not dispatched or released:
            store.release_lease(arrival)
        write_once(output / 'receipt.json', receipt)
    if not receipt['protocol_valid']:
        raise RuntimeError('动作协议或释放失败；停止，不重试')
    return receipt
