"""真实服务资格；探针也进入持久租约，控制者中断后不得丢失在途工作。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from .execution import write_once
from .lab import canonical, digest


async def qualify_engine(engine, panel, output, *, store, cleanups=None, application_result=None):
    from .profile import TRANSPORT_SCOPE

    if panel.get('qualification_scope') == TRANSPORT_SCOPE:
        from .transport import qualify_transport

        return await qualify_transport(
            engine,
            panel,
            output,
            store=store,
            cleanups=cleanups,
            application_result=application_result,
        )
    cases = panel['cases']
    if not cases or panel['required_passes'] != len(cases):
        raise ValueError('服务门要求完整面板全部通过')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('重复资格题目')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_once(
        output / 'input.json',
        {
            'panel': panel,
            'panel_digest': digest(panel),
            'worker': engine.identity,
            'version_id': engine.version_id,
            'artifact_digest': engine.binding.artifact_digest,
        },
    )
    results = []
    for case in cases:
        arrival = uuid.uuid4().hex
        attempt = uuid.uuid4().hex
        response = None
        release = None
        dispatched = False
        count_task = None
        started = time.monotonic()
        record = {
            'case_id': case['id'],
            'arrival_id': arrival,
            'attempt_id': attempt,
            'passed': False,
        }
        store.record_arrival(arrival, None)
        store.bind_arrival(
            arrival,
            engine.version_id,
            engine.identity,
            {'kind': 'service_qualification', 'panel_digest': digest(panel)},
        )
        body = {
            'model': engine.version_id,
            'messages': case['messages'],
            'max_tokens': panel['max_tokens'],
            'stream': False,
        }
        store.reserve_attempt(arrival, attempt, engine.version_id, digest(body))
        store.save_payload(arrival, body)
        try:
            async with asyncio.timeout(panel['deadline_seconds']):
                if not await engine.ready():
                    raise RuntimeError('worker_not_ready')
                count_task = asyncio.create_task(engine.count_tokens(body['messages']))
                count = await asyncio.shield(count_task)
                store.dispatched(arrival, engine.identity)
                dispatched = True
                response = await engine.open(attempt, body)
                raw = await response.aread()
                record['raw_response'] = raw.decode('utf-8')
                if response.status_code != 200:
                    raise RuntimeError('engine_rejected')
                data = json.loads(raw)
                choice = data['choices'][0]
                content = choice['message']['content']
                record.update(response=data, input_tokens_counted=count)
                if 'expected_text' in case:
                    correct = content.strip() == case['expected_text']
                else:
                    correct = canonical(json.loads(content)) == canonical(case['expected_json'])
                record['content_passed'] = correct
                record['passed'] = (
                    correct
                    and choice['finish_reason'] == 'stop'
                    and data['usage']['prompt_tokens'] == count
                )
        except Exception as exc:
            record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            if dispatched:
                try:
                    async with asyncio.timeout(10):
                        if response is not None:
                            await response.aclose()
                        release = await engine.wait_release(attempt, 9)
                except Exception as exc:
                    record['cleanup_error'] = type(exc).__name__
                store.cleanup(arrival, 'confirmed' if release else 'unconfirmed', release or {})
            record.update(release=release, elapsed_seconds=time.monotonic() - started)
            record['passed'] = record['passed'] and bool(release)
            store.finish(arrival, 'completed' if record['passed'] else 'failed', record)
            if count_task is not None and not count_task.done():

                async def finish_count(task=count_task, arrival_id=arrival):
                    try:
                        await task
                    except Exception:
                        pass
                    store.release_lease(arrival_id)

                cleanup = asyncio.create_task(finish_count())
                if cleanups is None:
                    await cleanup
                else:
                    cleanups.add(cleanup)
                    cleanup.add_done_callback(cleanups.discard)
            elif not dispatched or release:
                store.release_lease(arrival)
            write_once(output / (case['id'] + '.json'), record)
            results.append(record)
        if dispatched and not release:
            break
    receipt = {
        'passed': len(results) == len(cases) and all(r['passed'] for r in results),
        'version_id': engine.version_id,
        'panel_digest': digest(panel),
        'artifact_digest': engine.binding.artifact_digest,
        'worker': engine.identity,
        'cases': len(results),
        'passed_cases': sum(r['passed'] for r in results),
        'output': str(output),
        'scope': 'service functional qualification only, not research or production quality',
    }
    write_once(output / 'receipt.json', receipt)
    return receipt
