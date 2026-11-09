"""单版本工程预资格；不计算或替换旧应用输出分数。"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from pathlib import Path

from .execution import write_once
from .lab import digest, file_hash
from .profile import TRANSPORT_SCOPE


async def qualify_transport(
    engine, panel, output, *, store, cleanups=None, application_result=None
):
    if (
        panel.get('qualification_scope') != TRANSPORT_SCOPE
        or not engine.profile
        or not application_result
    ):
        raise ValueError('预资格缺少用途、配置或历史应用状态')
    cases = panel['cases']
    if (
        not cases
        or panel['required_passes'] != len(cases)
        or len({c['id'] for c in cases}) != len(cases)
    ):
        raise ValueError('预资格面板不完整')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    profile = engine.profile
    rows = []
    source_paths = [
        Path(__file__).with_name(n)
        for n in (
            'transport.py',
            'profile.py',
            'engine.py',
            'qualification.py',
            'runtime.py',
            'lab.py',
            'execution.py',
        )
    ]
    for path in source_paths:
        (output / path.name).write_bytes(path.read_bytes())
    write_once(
        output / 'input.json',
        {
            'panel': panel,
            'profile': profile.document,
            'profile_digest': profile.digest,
            'worker': engine.identity,
            'artifact_digest': engine.binding.artifact_digest,
            'application_history': application_result,
            'source_hashes': {p.name: file_hash(p) for p in source_paths},
        },
    )
    for case in cases:
        arrival = uuid.uuid4().hex
        attempt = uuid.uuid4().hex
        response = None
        released = None
        count_task = None
        dispatched = False
        started = time.monotonic()
        row = {'case_id': case['id'], 'arrival_id': arrival, 'attempt_id': attempt, 'passed': False}
        body = {
            'model': engine.version_id,
            'messages': case['messages'],
            'max_tokens': case['max_tokens'],
            'stream': case['stream'],
        }
        store.record_arrival(arrival, None)
        store.bind_arrival(
            arrival,
            engine.version_id,
            engine.identity,
            {
                'kind': 'qualification',
                'qualification_scope': TRANSPORT_SCOPE,
                'profile_digest': profile.digest,
            },
        )
        store.reserve_attempt(arrival, attempt, engine.version_id, digest(body))
        store.save_payload(arrival, body)
        try:
            async with asyncio.timeout(panel['deadline_seconds']):
                if not await engine.ready() or engine.profile.digest != profile.digest:
                    raise RuntimeError('worker_or_profile_changed')
                count_task = asyncio.create_task(engine.token_ids(body['messages']))
                ids = await asyncio.shield(count_task)
                remote_ids = await engine.engine_token_ids(body['messages'])
                if ids != remote_ids:
                    raise RuntimeError('token_ids_mismatch')
                wire = engine.wire_payload(body)
                row.update(
                    input_token_ids=ids,
                    engine_token_ids=remote_ids,
                    wire_payload=wire,
                    profile_digest=profile.digest,
                )
                store.save_wire(arrival, wire, profile.digest, ids)
                store.dispatched(arrival, engine.identity)
                dispatched = True
                response = await engine.open(attempt, body)
                if response.status_code != 200:
                    raise RuntimeError('engine_rejected')
                content = ''
                finish = None
                usage = None
                if case['stream']:
                    done = False
                    events = []
                    row['events'] = events
                    async for line in response.aiter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        events.append(data)
                        if data == '[DONE]':
                            done = True
                            break
                        event = json.loads(data)
                        if event.get('model') != engine.version_id:
                            raise RuntimeError('wrong_response_model')
                        if event.get('usage'):
                            usage = event['usage']
                        for choice in event.get('choices', []):
                            content += choice.get('delta', {}).get('content') or ''
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
                    if not done:
                        raise RuntimeError('missing_DONE')
                else:
                    raw = await response.aread()
                    row['raw_response'] = raw.decode()
                    data = json.loads(raw)
                    row['response'] = data
                    if data['model'] != engine.version_id:
                        raise RuntimeError('wrong_response_model')
                    content = data['choices'][0]['message']['content']
                    finish = data['choices'][0]['finish_reason']
                    usage = data['usage']
                row.update(content=content, finish_reason=finish, usage=usage)
                row['passed'] = (
                    isinstance(content, str)
                    and bool(content.strip())
                    and finish == case['expected_finish']
                    and usage['prompt_tokens'] == len(ids)
                    and 0 < usage['completion_tokens'] <= case['max_tokens']
                    and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
                    and (finish != 'length' or usage['completion_tokens'] == case['max_tokens'])
                )
        except Exception as exc:
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            if dispatched:
                try:
                    async with asyncio.timeout(10):
                        if response is not None:
                            await response.aclose()
                        released = await engine.wait_release(attempt, 9)
                except Exception as exc:
                    row['cleanup_error'] = type(exc).__name__
                store.cleanup(arrival, 'confirmed' if released else 'unconfirmed', released or {})
            expected_release = {'stop': 'FINISHED_STOPPED', 'length': 'FINISHED_LENGTH_CAPPED'}[
                case['expected_finish']
            ]
            row['passed'] = bool(
                row['passed'] and released and released['status'] == expected_release
            )
            row.update(release=released, elapsed_seconds=time.monotonic() - started)
            store.finish(arrival, 'completed' if row['passed'] else 'failed', row)
            if count_task is not None and not count_task.done():

                async def finish_count(task=count_task, aid=arrival):
                    try:
                        await task
                    except Exception:
                        pass
                    store.release_lease(aid)

                task = asyncio.create_task(finish_count())
                if cleanups is None:
                    await task
                else:
                    cleanups.add(task)
                    task.add_done_callback(cleanups.discard)
            elif not dispatched or released:
                store.release_lease(arrival)
            write_once(output / (case['id'] + '.json'), row)
            rows.append(row)
        if dispatched and not released:
            break
    receipt = {
        'qualification_scope': TRANSPORT_SCOPE,
        'passed': len(rows) == len(cases) and all(r['passed'] for r in rows),
        'version_id': engine.version_id,
        'panel_digest': digest(panel),
        'profile_digest': profile.digest,
        'artifact_digest': engine.binding.artifact_digest,
        'worker': engine.identity,
        'application_history': application_result,
        'cases': len(rows),
        'passed_cases': sum(r['passed'] for r in rows),
        'output': str(output),
        'system_acceptance': 'not_established_by_prequalification',
    }
    write_once(output / 'receipt.json', receipt)
    return receipt
