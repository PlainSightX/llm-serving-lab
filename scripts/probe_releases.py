"""双真实worker的TCP发布验收；全部阶段在新目录执行并保留失败。"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import secrets
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx

from serving_lab.engine import EngineBinding, process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from start_engine import start, stop
from probe_serving import probe


async def run(small, large, output):
    root = Path(__file__).resolve().parents[1]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bindings = [Path(small).resolve(), Path(large).resolve()]
    all_bindings = list(bindings)
    versions = [EngineBinding.load(p).version_id for p in bindings]
    a, b = versions
    port = 19083
    check_port_available('127.0.0.1', port)
    secret = secrets.token_hex(24)
    secret_path = out / 'admin-token'
    secret_path.write_text(secret)
    secret_path.chmod(0o600)
    old_small = next(
        (root / 'artifacts/r2-qualification-20261102-a/runtime/qualifications').glob(
            '*/receipt.json'
        )
    )
    config = {
        'bindings': [str(p) for p in bindings],
        'panel': str(root / 'fixtures/serving/r2-transport-v2.json'),
        'application_receipts': {
            a: str(old_small),
            b: str(root / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json'),
        },
        'admin_token_file': str(secret_path),
        'runtime': str(out / 'runtime'),
        'port': port,
        'limits': {
            'max_inflight': 1,
            'max_queue': 4,
            'deadline_seconds': 60,
            'cancel_grace_seconds': 10,
        },
    }
    source_paths = [
        'src/serving_lab/' + name + '.py'
        for name in (
            'online',
            'runtime',
            'releases',
            'parameters',
            'qualification',
            'transport',
            'profile',
            'engine',
            'engine_probe',
            'lab',
            'execution',
        )
    ]
    source_paths += [
        'scripts/serve_releases.py',
        'scripts/probe_releases.py',
        'scripts/start_engine.py',
        'scripts/probe_serving.py',
        'fixtures/serving/r2-transport-v2.json',
        'configs/serving/r2-small.json',
        'configs/serving/r2-large.json',
        'environments/serving-linux/uv.lock',
    ]
    hashes = {name: file_hash(root / name) for name in source_paths}
    for name in source_paths:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'source_hashes': hashes,
            'bindings': [json.loads(p.read_text()) for p in bindings],
            'scope': 'two local GPU workers, same physical GPU; functional release qualification, not cluster or speedup',
        },
    )
    process = None
    gateway_index = 0
    calls = []
    phases = []
    headers = {'Authorization': 'Bearer ' + secret}
    client = httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=300, trust_env=False)

    def launch_gateway():
        nonlocal process, gateway_index
        gateway_index += 1
        cfg = out / f'gateway-{gateway_index}.json'
        write_once(cfg, config)
        with (out / f'gateway-{gateway_index}.log').open('xb') as log:
            process = subprocess.Popen(
                [sys.executable, str(root / 'scripts/serve_releases.py'), '--config', str(cfg)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_once(out / f'gateway-{gateway_index}-process.json', process_identity(process.pid))

    def stop_gateway(kill=False):
        if process and process.poll() is None:
            process.send_signal(signal.SIGKILL if kill else signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    async def wait_gateway():
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError('gateway startup failed, see log')
            try:
                response = await client.get('/admin/releases', headers=headers, timeout=1)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError('gateway startup timeout')

    async def admin(path, body=None):
        response = await client.post(path, headers=headers, json=body or {})
        calls.append(
            {
                'path': path,
                'body': body,
                'status': response.status_code,
                'response': response.json(),
            }
        )
        write_once(out / f'admin-{len(calls):03}.json', calls[-1])
        return response

    async def state():
        return (await client.get('/admin/releases', headers=headers)).json()

    async def change(op, target, expected=200, **extra):
        before = await state()
        response = await admin(
            '/admin/route',
            {
                'operation': op,
                'target': target,
                'expected_generation': before['route']['generation'],
                **extra,
            },
        )
        assert response.status_code == expected, response.text
        return response.json()

    async def request(label, model='active', key=None):
        body = {
            'model': model,
            'messages': [{'role': 'user', 'content': '请只输出：OK'}],
            'max_tokens': 64,
        }
        response = await client.post(
            '/v1/chat/completions', json=body, headers={'Idempotency-Key': key} if key else {}
        )
        write_once(
            out / (label + '.json'),
            {
                'body': body,
                'status': response.status_code,
                'headers': dict(response.headers),
                'response': response.json(),
            },
        )
        return response

    def phase(name, **data):
        phases.append({'name': name, **data})
        write_once(out / f'phase-{len(phases):02}.json', phases[-1])
        print(name, flush=True)

    try:
        launch_gateway()
        await wait_gateway()
        assert (await client.post('/admin/route', json={})).status_code == 401
        await change('promote', b, expected=409)
        for version in versions:
            response = await admin(f'/admin/releases/{version}/qualify')
            assert response.status_code == 200 and response.json()['passed_cases'] == 4, (
                response.text
            )
            assert response.json()['qualification_scope'] == 'version_transport_ready_v2'
            assert response.json()['application_history']['status'] == 'failed'
        phase('both_real_versions_qualified', snapshot=await state())
        await change('promote', a)
        first = await request('baseline', key='cross-version-dedup')
        assert first.status_code == 200
        first_arrival = first.headers['X-Arrival-Id']
        wire = json.loads((out / 'runtime/wire-requests' / (first_arrival + '.json')).read_text())
        qual = next((out / 'runtime/qualifications').glob(a + '-*/plain.json'))
        direct = json.loads(qual.read_text())
        assert (
            wire['payload'] == direct['wire_payload']
            and wire['input_token_ids'] == direct['input_token_ids']
        )
        phase(
            'entry_fidelity',
            profile_digest=wire['profile_digest'],
            direct_wire_and_tokens_equal=True,
            independent_output_equal=first.json()['choices'][0]['message']['content']
            == direct['content'],
        )
        # 首段到达后切流，旧流与已经排队的请求必须仍固定旧实例。
        long_body = {
            'model': 'active',
            'messages': [
                {'role': 'user', 'content': '从1开始逐个输出整数直到1000，不要省略，不要解释。'}
            ],
            'stream': True,
            'max_tokens': 128,
        }
        async with client.stream('POST', '/v1/chat/completions', json=long_body) as cancelled:
            assert cancelled.status_code == 200
            cancel_arrival = cancelled.headers['X-Arrival-Id']
            async for line in cancelled.aiter_lines():
                if line.startswith('data:') and line[5:].strip() != '[DONE]':
                    event = json.loads(line[5:])
                    if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                        break
        for _ in range(200):
            if next(r for r in (await state())['versions'] if r['version'] == a)['pending'] == 0:
                break
            await asyncio.sleep(0.02)
        db = sqlite3.connect(out / 'runtime/service.sqlite')
        db.row_factory = sqlite3.Row
        cancelled_row = dict(
            db.execute('SELECT * FROM arrivals WHERE id=?', (cancel_arrival,)).fetchone()
        )
        db.close()
        release = json.loads(cancelled_row['cleanup_evidence'])
        assert cancelled_row['cleanup'] == 'confirmed' and release['status'] == 'FINISHED_ABORTED'
        phase('cancel', arrival=cancel_arrival, release=release)
        burst = await asyncio.gather(
            *[
                client.post('/v1/chat/completions', json={**long_body, 'stream': False})
                for _ in range(16)
            ]
        )
        statuses = [r.status_code for r in burst]
        assert 429 in statuses and 200 in statuses and set(statuses) <= {200, 429}
        phase('overload', statuses=statuses, arrivals=[r.headers['X-Arrival-Id'] for r in burst])
        async with client.stream('POST', '/v1/chat/completions', json=long_body) as response:
            assert response.status_code == 200 and response.headers['X-Model-Version'] == a
            old_stream_arrival = response.headers['X-Arrival-Id']
            events = []
            queued = None
            async for line in response.aiter_lines():
                events.append(line)
                if (
                    queued is None
                    and line.startswith('data:')
                    and line[5:].strip() != '[DONE]'
                    and any(
                        c.get('delta', {}).get('content')
                        for c in json.loads(line[5:]).get('choices', [])
                    )
                ):
                    queued = asyncio.create_task(request('queued-before-switch'))
                    for _ in range(100):
                        if (
                            next(r for r in (await state())['versions'] if r['version'] == a)[
                                'pending'
                            ]
                            >= 2
                        ):
                            break
                        await asyncio.sleep(0.005)
                    else:
                        raise RuntimeError('could not observe queued old lease')
                    await change('promote', b)
                    assert (await admin(f'/admin/releases/{a}/retire')).status_code == 409
            assert queued is not None and (await queued).headers['X-Model-Version'] == a
            assert 'data: [DONE]' in events
            write_once(
                out / 'old-stream.json', {'headers': dict(response.headers), 'events': events}
            )
        second = await request('new-after-switch')
        assert second.headers['X-Model-Version'] == b
        second_arrival = second.headers['X-Arrival-Id']
        second_wire = json.loads(
            (out / 'runtime/wire-requests' / (second_arrival + '.json')).read_text()
        )
        second_direct = json.loads(
            next((out / 'runtime/qualifications').glob(b + '-*/plain.json')).read_text()
        )
        assert (
            second_wire['payload'] == second_direct['wire_payload']
            and second_wire['input_token_ids'] == second_direct['input_token_ids']
        )
        phase(
            'second_entry_fidelity',
            profile_digest=second_wire['profile_digest'],
            direct_wire_and_tokens_equal=True,
            independent_output_equal=second.json()['choices'][0]['message']['content']
            == second_direct['content'],
        )
        assert (await request('dedup-after-switch', key='cross-version-dedup')).status_code == 409
        stale = await admin(
            '/admin/route', {'operation': 'promote', 'target': a, 'expected_generation': 0}
        )
        assert stale.status_code == 409
        phase('stream_queue_version_pinning_and_global_dedup')
        await change('rollback', a)
        assert (await request('rollback-live')).headers['X-Model-Version'] == a
        await change('canary', b, percentage=50)
        observed = []
        for i in range(16):
            response = await request(f'canary-{i:02}')
            assert response.status_code == 200
            observed.append(response.headers['X-Model-Version'])
        assert set(observed) == {a, b}
        phase('canary', counts={v: observed.count(v) for v in versions})
        await change('promote', b)
        assert (await admin(f'/admin/releases/{a}/drain', {'timeout_seconds': 5})).json()['drained']
        assert (await admin(f'/admin/releases/{a}/retire')).status_code == 200
        await change('rollback', a, expected=409)
        stop_gateway()
        stopped = stop(bindings[0])
        assert stopped['stopped']
        write_once(out / 'retired-worker-stop.json', stopped)
        new_dir = out / 'small-reloaded'
        start(root / 'configs/serving/r2-small.json', new_dir)
        bindings[0] = new_dir / 'binding.json'
        all_bindings.append(bindings[0])
        probe(bindings[0], out / 'small-reloaded-direct')
        config.update(bindings=[str(p) for p in bindings], allow_rebind=[a])
        launch_gateway()
        await wait_gateway()
        await change('rollback', a, expected=409)
        assert (await admin(f'/admin/releases/{a}/qualify')).status_code == 200
        await change('rollback', a)
        assert (await request('rollback-reloaded')).headers['X-Model-Version'] == a
        phase('retire_stop_reload_requalify_rollback', snapshot=await state())
        # 强制终止控制者，原调用保留unknown；重启不能补发同一key。
        crash_headers = {'Idempotency-Key': 'crash-inflight'}
        try:
            async with client.stream(
                'POST', '/v1/chat/completions', json=long_body, headers=crash_headers
            ) as response:
                async for line in response.aiter_lines():
                    if (
                        line.startswith('data:')
                        and line[5:].strip() != '[DONE]'
                        and any(
                            c.get('delta', {}).get('content')
                            for c in json.loads(line[5:]).get('choices', [])
                        )
                    ):
                        stop_gateway(kill=True)
                        break
        except httpx.HTTPError:
            pass
        launch_gateway()
        await wait_gateway()
        for _ in range(100):
            if (await client.get('/health/ready')).status_code == 200:
                break
            await asyncio.sleep(0.1)
        assert (await request('dedup-after-crash', key='crash-inflight')).status_code == 409
        phase('controller_crash_recovery')
        stop_gateway()
        config['limits']['deadline_seconds'] = 0.02
        launch_gateway()
        await wait_gateway()
        expired = await client.post('/v1/chat/completions', json={**long_body, 'stream': False})
        assert expired.status_code == 504, expired.text
        phase('deadline', arrival=expired.headers['X-Arrival-Id'], status=expired.status_code)
        stop_gateway()
        config['limits']['deadline_seconds'] = 60
        launch_gateway()
        await wait_gateway()
        stopped = stop(bindings[1])
        assert stopped['stopped']
        write_once(out / 'failed-worker-stop.json', stopped)
        before = (await state())['route']
        await change('promote', b, expected=503)
        assert (await state())['route'] == before
        assert (await request('healthy-active-after-candidate-exit')).status_code == 200
        phase('dead_candidate_keeps_active')
        fault_arrival = None
        fault_events = []
        injected = False
        try:
            async with client.stream(
                'POST',
                '/v1/chat/completions',
                json=long_body,
                headers={'Idempotency-Key': 'worker-fault'},
            ) as fault_response:
                assert fault_response.status_code == 200
                fault_arrival = fault_response.headers['X-Arrival-Id']
                injected = False
                async for line in fault_response.aiter_lines():
                    fault_events.append(line)
                    if not injected and line.startswith('data:') and line[5:].strip() != '[DONE]':
                        event = json.loads(line[5:])
                        if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                            owned = EngineBinding.load(bindings[0])
                            assert (
                                owned.alive()
                                and os.getpgid(owned.process['pid']) == owned.process['pid']
                            )
                            write_once(
                                out / 'active-worker-kill.json',
                                {
                                    'binding': str(bindings[0]),
                                    'process': owned.process,
                                    'signal': 'SIGKILL',
                                },
                            )
                            os.killpg(owned.process['pid'], signal.SIGKILL)
                            injected = True
                assert injected
        except httpx.RemoteProtocolError:
            pass
        assert (
            fault_arrival is not None and injected and (out / 'active-worker-kill.json').is_file()
        )
        phase('active_worker_exit', arrival=fault_arrival, events=fault_events)
        stop_gateway()
        assert stop(bindings[0])['stopped']
        recovered_dir = out / 'active-recovered'
        start(root / 'configs/serving/r2-small.json', recovered_dir)
        bindings[0] = recovered_dir / 'binding.json'
        all_bindings.append(bindings[0])
        probe(bindings[0], out / 'active-recovered-direct')
        config['bindings'] = [str(p) for p in bindings]
        launch_gateway()
        await wait_gateway()
        assert (await request('active-rebind-before-qualification')).status_code == 503
        assert (await admin(f'/admin/releases/{a}/qualify')).status_code == 200
        assert (await request('active-rebind-before-route-commit')).status_code == 503
        await change('promote', a)
        assert (await request('active-recovered')).status_code == 200
        assert (await request('dedup-worker-fault', key='worker-fault')).status_code == 409
        phase('active_worker_rebind_recovery', snapshot=await state())
        stop_gateway()
        db = sqlite3.connect(out / 'runtime/service.sqlite')
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute('SELECT * FROM arrivals')]
        db.close()
        assert any(r['state'] == 'unknown' and r['client_key'] == 'crash-inflight' for r in rows)
        assert all(r['lease_released'] == 1 for r in rows)
        by_id = {r['id']: r for r in rows}
        assert json.loads(by_id[first_arrival]['observation'])['response'] == first.json()
        assert json.loads(by_id[second_arrival]['observation'])['response'] == second.json()
        assert (
            by_id[fault_arrival]['state'] == 'failed'
            and by_id[fault_arrival]['cleanup'] == 'confirmed'
        )
        assert [e[5:].strip() for e in events if e.startswith('data:')] == json.loads(
            by_id[old_stream_arrival]['observation']
        )['events']
        required = {
            'both_real_versions_qualified',
            'entry_fidelity',
            'second_entry_fidelity',
            'cancel',
            'overload',
            'stream_queue_version_pinning_and_global_dedup',
            'canary',
            'retire_stop_reload_requalify_rollback',
            'controller_crash_recovery',
            'deadline',
            'dead_candidate_keeps_active',
            'active_worker_exit',
            'active_worker_rebind_recovery',
        }
        if {p['name'] for p in phases} != required:
            raise RuntimeError('完整系统验收场景缺失')
        write_once(out / 'arrivals.json', rows)
        assert all(file_hash(root / name) == value for name, value in hashes.items())
        result = {
            'passed': True,
            'phases': phases,
            'arrivals': len(rows),
            'unknown': sum(r['state'] == 'unknown' for r in rows),
            'source_hashes_verified': len(hashes),
            'qualification_scope': 'r2_engineering_acceptance_v2',
            'application_history': {
                v: next(p for p in phases if p['name'] == 'both_real_versions_qualified')[
                    'snapshot'
                ]['versions'][i]['identity']['application_history']
                for i, v in enumerate(versions)
            },
            'scope': 'single physical GPU, two distinct parameter versions; not application acceptance, cluster or learning',
        }
        write_once(out / 'summary.json', result)
        return result
    finally:
        await client.aclose()
        stop_gateway()
        cleanup = []
        for binding in all_bindings:
            try:
                cleanup.append({'binding': str(binding), 'result': stop(binding)})
            except Exception as exc:
                cleanup.append({'binding': str(binding), 'error': repr(exc)})
        write_once(out / 'cleanup.json', cleanup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', required=True)
    parser.add_argument('--large', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.small, args.large, args.output)), ensure_ascii=False, indent=2
        )
    )
