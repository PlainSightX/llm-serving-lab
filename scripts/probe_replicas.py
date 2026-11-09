"""同参数双vLLM实例的有限真实接入；不代表双节点或性能收益。"""

import argparse
import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
from pathlib import Path
import httpx
from serving_lab.engine import EngineBinding, process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.measurement_runtime import export_arrivals
from start_engine import start, stop
from probe_serving import probe


async def run(output):
    repo = Path(__file__).resolve().parents[1]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    handles = []
    bindings = []
    reserved = 0
    stages = []
    configs = []
    manifests = []
    completed = None
    sources = [
        'scripts/probe_replicas.py',
        'scripts/serve_replicas.py',
        'scripts/serve_worker.py',
        'scripts/start_engine.py',
        'scripts/probe_serving.py',
    ]
    sources += ['src/serving_lab/' + p.name for p in (repo / 'src/serving_lab').glob('*.py')]
    hashes = {name: file_hash(repo / name) for name in sources}
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'scope': 'real two same-parameter replicas on one local GPU; not cluster or optimization evidence',
            'source_hashes': hashes,
            'model_generation_reservation_cap': 80,
        },
    )

    def reserve(n):
        nonlocal reserved
        if reserved + n > 80:
            raise RuntimeError('bounded real generation budget exhausted')
        reserved += n

    def spawn(label, args):
        with (out / (label + '.log')).open('xb') as log:
            process = subprocess.Popen(
                [sys.executable, *map(str, args)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        handles.append((label, process))
        write_once(out / (label + '-process.json'), process_identity(process.pid))
        return process

    def stop_child(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

        async def wait_url(url, process, headers=None):
            for _ in range(400):
                if process.poll() is not None:
                    raise RuntimeError('owned service failed during startup')
                try:
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        return response.json()
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
            raise RuntimeError('service startup timeout')

        try:
            for port in (19080, 19082, 19180, 19182, 19183):
                check_port_available('127.0.0.1', port)
            token = secrets.token_hex(24)
            secret = out / 'worker-token'
            secret.write_text(token)
            secret.chmod(0o600)
            admin = secrets.token_hex(24)
            adminfile = out / 'admin-token'
            adminfile.write_text(admin)
            adminfile.chmod(0o600)
            for index, (port, bridge_port) in enumerate([(19080, 19180), (19082, 19182)], 1):
                cfg = json.loads((repo / 'configs/serving/r3-small.json').read_text())
                cfg['port'] = port
                cfg['gateway_port'] = 19183
                config = out / f'worker-{index}-config.json'
                write_once(config, cfg)
                configs.append(config)
                folder = out / f'worker-{index}'
                binding = folder / 'binding.json'
                try:
                    start(config, folder)
                finally:
                    if binding.exists():
                        bindings.append(binding)
                reserve(2)
                probe(binding, out / f'direct-{index}')
                manifest = out / f'worker-{index}-manifest.json'
                manifests.append(manifest)
                process = spawn(
                    f'bridge-{index}',
                    [
                        repo / 'scripts/serve_worker.py',
                        '--binding',
                        binding,
                        '--token-file',
                        secret,
                        '--port',
                        bridge_port,
                        '--manifest-output',
                        manifest,
                    ],
                )
                value = await wait_url(
                    f'http://127.0.0.1:{bridge_port}/identity',
                    process,
                    {'Authorization': 'Bearer ' + token},
                )
                assert value['ready'] and value['manifest'] == json.loads(manifest.read_text())
            ms = [json.loads(p.read_text()) for p in manifests]
            assert (
                ms[0]['parameter_digest'] == ms[1]['parameter_digest']
                and ms[0]['profile_digest'] == ms[1]['profile_digest']
                and ms[0]['identity'] != ms[1]['identity']
            )
            version = ms[0]['version_id']
            old = next(
                (repo / 'artifacts/r2-qualification-20261102-a/runtime/qualifications').glob(
                    '*/receipt.json'
                )
            )
            cfg = {
                'replicas': [
                    {
                        'id': f'replica-{i}',
                        'endpoint': f'http://127.0.0.1:{port}',
                        'manifest': str(manifests[i - 1]),
                        'token_file': str(secret),
                        'capacity': {'max_inflight': 1, 'max_queue': 2},
                    }
                    for i, port in [(1, 19180), (2, 19182)]
                ],
                'application_receipts': {version: str(old)},
                'panel': str(repo / 'fixtures/serving/r2-transport-v2.json'),
                'runtime': str(out / 'runtime'),
                'port': 19183,
                'admin_token_file': str(adminfile),
                'limits': {
                    'max_inflight': 2,
                    'max_queue': 4,
                    'deadline_seconds': 30,
                    'cancel_grace_seconds': 10,
                },
            }
            config = out / 'gateway.json'
            write_once(config, cfg)
            gateway = spawn('gateway', [repo / 'scripts/serve_replicas.py', '--config', config])
            base = 'http://127.0.0.1:19183'
            headers = {'Authorization': 'Bearer ' + admin}
            await wait_url(base + '/admin/releases', gateway, headers)

            async def manage(path, body=None, expected=200):
                response = await client.post(base + path, headers=headers, json=body or {})
                value = response.json()
                if response.status_code != expected:
                    raise RuntimeError(f'{path}: {response.status_code} {value}')
                stages.append(
                    {'operation': path, 'status': response.status_code, 'response': value}
                )
                return value

            for rid in ('replica-1', 'replica-2'):
                reserve(4)
                receipt = await manage('/admin/replicas/' + rid + '/qualify')
                assert receipt['passed']
            await manage(
                '/admin/route',
                {'operation': 'promote', 'target': version, 'expected_generation': 0},
            )
            body = {
                'model': 'active',
                'messages': [
                    {'role': 'user', 'content': '从1开始逐个输出整数直到1000，不要省略，不要解释。'}
                ],
                'max_tokens': 128,
                'stream': True,
            }
            response_records = []
            reserve(1)
            async with client.stream(
                'POST',
                base + '/v1/chat/completions',
                json=body,
                headers={'Idempotency-Key': 'stream-in-flight'},
            ) as response:
                assert response.status_code == 200
                rid = response.headers['x-replica-id']
                lines = response.aiter_lines()
                events = []
                async for line in lines:
                    if line.startswith('data:'):
                        events.append(line)
                        if line != 'data: [DONE]' and any(
                            c.get('delta', {}).get('content')
                            for c in json.loads(line[5:]).get('choices', [])
                        ):
                            break
                drained = await manage('/admin/replicas/' + rid + '/drain', expected=409)
                assert drained['pending'] >= 1
                reserve(1)
                other = await client.post(
                    base + '/v1/chat/completions', json={**body, 'max_tokens': 16, 'stream': False}
                )
                assert other.status_code == 200 and other.headers['x-replica-id'] != rid
                response_records.append({'headers': dict(other.headers), 'response': other.json()})
                events.extend([line async for line in lines if line.startswith('data:')])
                assert 'data: [DONE]' in events
                response_records.append({'headers': dict(response.headers), 'events': events})
            for _ in range(300):
                state = (await client.get(base + '/admin/releases', headers=headers)).json()
                if all(r['pending'] == 0 for r in state['replicas']):
                    break
                await asyncio.sleep(0.02)
            else:
                raise RuntimeError('real replicas did not drain')
            await manage('/admin/replicas/' + rid + '/retire')
            duplicate = await client.post(
                base + '/v1/chat/completions',
                json={**body, 'stream': False},
                headers={'Idempotency-Key': 'stream-in-flight'},
            )
            assert duplicate.status_code == 409
            write_once(out / 'responses.json', response_records)
            write_once(out / 'stages.json', stages)
            # 精确停止剩余副本的桥，验证新请求失败而非转向已退休副本；不制造自动重试。
            remaining = next(x for x in ('replica-1', 'replica-2') if x != rid)
            index = int(remaining[-1])
            bridge = next(p for label, p in handles if label == f'bridge-{index}')
            stop_child(bridge)
            failed = await client.post(
                base + '/v1/chat/completions', json={**body, 'stream': False, 'max_tokens': 16}
            )
            assert failed.status_code == 503
            write_once(
                out / 'offline-rejection.json',
                {'status': failed.status_code, 'response': failed.json()},
            )
            # 同一实际engine代际重启HTTP桥，不继承不同engine代际；已有不可变清单须逐字一致。
            recovered_manifest = out / f'worker-{index}-manifest-recovered.json'
            bridge = spawn(
                f'bridge-{index}-recovered',
                [
                    repo / 'scripts/serve_worker.py',
                    '--binding',
                    bindings[index - 1],
                    '--token-file',
                    secret,
                    '--port',
                    19180 if index == 1 else 19182,
                    '--manifest-output',
                    recovered_manifest,
                ],
            )
            recovered = await wait_url(
                f'http://127.0.0.1:{19180 if index == 1 else 19182}/identity',
                bridge,
                {'Authorization': 'Bearer ' + token},
            )
            assert recovered['manifest'] == ms[index - 1]
            reserve(1)
            ok = await client.post(
                base + '/v1/chat/completions', json={**body, 'stream': False, 'max_tokens': 16}
            )
            assert ok.status_code == 200 and ok.headers['x-replica-id'] == remaining
            write_once(
                out / 'recovered-request.json', {'headers': dict(ok.headers), 'response': ok.json()}
            )
            for _ in range(200):
                state = (await client.get(base + '/admin/releases', headers=headers)).json()
                if all(r['pending'] == 0 for r in state['replicas']):
                    break
                await asyncio.sleep(0.02)
            assert all(r['pending'] == 0 for r in state['replicas'])
            # 在确实有生成中的请求上杀掉原engine组；HTTP监督桥保留，用准确旧代际证明解除孤立lease。
            reserve(1)
            fault_body = {**body, 'stream': True}
            fault_events = []
            async with client.stream(
                'POST',
                base + '/v1/chat/completions',
                json=fault_body,
                headers={'Idempotency-Key': 'engine-kill-stream'},
            ) as response:
                assert response.status_code == 200 and response.headers['x-replica-id'] == remaining
                lines = response.aiter_lines()
                async for line in lines:
                    if line.startswith('data:'):
                        fault_events.append(line)
                        if line != 'data: [DONE]' and any(
                            c.get('delta', {}).get('content')
                            for c in json.loads(line[5:]).get('choices', [])
                        ):
                            break
                doomed = EngineBinding.load(bindings[index - 1])
                assert doomed.alive() and os.getpgid(doomed.process['pid']) == doomed.process['pid']
                write_once(
                    out / 'engine-kill-target.json',
                    {
                        'binding': str(bindings[index - 1]),
                        'process': doomed.process,
                        'manifest': ms[index - 1],
                    },
                )
                os.killpg(doomed.process['pid'], signal.SIGKILL)
                try:
                    fault_events.extend([line async for line in lines if line.startswith('data:')])
                except httpx.HTTPError as exc:
                    fault_events.append('transport_error:' + type(exc).__name__)
            assert 'data: [DONE]' not in fault_events
            write_once(
                out / 'engine-kill-stream.json',
                {'events': fault_events, 'arrival': response.headers['x-arrival-id']},
            )
            for _ in range(700):
                state = (await client.get(base + '/admin/releases', headers=headers)).json()
                victim = next(r for r in state['replicas'] if r['id'] == remaining)
                if victim['state'] == 'quarantined':
                    break
                await asyncio.sleep(0.02)
            assert victim['state'] == 'quarantined' and victim['pending'] >= 1
            reconciled = await manage('/admin/replicas/' + remaining + '/reconcile')
            assert reconciled['pending'] == 0
            stop_child(gateway)
            stop_child(bridge)
            replacement_folder = out / f'worker-{index}-rebound'
            replacement_binding = replacement_folder / 'binding.json'
            try:
                start(configs[index - 1], replacement_folder)
            finally:
                if replacement_binding.exists():
                    bindings.append(replacement_binding)
            reserve(2)
            probe(replacement_binding, out / 'rebound-direct')
            replacement_manifest = out / 'rebound-manifest.json'
            bridge = spawn(
                'bridge-rebound',
                [
                    repo / 'scripts/serve_worker.py',
                    '--binding',
                    replacement_binding,
                    '--token-file',
                    secret,
                    '--port',
                    19180 if index == 1 else 19182,
                    '--manifest-output',
                    replacement_manifest,
                ],
            )
            replacement = await wait_url(
                f'http://127.0.0.1:{19180 if index == 1 else 19182}/identity',
                bridge,
                {'Authorization': 'Bearer ' + token},
            )
            assert (
                replacement['manifest']['identity'] != ms[index - 1]['identity']
                and replacement['manifest']['parameter_digest'] == ms[index - 1]['parameter_digest']
            )
            cfg['replicas'][index - 1]['manifest'] = str(replacement_manifest)
            cfg['allow_rebind'] = [remaining]
            rebound_config = out / 'gateway-rebound.json'
            write_once(rebound_config, cfg)
            gateway = spawn(
                'gateway-rebound', [repo / 'scripts/serve_replicas.py', '--config', rebound_config]
            )
            state = await wait_url(base + '/admin/releases', gateway, headers)
            assert next(r['state'] for r in state['replicas'] if r['id'] == remaining) == 'loading'
            before_qual = await client.post(
                base + '/v1/chat/completions', json={**body, 'stream': False, 'max_tokens': 16}
            )
            assert before_qual.status_code == 503
            reserve(4)
            assert (await manage('/admin/replicas/' + remaining + '/qualify'))['passed']
            rejected_retry = await client.post(
                base + '/v1/chat/completions',
                json={**body, 'stream': False},
                headers={'Idempotency-Key': 'engine-kill-stream'},
            )
            assert rejected_retry.status_code == 409
            reserve(1)
            rebound = await client.post(
                base + '/v1/chat/completions', json={**body, 'stream': False, 'max_tokens': 16}
            )
            assert (
                rebound.status_code == 200
                and rebound.headers['x-worker-generation']
                == replacement['manifest']['identity']['startup_generation']
            )
            write_once(
                out / 'rebound-response.json',
                {'headers': dict(rebound.headers), 'response': rebound.json()},
            )
            write_once(out / 'all-stages.json', stages)
            export_arrivals(out / 'runtime/service.sqlite', out / 'arrivals.json')
            assert all(file_hash(repo / name) == sha for name, sha in hashes.items())
            completed = {
                'passed': True,
                'scope': 'local single-GPU same-version real replica integration',
                'reserved_generation_calls': reserved,
                'new_parameters': False,
                'cross_parameter_rollback': False,
                'same_parameters_two_actual_worker_generations': True,
                'engine_kill_reconciled_on_original_worker': True,
                'new_generation_requalified': True,
                'different_generation_fence': 'tested separately via real TCP contract fixtures',
                'terminal_rejections': {'duplicate': 409, 'all_eligible_bridges_offline': 503},
            }
        finally:
            processes = []
            for label, process in reversed(handles):
                try:
                    stop_child(process)
                    processes.append({'label': label, 'returncode': process.returncode})
                except Exception as exc:
                    processes.append({'label': label, 'error': repr(exc)})
            clean = []
            for path in bindings:
                try:
                    clean.append({'binding': str(path), 'result': stop(path)})
                except Exception as exc:
                    clean.append({'binding': str(path), 'error': repr(exc)})
            write_once(out / 'cleanup.json', {'processes': processes, 'workers': clean})
            if any('error' in r for r in processes + clean) or any(
                r['result'].get('remaining') for r in clean if 'result' in r
            ):
                raise RuntimeError('owned cleanup incomplete; do not accept run')
    if completed:
        write_once(out / 'summary.json', completed)
        return {'passed': True, 'output': str(out), 'reserved_generations': reserved}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(asyncio.run(run(a.output)), indent=2))
