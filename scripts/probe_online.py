"""真实TCP验收：保留每个场景的响应/账本，故障仅注入明确自有进程。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

from serving_lab.engine import EngineBinding, ReleaseJournal, process_identity
from serving_lab.execution import write_once
from serving_lab.lab import file_hash


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Gateway:
    def __init__(self, root, binding, runtime, config, port, label):
        self.root = root
        self.binding = binding
        self.runtime = runtime
        self.config = config
        self.port = port
        self.label = label

    async def start(self):
        command = [
            sys.executable,
            '-m',
            'serving_lab.online',
            '--binding',
            str(self.binding),
            '--runtime',
            str(self.runtime),
            '--config',
            str(self.config),
            '--port',
            str(self.port),
        ]
        with (self.root / (self.label + '.log')).open('xb') as log:
            self.process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.identity = process_identity(self.process.pid)
        write_once(
            self.root / (self.label + '-process.json'),
            {'command': command, 'process': self.identity},
        )
        async with httpx.AsyncClient(
            base_url=f'http://127.0.0.1:{self.port}', timeout=2, trust_env=False
        ) as c:
            for _ in range(150):
                if self.process.poll() is not None:
                    raise RuntimeError('gateway启动失败：' + self.label)
                try:
                    response = await c.get('/health/ready')
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        raise RuntimeError('gateway就绪超时：' + self.label)

    def stop(self, hard=False):
        if self.process.poll() is not None:
            return
        require(process_identity(self.process.pid) == self.identity, 'gateway PID身份变化')
        require(os.getpgid(self.process.pid) == self.process.pid, 'gateway进程组不符')
        os.killpg(self.process.pid, signal.SIGKILL if hard else signal.SIGTERM)
        try:
            self.process.wait(timeout=18)
        except subprocess.TimeoutExpired:
            if process_identity(self.process.pid) == self.identity:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)


def rows(runtime):
    conn = sqlite3.connect('file:' + str(runtime / 'service.sqlite') + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute('SELECT * FROM arrivals ORDER BY received,id')]
    finally:
        conn.close()


async def wait_row(runtime, arrival, *, clean=True):
    for _ in range(700):
        found = next((r for r in rows(runtime) if r['id'] == arrival), None)
        if found and found['state'] in {
            'completed',
            'failed',
            'cancelled',
            'unknown',
            'timed_out',
            'rejected',
        }:
            if not clean or found['cleanup'] in {'none', 'confirmed', 'unconfirmed'}:
                return found
        await asyncio.sleep(0.02)
    raise RuntimeError('终态/清理超时：' + arrival)


async def qualify(binding_path, config_path, output, exercise_worker):
    repo = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    binding_path = Path(binding_path).resolve()
    config_path = Path(config_path).resolve()
    binding = EngineBinding.load(binding_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    runtime = output / 'runtime'
    port = config['gateway_port']
    sources = {}
    for path in [
        *list((repo / 'src/serving_lab').glob('*.py')),
        Path(__file__),
        repo / 'scripts/start_engine.py',
        repo / 'environments/serving-linux/pyproject.toml',
        repo / 'environments/serving-linux/uv.lock',
        repo / 'pyproject.toml',
        config_path,
    ]:
        relative = path.relative_to(repo)
        target = output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[str(path)] = file_hash(path)
    write_once(
        output / 'qualification-input.json',
        {
            'binding_path': str(binding_path),
            'binding_sha256': binding.manifest_sha256,
            'config': config,
            'source_files': sources,
            'exercise_worker_exit': exercise_worker,
            'created': time.time(),
        },
    )
    gateways = []
    replacement = None
    results = {}
    gateway = Gateway(output, binding_path, runtime, config_path, port, 'gateway-first')
    gateways.append(gateway)
    base_url = f'http://127.0.0.1:{port}'

    def body(stream=False, long=False):
        prompt = (
            '请从1开始依次列出全部整数直到10000，中间不要省略，不要解释。'
            if long
            else '请只输出OK。'
        )
        return {
            'model': binding.version_id,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 128,
            'stream': stream,
        }

    def save(name, value):
        write_once(output / (name + '.json'), value)
        results[name] = value
        print(
            json.dumps({'case': name, 'passed': value.get('passed', False)}, ensure_ascii=False),
            flush=True,
        )

    async def call(client, name, payload, expected, **kwargs):
        started = time.perf_counter()
        response = await client.post('/v1/chat/completions', json=payload, **kwargs)
        arrival = response.headers.get('x-arrival-id')
        row = await wait_row(runtime, arrival)
        result = {
            'status': response.status_code,
            'request': payload,
            'response': response.text,
            'arrival_id': arrival,
            'row': row,
            'elapsed_seconds': time.perf_counter() - started,
            'passed': response.status_code == expected,
        }
        save(name, result)
        require(result['passed'], name + '状态码不符')
        return result

    try:
        await gateway.start()
        async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
            first = await call(
                client, 'normal', body(), 200, headers={'Idempotency-Key': 'normal-key'}
            )
            require(
                first['row']['state'] == 'completed' and first['row']['cleanup'] == 'confirmed',
                '正常请求释放未确认',
            )
            await call(client, 'duplicate', body(), 409, headers={'Idempotency-Key': 'normal-key'})
            await call(client, 'unknown-model', {**body(), 'model': 'unconfigured'}, 404)
            await call(
                client,
                'input-tokens',
                {**body(), 'messages': [{'role': 'user', 'content': 'hello ' * 2200}]},
                422,
            )
            await call(
                client,
                'message-limit',
                {**body(), 'messages': [{'role': 'user', 'content': 'x'}] * 33},
                422,
            )
            raw = await client.post('/v1/chat/completions', content=b'x' * 65537)
            rawrow = await wait_row(runtime, raw.headers['x-arrival-id'])
            save(
                'body-limit',
                {'passed': raw.status_code == 413, 'status': raw.status_code, 'row': rawrow},
            )
            require(raw.status_code == 413, '字节上限未拒绝')
            started = time.perf_counter()
            events = []
            ttft = None
            done = False
            async with client.stream(
                'POST', '/v1/chat/completions', json=body(stream=True)
            ) as response:
                arrival = response.headers['x-arrival-id']
                async for line in response.aiter_lines():
                    if not line.startswith('data:'):
                        continue
                    value = line[5:].strip()
                    events.append(value)
                    if value == '[DONE]':
                        done = True
                        break
                    obj = json.loads(value)
                    if (
                        any(c.get('delta', {}).get('content') for c in obj.get('choices', []))
                        and ttft is None
                    ):
                        ttft = time.perf_counter() - started
            row = await wait_row(runtime, arrival)
            save(
                'normal-stream',
                {
                    'passed': done
                    and row['state'] == 'completed'
                    and row['cleanup'] == 'confirmed',
                    'row': row,
                    'events': events,
                    'ttft_seconds': ttft,
                },
            )
            require(results['normal-stream']['passed'], '流式请求失败')
            journal = ReleaseJournal(
                binding.events_path, binding.worker_instance_id, binding.startup_generation
            )
            async with client.stream(
                'POST',
                '/v1/chat/completions',
                json=body(stream=True, long=True),
                headers={'Idempotency-Key': 'cancel-key'},
            ) as response:
                arrival = response.headers['x-arrival-id']
                first_chunk = None
                async for line in response.aiter_lines():
                    if line.startswith('data:') and line[5:].strip() != '[DONE]':
                        obj = json.loads(line[5:])
                        if any(c.get('delta', {}).get('content') for c in obj.get('choices', [])):
                            first_chunk = obj
                            break
                pending = next(r for r in rows(runtime) if r['id'] == arrival)
                before = journal.get(pending['attempt_id'])
                require(
                    first_chunk is not None and before is None, '取消前已自然结束，不能算取消资格'
                )
            closed = time.perf_counter()
            row = await wait_row(runtime, arrival)
            evidence = json.loads(row['cleanup_evidence'])
            save(
                'tcp-cancel',
                {
                    'passed': row['state'] == 'cancelled'
                    and row['cleanup'] == 'confirmed'
                    and evidence['status'] == 'FINISHED_ABORTED',
                    'row': row,
                    'first_content': first_chunk,
                    'release_before_close': before,
                    'close_to_confirmation_seconds': time.perf_counter() - closed,
                },
            )
            require(results['tcp-cancel']['passed'], '真实TCP取消未通过')
            paired = []
            async with httpx.AsyncClient(
                base_url=binding.endpoint, timeout=60, trust_env=False
            ) as direct:
                for index in range(3):
                    payload = body()
                    observed = {}
                    for route in ('direct', 'gateway') if index % 2 == 0 else ('gateway', 'direct'):
                        sent = time.perf_counter()
                        if route == 'direct':
                            r = await direct.post(
                                '/v1/chat/completions',
                                json={
                                    **payload,
                                    'temperature': 0,
                                    'chat_template_kwargs': {'enable_thinking': False},
                                },
                            )
                        else:
                            r = await client.post('/v1/chat/completions', json=payload)
                        require(r.status_code == 200, '配对基线响应失败')
                        observed[route] = {
                            'elapsed_seconds': time.perf_counter() - sent,
                            'response': r.json(),
                        }
                    a = observed['direct']['response']
                    b = observed['gateway']['response']
                    matched = (
                        a['usage']['prompt_tokens'] == b['usage']['prompt_tokens']
                        and a['usage']['completion_tokens'] == b['usage']['completion_tokens']
                        and a['choices'][0]['message']['content']
                        == b['choices'][0]['message']['content']
                    )
                    paired.append({'input': payload, 'routes': observed, 'matched_output': matched})
            save(
                'paired-baseline',
                {
                    'passed': all(p['matched_output'] for p in paired),
                    'pairs': paired,
                    'scope': 'same-input qualification; three pairs do not establish a performance improvement',
                },
            )
            require(results['paired-baseline']['passed'], '配对输入/输出不一致，不能比较入口开销')

            async def load(i):
                r = await client.post(
                    '/v1/chat/completions',
                    json=body(long=True),
                    headers={'Idempotency-Key': 'load-' + str(i)},
                )
                return {'status': r.status_code, 'arrival_id': r.headers.get('x-arrival-id')}

            overloaded = await asyncio.gather(*(load(i) for i in range(24)))
            saved = [await wait_row(runtime, r['arrival_id']) for r in overloaded]
            save(
                'overload',
                {
                    'passed': any(r['status'] == 429 for r in overloaded)
                    and all(r['status'] in {200, 429} for r in overloaded),
                    'responses': overloaded,
                    'rows': saved,
                },
            )
            require(results['overload']['passed'], '实际过载行为不符')
            async with client.stream(
                'POST',
                '/v1/chat/completions',
                json=body(stream=True, long=True),
                headers={'Idempotency-Key': 'crash-key'},
            ) as response:
                arrival = response.headers['x-arrival-id']
                async for line in response.aiter_lines():
                    if line.startswith('data:') and line[5:].strip() != '[DONE]':
                        obj = json.loads(line[5:])
                        if any(c.get('delta', {}).get('content') for c in obj.get('choices', [])):
                            break
                crashed = next(r for r in rows(runtime) if r['id'] == arrival)
                require(
                    crashed['state'] == 'running' and journal.get(crashed['attempt_id']) is None,
                    '崩溃前请求已完成',
                )
                gateway.stop(hard=True)
            restarted = Gateway(
                output, binding_path, runtime, config_path, port, 'gateway-recovered'
            )
            gateways.append(restarted)
            await restarted.start()
            recovered = await wait_row(runtime, arrival)
            duplicate = await client.post(
                '/v1/chat/completions', json=body(), headers={'Idempotency-Key': 'crash-key'}
            )
            save(
                'controller-recovery',
                {
                    'passed': recovered['state'] == 'unknown'
                    and recovered['cleanup'] == 'confirmed'
                    and duplicate.status_code == 409,
                    'before': crashed,
                    'after': recovered,
                    'duplicate_status': duplicate.status_code,
                },
            )
            require(results['controller-recovery']['passed'], '同目录恢复/去重不符')
            if exercise_worker:
                require(binding.alive(), '待故障worker身份不符')
                async with client.stream(
                    'POST', '/v1/chat/completions', json=body(stream=True, long=True)
                ) as response:
                    arrival = response.headers['x-arrival-id']
                    lines = response.aiter_lines()
                    async for line in lines:
                        if line.startswith('data:') and line[5:].strip() != '[DONE]':
                            obj = json.loads(line[5:])
                            if any(
                                c.get('delta', {}).get('content') for c in obj.get('choices', [])
                            ):
                                break
                    pending = next(r for r in rows(runtime) if r['id'] == arrival)
                    require(journal.get(pending['attempt_id']) is None, '故障前请求已完成')
                    require(
                        os.getpgid(binding.process['pid']) == binding.process['pid'],
                        'worker进程组不符',
                    )
                    os.killpg(binding.process['pid'], signal.SIGKILL)
                    with contextlib.suppress(httpx.HTTPError):
                        async for _ in lines:
                            pass
                row = await wait_row(runtime, arrival)
                health = await client.get('/health/ready')
                save(
                    'worker-exit',
                    {
                        'passed': row['state'] in {'failed', 'cancelled'}
                        and health.status_code == 503,
                        'row': row,
                        'ready_status': health.status_code,
                    },
                )
                require(results['worker-exit']['passed'], 'worker退出后仍ready')
                from start_engine import start

                replacement_dir = output / 'replacement-engine'
                start(config_path, replacement_dir)
                replacement = replacement_dir / 'binding.json'
                from probe_serving import probe

                probe(replacement, output / 'replacement-direct')
                old_ready = await client.get('/health/ready')
                require(old_ready.status_code == 503, '旧gateway自动继承同endpoint的新worker')
                restarted.stop()
                rebound = Gateway(
                    output, replacement, runtime, config_path, port, 'gateway-rebound'
                )
                gateways.append(rebound)
                await rebound.start()
                response = await client.post('/v1/chat/completions', json=body())
                save(
                    'worker-rebind',
                    {
                        'passed': response.status_code == 200,
                        'old_binding_status': old_ready.status_code,
                        'new_status': response.status_code,
                        'new_binding': str(replacement),
                        'old_failed_row_after_reconcile': next(
                            r for r in rows(runtime) if r['id'] == arrival
                        ),
                    },
                )
                require(results['worker-rebind']['passed'], '显式新binding恢复失败')
            deadline_config = json.loads(config_path.read_text(encoding='utf-8'))
            deadline_config['limits'].update(max_inflight=1, max_queue=4, deadline_seconds=0.35)
            profile = output / 'deadline-profile.json'
            write_once(profile, deadline_config)
            deadline_runtime = output / 'deadline-runtime'
            deadline_gateway = Gateway(
                output,
                replacement or binding_path,
                deadline_runtime,
                profile,
                port + 1,
                'gateway-deadline',
            )
            gateways.append(deadline_gateway)
            await deadline_gateway.start()
            async with httpx.AsyncClient(
                base_url=f'http://127.0.0.1:{port + 1}', timeout=20, trust_env=False
            ) as dc:
                deadline_responses = await asyncio.gather(
                    *(dc.post('/v1/chat/completions', json=body(long=True)) for _ in range(5))
                )
            deadline_rows = [
                await wait_row(deadline_runtime, r.headers['x-arrival-id'])
                for r in deadline_responses
            ]
            save(
                'deadline-queue',
                {
                    'passed': all(r.status_code == 504 for r in deadline_responses)
                    and any(
                        r['state'] == 'timed_out' and not r['dispatched'] for r in deadline_rows
                    )
                    and all(r['cleanup'] in {'none', 'confirmed'} for r in deadline_rows),
                    'statuses': [r.status_code for r in deadline_responses],
                    'rows': deadline_rows,
                    'profile': deadline_config['limits'],
                },
            )
            require(results['deadline-queue']['passed'], '真实deadline/排队到期资格失败')
        for source, expected in sources.items():
            require(file_hash(source) == expected, '运行期间源码变化：' + source)
        write_once(
            output / 'summary.json',
            {
                'passed': True,
                'cases': {k: v['passed'] for k, v in results.items()},
                'evidence': 'real_TCP_vllm_single_GPU_qualification',
                'research_claim': False,
                'cluster_claim': False,
                'runtime': str(runtime),
                'arrival_count': len(rows(runtime)),
            },
        )
        return {'passed': True, 'cases': len(results), 'output': str(output)}
    finally:
        for gateway in reversed(gateways):
            gateway.stop()
        if replacement:
            from start_engine import stop

            stop(replacement)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--config', default='configs/serving/local-smoke.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--exercise-worker-exit', action='store_true')
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(qualify(args.binding, args.config, args.output, args.exercise_worker_exit)),
            ensure_ascii=False,
            indent=2,
        )
    )
