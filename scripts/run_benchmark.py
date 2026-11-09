"""固定R3矩阵与真实版本发布；准备、正式测量、诊断的记录严格分开。"""

from __future__ import annotations
import argparse
import asyncio
import csv
import json
import random
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

from serving_lab.benchmark import (
    window,
    summarize,
    parse_metrics,
    metric_delta,
    metric_series_delta,
)
from serving_lab.engine import EngineBinding, HttpEngine, process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.measurement_runtime import prepare_runtime, snapshot_runtimes, export_arrivals
from start_engine import start, stop
from probe_serving import probe


def create_cases(tokenizer, repeat, count, config):
    combinations = [(i, o) for i in config['input_tokens'] for o in config['output_budgets']]
    shapes = combinations * (count // 4)
    random.Random(config['seed'] + repeat).shuffle(shapes)
    result = []
    for index, (target, budget) in enumerate(shapes):
        prefix = f'ID:{repeat:02}-{index:03}\n'
        suffix = '\n从1开始逐个输出整数直到1000，不要省略，不要解释。'

        def messages(padding):
            return [{'role': 'user', 'content': prefix + ' x' * padding + suffix}]

        def ids(msg):
            return tokenizer.apply_chat_template(
                msg, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )

        padding = max(0, target - len(ids(messages(0))))
        for _ in range(20):
            body = messages(padding)
            tokens = ids(body)
            difference = target - len(tokens)
            if not difference:
                break
            padding = max(0, padding + difference)
        if len(tokens) != target:
            raise ValueError('无法在执行前固定输入长度')
        result.append(
            {
                'id': f'r{repeat}-{index:03}',
                'shape': f'in{target}-out{budget}',
                'messages': body,
                'input_tokens': target,
                'output_budget': budget,
                'input_token_ids': tokens,
            }
        )
    return result


def cpu_sample(processes):
    result = {}
    for name, process in processes.items():
        if process.poll() is None:
            p = psutil.Process(process.pid)
            times = p.cpu_times()
            result[name] = {'user': times.user, 'system': times.system, 'rss': p.memory_info().rss}
    return result


async def run(output, config_path, runtime_base=None):
    repo = Path(__file__).resolve().parents[1]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    gateways = {}
    engines = {}
    bindings = {}
    telemetry = None
    runtime_root = prepare_runtime(out, runtime_base)
    for port in (19080, 19082, 19083, 19084):
        check_port_available('127.0.0.1', port)
    sources = [
        'src/serving_lab/' + n + '.py'
        for n in (
            'benchmark',
            'online',
            'engine',
            'profile',
            'transport',
            'qualification',
            'runtime',
            'releases',
            'parameters',
            'measurement_runtime',
        )
    ]
    sources += [
        'scripts/run_benchmark.py',
        'scripts/start_engine.py',
        'scripts/serve_releases.py',
        'scripts/probe_serving.py',
        'configs/benchmarks/r3-local.json',
        'configs/serving/r3-small.json',
        'configs/serving/r3-large.json',
        'fixtures/serving/r2-transport-v2.json',
        'environments/serving-linux/uv.lock',
    ]
    hashes = {p: file_hash(repo / p) for p in sources}
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'config': config,
            'source_hashes': hashes,
            'scope': 'local engineering measurement; no application quality or speedup claim',
        },
    )
    token = secrets.token_hex(24)
    secret = out / 'admin-token'
    secret.write_text(token)
    secret.chmod(0o600)
    admin_headers = {'Authorization': 'Bearer ' + token}
    gateway_configs = {}
    summaries = []
    window_index = 0
    sampler_log = None
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:

        async def snapshot(name):
            result = {}
            for version, engine in engines.items():
                response = await client.get(engine.binding.endpoint + '/metrics')
                response.raise_for_status()
                (out / (name + '-' + version + '-metrics.txt')).write_text(response.text)
                result[version] = parse_metrics(response.text)
            return result

        async def admin(which, path, body=None):
            base = f'http://127.0.0.1:{gateway_configs[which]["port"]}'
            response = await client.post(base + path, headers=admin_headers, json=body or {})
            if response.status_code != 200:
                raise RuntimeError(f'{which}{path}: {response.status_code} {response.text}')
            return response.json()

        async def state(which):
            response = await client.get(
                f'http://127.0.0.1:{gateway_configs[which]["port"]}/admin/releases',
                headers=admin_headers,
            )
            response.raise_for_status()
            return response.json()

        async def quiescent():
            for _ in range(200):
                states = await asyncio.gather(*(state(which) for which in gateways))
                if all(v['pending'] == 0 for item in states for v in item['versions']):
                    return
                await asyncio.sleep(0.05)
            raise RuntimeError('网关未排空，不能启动下一测量窗口')

        async def launch(which, runtime=None, port=None):
            old_small = next(
                (repo / 'artifacts/r2-qualification-20261102-a/runtime/qualifications').glob(
                    '*/receipt.json'
                )
            )
            versions = list(engines)
            a, b = versions
            cfg = {
                'bindings': [str(bindings[v]) for v in versions],
                'panel': str(repo / 'fixtures/serving/r2-transport-v2.json'),
                'application_receipts': {
                    a: str(old_small),
                    b: str(
                        repo / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json'
                    ),
                },
                'runtime': str(runtime or runtime_root / (which + '-runtime')),
                'port': port or (19083 if which == 'control' else 19084),
                'runtime_receipt': str(out / (which + '-runtime-settings.json')),
                'admin_token_file': str(secret),
                'limits': config['gateway_limits'],
            }
            gateway_configs[which] = cfg
            cfgpath = out / (which + '-config.json')
            write_once(cfgpath, cfg)
            with (out / (which + '.log')).open('xb') as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(repo / 'scripts/serve_releases.py'),
                        '--config',
                        str(cfgpath),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            gateways[which] = process
            write_once(out / (which + '-process.json'), process_identity(process.pid))
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError('网关启动失败：' + which)
                try:
                    await state(which)
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.5)
            else:
                raise RuntimeError('网关启动超时')
            for version in versions:
                receipt = await admin(which, f'/admin/releases/{version}/qualify')
                if not receipt['passed']:
                    raise RuntimeError('预资格未通过')
            await admin(
                which,
                '/admin/route',
                {'operation': 'promote', 'target': a, 'expected_generation': 0},
            )

        async def make_requests(cases, versions, path, label, alias=False):
            requests = []
            for i, (case, version) in enumerate(zip(cases, versions)):
                engine = engines[version]
                if await engine.token_ids(case['messages']) != case['input_token_ids']:
                    raise ValueError('两版本tokenizer不匹配冻结输入')
                body = {
                    'model': 'active' if alias else version,
                    'messages': case['messages'],
                    'max_tokens': case['output_budget'],
                    'stream': True,
                }
                rid = label + '-' + str(i)
                requests.append(
                    {
                        **case,
                        'id': rid,
                        'expected_version': version,
                        'path': path,
                        'expected_profile': engine.profile.digest,
                        'allowed_versions': list(engines) if alias else [version],
                        'profiles': {v: e.profile.digest for v, e in engines.items()},
                        'body': engine.wire_payload({**body, 'model': version})
                        if path == 'direct'
                        else body,
                        'headers': {'X-Request-Id': rid}
                        if path == 'direct'
                        else {'Idempotency-Key': rid},
                    }
                )
            return requests

        async def measure(label, cases, versions, path, rate, *, which='control', publish=None):
            nonlocal window_index
            window_index += 1
            folder = out / 'windows' / label
            folder.mkdir(parents=True, exist_ok=False)
            requests = await make_requests(cases, versions, path, label, alias=publish is not None)
            offsets = [i / rate for i in range(len(cases))]
            if label.startswith('release-'):
                offsets = [
                    value + (config['release_gap_seconds'] if i >= len(cases) // 2 else 0)
                    for i, value in enumerate(offsets)
                ]
            await quiescent()
            before = await snapshot(label + '-before')
            cpu_before = cpu_sample(gateways)
            write_once(
                folder / 'input.json',
                {
                    'requests': requests,
                    'offsets': offsets,
                    'rate': rate,
                    'path': path,
                    'gateway': which,
                    'publish': publish,
                    'diagnostic': False,
                    'profile_trace': False,
                    'profiles': {v: e.profile.document for v, e in engines.items()},
                },
            )

            async def publish_action(started):
                await asyncio.sleep(
                    max(
                        0,
                        started
                        + config['release_switch_seconds']
                        - asyncio.get_running_loop().time(),
                    )
                )
                before_state = await state(which)
                began = asyncio.get_running_loop().time() - started
                route = await admin(
                    which,
                    '/admin/route',
                    {
                        'operation': 'promote',
                        'target': publish,
                        'expected_generation': before_state['route']['generation'],
                    },
                )
                acknowledged = asyncio.get_running_loop().time() - started
                return {
                    'started_offset': began,
                    'acknowledged_offset': acknowledged,
                    'route': route,
                    'before': before_state['route'],
                }

            endpoint = (
                engines[versions[0]].binding.endpoint
                if path == 'direct'
                else f'http://127.0.0.1:{gateway_configs[which]["port"]}'
            )
            result = await window(
                endpoint, requests, offsets, config, action=publish_action if publish else None
            )
            cpu_after = cpu_sample(gateways)
            settle_start = time.monotonic()
            if path == 'direct':
                if any(r['sent'] and r['status'] != 200 for r in result['rows']):
                    write_once(folder / 'result.json', result)
                    raise RuntimeError('直连出现非200/连接失败，停止窗口串行推进并清理自有worker')
                sent = [
                    r
                    for r in result['rows']
                    if r['sent'] and (r['status'] == 200 or r['status'] is None)
                ]
                evidence = await asyncio.gather(
                    *(engines[r['expected_version']].wait_release(r['id'], 10) for r in sent)
                )
                result['direct_release_evidence'] = {
                    r['id']: event for r, event in zip(sent, evidence)
                }
                if not all(evidence):
                    write_once(folder / 'result.json', result)
                    raise RuntimeError('直连请求释放未确认，不能继续窗口')
            await quiescent()
            result['settle_seconds'] = time.monotonic() - settle_start
            after = await snapshot(label + '-after')
            summary = summarize(
                result['rows'],
                result['elapsed_seconds'],
                config['slo'],
                interval=1 / rate,
                lag_fraction=config['valid_schedule_p95_interval_fraction'],
            )
            summary.update(
                label=label,
                path=path,
                rate=rate,
                diagnostic=False,
                profile_trace=False,
                publish=publish,
                versions=sorted(set(versions)),
                settle_seconds=result['settle_seconds'],
                cpu_before=cpu_before,
                cpu_after=cpu_after,
                engine_metrics={
                    v: {
                        name: metric_delta(before[v], after[v], name)
                        for name in (
                            'vllm:request_queue_time_seconds_sum',
                            'vllm:request_queue_time_seconds_count',
                            'vllm:request_prefill_time_seconds_sum',
                            'vllm:request_decode_time_seconds_sum',
                            'vllm:prefix_cache_hits_total',
                            'vllm:prefix_cache_queries_total',
                            'vllm:num_preemptions_total',
                            'vllm:prompt_tokens_total',
                            'vllm:generation_tokens_total',
                        )
                    }
                    for v in engines
                },
            )
            summary['engine_finishes'] = {
                v: metric_series_delta(before[v], after[v], 'vllm:request_success_total')
                for v in engines
            }
            if label.startswith('release-'):
                summary['planned_version_matches'] = all(
                    r.get('planned_version_match') for r in result['rows'] if r['status'] == 200
                )
                summary['switch_completed_in_gap'] = not publish or bool(
                    result['action']
                    and not result['action'].get('error')
                    and result['action']['acknowledged_offset'] < offsets[len(cases) // 2]
                )
                summary['matched_plan_valid'] = (
                    summary['planned_version_matches'] and summary['switch_completed_in_gap']
                )
            write_once(folder / 'result.json', result)
            write_once(folder / 'summary.json', summary)
            summaries.append(summary)
            if publish and (not result['action'] or result['action'].get('error')):
                raise RuntimeError('发布操作失败，原窗口已保留')
            print(
                json.dumps(
                    {
                        'window': label,
                        'valid': summary['protocol_valid'],
                        'planned': summary['planned'],
                        'lag_ok': summary['schedule_valid'],
                        'ttft_median': summary['ttft_scheduled']['median'],
                        'slo': summary['slo_fraction'],
                    }
                ),
                flush=True,
            )
            return summary

        try:
            # 任何失败都只回收本包明确的binding，不触碰其他服务。
            for size in ('small', 'large'):
                folder = out / (size + '-worker')
                path = folder / 'binding.json'
                startup = time.monotonic()
                try:
                    start(repo / f'configs/serving/r3-{size}.json', folder)
                finally:
                    if path.exists():
                        bindings[EngineBinding.load(path).version_id] = path
                probe(path, out / (size + '-direct'))
                binding = EngineBinding.load(path)
                engine = HttpEngine(binding, profile=True)
                engines[binding.version_id] = engine
                write_once(
                    out / (size + '-preparation.json'),
                    {
                        'seconds': time.monotonic() - startup,
                        'binding': str(path),
                        'profile': engine.profile.document,
                    },
                )
                log = (folder / 'engine.log').read_text()
                if 'enable_prefix_caching=False' not in log:
                    raise RuntimeError('实际prefix cache未确认关闭')
            a, b = list(engines)
            cases = {
                rep: create_cases(engines[a].tokenizer, rep, config['steady_requests'], config)
                for rep in range(config['repeats'])
            }
            release_cases = {
                rep: create_cases(
                    engines[a].tokenizer, 100 + rep, config['release_requests'], config
                )
                for rep in range(config['repeats'])
            }
            write_once(out / 'workloads.json', {'steady': cases, 'release': release_cases})
            await launch('control')
            await launch('treatment')
            sampler_log = (out / 'gpu.csv').open('xb')
            telemetry = subprocess.Popen(
                [
                    'nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv',
                    '--loop-ms=1000',
                ],
                stdout=sampler_log,
                stderr=subprocess.STDOUT,
            )
            write_once(out / 'telemetry-process.json', process_identity(telemetry.pid))
            # 四种形状各一次，全局预热；不计入正式1152请求。
            warm = []
            for shape in sorted({c['shape'] for c in cases[0]}):
                warm.append(next(c for c in cases[0] if c['shape'] == shape))
            for version in (a, b):
                for path, which in (
                    ('direct', 'control'),
                    ('gateway', 'control'),
                    ('gateway', 'treatment'),
                ):
                    req = await make_requests(
                        warm, [version] * 4, path, 'warm-' + version + '-' + path + '-' + which
                    )
                    endpoint = (
                        engines[version].binding.endpoint
                        if path == 'direct'
                        else f'http://127.0.0.1:{gateway_configs[which]["port"]}'
                    )
                    warmed = await window(endpoint, req, [i for i in range(4)], config)
                    write_once(out / (f'warm-{version}-{path}-{which}.json'), warmed)
                    if not all(row['protocol_valid'] for row in warmed['rows']):
                        raise RuntimeError('预热协议未通过，停止正式测量')
                    if path == 'direct' and not all(
                        await asyncio.gather(
                            *(
                                engines[version].wait_release(row['id'], 10)
                                for row in warmed['rows']
                            )
                        )
                    ):
                        raise RuntimeError('直连预热释放未确认')
            for rep in range(config['repeats']):
                for model_index, version in enumerate((a, b)):
                    for rate_index, rate in enumerate(config['rates']):
                        order = (
                            ['direct', 'gateway']
                            if (rep + model_index + rate_index) % 2 == 0
                            else ['gateway', 'direct']
                        )
                        for path in order:
                            await measure(
                                f'steady-r{rep}-{version}-rate{rate}-{path}',
                                cases[rep],
                                [version] * len(cases[rep]),
                                path,
                                rate,
                            )
            for rep in range(config['repeats']):
                source, target = (a, b) if rep % 2 == 0 else (b, a)
                planned = [source] * (config['release_requests'] // 2) + [target] * (
                    config['release_requests'] // 2
                )
                order = ['control', 'treatment'] if rep % 2 == 0 else ['treatment', 'control']
                for which in order:
                    await measure(
                        f'release-r{rep}-{which}',
                        release_cases[rep],
                        planned,
                        'gateway',
                        config['release_rate'],
                        which=which,
                        publish=target if which == 'treatment' else None,
                    )
            formal = [s for s in summaries if not s['diagnostic']]
            assert sum(s['planned'] for s in formal) == config['formal_requests']
            # 事先定义选择规则：在有效稳态gateway窗口中，选择TTFT中位数最大的一个。
            eligible = [
                s
                for s in formal
                if s['label'].startswith('steady-')
                and s['path'] == 'gateway'
                and s['schedule_valid']
                and s['protocol_valid']
            ]
            if not eligible:
                raise RuntimeError('无合格窗口可定位，保留结果并停止')
            selected = max(eligible, key=lambda s: s['ttft_scheduled']['median'])
            write_once(
                out / 'diagnostic-selection.json',
                {
                    'rule': 'maximum median TTFT among schedule-valid steady gateway windows',
                    'selected': selected['label'],
                },
            )
            await quiescent()
            for which, cfg in gateway_configs.items():
                export_arrivals(
                    Path(cfg['runtime']) / 'service.sqlite', out / (which + '-arrivals.json')
                )
            assert all(file_hash(repo / p) == h for p, h in hashes.items())
            write_once(
                out / 'summary.json',
                {
                    'integrity_completed': True,
                    'formal_requests': sum(s['planned'] for s in formal),
                    'formal_windows': len(formal),
                    'schedule_valid_windows': sum(s['schedule_valid'] for s in formal),
                    'windows': summaries,
                    'scope': config['scope'],
                    'diagnostic_status': 'run separate thread-tagged diagnosis after cleanup',
                },
            )
            return {
                'output': str(out),
                'formal_requests': config['formal_requests'],
                'formal_windows': len(formal),
            }
        finally:
            for which, process in gateways.items():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            telemetry_before_stop = telemetry.poll() if telemetry else None
            if telemetry and telemetry.poll() is None:
                telemetry.terminate()
                telemetry.wait(timeout=5)
            if sampler_log:
                sampler_log.close()
            if telemetry:
                text = (out / 'gpu.csv').read_text(errors='replace')
                lines = text.splitlines()
                samples = (
                    list(csv.DictReader(lines))
                    if lines and lines[0].startswith('timestamp')
                    else []
                )
                write_once(
                    out / 'telemetry-summary.json',
                    {
                        'samples': len(samples),
                        'returncode_before_cleanup': telemetry_before_stop,
                        'returncode_after_cleanup': telemetry.returncode,
                        'header': lines[0] if lines else None,
                        'missing_values_preserved': True,
                        'usable': bool(samples) and telemetry_before_stop is None,
                    },
                )
            for engine in engines.values():
                await engine.close()
            cleanup = []
            for version, path in bindings.items():
                try:
                    cleanup.append({'version': version, 'binding': str(path), 'result': stop(path)})
                except Exception as exc:
                    cleanup.append({'version': version, 'error': repr(exc)})
            write_once(out / 'cleanup.json', cleanup)
            snapshot_runtimes(out, runtime_root, gateways)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/benchmarks/r3-local.json')
    parser.add_argument('--runtime-base')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.output, args.config, args.runtime_base)), indent=2))
