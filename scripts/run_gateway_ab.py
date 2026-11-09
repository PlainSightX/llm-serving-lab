"""网关热路径A/B测量：同一对vLLM引擎上，直连与多个冻结源码网关按轮换顺序交替测量。

与R3/R4使用同一负载生成器、同一固定输入（seed相同）、同一SLO与网关限额；差别在于：
- 每个网关从输出目录内的冻结源码副本启动（PYTHONPATH指向副本），运行身份可追溯；
- 稳态窗口在直连与各网关之间按排列轮换，避免固定先后顺序造成系统漂移偏差；
- 发布窗口在每个网关上都执行“别名请求 + 中途切换active”，比较发布期间行为。
只用于本机工程测量，不构成生产容量、应用质量或集群结论。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from serving_lab.benchmark import (
    metric_delta,
    metric_series_delta,
    parse_metrics,
    summarize,
    window,
)
from serving_lab.engine import EngineBinding, HttpEngine, check_port_available, process_identity
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.measurement_runtime import export_arrivals, prepare_runtime, snapshot_runtimes
from probe_serving import probe
from run_benchmark import cpu_sample, create_cases
from start_engine import start, stop

ENGINE_CONFIGS = {
    'small': 'configs/serving/r3-small.json',
    'large': 'configs/serving/r3-large.json',
}
FIRST_GATEWAY_PORT = 19083


def freeze_source(name, root, out):
    """把网关源码复制进证据目录，网关只从副本加载，测量期间改工作树也不影响本次运行。"""
    target = out / 'gateway-sources' / name
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(root / 'src' / 'serving_lab', target / 'src' / 'serving_lab', ignore=ignore)
    shutil.copytree(root / 'scripts', target / 'scripts', ignore=ignore)
    files = {
        p.relative_to(target).as_posix(): file_hash(p)
        for p in sorted(target.rglob('*'))
        if p.is_file()
    }
    return target, files


def gateway_env(frozen):
    return {**os.environ, 'PYTHONPATH': str(frozen / 'src')}


def profile_document_for(document, frozen):
    """网关profile身份包含其engine.py/profile.py摘要；用冻结副本重建该网关应报告的文档。"""
    rebuilt = json.loads(json.dumps(document))
    rebuilt['implementation'] = {
        name: file_hash(frozen / 'src' / 'serving_lab' / name)
        for name in ('profile.py', 'engine.py')
    }
    return rebuilt


async def run(args):
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.repeats is not None:
        config = {**config, 'repeats': args.repeats}
    specs = {}
    for value in args.gateway:
        name, path = value.split('=', 1)
        if name in specs or name == 'direct' or not name.isidentifier():
            raise ValueError('网关名必须唯一且不能叫direct')
        specs[name] = Path(path).resolve()
    notes = dict(value.split('=', 1) for value in args.note or [])
    runtime_root = prepare_runtime(out, args.runtime_base)
    gateway_ports = {name: FIRST_GATEWAY_PORT + i for i, name in enumerate(specs)}
    for port in (19080, 19082, *gateway_ports.values()):
        check_port_available('127.0.0.1', port)

    frozen = {}
    for name, root in specs.items():
        target, files = freeze_source(name, root, out)
        resolved = subprocess.check_output(
            [sys.executable, '-c', 'import serving_lab;print(serving_lab.__file__)'],
            env=gateway_env(target),
            cwd=repo,
            text=True,
        ).strip()
        if Path(resolved).resolve() != (target / 'src' / 'serving_lab' / '__init__.py').resolve():
            raise RuntimeError(f'{name}网关没有加载冻结副本：{resolved}')
        frozen[name] = {
            'source_root': str(root),
            'frozen_root': str(target),
            'files': files,
            'import_resolved': resolved,
            'port': gateway_ports[name],
            'note': notes.get(name),
        }
    runner_sources = [
        'scripts/run_gateway_ab.py',
        'scripts/run_benchmark.py',
        'scripts/start_engine.py',
        'scripts/probe_serving.py',
        'src/serving_lab/benchmark.py',
        'src/serving_lab/engine.py',
        'src/serving_lab/profile.py',
        'src/serving_lab/measurement_runtime.py',
        str(Path(args.config).resolve().relative_to(repo)).replace('\\', '/'),
        *ENGINE_CONFIGS.values(),
        'fixtures/serving/r2-transport-v2.json',
        'environments/serving-linux/uv.lock',
    ]
    runner_hashes = {p: file_hash(repo / p) for p in runner_sources}
    write_once(
        out / 'input.json',
        {
            'config': config,
            'runner_source_hashes': runner_hashes,
            'gateways': frozen,
            'scope': config['scope'],
        },
    )
    token = secrets.token_hex(24)
    secret = out / 'admin-token'
    secret.write_text(token)
    secret.chmod(0o600)
    admin_headers = {'Authorization': 'Bearer ' + token}
    engines, bindings, gateways, gateway_configs, profiles = {}, {}, {}, {}, {}
    summaries = []
    telemetry = sampler_log = None

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
            for _ in range(400):
                states = await asyncio.gather(*(state(which) for which in gateways))
                if all(v['pending'] == 0 for item in states for v in item['versions']):
                    return
                await asyncio.sleep(0.05)
            raise RuntimeError('网关未排空，不能启动下一测量窗口')

        async def launch(which):
            versions = list(engines)
            a, b = versions
            old_small = next(
                (repo / 'artifacts/r2-qualification-20261102-a/runtime/qualifications').glob(
                    '*/receipt.json'
                )
            )
            cfg = {
                'bindings': [str(bindings[v]) for v in versions],
                'panel': str(repo / 'fixtures/serving/r2-transport-v2.json'),
                'application_receipts': {
                    a: str(old_small),
                    b: str(
                        repo / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json'
                    ),
                },
                'runtime': str(runtime_root / (which + '-runtime')),
                'port': gateway_ports[which],
                'runtime_receipt': str(out / (which + '-runtime-settings.json')),
                'admin_token_file': str(secret),
                'limits': config['gateway_limits'],
            }
            gateway_configs[which] = cfg
            cfgpath = out / (which + '-config.json')
            write_once(cfgpath, cfg)
            script = Path(frozen[which]['frozen_root']) / 'scripts' / 'serve_releases.py'
            with (out / (which + '.log')).open('xb') as log:
                process = subprocess.Popen(
                    [sys.executable, str(script), '--config', str(cfgpath)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=gateway_env(Path(frozen[which]['frozen_root'])),
                    cwd=repo,
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
            reported = {
                v['version']: v['identity']['profile_digest']
                for v in (await state(which))['versions']
            }
            documents = {
                v: profile_document_for(
                    engines[v].profile.document, Path(frozen[which]['frozen_root'])
                )
                for v in versions
            }
            if {v: digest(d) for v, d in documents.items()} != reported:
                raise RuntimeError(f'{which}网关报告的profile与冻结源码不一致')
            profiles[which] = {'digests': reported, 'documents': documents}

        async def make_requests(cases, versions, path, label, alias=False):
            requests = []
            for i, (case, version) in enumerate(zip(cases, versions)):
                engine = engines[version]
                if await engine.token_ids(case['messages']) != case['input_token_ids']:
                    raise ValueError('tokenizer不匹配冻结输入')
                body = {
                    'model': 'active' if alias else version,
                    'messages': case['messages'],
                    'max_tokens': case['output_budget'],
                    'stream': True,
                }
                rid = label + '-' + str(i)
                item = {
                    **case,
                    'id': rid,
                    'expected_version': version,
                    'allowed_versions': list(engines) if alias else [version],
                }
                if path == 'direct':
                    item.update(
                        path='direct',
                        expected_profile=engine.profile.digest,
                        body=engine.wire_payload({**body, 'model': version}),
                        headers={'X-Request-Id': rid},
                    )
                else:
                    item.update(
                        path='gateway',
                        gateway=path,
                        expected_profile=profiles[path]['digests'][version],
                        profiles=profiles[path]['digests'],
                        body=body,
                        headers={'Idempotency-Key': rid},
                    )
                requests.append(item)
            return requests

        async def measure(label, cases, versions, path, rate, *, publish=None):
            folder = out / 'windows' / label
            folder.mkdir(parents=True, exist_ok=False)
            requests = await make_requests(cases, versions, path, label, alias=publish is not None)
            offsets = [i / rate for i in range(len(cases))]
            if publish:
                half = len(cases) // 2
                offsets = [
                    v + (config['release_gap_seconds'] if i >= half else 0)
                    for i, v in enumerate(offsets)
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
                    'publish': publish,
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
                before_state = await state(path)
                began = asyncio.get_running_loop().time() - started
                route = await admin(
                    path,
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
                }

            endpoint = (
                engines[versions[0]].binding.endpoint
                if path == 'direct'
                else f'http://127.0.0.1:{gateway_configs[path]["port"]}'
            )
            result = await window(
                endpoint, requests, offsets, config, action=publish_action if publish else None
            )
            cpu_after = cpu_sample(gateways)
            settle_start = time.monotonic()
            if path == 'direct':
                if any(r['sent'] and r['status'] != 200 for r in result['rows']):
                    write_once(folder / 'result.json', result)
                    raise RuntimeError('直连出现非200/连接失败，停止并清理自有worker')
                sent = [r for r in result['rows'] if r['sent']]
                evidence = await asyncio.gather(
                    *(engines[r['expected_version']].wait_release(r['id'], 10) for r in sent)
                )
                result['direct_release_evidence'] = {r['id']: e for r, e in zip(sent, evidence)}
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
                            'vllm:num_preemptions_total',
                            'vllm:prompt_tokens_total',
                            'vllm:generation_tokens_total',
                        )
                    }
                    for v in engines
                },
                engine_finishes={
                    v: metric_series_delta(before[v], after[v], 'vllm:request_success_total')
                    for v in engines
                },
            )
            if publish:
                summary['planned_version_matches'] = all(
                    r.get('planned_version_match') for r in result['rows'] if r['status'] == 200
                )
                summary['switch_completed_in_gap'] = bool(
                    result['action']
                    and not result['action'].get('error')
                    and result['action']['acknowledged_offset'] < offsets[len(cases) // 2]
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
            for size, engine_config in ENGINE_CONFIGS.items():
                folder = out / (size + '-worker')
                path = folder / 'binding.json'
                startup = time.monotonic()
                try:
                    start(repo / engine_config, folder)
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
                if 'enable_prefix_caching=False' not in (folder / 'engine.log').read_text():
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
            for which in specs:
                await launch(which)
            write_once(out / 'profiles.json', profiles)
            sampler_log = (out / 'gpu.csv').open('xb')
            telemetry = subprocess.Popen(
                [
                    'nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv',
                    '--loop-ms=1000',
                ],
                stdout=sampler_log,
                stderr=subprocess.STDOUT,
            )
            # 每种形状各一次全局预热，不计入正式请求。
            warm = [
                next(c for c in cases[0] if c['shape'] == shape)
                for shape in sorted({c['shape'] for c in cases[0]})
            ]
            for version in (a, b):
                for path in ('direct', *specs):
                    requests = await make_requests(
                        warm, [version] * len(warm), path, f'warm-{version}-{path}'
                    )
                    endpoint = (
                        engines[version].binding.endpoint
                        if path == 'direct'
                        else f'http://127.0.0.1:{gateway_configs[path]["port"]}'
                    )
                    warmed = await window(endpoint, requests, list(range(len(warm))), config)
                    write_once(out / f'warm-{version}-{path}.json', warmed)
                    if not all(row['protocol_valid'] for row in warmed['rows']):
                        raise RuntimeError('预热协议未通过，停止正式测量')
                    if path == 'direct' and not all(
                        await asyncio.gather(
                            *(engines[version].wait_release(r['id'], 10) for r in warmed['rows'])
                        )
                    ):
                        raise RuntimeError('直连预热释放未确认')
                    await quiescent()
            paths = ['direct', *specs]
            orders = list(itertools.permutations(paths))
            combo = 0
            for rep in range(config['repeats']):
                for version in (a, b):
                    for rate in config['rates']:
                        for path in orders[combo % len(orders)]:
                            await measure(
                                f'steady-r{rep}-{version}-rate{rate}-{path}',
                                cases[rep],
                                [version] * len(cases[rep]),
                                path,
                                rate,
                            )
                        combo += 1
            for rep in range(config['repeats']):
                source, target = (a, b) if rep % 2 == 0 else (b, a)
                half = config['release_requests'] // 2
                planned = [source] * half + [target] * half
                ordered = list(specs) if rep % 2 == 0 else list(reversed(list(specs)))
                for which in ordered:
                    route = (await state(which))['route']
                    if route['active'] != source or route['canary'] is not None:
                        raise RuntimeError('发布窗口起点不是预定active版本')
                    await measure(
                        f'release-r{rep}-{which}',
                        release_cases[rep],
                        planned,
                        which,
                        config['release_rate'],
                        publish=target,
                    )
            await quiescent()
            for which, cfg in gateway_configs.items():
                export_arrivals(
                    Path(cfg['runtime']) / 'service.sqlite', out / (which + '-arrivals.json')
                )
            if any(file_hash(repo / p) != h for p, h in runner_hashes.items()):
                raise RuntimeError('测量期间测量器源码发生变化')
            for name, item in frozen.items():
                root = Path(item['frozen_root'])
                if any(file_hash(root / p) != h for p, h in item['files'].items()):
                    raise RuntimeError(f'{name}冻结源码在运行期间被修改')
            write_once(
                out / 'summary.json',
                {
                    'integrity_completed': True,
                    'formal_requests': sum(s['planned'] for s in summaries),
                    'formal_windows': len(summaries),
                    'schedule_valid_windows': sum(s['schedule_valid'] for s in summaries),
                    'windows': summaries,
                    'scope': config['scope'],
                },
            )
            return {'output': str(out), 'formal_requests': sum(s['planned'] for s in summaries)}
        finally:
            for which, process in gateways.items():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            if telemetry and telemetry.poll() is None:
                telemetry.terminate()
                telemetry.wait(timeout=5)
            if sampler_log:
                sampler_log.close()
            if telemetry:
                lines = (out / 'gpu.csv').read_text(errors='replace').splitlines()
                samples = (
                    list(csv.DictReader(lines))
                    if lines and lines[0].startswith('timestamp')
                    else []
                )
                write_once(
                    out / 'telemetry-summary.json',
                    {'samples': len(samples), 'header': lines[0] if lines else None},
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
    parser.add_argument('--config', default='configs/benchmarks/gateway-ab-local.json')
    parser.add_argument(
        '--runtime-base', required=True, help='本机ext4目录，SQLite与请求文件放在这里'
    )
    parser.add_argument(
        '--gateway', action='append', required=True, help='NAME=源码根目录（含src与scripts）'
    )
    parser.add_argument('--note', action='append', help='NAME=说明，例如对应的git提交')
    parser.add_argument('--repeats', type=int, help='仅用于短诊断；正式A/B使用配置中的重复次数')
    print(json.dumps(asyncio.run(run(parser.parse_args())), indent=2, ensure_ascii=False))
