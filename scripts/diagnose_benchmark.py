"""仅复核B所选32请求的同步阻塞调用链；不重跑或替换正式性能矩阵。"""

import argparse
import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from serving_lab.paths import lab_root

import httpx

from serving_lab.benchmark import window, summarize
from serving_lab.engine import EngineBinding, process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.measurement_runtime import prepare_runtime, snapshot_runtimes, export_arrivals
from start_engine import start, stop
from probe_serving import probe
from storage_benchmark import storage_diagnosis


async def run(original, output, runtime_base=None):
    repo = Path(__file__).resolve().parents[1]
    original = Path(original).resolve()
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((original / 'input.json').read_text())['config']
    runtime_root = prepare_runtime(out, runtime_base)
    selected = json.loads((original / 'diagnostic-selection.json').read_text())['selected']
    chosen = json.loads((original / 'windows' / selected / 'input.json').read_text())
    requests = []
    for i, item in enumerate(chosen['requests']):
        requests.append(
            {
                **item,
                'source_request_id': item['id'],
                'id': f'c-diagnostic-{i}',
                'headers': {'Idempotency-Key': f'c-diagnostic-{i}'},
            }
        )
    sources = [
        'scripts/diagnose_benchmark.py',
        'scripts/instrument_gateway.py',
        'scripts/storage_benchmark.py',
        'scripts/start_engine.py',
        'scripts/serve_releases.py',
    ]
    sources += [
        'src/serving_lab/' + name + '.py'
        for name in (
            'benchmark',
            'runtime',
            'online',
            'profile',
            'engine',
            'releases',
            'transport',
            'qualification',
            'parameters',
            'measurement_runtime',
        )
    ]
    hashes = {name: file_hash(repo / name) for name in sources}
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'source_formal_run': str(original),
            'selected_window': selected,
            'selected_input_sha256': file_hash(original / 'windows' / selected / 'input.json'),
            'requests': requests,
            'source_hashes': hashes,
            'scope': '32 request diagnostic only; formal B measurements unchanged',
        },
    )
    bindings = []
    process = None
    token = secrets.token_hex(24)
    secret = out / 'admin-token'
    secret.write_text(token)
    secret.chmod(0o600)
    headers = {'Authorization': 'Bearer ' + token}
    base = 'http://127.0.0.1:19083'
    trace = out / 'timing.json'
    try:
        for port in (19080, 19082, 19083):
            check_port_available('127.0.0.1', port)
        for size in ('small', 'large'):
            folder = out / (size + '-worker')
            binding = folder / 'binding.json'
            try:
                start(repo / f'configs/serving/r3-{size}.json', folder)
            finally:
                if binding.exists():
                    bindings.append(binding)
            probe(binding, out / (size + '-direct'))
        versions = [EngineBinding.load(p).version_id for p in bindings]
        a, b = versions
        cfg = {
            'bindings': [str(p) for p in bindings],
            'panel': str(repo / 'fixtures/serving/r2-transport-v2.json'),
            'application_receipts': {
                a: str(
                    next(
                        (
                            repo / 'artifacts/r2-qualification-20261102-a/runtime/qualifications'
                        ).glob('*/receipt.json')
                    )
                ),
                b: str(repo / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json'),
            },
            'runtime': str(runtime_root / 'control-runtime' if runtime_base else out / 'runtime'),
            'runtime_receipt': str(out / 'runtime-settings.json'),
            'port': 19083,
            'limits': config['gateway_limits'],
            'admin_token_file': str(secret),
        }
        write_once(out / 'gateway.json', cfg)
        env = dict(os.environ)
        env['SERVING_LAB_TIMING_PATH'] = str(trace)
        with (out / 'gateway.log').open('xb') as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(repo / 'scripts/instrument_gateway.py'),
                    '--config',
                    str(out / 'gateway.json'),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_once(out / 'gateway-process.json', process_identity(process.pid))
        async with httpx.AsyncClient(base_url=base, timeout=300, trust_env=False) as client:

            async def state():
                r = await client.get('/admin/releases', headers=headers)
                r.raise_for_status()
                return r.json()

            async def empty():
                for _ in range(400):
                    if all(v['pending'] == 0 for v in (await state())['versions']):
                        return
                    await asyncio.sleep(0.025)
                raise RuntimeError('诊断结束但租约未排空')

            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError('诊断网关启动失败')
                try:
                    await state()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.5)
            else:
                raise RuntimeError('诊断网关启动超时')
            for version in versions:
                r = await client.post(
                    f'/admin/releases/{version}/qualify', headers=headers, json={}
                )
                r.raise_for_status()
                assert r.json()['passed'] and r.json()['profile_digest'] == next(
                    item['profiles'][version] for item in requests
                )
            r = await client.post(
                '/admin/route',
                headers=headers,
                json={'operation': 'promote', 'target': a, 'expected_generation': 0},
            )
            r.raise_for_status()
            warm = []
            for i, shape in enumerate(sorted({item['shape'] for item in requests})):
                item = next(r for r in requests if r['shape'] == shape)
                warm.append(
                    {**item, 'id': f'c-warm-{i}', 'headers': {'Idempotency-Key': f'c-warm-{i}'}}
                )
            warmed = await window(base, warm, list(range(4)), config)
            write_once(out / 'warm.json', warmed)
            assert all(r['protocol_valid'] for r in warmed['rows'])
            await empty()
            os.kill(process.pid, signal.SIGUSR1)
            for _ in range(500):
                if trace.with_suffix('.start.json').exists():
                    break
                await asyncio.sleep(0.01)
            else:
                raise RuntimeError('计时器没有启动回执')
            result = await window(base, requests, chosen['offsets'], config)
            write_once(out / 'result.json', result)
            await empty()
            drained = time.monotonic()
            os.kill(process.pid, signal.SIGUSR2)
            for _ in range(500):
                if trace.with_suffix('.stop.json').exists():
                    break
                await asyncio.sleep(0.01)
            else:
                raise RuntimeError('计时器没有停止回执')
            timing = json.loads(trace.read_text())
            records = timing['records']
            assert timing['started'] <= result['started_monotonic'] and timing['ended'] >= drained
            assert all(
                r['thread'] == timing['owner_thread']
                and r['native_thread'] == timing['owner_native']
                for r in records
            )
            counts = Counter(r['operation'] for r in records)
            assert all(row['protocol_valid'] for row in result['rows'])
            assert (
                counts['record_arrival']
                == counts['bind_arrival']
                == counts['reserve_attempt']
                == counts['dispatched']
                == counts['release_lease']
                == 32
            )
            exits = [r for r in records if r['operation'] == 'sqlite_transaction_exit']
            assert all(r['caller_file'].endswith('runtime.py') and r['arrival_id'] for r in exits)
            for left, right in zip(
                sorted(exits, key=lambda r: r['start']), sorted(exits, key=lambda r: r['start'])[1:]
            ):
                assert left['end'] <= right['start']
            write_once(
                out / 'timing-summary.json',
                {
                    'scope': 'main-thread synchronous wall time, not CPU percentage or end-to-end speedup',
                    'requests': 32,
                    'thread_ids_verified': True,
                    'trace_seconds': timing['ended'] - timing['started'],
                    'client_window_seconds': result['elapsed_seconds'],
                    'drained_monotonic': drained,
                    'transaction_exit_seconds': sum(r['seconds'] for r in exits),
                    'commit_seconds': sum(r['seconds'] for r in exits if r['outcome'] == 'commit'),
                    'operation_counts': dict(counts),
                    'operation_seconds': {
                        name: sum(r['seconds'] for r in records if r['operation'] == name)
                        for name in counts
                    },
                    'protocol': summarize(
                        result['rows'],
                        result['elapsed_seconds'],
                        config['slo'],
                        interval=1 / chosen['rate'],
                    ),
                },
            )
            export_arrivals(Path(cfg['runtime']) / 'service.sqlite', out / 'arrivals.json')
            fraction = sum(r['seconds'] for r in exits) / (timing['ended'] - timing['started'])
            write_once(
                out / 'storage-diagnostic-trigger.json',
                {
                    'basis': 'verified thread-tagged timing',
                    'fraction': fraction,
                    'triggered': fraction >= 0.1,
                },
            )
            if fraction >= 0.1:
                native = lab_root() / 'r3-diagnostics' / out.name / 'runtime'
                write_once(
                    out / 'local-runtime-owner.json',
                    {
                        'path': str(native),
                        'owner': 'R3 root',
                        'retention': 'original CPU storage evidence',
                    },
                )
                write_once(
                    out / 'storage-diagnostic.json',
                    storage_diagnosis(
                        out / 'storage-mounted', native, config['diagnostic_requests']
                    ),
                )
            assert all(file_hash(repo / name) == value for name, value in hashes.items())
            return {
                'scope': 'independent32-request timed diagnosis',
                'verified': True,
                'output': str(out),
            }
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        cleanup = []
        for path in bindings:
            try:
                cleanup.append({'binding': str(path), 'result': stop(path)})
            except Exception as exc:
                cleanup.append({'binding': str(path), 'error': repr(exc)})
        write_once(out / 'cleanup.json', cleanup)
        snapshot_runtimes(out, runtime_root, ['control'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-base')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.source, args.output, args.runtime_base)), indent=2))
