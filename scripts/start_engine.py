"""在独立Linux环境启动一个受控vLLM实例，记录实际进程与参数身份。"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

from serving_lab.paths import expand
from serving_lab.engine import EngineBinding, process_identity, check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash, inspect_base
from serving_lab.worker_protocol import supervisor_identity


def cache_options(config):
    """显式缓存只控制KV容量；不把utilization误写成整个进程的显存上限。"""
    if 'kv_cache_memory_bytes' not in config:
        return []
    value = config['kv_cache_memory_bytes']
    if type(value) is not int or value <= 0:
        raise ValueError('kv_cache_memory_bytes必须为正整数bytes')
    return ['--kv-cache-memory-bytes', str(value)]


def observe_startup_processes(binding_path):
    """启动期间逐次保存出生身份，避免API父进程退出后遗失已启动的worker。"""
    binding = EngineBinding.load(binding_path)
    directory = binding.manifest_path.parent
    if json.loads((directory / 'startup-supervisor.json').read_text()) != supervisor_identity():
        raise RuntimeError('启动进程监督环境发生变化')
    path = directory / 'startup-processes.jsonl'
    recorded = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    expected = {
        f'SERVING_LAB_WORKER_ID={binding.worker_instance_id}'.encode(),
        f'SERVING_LAB_GENERATION={binding.startup_generation}'.encode(),
    }
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError):
            pid = int(entry.name)
            if os.getpgid(pid) != binding.process['pid']:
                continue
            identity = process_identity(pid)
            # 同一组且继承本次不可变随机代际；不靠PID数字认领进程。
            if (
                not identity
                or identity in recorded
                or not expected <= set((entry / 'environ').read_bytes().split(b'\0'))
            ):
                continue
            if process_identity(pid) != identity:
                continue
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(identity) + '\n')
            recorded.append(identity)
    return recorded


def stop(binding_path):
    if sys.platform != 'linux':
        raise RuntimeError('停止引擎只允许在对应Linux环境执行，不能从Windows推断Linux进程已退出')
    binding = EngineBinding.load(binding_path)
    pid = binding.process['pid']
    supervisor_path = binding.manifest_path.parent / 'startup-supervisor.json'
    if (
        supervisor_path.exists()
        and json.loads(supervisor_path.read_text()) != supervisor_identity()
    ):
        raise RuntimeError('停止操作不在原启动监督环境，不能推断原进程已退出')
    identity_path = binding.manifest_path.parent / 'runtime-identity.json'
    owned = (
        json.loads(identity_path.read_text(encoding='utf-8'))['processes']
        if identity_path.exists()
        else [binding.process]
    )
    startup_path = binding.manifest_path.parent / 'startup-processes.jsonl'
    if startup_path.exists():
        for line in startup_path.read_text().splitlines():
            identity = json.loads(line)
            if identity not in owned:
                owned.append(identity)
    if binding.alive():
        if os.getpgid(pid) != pid:
            raise RuntimeError('进程组与自有启动身份不一致')
        # 在已确认的自有session仍存活时，固定整个进程组的出生身份，不能只等API leader。
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            with contextlib.suppress(OSError, ValueError):
                if os.getpgid(int(entry.name)) == pid:
                    identity = process_identity(int(entry.name))
                    if identity and identity not in owned:
                        owned.append(identity)

    def alive_members():
        return [p for p in owned if process_identity(p['pid']) == p]

    live = alive_members()
    if not live:
        return {'stopped': True, 'already_exited': True, 'processes': owned}
    if any(os.getpgid(p['pid']) != pid for p in live):
        raise RuntimeError('自有worker进程组发生变化')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while alive_members() and time.monotonic() < deadline:
        time.sleep(0.2)
    if alive_members():
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while alive_members() and time.monotonic() < deadline:
            time.sleep(0.1)
    return {
        'stopped': not alive_members(),
        'processes': owned,
        'remaining': alive_members(),
        'worker_instance_id': binding.worker_instance_id,
    }


def start(config_path, run_dir):
    if sys.platform != 'linux':
        raise RuntimeError('引擎启动只允许独立Linux环境')
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    cache_args = cache_options(config)
    if config['host'] != '127.0.0.1' or config['vllm_version'] != '0.11.2':
        raise ValueError('不符合R1固定运行合同')
    if importlib.metadata.version('vllm') != config['vllm_version']:
        raise RuntimeError('引擎版本与配置不符')
    directory = Path(run_dir).resolve()
    if directory.exists():
        raise ValueError('运行目录已存在，禁止覆盖原证据')
    check_port_available(config['host'], config['port'])
    model_path = (repo / config['model_path']).resolve()
    base = inspect_base(model_path)
    directory.mkdir(parents=True)
    instance = uuid.uuid4().hex
    generation = uuid.uuid4().hex
    endpoint = f'http://{config["host"]}:{config["port"]}'
    events = directory / 'release-events.jsonl'
    from serving_lab import engine_probe

    probe_source = Path(engine_probe.__file__).resolve()
    (directory / 'source_engine_probe.py').write_bytes(probe_source.read_bytes())
    env = dict(os.environ)
    env.update(
        VLLM_PLUGINS='serving_lab_release_probe',
        SERVING_LAB_RELEASE_EVENTS=str(events),
        SERVING_LAB_WORKER_ID=instance,
        SERVING_LAB_GENERATION=generation,
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1',
        DO_NOT_TRACK='1',
        TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',
    )
    if config.get('enable_torch_profiler') is True:
        profiles = directory / 'torch-profiles'
        profiles.mkdir()
        env['VLLM_TORCH_PROFILER_DIR'] = str(profiles)
    compiler = None
    if config.get('compiler_root'):
        compiler_root = expand(config['compiler_root']).resolve()
        compiler = json.loads((compiler_root / 'receipt.json').read_text())
        if compiler.get('cpu_compile_link_and_load') != 'C and C++ passed':
            raise ValueError('编译器未通过CPU资格')
        for name in ('cc', 'c++'):
            if not (compiler_root / 'bin' / name).is_file():
                raise ValueError('缺少编译器wrapper')
        cache = expand(config['compile_cache_root']).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        env.update(
            CC=str(compiler_root / 'bin/cc'),
            CXX=str(compiler_root / 'bin/c++'),
            VLLM_CACHE_ROOT=str(cache / 'vllm'),
            TRITON_CACHE_DIR=str(cache / 'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'),
        )
        compiler = {
            **compiler,
            'manifest_sha256': file_hash(compiler_root / 'package-manifest.json'),
            'cc_wrapper_sha256': file_hash(compiler_root / 'bin/cc'),
            'cxx_wrapper_sha256': file_hash(compiler_root / 'bin/c++'),
            'compile_cache_root': str(cache),
        }
    command = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        str(model_path),
        '--served-model-name',
        config['version_id'],
        '--host',
        config['host'],
        '--port',
        str(config['port']),
        '--dtype',
        config['dtype'],
        '--max-model-len',
        str(config['max_model_len']),
        '--max-num-seqs',
        str(config['max_num_seqs']),
        '--max-num-batched-tokens',
        '4096',
        '--gpu-memory-utilization',
        str(config['gpu_memory_utilization']),
        '--generation-config',
        'vllm',
        '--enable-request-id-headers',
    ]
    command.extend(cache_args)
    if config['enforce_eager']:
        command.append('--enforce-eager')
    if 'enable_prefix_caching' in config:
        if type(config['enable_prefix_caching']) is not bool:
            raise ValueError('prefix cache配置必须为bool')
        command.append(
            '--enable-prefix-caching'
            if config['enable_prefix_caching']
            else '--no-enable-prefix-caching'
        )
    write_once(
        directory / 'launch.json',
        {
            'command': command,
            'config': config,
            'base_identity': base,
            'worker_instance_id': instance,
            'startup_generation': generation,
            'probe_sha256': file_hash(probe_source),
            'versions': {
                x: importlib.metadata.version(x)
                for x in ('vllm', 'torch', 'transformers', 'llm-serving-lab')
            },
            'python': sys.version,
            'executable': sys.executable,
            'compiler': compiler,
            'created': time.time(),
        },
    )
    write_once(directory / 'startup-supervisor.json', supervisor_identity())
    with (directory / 'engine.log').open('xb') as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    identity = process_identity(process.pid)
    if not identity:
        raise RuntimeError('未获得实际启动进程身份')
    write_once(
        directory / 'binding.json',
        {
            'version_id': config['version_id'],
            'endpoint': endpoint,
            'served_model': config['version_id'],
            'artifact_digest': digest(base['files']),
            'worker_instance_id': instance,
            'startup_generation': generation,
            'process': identity,
            'model_path': str(model_path),
            'events_path': str(events),
        },
    )
    print(
        json.dumps(
            {'state': 'starting', 'pid': process.pid, 'binding': str(directory / 'binding.json')},
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.monotonic()
    while time.monotonic() - started < 300:
        observe_startup_processes(directory / 'binding.json')
        if process.poll() is not None:
            write_once(
                directory / 'startup-failed.json',
                {
                    'exit_code': process.returncode,
                    'elapsed': time.monotonic() - started,
                    'cleanup': stop(directory / 'binding.json'),
                },
            )
            raise RuntimeError('真实引擎启动失败，见engine.log；不回退模拟')
        try:
            response = httpx.get(endpoint + '/health', timeout=2, trust_env=False)
            if response.status_code == 200:
                models = httpx.get(endpoint + '/v1/models', timeout=2, trust_env=False).json()
                if config['version_id'] in [m['id'] for m in models['data']]:
                    result = {
                        'state': 'ready',
                        'elapsed_seconds': time.monotonic() - started,
                        'models': models,
                        'pid': process.pid,
                        'binding': str(directory / 'binding.json'),
                    }
                    write_once(directory / 'ready.json', result)
                    return result
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        time.sleep(1)
    stop(directory / 'binding.json')
    write_once(directory / 'startup-timeout.json', {'timeout_seconds': 300})
    raise RuntimeError('启动超时，已停止确切自有引擎')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/serving/local-smoke.json')
    parser.add_argument('--run-dir')
    parser.add_argument('--stop')
    args = parser.parse_args()
    result = stop(args.stop) if args.stop else start(args.config, args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
