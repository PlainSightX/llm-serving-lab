"""在同一容器内监督native引擎和TLS桥，并保留每次启动的独立身份。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from serving_lab.engine import EngineBinding
from serving_lab.execution import write_once
from start_engine import start, stop
from probe_serving import probe


class StopRequested(BaseException):
    """SIGTERM进入finally，容器等待自有进程完成清理。"""


def run(
    config, state, secrets, port=8443, *, shutdown_state=None, profile_policy='q2_public_choice_v1'
):
    if profile_policy not in {
        'q2_public_choice_v1',
        'greedy_non_thinking_v1',
        'n2_length_controlled_v1',
    }:
        raise ValueError('未知worker profile')
    if shutdown_state is None:
        shutdown_state = {'cleanup_started': False}
    state, secrets = Path(state), Path(secrets)
    directory = state / ('generation-' + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    binding = directory / 'native/binding.json'
    bridge = None
    try:
        start(config, binding.parent)
        # 两次bootstrap在桥资格之前产生真实Scheduler释放记录。
        write_once(directory / 'bootstrap-reservation.json', {'reserved_calls': 2})
        probe(binding, directory / 'bootstrap')
        manifest = directory / 'manifest.json'
        command = [
            sys.executable,
            str(Path(__file__).with_name('serve_worker.py')),
            '--binding',
            str(binding),
            '--manifest-output',
            str(manifest),
            '--token-file',
            str(secrets / 'worker-token'),
            '--host',
            '0.0.0.0',
            '--port',
            str(port),
            '--tls-ca',
            str(secrets / 'ca.pem'),
            '--tls-cert',
            str(secrets / 'tls.pem'),
            '--tls-key',
            str(secrets / 'tls.key'),
            '--profile-policy',
            profile_policy,
        ]
        with (directory / 'bridge.log').open('xb') as log:
            bridge = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        pointer = state / 'current.json'
        temporary = state / ('pointer-' + uuid.uuid4().hex)
        temporary.write_text(
            json.dumps({'run': str(directory), 'manifest': str(manifest), 'binding': str(binding)}),
            encoding='utf-8',
        )
        os.replace(temporary, pointer)
        while bridge.poll() is None:
            if not EngineBinding.load(binding).alive():
                raise RuntimeError('native引擎退出，停止旧代际桥')
            time.sleep(0.25)
        raise RuntimeError(f'worker桥退出: {bridge.returncode}')
    except StopRequested:
        write_once(directory / 'shutdown.json', {'reason': 'SIGTERM or SIGINT'})
    except BaseException as error:
        write_once(
            directory / 'failure.json', {'error_type': type(error).__name__, 'message': str(error)}
        )
        raise
    finally:
        shutdown_state['cleanup_started'] = True
        cleanup = {'engine': None, 'bridge_returncode': None}
        try:
            if binding.exists():
                try:
                    cleanup['engine'] = stop(binding)
                except BaseException as error:
                    cleanup['engine_error'] = {'type': type(error).__name__, 'message': str(error)}
                    raise
            if (
                bridge is not None
                and bridge.poll() is None
                and cleanup['engine']
                and cleanup['engine']['stopped']
            ):
                # 覆盖10秒清理与就绪端点更新。总清理上界70秒，低于Pod的75秒宽限。
                time.sleep(20)
        finally:
            if bridge is not None:
                if bridge.poll() is None:
                    bridge.terminate()
                    try:
                        bridge.wait(timeout=25)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.wait(timeout=5)
                cleanup['bridge_returncode'] = bridge.returncode
            write_once(directory / 'cleanup.json', cleanup)
        if cleanup['engine'] and not cleanup['engine']['stopped']:
            raise RuntimeError('native自有进程清理未完成')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--state', default='/state')
    parser.add_argument('--secrets', default='/secrets')
    parser.add_argument('--port', type=int, default=8443)
    parser.add_argument(
        '--profile-policy',
        choices=['q2_public_choice_v1', 'greedy_non_thinking_v1', 'n2_length_controlled_v1'],
        default='q2_public_choice_v1',
    )
    args = parser.parse_args()
    shutdown_state = {'cleanup_started': False}

    def requested(signum, frame):
        # 后续停止信号归并到正在执行的有界清理，避免打断原监督证明窗口。
        if not shutdown_state['cleanup_started']:
            raise StopRequested()

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    run(
        args.config,
        args.state,
        args.secrets,
        args.port,
        shutdown_state=shutdown_state,
        profile_policy=args.profile_policy,
    )


if __name__ == '__main__':
    main()
