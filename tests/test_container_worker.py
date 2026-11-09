"""容器监督者在启动中断时仍使用本轮binding清理。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_start_failure_is_retained_and_owned_binding_stopped(tmp_path, monkeypatch):
    import run_worker

    seen = []

    def interrupted_start(config, run):
        run.mkdir()
        (run / 'binding.json').write_text('{}')
        raise RuntimeError('injected startup failure')

    monkeypatch.setattr(run_worker, 'start', interrupted_start)
    monkeypatch.setattr(
        run_worker, 'stop', lambda path: seen.append(path) or {'stopped': True, 'remaining': []}
    )
    with pytest.raises(RuntimeError, match='injected startup failure'):
        run_worker.run('config.json', tmp_path / 'state', tmp_path / 'secrets')
    assert len(seen) == 1
    folder = seen[0].parent.parent
    assert json.loads((folder / 'failure.json').read_text())['message']
    assert json.loads((folder / 'cleanup.json').read_text())['engine']['stopped']
    assert not (tmp_path / 'state/current.json').exists()


@pytest.mark.parametrize('stop_fails', [False, True])
@pytest.mark.parametrize('signal_during_cleanup', [False, True])
def test_native_exit_keeps_original_bridge_until_termination_proof_can_be_read(
    tmp_path, monkeypatch, stop_fails, signal_during_cleanup
):
    import run_worker
    from types import SimpleNamespace

    events = []
    handlers = {}

    def start(config, directory):
        directory.mkdir()
        (directory / 'binding.json').write_text('{}')

    class Bridge:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append('bridge_terminated')
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(run_worker, 'start', start)
    monkeypatch.setattr(run_worker, 'probe', lambda *args: None)
    monkeypatch.setattr(run_worker.subprocess, 'Popen', lambda *args, **kwargs: Bridge())
    monkeypatch.setattr(
        run_worker.EngineBinding, 'load', lambda path: SimpleNamespace(alive=lambda: False)
    )

    def stop(path):
        events.append('engine_stopped')
        if stop_fails:
            raise RuntimeError('injected stop failure')
        return {'stopped': True}

    monkeypatch.setattr(run_worker, 'stop', stop)

    def sleep(seconds):
        events.append(('reconciliation_window', seconds))
        if signal_during_cleanup:
            handlers[run_worker.signal.SIGTERM](run_worker.signal.SIGTERM, None)

    monkeypatch.setattr(run_worker.time, 'sleep', sleep)
    monkeypatch.setattr(
        run_worker.signal, 'signal', lambda signum, handler: handlers.__setitem__(signum, handler)
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_worker.py',
            '--config',
            'config.json',
            '--state',
            str(tmp_path / 'state'),
            '--secrets',
            str(tmp_path / 'secrets'),
        ],
    )
    expected_error = 'injected stop failure' if stop_fails else 'native引擎退出'
    with pytest.raises(RuntimeError, match=expected_error):
        run_worker.main()
    expected = ['engine_stopped']
    if not stop_fails:
        expected.append(('reconciliation_window', 20))
    assert events == expected + ['bridge_terminated']
    if stop_fails:
        cleanup = next((tmp_path / 'state').glob('generation-*/cleanup.json'))
        assert json.loads(cleanup.read_text())['engine_error']['type'] == 'RuntimeError'


def test_stop_signal_still_interrupts_startup_before_cleanup(tmp_path, monkeypatch):
    import run_worker

    handlers = {}
    stopped = []

    def start(config, directory):
        directory.mkdir()
        (directory / 'binding.json').write_text('{}')
        handlers[run_worker.signal.SIGTERM](run_worker.signal.SIGTERM, None)
        raise AssertionError('启动停止信号被错误忽略')

    monkeypatch.setattr(run_worker, 'start', start)
    monkeypatch.setattr(run_worker, 'stop', lambda path: stopped.append(path) or {'stopped': True})
    monkeypatch.setattr(
        run_worker.signal, 'signal', lambda signum, handler: handlers.__setitem__(signum, handler)
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run_worker.py',
            '--config',
            'config.json',
            '--state',
            str(tmp_path / 'state'),
            '--secrets',
            str(tmp_path / 'secrets'),
        ],
    )
    run_worker.main()
    assert len(stopped) == 1
    assert list((tmp_path / 'state').glob('generation-*/shutdown.json'))
    assert not (tmp_path / 'state/current.json').exists()
