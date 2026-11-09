"""启动前拒绝无效缓存配置；父进程提前退出也不能遗失已观察子进程。"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import start_engine
from serving_lab.engine import process_identity
from serving_lab.worker_protocol import supervisor_identity
from test_engine import launch_record


def test_explicit_cache_budget_validated_and_forwarded():
    assert hasattr(start_engine, 'cache_options')
    assert start_engine.cache_options({}) == []
    assert start_engine.cache_options({'kv_cache_memory_bytes': 768 * 1024**2}) == [
        '--kv-cache-memory-bytes',
        '805306368',
    ]
    for invalid in (True, False, 0, -1, '805306368', 1.5, None):
        with pytest.raises(ValueError):
            start_engine.cache_options({'kv_cache_memory_bytes': invalid})


@pytest.mark.skipif(sys.platform != 'linux', reason='需要Linux真实进程与出生身份')
def test_stop_remembers_child_after_startup_parent_exits(tmp_path):
    assert hasattr(start_engine, 'observe_startup_processes')
    child_file, exit_file = tmp_path / 'child', tmp_path / 'exit-parent'
    code = """import subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
Path(sys.argv[1]).write_text(str(child.pid))
while not Path(sys.argv[2]).exists(): time.sleep(.01)
"""
    env = dict(
        os.environ, SERVING_LAB_WORKER_ID='test-worker', SERVING_LAB_GENERATION='test-generation'
    )
    parent = subprocess.Popen(
        [sys.executable, '-c', code, str(child_file), str(exit_file)],
        env=env,
        start_new_session=True,
    )
    child_identity = None
    try:
        deadline = time.monotonic() + 5
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_file.exists()
        child_identity = process_identity(int(child_file.read_text()))
        assert child_identity
        data = {
            'version_id': 'test',
            'served_model': 'test',
            'endpoint': 'http://127.0.0.1:19080',
            'worker_instance_id': 'test-worker',
            'startup_generation': 'test-generation',
            'process': process_identity(parent.pid),
            'model_path': str(tmp_path),
            'events_path': str(tmp_path / 'events.jsonl'),
        }
        launch_record(tmp_path, data)
        binding = tmp_path / 'binding.json'
        binding.write_text(json.dumps(data))
        owner = tmp_path / 'startup-supervisor.json'
        owner.write_text(json.dumps(supervisor_identity()))
        start_engine.observe_startup_processes(binding)
        assert child_identity in [
            json.loads(s) for s in (tmp_path / 'startup-processes.jsonl').read_text().splitlines()
        ]
        exit_file.touch()
        parent.wait(timeout=5)
        assert (
            process_identity(parent.pid) is None
            and process_identity(child_identity['pid']) == child_identity
        )
        # 当前监督环境不符时，不能冒称退出或发信号。
        original = supervisor_identity()
        owner.write_text(json.dumps({**original, 'pid_namespace': 'different'}))
        with pytest.raises(RuntimeError, match='监督环境'):
            start_engine.stop(binding)
        assert process_identity(child_identity['pid']) == child_identity
        owner.write_text(json.dumps(original))
        result = start_engine.stop(binding)
        assert (
            result['stopped'] and not result['remaining'] and child_identity in result['processes']
        )
        assert process_identity(child_identity['pid']) is None
    finally:
        # 测试失败也只清理准确出生身份；不影响其他进程。
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
        if child_identity and process_identity(child_identity['pid']) == child_identity:
            os.kill(child_identity['pid'], 9)
