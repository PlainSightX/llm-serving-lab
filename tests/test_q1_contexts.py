"""消费者需核对真正被子进程导入的文件，不能只检查旁边的记录副本。"""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def test_execution_module_drift_is_rejected(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/q1_contexts.py'
    spec = importlib.util.spec_from_file_location('q1_test_contexts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = b'# frozen protocol\n'
    runner = b'# frozen runner\n'
    (tmp_path / 'source_task_protocol.py').write_bytes(protocol)
    (tmp_path / 'source_q1_qualify.py').write_bytes(runner)
    actual = tmp_path / 'source-code/serving_lab/task_protocol.py'
    actual.parent.mkdir(parents=True)
    actual.write_bytes(protocol)
    identity = {
        'task_protocol_sha256': hashlib.sha256(protocol).hexdigest(),
        'runner_sha256': hashlib.sha256(runner).hexdigest(),
    }
    module.verify_execution_sources(tmp_path, identity)
    actual.write_bytes(b'# changed actual child module\n')
    with pytest.raises(ValueError):
        module.verify_execution_sources(tmp_path, identity)
