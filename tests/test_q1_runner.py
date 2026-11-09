"""只测试容器所有权/异常清理合同，不启动Docker或模型。"""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/q1_run.py'
    spec = importlib.util.spec_from_file_location('q1_test_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_docker(monkeypatch, module, *, timeout=False):
    calls = []
    identity = {}
    running = True

    def output(args, **kwargs):
        calls.append(args)
        if args[1] == 'create':
            identity['name'] = args[args.index('--name') + 1]
            identity['labels'] = {
                args[i + 1].split('=', 1)[0]: args[i + 1].split('=', 1)[1]
                for i, x in enumerate(args)
                if x == '--label'
            }
            return 'exact-owned-id\n'
        return json.dumps(
            [
                {
                    'Name': '/' + identity['name'],
                    'Config': {'Labels': identity['labels']},
                    'State': {'Running': running, 'ExitCode': 0},
                }
            ]
        )

    def run(args, **kwargs):
        nonlocal running
        calls.append(args)
        if args[1] == 'start' and timeout:
            raise subprocess.TimeoutExpired(args, 1)
        if args[1] == 'stop':
            running = False
        return SimpleNamespace(stdout=b'{}', stderr=b'', returncode=0)

    monkeypatch.setattr(module.subprocess, 'check_output', output)
    monkeypatch.setattr(module.subprocess, 'run', run)
    return calls


def test_launch_write_failure_still_cleans_exact_created_container(tmp_path, monkeypatch):
    module = runner_module()
    calls = fake_docker(monkeypatch, module)
    with pytest.raises(OSError):
        module.owned_container('image', ['-c', 'pass'], output=tmp_path / 'missing', label='probe')
    assert ['docker', 'stop', '--time', '5', 'exact-owned-id'] in calls
    assert ['docker', 'rm', 'exact-owned-id'] in calls


def test_probe_timeout_also_cleans_exact_container(tmp_path, monkeypatch):
    module = runner_module()
    calls = fake_docker(monkeypatch, module, timeout=True)
    with pytest.raises(subprocess.TimeoutExpired):
        module.owned_container('image', ['-c', 'pass'], output=tmp_path, label='probe', timeout=1)
    assert ['docker', 'stop', '--time', '5', 'exact-owned-id'] in calls
    assert ['docker', 'rm', 'exact-owned-id'] in calls
