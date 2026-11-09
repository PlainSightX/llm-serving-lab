"""Windows研究入口复用已有容器桥，挂载路径不依赖wslpath。"""

import importlib.util
from pathlib import Path


def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('workflow_windows_bridge', scripts / 'q2_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_windows_mount_path_uses_existing_absolute_path(tmp_path, monkeypatch):
    module = runner(monkeypatch)
    monkeypatch.setattr(module.sys, 'platform', 'win32')

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError('wslpath called')

    monkeypatch.setattr(module.subprocess, 'check_output', unexpected_conversion)
    assert module.windows(tmp_path) == str(tmp_path.resolve())


def test_wsl_mount_path_keeps_existing_conversion(tmp_path, monkeypatch):
    module = runner(monkeypatch)
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    calls = []

    def converted(command, **kwargs):
        calls.append(command)
        return 'D:\\project\n'

    monkeypatch.setattr(module.subprocess, 'check_output', converted)
    assert module.windows(tmp_path) == 'D:\\project'
    assert calls == [['wslpath', '-w', str(tmp_path.resolve())]]
