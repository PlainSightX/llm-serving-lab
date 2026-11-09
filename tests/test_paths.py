"""本机运行根与配置路径展开：公开仓库不写死个人目录。"""

from pathlib import Path

from serving_lab.paths import expand, lab_root


def test_lab_root_defaults_to_home_and_honours_override(monkeypatch, tmp_path):
    monkeypatch.delenv('SERVING_LAB_ROOT', raising=False)
    assert lab_root() == Path.home() / '.local/share/llm-serving-lab'
    monkeypatch.setenv('SERVING_LAB_ROOT', str(tmp_path))
    assert lab_root() == tmp_path


def test_expand_handles_tilde_and_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv('LAB_TEST_ROOT', str(tmp_path))
    assert expand('$LAB_TEST_ROOT/compiler') == tmp_path / 'compiler'
    assert expand('~/x') == Path.home() / 'x'
    assert expand('/absolute/path') == Path('/absolute/path')
