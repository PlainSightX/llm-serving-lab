"""组内参数可跨场景重复，单场景配额保持实例与参数多样性。"""

import importlib.util
from pathlib import Path


def test_scene_strata_allow_same_parameter_across_different_layouts(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('workflow_data_test', scripts / 'q2_workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {'key': 'a-x', 'scene': 1, 'parameter_digest': 'x'},
        {'key': 'b-x', 'scene': 2, 'parameter_digest': 'x'},
        {'key': 'a-y', 'scene': 1, 'parameter_digest': 'y'},
    ]
    selected = module.select_scene_strata(rows, {1, 2}, 1)
    assert {row['key'] for row in selected} == {'a-x', 'b-x'}
