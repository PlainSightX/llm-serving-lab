"""审计覆盖不能靠缺失记录或重复记录制造完整比较。"""

import importlib.util
from pathlib import Path
import pytest


def auditor_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/n1_audit.py'
    spec = importlib.util.spec_from_file_location('n1_audit_guard_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raw_response_from_another_request_is_rejected():
    module = auditor_module()
    module.check_response_identity({'id': 'chatcmpl-a'}, 'a')
    with pytest.raises(ValueError):
        module.check_response_identity({'id': 'chatcmpl-b'}, 'a')


def test_wrong_environment_or_continuing_past_stagnation_is_rejected():
    import copy

    module = auditor_module()
    view = {'feedback': 'same', 'admissible_commands': ['look']}
    rows = [
        {
            'step': i,
            'action': 'look',
            'before': view,
            'after': view,
            'format_valid': True,
            'admissible': True,
        }
        for i in range(1, 5)
    ]
    env = {
        'trace': rows,
        'result': {
            'steps': 4,
            'max_steps': 100,
            'success': False,
            'termination': 'stagnation_terminated',
            'format_valid_rate': 1.0,
            'admissible_rate': 1.0,
        },
    }
    launch = {'image': 'fixed', 'command': ['fixed', '/runner.py']}
    module.check_environment_contract(env, launch, 'fixed')
    with pytest.raises(ValueError):
        module.check_environment_contract(env, {**launch, 'image': 'different'}, 'fixed')
    bad = copy.deepcopy(env)
    bad['trace'].append({**rows[-1], 'step': 5})
    bad['result']['steps'] = 5
    with pytest.raises(ValueError):
        module.check_environment_contract(bad, launch, 'fixed')
    bad = copy.deepcopy(env)
    bad['result']['max_steps'] = 101
    with pytest.raises(ValueError):
        module.check_environment_contract(bad, launch, 'fixed')


def test_coverage_rejects_missing_and_duplicate_cases():
    path = Path(__file__).resolve().parents[1] / 'scripts/n1_audit.py'
    assert path.is_file()
    spec = importlib.util.spec_from_file_location('n1_audit_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = [{'task_id': 'a', 'condition': 'script'}, {'task_id': 'b', 'condition': 'script'}]
    module.check_coverage(expected, [('script', 'a'), ('script', 'b')])
    with pytest.raises(ValueError):
        module.check_coverage(expected, [('script', 'a')])
    with pytest.raises(ValueError):
        module.check_coverage(expected, [('script', 'a'), ('script', 'a'), ('script', 'b')])


def test_release_import_rejects_same_ids_with_a_different_application_dataset():
    path = Path(__file__).resolve().parents[1] / 'scripts/import_n1_release.py'
    spec = importlib.util.spec_from_file_location('n1_import_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = {
        'source_verification': {'data_manifest_sha256': 'original'},
        'policy': {'actor_sources': {'actor.py': 'bytes'}},
    }
    initial = {
        'mode': 'application',
        'data_sha256': 'changed',
        'source_hashes': {'src/serving_lab/actor.py': 'bytes'},
        'environment_contract_recorded': True,
    }
    with pytest.raises(ValueError):
        module.verify_application_contract(parent, initial)
    initial['data_sha256'] = 'original'
    module.verify_application_contract(parent, initial)
    initial['environment_contract_recorded'] = False
    with pytest.raises(ValueError):
        module.verify_application_contract(parent, initial)
