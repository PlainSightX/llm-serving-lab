"""研究入口在接触模型前绑定真实参数，澄清试点使用固定元数据子集。"""

import importlib.util
import json
from pathlib import Path

import pytest

from serving_lab.workflow import WorkflowActor, workflow_spec


def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        'workflow_probe_test', scripts / 'q2_workflow_probe.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_same_tokenizer_different_parameters_rejected_before_client_creation(
    tmp_path, monkeypatch
):
    module = runner(monkeypatch)
    preparation, data, model = [tmp_path / name for name in ('preparation', 'data', 'model')]
    for path in (preparation, data, model):
        path.mkdir()
    for path in (
        data / 'manifest.json',
        preparation / 'memory.json',
        preparation / 'retrieval.json',
    ):
        path.write_text('{}')
    token = tmp_path / 'token'
    token.write_text('test-only-token-value')
    plan = {
        'source_hashes': {},
        'data': str(data),
        'model': str(model),
        'tokenizer_files': {},
        'data_manifest_sha256': module.file_hash(data / 'manifest.json'),
        'memory_sha256': module.file_hash(preparation / 'memory.json'),
        'retrieval_sha256': module.file_hash(preparation / 'retrieval.json'),
        'reference_model': {'artifact_digest': 'a' * 64, 'parameter_digest': 'b' * 64},
    }
    (preparation / 'plan.json').write_text(json.dumps(plan))
    config = {
        'endpoint': 'https://localhost:1',
        'token_file': str(token),
        'tls': {},
        'manifest': {'artifact_digest': 'a' * 64, 'parameter_digest': 'c' * 64},
    }
    config_path = tmp_path / 'engine.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(module, 'validate_workflow_manifest', lambda data: {})
    monkeypatch.setattr(module, 'tokenizer_at', lambda model: None)

    def unexpected_client(*args, **kwargs):
        raise AssertionError('client created before model binding')

    monkeypatch.setattr(module, 'RemoteEngine', unexpected_client)
    with pytest.raises(ValueError, match='参数身份'):
        await module.run(preparation, config_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_workflow_actor_explicitly_links_new_fields_to_task_success():
    initial = {
        'feedback': 'Your task is to: put a hot apple on countertop.',
        'admissible_commands': ['look'],
    }
    prepared = WorkflowActor(workflow_spec('preparation-v1', 'heat')).prepare(
        initial, initial, [], lambda messages: list(json.dumps(messages)), max_input=20000
    )
    system = prepared['messages'][0]['content']
    assert 'CURRENT_WORKFLOW' in system and 'PAST_VERIFIED_EXPERIENCE' in system
    assert prepared['context']['actor'] == 'public_workflow_goal_v3'
    document = json.loads(prepared['messages'][1]['content'])
    assert workflow_spec('preparation-v1', 'heat')['instruction'] in document['GOAL']


def test_instruction_probe_selects_one_base_per_family_across_all_workflows(monkeypatch):
    module = runner(monkeypatch)
    tasks = [
        {
            'task_id': f'{phase}-{base}',
            'base_id': f'base-{base}',
            'role': 'dev',
            'phase': phase,
            'family': 'heat' if base < 3 else 'cool',
        }
        for phase in (1, 2, 3)
        for base in range(6)
    ]
    cases = module.probe_cases(tasks, 'instruction_check')
    indexed = {task['task_id']: task for task in tasks}
    assert len(cases) == 12 and len({case['task_id'] for case in cases}) == 6
    assert {indexed[case['task_id']]['base_id'] for case in cases} == {'base-0', 'base-3'}
    assert {case['condition'] for case in cases} == {'frozen_workflow', 'trajectory_memory'}
