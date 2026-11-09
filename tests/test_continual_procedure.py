"""公告与查询边界不能把新规则直接泄露到无记忆查询中。"""

import json
import pytest
from serving_lab.workflow import workflow_spec


def test_query_does_not_repeat_complete_rule_but_explicit_control_does():
    from serving_lab.continual_procedure import ProcedureActor

    workflow = workflow_spec('preparation-v2', 'heat')
    view = {
        'feedback': 'Your task is to: put a hot apple in countertop.',
        'admissible_commands': ['look', 'inventory'],
    }
    tokenizer = lambda messages: list(range(len(json.dumps(messages))))
    plain = ProcedureActor(workflow).prepare(view, view, [], tokenizer, max_input=10000)
    explicit = ProcedureActor(workflow, explicit=True).prepare(
        view, view, [], tokenizer, max_input=10000
    )
    document = json.loads(plain['messages'][1]['content'])
    assert document['PROCEDURE'] == {'name': 'preparation', 'revision': 2, 'thermal': 'heat'}
    assert workflow['instruction'] not in str(plain['messages'])
    assert 'operations' not in document['PROCEDURE']
    assert workflow['instruction'] in str(explicit['messages'])


def test_memory_rejects_future_stage_or_an_unrelated_revision():
    from serving_lab.continual_procedure import validate_memory_snapshot

    workflow = workflow_spec('preparation-v2', 'heat')
    good = {
        'stage': 3,
        'workflow': workflow,
        'source_role': 'train',
        'entry_digests': ['a'],
        'prompt': [{'rule': workflow}],
    }
    assert validate_memory_snapshot(good, workflow, 3) == good
    with pytest.raises(ValueError):
        validate_memory_snapshot(good, workflow, 2)
    with pytest.raises(ValueError):
        validate_memory_snapshot({**good, 'source_role': 'future'}, workflow, 3)
    with pytest.raises(ValueError):
        validate_memory_snapshot(good, workflow_spec('preparation-v1', 'heat'), 3)


def test_qualification_cannot_repeat_easy_sources_under_different_labels():
    from serving_lab.continual_procedure import qualification_cases, validate_qualification_cases

    tasks = {
        str(i): {
            'task_id': str(i),
            'role': 'dev',
            'phase': i // 6 + 1,
            'base_id': str(i % 6),
            'workflow': {'thermal': 'heat' if i % 2 else 'cool'},
        }
        for i in range(18)
    }
    cases = qualification_cases(tasks)
    assert len(validate_qualification_cases(cases, tasks)) == 36
    import copy

    changed = copy.deepcopy(cases)
    changed[2]['source_task_id'] = changed[0]['source_task_id']
    with pytest.raises(ValueError):
        validate_qualification_cases(changed, tasks)
    changed = copy.deepcopy(cases)
    changed[0]['phase'] = 3
    with pytest.raises(ValueError):
        validate_qualification_cases(changed, tasks)


def test_actor_whitelist_rejects_a_self_consistent_different_candidate():
    from serving_lab.continual_procedure import ACTOR_IDENTITIES, validate_actor_binding
    from serving_lab.task_protocol import digest

    artifact, parameters = ACTOR_IDENTITIES['C0-Qwen3-1.7B']
    profile = {'name': 'fixed'}
    binding = {'artifact_digest': artifact, 'parameter_digest': parameters, 'profile': profile}
    manifest = {**binding, 'profile_digest': digest(profile)}
    validate_actor_binding('C0-Qwen3-1.7B', manifest, binding)
    with pytest.raises(ValueError):
        validate_actor_binding(
            'C0-Qwen3-1.7B',
            {**manifest, 'artifact_digest': 'other'},
            {**binding, 'artifact_digest': 'other'},
        )
