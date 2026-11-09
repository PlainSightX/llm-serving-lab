"""N1的阶段边界、公开状态与监督因果关系。"""

import copy

import pytest

from serving_lab.task_protocol import canonical, digest
from test_workflow_memory import entry, source


def test_script_and_combined_keep_source_stage_and_whole_budget():
    from serving_lab.workflow_study import retrieve_procedural

    old, future = entry(1), entry(30, 'preparation-v2')
    result = retrieve_procedural(
        [old, future],
        'apple',
        old['workflow'],
        cutoff=20,
        tokenize=list,
        representation='script',
        token_budget=5000,
    )
    assert [r['source_task_id'] for r in result['selected']] == ['source-1']
    assert result['prompt'][0]['builder'] == 'deterministic_public_rule_projection_v1'
    assert 'apple 1' not in canonical(result['prompt'])
    assert result['prompt'][0]['operations'] == ['clean', 'heat']
    exact = len(canonical(result['prompt']))
    empty = retrieve_procedural(
        [old],
        'apple',
        old['workflow'],
        cutoff=20,
        tokenize=list,
        representation='script',
        token_budget=exact - 1,
    )
    assert empty['prompt'] == []
    combined = retrieve_procedural(
        [old],
        'apple',
        old['workflow'],
        cutoff=20,
        tokenize=list,
        representation='combined',
        token_budget=5000,
    )
    assert combined['prompt'][0]['trajectory'] == old['public']['trajectory']
    with pytest.raises(ValueError):
        retrieve_procedural(
            [old], 'apple', old['workflow'], cutoff=20, tokenize=list, representation='unknown'
        )


def test_decision_state_excludes_the_answer_and_future_feedback():
    from serving_lab.workflow_study import decision_state

    record, task = source()
    record['role'] = 'dev'
    record['eligible_for_training'] = False
    for row in record['trace']:
        row['after']['admissible_commands'] = ['look', 'inventory']
    record['record_hash'] = digest({k: v for k, v in record.items() if k != 'record_hash'})
    state = decision_state(record)
    assert state['expected_action'] == 'clean apple 1 with sinkbasin 1'
    assert [r['action'] for r in state['public']['trace']] == ['take apple 1 from countertop 1']
    assert 'You clean' not in canonical(state['public'])
    assert 'expected_action' not in canonical(state['public'])
    broken = copy.deepcopy(record)
    broken['trace'][1]['action'] = 'look'
    with pytest.raises(ValueError):
        decision_state(broken)


def test_stage_one_supervision_never_uses_dev_or_future_and_masks_future_feedback():
    from serving_lab.workflow_study import workflow_samples

    record, task = source()
    for row in record['trace']:
        row['after']['admissible_commands'] = ['look', 'inventory']
    record['record_hash'] = digest({k: v for k, v in record.items() if k != 'record_hash'})
    rows = workflow_samples(
        record, task, 'c' * 64, 'd' * 64, lambda ms: list(canonical(ms)), max_input=10000
    )
    clean = next(r for r in rows if r['messages'][-1]['content'].startswith('clean '))
    assert 'You clean' not in canonical(clean['messages'][:-1])
    assert clean['cohort'] == 'workflow_stage1'
    assert len(rows) == 4
    for field, value in [('role', 'dev'), ('phase', 2)]:
        bad_record, bad_task = copy.deepcopy(record), copy.deepcopy(task)
        bad_record[field] = value
        bad_task[field] = value
        bad_record['record_hash'] = digest(
            {k: v for k, v in bad_record.items() if k != 'record_hash'}
        )
        with pytest.raises(ValueError):
            workflow_samples(
                bad_record,
                bad_task,
                'c' * 64,
                'd' * 64,
                lambda ms: list(canonical(ms)),
                max_input=10000,
            )


def test_replay_mix_is_deterministic_and_bounded_by_one_sample():
    from serving_lab.workflow_study import replay_indices

    counts = [4, 7, 8]
    result = replay_indices(counts, target_tokens=50, seed=5)
    assert result == replay_indices(counts, target_tokens=50, seed=5)
    assert 50 <= sum(counts[i] for i in result) < 58
    assert all(0 <= i < 3 for i in result)
    with pytest.raises(ValueError):
        replay_indices([0], 50, 5)


def test_c1_parent_binding_rejects_same_tokenizer_wrong_weights():
    from serving_lab.workflow_training import validate_parent

    base = {'files': {'config.json': 'a', 'model.safetensors': 'b'}}
    binding = {
        'base_files': base['files'],
        'base_artifact_digest': digest(base['files']),
        'dataset_manifest_sha256': 'c' * 64,
    }
    validate_parent(binding, base, 'c' * 64, expected_parent=digest(base['files']))
    wrong = {'files': {'config.json': 'a', 'model.safetensors': 'different'}}
    with pytest.raises(ValueError):
        validate_parent(binding, wrong, 'c' * 64, expected_parent=digest(base['files']))
    with pytest.raises(ValueError):
        validate_parent(binding, base, 'd' * 64, expected_parent=digest(base['files']))


def test_cloud_source_mapping_preserves_repo_boundary(tmp_path):
    from serving_lab.workflow_training import source_path

    path = source_path('D:\\work\\llm-serving-lab\\artifacts\\source\\manifest.json', root=tmp_path)
    assert path == tmp_path / 'artifacts/source/manifest.json'
    with pytest.raises(ValueError):
        source_path('D:\\elsewhere\\secret', root=tmp_path)
    with pytest.raises(ValueError):
        source_path('D:\\work\\llm-serving-lab\\..\\secret', root=tmp_path)
