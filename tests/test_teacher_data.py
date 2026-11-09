"""教师数据边界：场景隔离、来源身份及未来反馈不得混入监督输入。"""

import importlib
import importlib.util

import pytest

from serving_lab.task_protocol import FAMILIES, digest


def module():
    assert importlib.util.find_spec('serving_lab.teacher_data') is not None
    return importlib.import_module('serving_lab.teacher_data')


def candidates():
    return [
        {
            'key': f'train/{family}-obj-{scene}/trial',
            'family': family,
            'scene': scene,
            'parameter_digest': digest([family, scene]),
            'scene_partition': 'dev' if scene % 3 == 0 else 'train',
        }
        for family in FAMILIES
        for scene in range(1, 37)
    ]


def test_split_preserves_full_family_counts_and_rejects_insufficient_disjoint_data():
    m = module()
    selected = m.select_candidates(candidates(), seed=20261103)
    assert len(selected) == 72
    assert len([t for t in selected if t['role'] == 'train']) == 48
    assert {t['scene'] for t in selected if t['role'] == 'train'}.isdisjoint(
        {t['scene'] for t in selected if t['role'] == 'dev'}
    )
    assert {t['parameter_digest'] for t in selected if t['role'] == 'train'}.isdisjoint(
        {t['parameter_digest'] for t in selected if t['role'] == 'dev'}
    )
    assert m.select_candidates(list(reversed(candidates())), seed=20261103) == selected
    with pytest.raises(ValueError):
        m.select_candidates(candidates()[:12], seed=20261103)


def test_action_samples_exclude_future_feedback_dev_and_diagnostic_origins():
    m = module()
    initial = {
        'feedback': 'Your task is to: put a candle on toilet.',
        'admissible_commands': ['go to desk 1'],
    }
    after = {'feedback': 'FUTURE_ONLY', 'admissible_commands': ['look']}
    record = {
        'task_id': 'new_train',
        'role': 'train',
        'origin': 'public_policy',
        'purpose': 'teacher_coldstart_source',
        'source_contract_digest': 'a' * 64,
        'eligible_for_training': True,
        'eligible_for_primary_history': False,
        'initial': initial,
        'trace': [{'step': 1, 'before': initial, 'action': 'go to desk 1', 'after': after}],
        'result': {'success': True, 'steps': 1},
    }
    record['record_hash'] = digest(record)
    tokenize = lambda ms: list(range(10))
    samples = m.action_samples(record, tokenize, 'a' * 64)
    assert len(samples) == 1 and samples[0]['messages'][-1] == {
        'role': 'assistant',
        'content': 'go to desk 1',
    }
    assert 'FUTURE_ONLY' not in str(samples[0]['messages'])
    for change in (
        {'role': 'dev'},
        {'purpose': 'actor_diagnostic'},
        {'source_contract_digest': 'b' * 64},
        {'eligible_for_training': False},
    ):
        wrong = {**record, **change}
        wrong['record_hash'] = digest({k: v for k, v in wrong.items() if k != 'record_hash'})
        with pytest.raises(ValueError):
            m.action_samples(wrong, tokenize, 'a' * 64)


def test_supervision_does_not_learn_rejected_or_public_no_effect_action():
    m = module()
    initial = {
        'feedback': 'Your task is to: put a candle on toilet.',
        'admissible_commands': ['go to desk 1'],
    }
    record = {
        'task_id': 'new_train',
        'role': 'train',
        'origin': 'public_policy',
        'purpose': 'teacher_coldstart_source',
        'source_contract_digest': 'a' * 64,
        'eligible_for_training': True,
        'eligible_for_primary_history': False,
        'initial': initial,
        'trace': [
            {
                'step': 1,
                'before': initial,
                'action': 'go to desk 1',
                'after': {**initial, 'feedback': 'Nothing happens.'},
            }
        ],
        'result': {'success': True, 'steps': 1},
    }
    record['record_hash'] = digest(record)
    assert m.action_samples(record, lambda ms: [1], 'a' * 64) == []


async def test_old_task_cannot_be_relabelled_by_new_teacher_episode(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from q2_run import episode

    with pytest.raises(ValueError, match='公开教师'):
        await episode(
            {'role': 'support'},
            tmp_path,
            tmp_path / 'must_not_exist',
            'image',
            purpose='teacher_coldstart_source',
            source_contract_digest='a' * 64,
        )
    assert not (tmp_path / 'must_not_exist').exists()


def test_teacher_understands_public_look_at_and_hot_goal_variants():
    from serving_lab.public_tasks import PublicPolicy

    p = PublicPolicy()
    action = p.choose(
        {
            'feedback': 'Your task is to: look at pencil under the desklamp.',
            'admissible_commands': ['take pencil 1 from desk 1', 'go to shelf 1'],
        }
    )
    assert action == 'take pencil 1 from desk 1'
    assert (
        p.choose(
            {
                'feedback': 'You pick up the pencil 1 from the desk 1.',
                'admissible_commands': ['use desklamp 1', 'move pencil 1 to desk 1'],
            }
        )
        == 'use desklamp 1'
    )
    p = PublicPolicy()
    assert (
        p.choose(
            {
                'feedback': 'Your task is to: put a hot apple in garbagecan.',
                'admissible_commands': ['take apple 1 from countertop 1'],
            }
        )
        == 'take apple 1 from countertop 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You pick up the apple 1 from the countertop 1.',
                'admissible_commands': ['go to microwave 1', 'go to garbagecan 1'],
            }
        )
        == 'go to microwave 1'
    )


def test_teacher_returns_to_previously_seen_goal_lamp_after_finding_object():
    from serving_lab.public_tasks import PublicPolicy

    p = PublicPolicy()
    assert (
        p.choose(
            {
                'feedback': 'Your task is to: examine the laptop with the desklamp.',
                'admissible_commands': ['go to sidetable 1', 'go to sofa 1'],
            }
        )
        == 'go to sidetable 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You arrive at sidetable 1. On the sidetable 1, you see a desklamp 1.',
                'admissible_commands': ['use desklamp 1', 'go to sofa 1'],
            }
        )
        == 'go to sofa 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You arrive at sofa 1. On the sofa 1, you see a laptop 1.',
                'admissible_commands': ['take laptop 1 from sofa 1', 'go to sidetable 1'],
            }
        )
        == 'take laptop 1 from sofa 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You pick up the laptop 1 from the sofa 1.',
                'admissible_commands': ['go to sidetable 1'],
            }
        )
        == 'go to sidetable 1'
    )


def test_matching_tokenizer_and_self_consistent_receipt_do_not_qualify_wrong_base():
    m = module()
    assert hasattr(m, 'validate_base_binding')
    wrong_base = {
        'path': 'wrong-model',
        'files': {
            'tokenizer_config.json': 'same-tokenizer',
            'model.safetensors': 'different-weights',
        },
    }
    binding = {
        'base': wrong_base,
        'base_artifact_digest': digest(wrong_base['files']),
        'base_revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'dataset_manifest_sha256': 'a' * 64,
    }
    with pytest.raises(ValueError):
        m.validate_base_binding(binding, wrong_base, 'a' * 64)
