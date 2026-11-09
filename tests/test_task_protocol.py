import copy

import pytest


def reference_record():
    from serving_lab.task_protocol import Episode

    class Env:
        def reset(self):
            return {'feedback': 'start', 'admissible_commands': ['finish'], 'won': False}

        def step(self, action):
            return {'feedback': 'end', 'admissible_commands': [], 'won': True}, 1, True

        def close(self):
            pass

    episode = Episode(
        {'task_id': 's', 'role': 'support', 'family': 'f', 'sequence': 1},
        Env(),
        max_steps=2,
        origin='privileged_reference',
        reference_method='planner',
    )
    episode.reset()
    episode.step('finish')
    record = episode.record()
    episode.close()
    return record


def test_public_view_is_a_whitelist_and_cannot_mutate_private_state():
    from serving_lab.task_protocol import public_view

    state = {
        'feedback': 'room',
        'admissible_commands': ['look'],
        'won': True,
        'facts': ['SECRET'],
        '_facts': ['SECRET'],
        'extra.expert_plan': ['SECRET'],
        'policy_commands': ['SECRET'],
        'extra.gamefile': 'SECRET',
    }
    view = public_view(state)
    assert view == {'feedback': 'room', 'admissible_commands': ['look']}
    view['admissible_commands'].append('changed')
    assert state['admissible_commands'] == ['look']


def test_episode_requires_execution_before_feedback_and_rejects_post_terminal():
    from serving_lab.task_protocol import Episode

    class Env:
        def reset(self):
            return {'feedback': 'start', 'admissible_commands': ['finish'], 'won': False}

        def step(self, action):
            return {'feedback': 'end', 'admissible_commands': [], 'won': True}, 1, True

        def close(self):
            pass

    episode = Episode(
        {'task_id': 's', 'role': 'support', 'family': 'f', 'sequence': 1},
        Env(),
        max_steps=2,
        origin='scripted_control',
    )
    with pytest.raises(ValueError):
        episode.result()
    assert 'won' not in episode.reset()
    with pytest.raises(ValueError):
        episode.result()
    assert 'won' not in episode.step('finish')
    assert episode.result()['success'] is True
    with pytest.raises(ValueError):
        episode.step('finish')
    episode.close()


def test_constant_legal_actions_are_not_task_success_and_protected_cannot_run():
    from serving_lab.task_protocol import Episode

    class Env:
        def reset(self):
            return {'feedback': 'start', 'admissible_commands': ['look'], 'won': False}

        def step(self, action):
            return self.reset(), 0, False

        def close(self):
            pass

    task = {'task_id': 's', 'role': 'support', 'family': 'f', 'sequence': 1}
    episode = Episode(task, Env(), max_steps=2, origin='constant_look')
    episode.reset()
    episode.step('look')
    episode.step('look')
    result = episode.result()
    assert result['format_valid_rate'] == 1 and result['admissible_rate'] == 1
    assert result['success'] is False and result['termination'] == 'step_budget'
    with pytest.raises(ValueError):
        Episode({**task, 'role': 'protected'}, Env(), max_steps=2, origin='reference')


def test_common_history_rejects_future_protected_failed_and_false_model_attribution():
    from serving_lab.task_protocol import history_material

    record = reference_record()
    unsealed = copy.deepcopy(record)
    unsealed.pop('record_hash')

    with pytest.raises(ValueError):
        history_material(unsealed, cutoff=2)
    material = history_material(record, cutoff=2)
    assert material['origin'] == 'privileged_reference' and 'model_generated' not in material
    assert material['eligible_for_parameter_training'] is False
    assert material['eligible_for_primary_comparison_history'] is False
    assert material['history_role'] == 'oracle_diagnostic_only'
    with pytest.raises(ValueError):
        history_material(record, cutoff=2, purpose='parameter_training')
    future = copy.deepcopy(record)
    future['feedback_available_sequence'] = 3
    protected = copy.deepcopy(record)
    protected['task']['role'] = 'protected'
    failed = copy.deepcopy(record)
    failed['result']['success'] = False
    for bad in [future, protected, failed]:
        with pytest.raises(ValueError):
            history_material(bad, cutoff=2)


def test_invalid_format_consumes_budget_without_environment_execution():
    from serving_lab.task_protocol import Episode

    class Env:
        calls = 0

        def reset(self):
            return {'feedback': 'start', 'admissible_commands': ['look'], 'won': False}

        def step(self, action):
            self.calls += 1
            return self.reset(), 0, False

        def close(self):
            pass

    env = Env()
    episode = Episode(
        {'task_id': 's', 'role': 'target', 'family': 'f', 'sequence': 2},
        env,
        max_steps=1,
        origin='scripted_control',
    )
    episode.reset()
    episode.step('look\nfinish')
    assert env.calls == 0 and episode.result()['steps'] == 1
    assert episode.result()['format_valid_rate'] == 0 and episode.result()['success'] is False


def test_archive_member_cannot_escape_target(tmp_path):
    from serving_lab.task_protocol import checked_target

    for name in ['../escape', '/absolute', 'a/../../escape', 'C:/outside', 'a\\..\\..\\escape']:
        with pytest.raises(ValueError):
            checked_target(tmp_path, name)
    assert checked_target(tmp_path, 'train/task/file.json') == tmp_path / 'train/task/file.json'


def test_context_only_includes_public_observation_and_diagnostic_history():
    from serving_lab.task_protocol import build_context, history_material

    current = {
        'feedback': 'target room',
        'admissible_commands': ['look'],
        'facts': 'PRIVATE',
        'won': True,
    }
    history = history_material(reference_record(), cutoff=2)
    context = build_context(current, history, condition='oracle_related', cutoff=2)
    import json

    assert 'PRIVATE' not in json.dumps(context['messages'])
    assert context['metadata']['model_trial_ready'] is False
    assert context['metadata']['source_ids'] == ['s']
    assert build_context(current, None, condition='none', cutoff=2)['metadata']['source_ids'] == []
    with pytest.raises(ValueError):
        build_context(
            current, history, condition='oracle_related', cutoff=2, purpose='primary_comparison'
        )


def test_context_rejects_unsealed_forged_and_wrong_cutoff_material():
    from serving_lab.task_protocol import build_context, history_material

    current = {'feedback': 'now', 'admissible_commands': ['look']}
    forged = {
        'source_task_id': 'protected_future',
        'origin': 'model_generated',
        'initial_observation': 'LEAK',
        'trajectory': [],
    }
    with pytest.raises(ValueError):
        build_context(current, forged, condition='oracle_related', cutoff=2)
    good = history_material(reference_record(), cutoff=3)
    with pytest.raises(ValueError):
        build_context(current, good, condition='oracle_related', cutoff=2)
    good['trajectory'][0]['action'] = 'changed'
    with pytest.raises(ValueError):
        build_context(current, good, condition='oracle_related', cutoff=3)


def test_target_record_hash_and_frozen_identity_are_verified():
    from serving_lab.task_protocol import validate_record, digest

    record = reference_record()
    expected = copy.deepcopy(record['task'])
    validate_record(record, expected)
    changed = copy.deepcopy(record)
    changed['initial']['feedback'] = 'rewritten'
    with pytest.raises(ValueError):
        validate_record(changed, expected)
    changed = copy.deepcopy(record)
    changed['task']['sequence'] = 99
    changed['record_hash'] = digest({k: v for k, v in changed.items() if k != 'record_hash'})
    with pytest.raises(ValueError):
        validate_record(changed, expected)


def test_context_rejects_unsuccessful_material_even_with_valid_digest():
    from serving_lab.task_protocol import build_context, history_material, digest

    history = history_material(reference_record(), cutoff=2)
    history['verified_success'] = False
    history['material_hash'] = digest({k: v for k, v in history.items() if k != 'material_hash'})
    with pytest.raises(ValueError):
        build_context(
            {'feedback': 'now', 'admissible_commands': ['look']},
            history,
            condition='oracle_related',
            cutoff=2,
        )


def test_manifest_rejects_empty_and_protected_only_before_execution():
    from serving_lab.task_protocol import validate_manifest

    for tasks in ([], [{'role': 'protected', 'task_id': 'p', 'sequence': 1}]):
        with pytest.raises(ValueError):
            validate_manifest(
                {'name': 'q1-alfworld-text-qualification-v1', 'tasks': tasks, 'max_steps': 100}
            )
