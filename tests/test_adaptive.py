import json

import pytest


def task(i, split='train', seq=1):
    return {
        'task_id': i,
        'sequence': seq,
        'family': i,
        'split': split,
        'rule_version': 'v1',
        'messages': [{'role': 'user', 'content': f'任务{i}：返回结构化操作计划'}],
        'expected': {'actions': ['check_units', 'compare']},
    }


def test_feedback_requires_attempt_and_no_holdout_export(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path) as lab:
        lab.import_tasks([task('a'), task('h', 'holdout', 2)])
        with pytest.raises(LabError):
            lab.feedback('missing')
        lab.record('a1', 'a', 'baseline', {'actions': ['check_units', 'compare']})
        lab.feedback('a1')
        lab.record('h1', 'h', 'baseline', {'actions': ['check_units', 'compare']})
        lab.feedback('h1')
        manifest = lab.export_training(tmp_path / 'export', cutoff=2)
        assert manifest['task_ids'] == ['a']
        rows = [json.loads(s) for s in (tmp_path / 'export/train.jsonl').read_text().splitlines()]
        assert 'expected' not in rows[0]


def test_future_failure_and_duplicate_excluded(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path) as lab:
        lab.import_tasks([task('a'), task('b', seq=3)])
        lab.record('bad', 'a', 'baseline', {'actions': ['compare']})
        lab.feedback('bad')
        lab.record('future', 'b', 'baseline', {'actions': ['check_units', 'compare']})
        lab.feedback('future')
        with pytest.raises(LabError):
            lab.record('bad', 'a', 'baseline', {})
        with pytest.raises(LabError):
            lab.export_training(tmp_path / 'empty', cutoff=1)
        with pytest.raises(LabError):
            lab.import_tasks([dict(task('a'), expected={})])


def test_only_one_writer(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path):
        with pytest.raises(LabError):
            Lab(tmp_path)


def test_delayed_feedback_cannot_travel_back_in_time(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path) as lab:
        lab.import_tasks([task('a'), task('b', seq=3)])
        lab.record('a1', 'a', 'baseline', {'actions': ['check_units', 'compare']})
        lab.record('b1', 'b', 'baseline', {})
        lab.feedback('a1')
        with pytest.raises(LabError):
            lab.export_training(tmp_path / 'past', cutoff=1)


def test_cross_split_same_family_rejected(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path) as lab:
        lab.import_tasks([task('a')])
        with pytest.raises(LabError):
            lab.import_tasks([dict(task('b', 'holdout', 2), family='a')])


def test_future_task_exposure_and_arm_switch_rejected(tmp_path):
    from serving_lab.lab import Lab, LabError

    with Lab(tmp_path) as lab:
        lab.import_tasks([task('a'), task('b', seq=3)])
        with pytest.raises(LabError):
            lab.public_task('b')
        assert 'expected' not in lab.public_task('a')
        lab.record('a1', 'a', 'baseline', {}, arm='frozen')
        with pytest.raises(LabError):
            lab.record('a2', 'a', 'baseline', {}, arm='retry')
        with pytest.raises(LabError):
            lab.record('fake', 'a', 'baseline', {}, actor='http_model')
        with pytest.raises(LabError):
            lab.import_tasks([task('new', seq=4)])


def test_json_boolean_does_not_count_as_integer():
    from serving_lab.lab import check_output

    assert not check_output({'expected': {'value': 1}}, {'value': True})['passed']
