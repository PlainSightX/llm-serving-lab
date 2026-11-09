import pytest

from serving_lab.lab import Lab, LabError
from serving_lab.execution import begin_execution, DEFAULT_GENERATION
from test_execution import base_dir


def test_pilot_freezes_two_arms_and_never_imports_protected_panel(tmp_path):
    from serving_lab.study import prepare, verify_study, report

    folder = tmp_path / 'study'
    config = prepare(folder, base_dir(tmp_path))
    assert set(config['arms']) == {'frozen', 'memory'}
    assert config['stage'] == 'business_protocol_pilot'
    for arm in config['arms']:
        with Lab(folder / arm) as lab:
            assert (
                lab.db.execute("SELECT count(*) FROM tasks WHERE split='holdout'").fetchone()[0]
                == 0
            )
            assert lab.db.execute('SELECT count(*) FROM tasks').fetchone()[0] == 20
    assert verify_study(folder)['dataset']['split_counts']['holdout'] == 10
    summary = report(folder)
    assert all(
        v['completed_model_calls'] == 0 and v['reserved_attempts'] == 0 and v['passed'] == 0
        for v in summary['arms'].values()
    )
    assert summary['all_scheduled_completed'] is False
    with pytest.raises(LabError):
        prepare(folder, tmp_path / 'base')


def test_frozen_input_change_is_rejected(tmp_path):
    from serving_lab.study import prepare, verify_study

    folder = tmp_path / 'study'
    prepare(folder, base_dir(tmp_path))
    path = folder / 'pilot.jsonl'
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(LabError):
        verify_study(folder)


def test_study_rejects_bypassing_its_frozen_arm_contract(tmp_path):
    from serving_lab.study import prepare, report

    folder = tmp_path / 'study'
    config = prepare(folder, base_dir(tmp_path))
    with Lab(folder / 'memory') as lab:
        task = lab.db.execute('SELECT id FROM tasks ORDER BY seq').fetchone()[0]
        spec = {
            **DEFAULT_GENERATION,
            'max_new_tokens': config['max_new_tokens'],
            'max_input_tokens': config['max_input_tokens'],
        }
        with pytest.raises(LabError):
            begin_execution(
                lab, 'wrong', task, 'base', generation=spec, arm='memory', memory_policy='none'
            )
        with pytest.raises(LabError):
            begin_execution(
                lab,
                'wrong',
                task,
                'base',
                generation=spec,
                arm='frozen',
                memory_policy='episodic_v1',
            )
        begin_execution(
            lab, 'valid', task, 'base', generation=spec, arm='memory', memory_policy='episodic_v1'
        )
        with lab.db:
            lab.db.execute("UPDATE attempts SET arm='frozen'")
    with pytest.raises(LabError):
        report(folder)
