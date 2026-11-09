import json

import pytest

from serving_lab.lab import Lab, LabError, canonical
from serving_lab.execution import begin_execution, verify_execution
from test_execution import base_dir, complete_contract_fixture
from serving_lab.business import build_suite


def test_memory_uses_only_past_verified_training_with_current_rules(tmp_path):
    from serving_lab.memory import build_context

    data = build_suite()
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks(data)
        lab.register_base('base', base_dir(tmp_path))
        for t in data[:6]:
            begin_execution(lab, 'r_' + t['task_id'], t['task_id'], 'base')
            complete_contract_fixture(lab, 'r_' + t['task_id'], t['evaluation']['reference_plan'])
            lab.feedback('r_' + t['task_id'])
        ctx = build_context(lab, data[6]['task_id'], 'episodic_v1')
        assert ctx['entries'] and len(ctx['entries']) <= 3
        assert all(e['task_id'] in {t['task_id'] for t in data[:4]} for e in ctx['entries'])
        assert all('expected' not in e and 'evaluation' not in e for e in ctx['entries'])
        assert not build_context(lab, data[10]['task_id'], 'episodic_v1')['entries']
        assert build_context(lab, data[6]['task_id'], 'none')['entries'] == []


def test_request_binds_memory_snapshot_and_rejects_switch_or_tamper(tmp_path):
    from serving_lab.memory import input_messages

    data = build_suite()[:2]
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks(data)
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'r1', data[0]['task_id'], 'base', memory_policy='episodic_v1')
        complete_contract_fixture(lab, 'r1', data[0]['evaluation']['reference_plan'])
        lab.feedback('r1')
        with pytest.raises(LabError):
            begin_execution(lab, 'wrong', data[1]['task_id'], 'base', memory_policy='none')
        req = begin_execution(lab, 'r2', data[1]['task_id'], 'base', memory_policy='episodic_v1')
        assert req['context']['entries'][0]['attempt_id'] == 'r1'
        assert input_messages(req) != req['task']['messages']
        complete_contract_fixture(lab, 'r2', data[1]['evaluation']['reference_plan'])
        assert verify_execution(lab, 'r2')['complete']
        verdict = json.loads(
            lab.db.execute('SELECT body FROM feedback WHERE attempt_id=?', ('r1',)).fetchone()[0]
        )
        verdict['available_sequence'] = 99
        with lab.db:
            lab.db.execute(
                'UPDATE feedback SET body=? WHERE attempt_id=?', (canonical(verdict), 'r1')
            )
        with pytest.raises(LabError):
            verify_execution(lab, 'r2')


def test_manual_success_is_not_actual_model_memory(tmp_path):
    from serving_lab.memory import build_context

    data = build_suite()[:2]
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks(data)
        lab.record('manual', data[0]['task_id'], 'manual', data[0]['evaluation']['reference_plan'])
        lab.feedback('manual')
        assert build_context(lab, data[1]['task_id'], 'episodic_v1')['entries'] == []


def test_resume_repairs_only_missing_training_feedback_without_reexecution(tmp_path):
    from serving_lab.execution import resume_existing, DEFAULT_GENERATION

    data = build_suite()[:1]
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks(data)
        lab.register_base('base', base_dir(tmp_path))
        begin_execution(lab, 'r1', data[0]['task_id'], 'base')
        complete_contract_fixture(lab, 'r1', data[0]['evaluation']['reference_plan'])
        assert lab.db.execute('SELECT count(*) FROM feedback').fetchone()[0] == 0
        result = resume_existing(lab, data[0]['task_id'], 'base', DEFAULT_GENERATION, 'frozen')
        assert result['reused'] and result['new_model_call'] is False
        assert lab.db.execute('SELECT count(*) FROM feedback').fetchone()[0] == 1
        resume_existing(lab, data[0]['task_id'], 'base', DEFAULT_GENERATION, 'frozen')
        assert lab.db.execute('SELECT count(*) FROM feedback').fetchone()[0] == 1
