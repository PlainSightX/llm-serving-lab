import copy
import json
import subprocess
import sys

import pytest

from serving_lab.lab import Lab, LabError, check_output


def case():
    return {
        'request': {'candidate': 'new', 'reference': 'old', 'metric': 'mae'},
        'rule': {'version': 'v2', 'minimum': 0.75},
        'expected_times': ['t1', 't2', 't3', 't4'],
        'observations': [{'time': f't{i}', 'kw': v} for i, v in enumerate([100, 200, 300, 400], 1)],
        'runs': [
            {
                'id': 'new',
                'unit': 'MW',
                'rows': [
                    {'time': 't1', 'value': 0.11},
                    {'time': 't2', 'value': 0.18},
                    {'time': 't3', 'value': 0.33},
                ],
            },
            {
                'id': 'old',
                'unit': 'kW',
                'rows': [
                    {'time': 't1', 'value': 120},
                    {'time': 't2', 'value': 240},
                    {'time': 't3', 'value': 330},
                    {'time': 't4', 'value': 400},
                ],
            },
        ],
    }


def plan():
    return {
        'operation': 'compare',
        'candidate': 'new',
        'reference': 'old',
        'unit': 'kW',
        'support': 'intersection',
        'metric': 'mae',
        'gate': {'basis': 'expected_fraction', 'threshold': 0.75},
    }


def test_real_business_calculation_and_safety_steps():
    from serving_lab.business import execute_plan, oracle, verify_plan

    c = case()
    result = execute_plan(c, plan())
    assert result['candidate_error'] == pytest.approx(20)
    assert result['reference_error'] == pytest.approx(30)
    assert result['coverage'] == 0.75 and result['winner'] == 'new'
    assert result == oracle(c)
    assert verify_plan(c, plan())['passed']
    for key, value in [
        ('candidate', 'old'),
        ('unit', 'MW'),
        ('support', 'per_run'),
        ('metric', 'rmse'),
    ]:
        p = plan()
        p[key] = value
        assert not verify_plan(c, p)['passed']
    p = plan()
    p['gate'] = {'basis': 'min_count', 'threshold': 2}
    assert not verify_plan(c, p)['passed']


def test_coverage_refusal_and_obsolete_rule_are_separate():
    from serving_lab.business import execute_plan, verify_plan

    c = case()
    c['rule']['minimum'] = 0.8
    p = plan()
    p['gate']['threshold'] = 0.8
    p['operation'] = 'abstain'
    result = execute_plan(c, p)
    assert result['status'] == 'insufficient_coverage' and result['winner'] is None
    assert verify_plan(c, p)['passed']
    c['rule'] = {'version': 'v1', 'minimum': 2}
    p['gate'] = {'basis': 'min_count', 'threshold': 2}
    p['operation'] = 'compare'
    assert execute_plan(c, p)['status'] == 'compared'
    assert verify_plan(c, p)['passed']


def test_metric_reversal_empty_support_and_numeric_equivalence():
    from serving_lab.business import execute_plan, oracle, verify_plan

    c = case()
    c['rule'] = {'version': 'v1', 'minimum': 2}
    c['observations'] = [{'time': t, 'kw': 0} for t in c['expected_times']]
    c['runs'] = [
        {
            'id': 'new',
            'unit': 'kW',
            'rows': [{'time': t, 'value': v} for t, v in zip(c['expected_times'], [0, 0, 0, 40])],
        },
        {
            'id': 'old',
            'unit': 'kW',
            'rows': [{'time': t, 'value': 12} for t in c['expected_times']],
        },
    ]
    p = plan()
    p['gate'] = {'basis': 'min_count', 'threshold': 2.0}
    assert verify_plan(c, p)['passed']
    assert execute_plan(c, p)['winner'] == 'new'
    c['request']['metric'] = 'rmse'
    p['metric'] = 'rmse'
    assert execute_plan(c, p)['winner'] == 'old' and execute_plan(c, p) == oracle(c)
    c['runs'][0]['rows'] = c['runs'][0]['rows'][:1]
    c['runs'][1]['rows'] = c['runs'][1]['rows'][1:]
    assert oracle(c)['times'] == [] and execute_plan(c, p)['winner'] is None


def test_common_support_changes_winner():
    from serving_lab.business import execute_plan

    c = case()
    c['rule'] = {'version': 'v1', 'minimum': 2}
    c['observations'] = [{'time': t, 'kw': 0} for t in c['expected_times']]
    c['runs'] = [
        {
            'id': 'new',
            'unit': 'kW',
            'rows': [
                {'time': 't1', 'value': 10},
                {'time': 't2', 'value': 10},
                {'time': 't3', 'value': 100},
            ],
        },
        {
            'id': 'old',
            'unit': 'kW',
            'rows': [{'time': 't1', 'value': 15}, {'time': 't2', 'value': 15}],
        },
    ]
    p = plan()
    p['gate'] = {'basis': 'min_count', 'threshold': 2}
    assert execute_plan(c, p)['winner'] == 'new'  # 分别按各自样本算则new的MAE=40，错误地选old。


def test_business_task_public_binding_and_bad_data(tmp_path):
    from serving_lab.business import make_task

    t = make_task(case(), task_id='business1', sequence=1, family='f1', split='train')
    with Lab(tmp_path / 'lab') as lab:
        lab.import_tasks([t])
        public = lab.public_task(t['task_id'])
        assert 'evaluation' not in public and 'expected' not in public
        assert check_output(t, plan())['passed']
    bad = copy.deepcopy(t)
    bad['messages'][1]['content'] = 'different data'
    with Lab(tmp_path / 'bad') as lab:
        with pytest.raises(LabError):
            lab.import_tasks([bad])
    c = case()
    c['runs'][0]['rows'].append(c['runs'][0]['rows'][0])
    with pytest.raises(LabError):
        make_task(c, task_id='dup', sequence=2, family='f2', split='train')


def test_task_suite_splits_are_structural_and_protected():
    from serving_lab.business import build_suite, audit_suite

    suite = build_suite()
    audit = audit_suite(suite)
    assert audit['origin'] == 'synthetic_protocol_pilot'
    assert audit['cross_split_object_overlap'] == 0
    assert {t['split'] for t in suite} == {'train', 'dev', 'holdout'}
    assert {t['rule_version'] for t in suite} == {'wind_v1', 'wind_v2'}
    assert all(check_output(t, t['evaluation']['reference_plan'])['passed'] for t in suite)
    bad = copy.deepcopy(suite)
    h = next(t for t in bad if t['split'] == 'holdout')
    h['evaluation']['object_id'] = bad[0]['evaluation']['object_id']
    with pytest.raises(LabError):
        audit_suite(bad)


def test_current_rule_only_and_data_dependent_operation():
    from serving_lab.business import make_task, reference_plan, verify_plan

    c = case()
    c['rule'] = {'version': 'v1', 'minimum': 2}
    t = make_task(c, task_id='v1', sequence=1, family='v1', split='train')
    assert '规则v2' not in t['messages'][0]['content']
    assert '共同样本数除以' not in t['messages'][0]['content']
    p = reference_plan(c)
    assert p['operation'] == 'compare'
    c['runs'][0]['rows'] = []
    assert reference_plan(c)['operation'] == 'abstain'
    assert not verify_plan(c, p)['passed']


def test_optimized_python_cannot_skip_case_validation():
    c = case()
    c['runs'][0]['unit'] = 'UNSUPPORTED'
    code = 'import json,sys; from serving_lab.business import validate_case; validate_case(json.load(sys.stdin))'
    result = subprocess.run(
        [sys.executable, '-O', '-c', code], input=json.dumps(c), text=True, capture_output=True
    )
    assert result.returncode != 0 and 'LabError' in result.stderr
