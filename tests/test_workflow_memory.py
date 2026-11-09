"""流程记忆的来源、时间、版本和完整事件预算边界。"""

import copy

import pytest

from serving_lab.task_protocol import digest, canonical
from serving_lab.workflow import workflow_spec, assess_workflow
from serving_lab.workflow_memory import memory_entry, retrieve


def source(sequence=1, variant='preparation-v1'):
    workflow = workflow_spec(variant, 'heat')
    trace = [
        {
            'action': 'take apple 1 from countertop 1',
            'after': {'feedback': 'You pick up the apple 1 from the countertop 1.'},
        }
    ]
    for operation in workflow['operations']:
        tool = {'clean': 'sinkbasin', 'heat': 'microwave'}[operation]
        trace.append(
            {
                'action': f'{operation} apple 1 with {tool} 1',
                'after': {'feedback': f'You {operation} the apple 1 using the {tool} 1.'},
            }
        )
    trace.append(
        {
            'action': 'move apple 1 to countertop 1',
            'after': {'feedback': 'You move the apple 1 to the countertop 1.'},
        }
    )
    initial_feedback = 'Your task is to: put a hot apple on countertop.'
    previous = initial_feedback
    for row in trace:
        row['before'] = {'feedback': previous, 'admissible_commands': [row['action']]}
        previous = row['after']['feedback']
    task = {
        'task_id': 'source-' + str(sequence),
        'role': 'train',
        'sequence': sequence,
        'phase': 1 if variant == 'preparation-v1' else 3,
        'workflow': workflow,
    }
    record = {
        **task,
        'condition': 'workflow_teacher',
        'purpose': 'workflow_teacher_source',
        'origin': 'public_policy',
        'eligible_for_training': True,
        'source_contract_digest': 'c' * 64,
        'initial': {'feedback': initial_feedback, 'admissible_commands': []},
        'trace': trace,
        'result': assess_workflow(trace, workflow, True),
    }
    record['record_hash'] = digest(record)
    return record, task


def entry(sequence=1, variant='preparation-v1'):
    record, task = source(sequence, variant)
    return memory_entry(record, task, 'c' * 64, 'd' * 64)


def test_memory_rejects_nontraining_and_counterfactual_sources():
    for key, value in [
        ('role', 'dev'),
        ('condition', 'opposite_order'),
        ('purpose', 'actor_diagnostic'),
    ]:
        record, task = source()
        record[key] = value
        record['record_hash'] = digest({k: v for k, v in record.items() if k != 'record_hash'})
        with pytest.raises(ValueError):
            memory_entry(record, task, 'c' * 64, 'd' * 64)


def test_future_and_obsolete_entries_cannot_enter_current_prompt():
    entries = [entry(1), entry(10, 'preparation-v2'), entry(30, 'preparation-v2')]
    result = retrieve(
        entries,
        'put a hot apple on countertop',
        workflow_spec('preparation-v2', 'heat'),
        cutoff=20,
        tokenize=lambda text: list(text),
        token_budget=5000,
    )
    assert [row['source_task_id'] for row in result['selected']] == ['source-10']
    assert all(item['workflow']['revision'] == 2 for item in result['prompt'])


def test_exact_cutoff_and_corrupt_source_are_rejected():
    candidate = entry(10)
    result = retrieve(
        [candidate],
        'apple',
        workflow_spec('preparation-v1', 'heat'),
        cutoff=10,
        tokenize=lambda text: list(text),
        token_budget=5000,
    )
    assert result['prompt'] == []
    corrupted = copy.deepcopy(candidate)
    corrupted['sequence'] = 0
    with pytest.raises(ValueError):
        retrieve(
            [corrupted],
            'apple',
            workflow_spec('preparation-v1', 'heat'),
            cutoff=20,
            tokenize=lambda text: list(text),
            token_budget=5000,
        )


def test_memory_budget_keeps_whole_experience_or_reports_no_fit():
    candidate = entry()
    workflow = workflow_spec('preparation-v1', 'heat')
    full = retrieve(
        [candidate],
        'apple',
        workflow,
        cutoff=2,
        tokenize=lambda text: list(text),
        token_budget=5000,
    )
    exact = len(canonical(full['prompt']))
    fitting = retrieve(
        [candidate],
        'apple',
        workflow,
        cutoff=2,
        tokenize=lambda text: list(text),
        token_budget=exact,
    )
    assert fitting['prompt'] == full['prompt']
    missing = retrieve(
        [candidate],
        'apple',
        workflow,
        cutoff=2,
        tokenize=lambda text: list(text),
        token_budget=exact - 1,
    )
    assert missing['prompt'] == [] and missing['eligible_count'] == 1
    assert missing['budget_skipped'] == 1


def test_budget_cannot_understate_even_empty_serialization():
    with pytest.raises(ValueError):
        retrieve(
            [],
            'apple',
            workflow_spec('preparation-v1', 'heat'),
            cutoff=1,
            tokenize=lambda text: list(text),
            token_budget=1,
        )
