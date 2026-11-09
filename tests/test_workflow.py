"""流程评价核对成功反馈、物品身份和操作顺序，物理目标单独保留。"""

import pytest

from serving_lab.workflow import workflow_spec, assess_workflow, observed_operations


def trace(actions):
    rows = []
    for action, feedback in actions:
        rows.append(
            {
                'action': action,
                'admissible': True,
                'before': {'feedback': '', 'admissible_commands': [action]},
                'after': {'feedback': feedback, 'admissible_commands': []},
            }
        )
    return rows


def target_trace(operations):
    actions = [('take apple 1 from countertop 1', 'You pick up the apple 1 from the countertop 1.')]
    tools = {'clean': 'sinkbasin', 'heat': 'microwave', 'cool': 'fridge'}
    for operation in operations:
        if operation == 'inventory':
            actions.append(('inventory', 'You are carrying: an apple 1.'))
        else:
            tool = tools[operation]
            actions.append(
                (
                    f'{operation} apple 1 with {tool} 1',
                    f'You {operation} the apple 1 using the {tool} 1.',
                )
            )
    actions.append(('move apple 1 to countertop 1', 'You move the apple 1 to the countertop 1.'))
    return trace(actions)


def test_physical_goal_does_not_override_missing_workflow():
    result = assess_workflow(target_trace(['heat']), workflow_spec('preparation-v1', 'heat'), True)
    assert result['base_success'] and not result['success']


def test_rule_reversal_distinguishes_same_physical_goal():
    earlier = target_trace(['clean', 'heat'])
    assert assess_workflow(earlier, workflow_spec('preparation-v1', 'heat'), True)['success']
    changed = assess_workflow(earlier, workflow_spec('preparation-v2', 'heat'), True)
    assert not changed['success'] and changed['obsolete_order_observed']
    assert assess_workflow(
        target_trace(['heat', 'clean']), workflow_spec('preparation-v2', 'heat'), True
    )['success']


def test_failed_and_different_object_operations_cannot_satisfy_workflow():
    rows = target_trace(['heat'])
    rows[1:1] = trace(
        [
            ('clean apple 2 with sinkbasin 1', 'You clean the apple 2 using the sinkbasin 1.'),
            ('clean apple 1 with sinkbasin 1', 'Nothing happens.'),
        ]
    )
    assert observed_operations(rows, 'apple 1') == ['heat']
    assert not assess_workflow(rows, workflow_spec('preparation-v1', 'heat'), True)['success']


def test_inventory_requires_public_evidence_for_held_target():
    spec = workflow_spec('inspection-v1', 'cool')
    assert assess_workflow(target_trace(['cool', 'inventory']), spec, True)['success']
    rows = target_trace(['cool', 'inventory'])
    rows[2]['after']['feedback'] = 'You are carrying: a tomato 1.'
    assert not assess_workflow(rows, spec, True)['success']


def test_success_requires_terminal_physical_goal_and_exact_processing_order():
    spec = workflow_spec('preparation-v1', 'heat')
    assert not assess_workflow(target_trace(['clean', 'heat']), spec, False)['success']
    assert not assess_workflow(target_trace(['heat', 'clean', 'heat']), spec, True)['success']
    assert assess_workflow(target_trace(['inventory', 'clean', 'heat']), spec, True)['success']


def test_unknown_workflow_fails_before_environment_use():
    with pytest.raises(ValueError):
        workflow_spec('unknown', 'heat')
    with pytest.raises(ValueError):
        workflow_spec('preparation-v1', 'arbitrary-command')


def test_intermediate_placement_preserves_same_object_processing():
    rows = target_trace(['clean', 'heat'])
    rows[2:2] = trace(
        [
            ('move apple 1 to shelf 1', 'You move the apple 1 to the shelf 1.'),
            ('take apple 1 from shelf 1', 'You pick up the apple 1 from the shelf 1.'),
        ]
    )
    assert assess_workflow(rows, workflow_spec('preparation-v1', 'heat'), True)['success']
