"""覆盖清单身份、重复回合及桥超时原件留存的具体回归。"""

import asyncio
import copy
import importlib.util
from pathlib import Path

import pytest

from serving_lab.task_protocol import digest
from serving_lab.workflow import (
    VARIANTS,
    BASE_COUNTS,
    THERMAL_FAMILIES,
    workflow_spec,
    validate_workflow_manifest,
    assess_workflow,
)


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        'workflow_integrity_runner', scripts / 'q2_workflow.py'
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def manifest():
    contract = {'origin': 'teacher_public_program', 'teacher_policy_sha256': '0' * 64}
    bases = []
    for role_index, (role, count) in enumerate(BASE_COUNTS.items()):
        for index in range(count):
            base = role + str(index)
            bases.append(
                {
                    'base_id': base,
                    'role': role,
                    'scene': [14, 16, 25][index % 3]
                    if role == 'future'
                    else 100 * (role_index + 1) + index,
                    'family': tuple(THERMAL_FAMILIES)[index % 2],
                    'source_split': 'train',
                    'source_key': 'train/' + base,
                    'parameter_digest': base,
                    'game_path': f'games/{base}/game.tw-pddl',
                    'files': {
                        f'games/{base}/game.tw-pddl': '0' * 64,
                        f'games/{base}/traj_data.json': '0' * 64,
                    },
                }
            )
    tasks = []
    for phase, variant in enumerate(VARIANTS, 1):
        for base in bases:
            tasks.append(
                {
                    **base,
                    'task_id': f'{phase}-{base["base_id"]}',
                    'sequence': len(tasks) + 1,
                    'phase': phase,
                    'variant': variant,
                    'workflow': workflow_spec(variant, THERMAL_FAMILIES[base['family']]),
                }
            )
    return {
        'name': 'q2-public-workflows-v1',
        'max_steps': 100,
        'source_contract': contract,
        'source_contract_digest': digest(contract),
        'excluded_source_keys': [],
        'tasks': tasks,
    }


@pytest.mark.parametrize('corruption', ['phase', 'duplicate_variant', 'source_alias'])
def test_manifest_rejects_phase_variant_and_source_aliases(corruption):
    data = manifest()
    assert len(validate_workflow_manifest(data)) == 126
    if corruption == 'phase':
        data['tasks'][42]['phase'] = 99
    elif corruption == 'duplicate_variant':
        replacement = copy.deepcopy(data['tasks'][2])
        replacement.update(task_id=data['tasks'][0]['task_id'], sequence=1)
        data['tasks'][0] = replacement
    else:
        for task in data['tasks']:
            if task['base_id'] == 'train1':
                task['source_key'] = 'train/train0'
    with pytest.raises(ValueError):
        validate_workflow_manifest(data)


def test_audit_rejects_repeated_successes_in_place_of_missing_tasks(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    data = manifest()
    task = data['tasks'][0]
    root = tmp_path / 'run'
    data_root = tmp_path / 'data'
    paths = [
        root / condition / ('copy-' + str(index)) / 'record.json'
        for condition, count in [
            ('workflow_teacher', 90),
            ('base_policy', 18),
            ('opposite_order', 12),
        ]
        for index in range(count)
    ]
    original_glob = Path.glob
    monkeypatch.setattr(
        Path,
        'glob',
        lambda path, pattern: iter(paths) if path == root else original_glob(path, pattern),
    )
    monkeypatch.setattr(runner, 'file_hash', lambda path: '0' * 64)
    monkeypatch.setattr(runner, 'verify_files', lambda task, root: None)

    def read(path):
        if path == data_root / 'manifest.json':
            return data
        if path.name == 'manifest-anchor.json':
            return {'sha256': '0' * 64}
        if path == root / 'input.json':
            return {
                'manifest_sha256': '0' * 64,
                'source_hashes': {'src/serving_lab/workflow.py': '0' * 64},
                'image': runner.IMAGE,
                'planned_episodes': 120,
            }
        condition = path.parent.parent.name
        operations = {
            'workflow_teacher': ['clean', 'heat'],
            'base_policy': ['heat'],
            'opposite_order': ['heat', 'clean'],
        }[condition]
        actions = [
            ('take apple 1 from countertop 1', 'You pick up the apple 1 from the countertop 1.')
        ]
        for operation in operations:
            tool = {'clean': 'sinkbasin', 'heat': 'microwave'}[operation]
            actions.append(
                (
                    f'{operation} apple 1 with {tool} 1',
                    f'You {operation} the apple 1 using the {tool} 1.',
                )
            )
        actions.append(
            ('move apple 1 to countertop 1', 'You move the apple 1 to the countertop 1.')
        )
        trace = [
            {
                'step': i,
                'action': action,
                'before': {'feedback': '', 'admissible_commands': [action]},
                'after': {'feedback': feedback, 'admissible_commands': []},
                'receipt': None,
            }
            for i, (action, feedback) in enumerate(actions, 1)
        ]
        result = {**assess_workflow(trace, task['workflow'], True), 'steps': len(trace)}
        record = {
            **{key: task[key] for key in ('task_id', 'role', 'phase', 'sequence', 'workflow')},
            'condition': condition,
            'source_contract_digest': data['source_contract_digest'],
            'initial': trace[0]['before'],
            'trace': trace,
            'result': result,
            'eligible_for_training': condition == 'workflow_teacher',
        }
        record['record_hash'] = digest(record)
        if path.name == 'record.json':
            return record
        if path.name == 'environment.json':
            return {**record, 'reference_enabled': False}
        if path.name == 'launch.json':
            return {'image': runner.IMAGE, 'container_id': 'same-container'}
        if path.name == 'exit.json':
            return {'Running': False, 'ExitCode': 0}
        raise AssertionError(path)

    monkeypatch.setattr(runner, 'read', read)
    with pytest.raises(AssertionError):
        runner.audit(data_root, root)


@pytest.mark.parametrize('cleanup_fails', [False, True])
async def test_timeout_preserves_completed_trace_pending_action_and_primary_error(
    tmp_path, monkeypatch, cleanup_fails
):
    runner = module(monkeypatch)
    workflow = workflow_spec('preparation-v1', 'heat')
    view = {
        'feedback': 'Your task is to: put a hot apple on countertop.',
        'admissible_commands': ['look'],
    }

    class Bridge:
        steps = 0
        closed = False

        def read(self):
            return {'public': view, 'workflow': workflow, 'terminal': False}

        async def step(self, action):
            self.steps += 1
            if self.steps == 2:
                raise asyncio.TimeoutError('test bridge timeout')
            return self.read()

        def close(self):
            self.closed = True
            if cleanup_fails:
                raise RuntimeError('test cleanup failure')

    bridge = Bridge()
    output = tmp_path / 'episode'

    async def opened(*args, **kwargs):
        output.mkdir()
        return bridge

    monkeypatch.setattr(runner, 'open_owned_bridge', opened)
    task = {'task_id': 't', 'role': 'dev', 'phase': 1, 'sequence': 1, 'workflow': workflow}
    with pytest.raises(asyncio.TimeoutError):
        await runner.run_episode(task, tmp_path, output, 'workflow_teacher', 'c' * 64)
    failure = runner.read(output / 'failure.json')
    assert len(failure['trace']) == 1 and failure['pending_action'] == 'look'
    assert failure['error']['type'] == 'TimeoutError' and bridge.closed
    assert (output / 'cleanup-failure.json').exists() == cleanup_fails
