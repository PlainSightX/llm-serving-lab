"""准备流程任务流，并在隔离CPU环境中采集公开教师与反事实控制。"""

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.public_tasks import PublicPolicy
from serving_lab.task_protocol import canonical, checked_target, digest, verify_files
from serving_lab.workflow import (
    BASE_COUNTS,
    THERMAL_FAMILIES,
    VARIANTS,
    WorkflowPolicy,
    assess_workflow,
    validate_workflow_manifest,
    workflow_spec,
)
from q1_prepare import CONFIG, archive_index
from q2_run import open_owned_bridge

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / 'artifacts/q1-data-20261102'
COLD = REPO / 'artifacts/q2-teacher-data-20261103-b'
IMAGE = CONFIG['image_id']
PUBLIC_OBJECTS = {'Apple', 'Cup', 'Egg', 'Mug', 'Plate', 'Potato', 'Tomato'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot(root):
    names = [
        'scripts/q2_workflow.py',
        'scripts/q2_environment.py',
        'scripts/q2_run.py',
        'scripts/q1_prepare.py',
    ]
    names += ['src/serving_lab/' + path.name for path in (REPO / 'src/serving_lab').glob('*.py')]
    hashes = {name: file_hash(REPO / name) for name in names}
    for name in names:
        target = root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    return hashes


def select_scene_strata(rows, scenes, per_scene):
    """同组跨场景可重复参数以观察布局迁移，单场景内选择不同参数。"""
    selected = []
    for scene in sorted(scenes):
        parameters, chosen = set(), []
        for row in rows:
            if row['scene'] != scene or row['parameter_digest'] in parameters:
                continue
            chosen.append(row)
            parameters.add(row['parameter_digest'])
            if len(chosen) == per_scene:
                break
        if len(chosen) != per_scene:
            raise ValueError('场景内可执行实例不足以满足配额')
        selected.extend(chosen)
    return selected


def select_bases(candidates, cold, attempt=0):
    """按元数据固定选择，评价、训练、开发在场景及参数上隔离。"""
    known_scenes = {
        task['scene'] for task in cold['tasks'] if task['role'] == 'train' and task['scene'] < 100
    }
    specifications = [('future', 6, 2), ('dev', 3, 1), ('train', 12, 2)]
    rank = lambda row: hashlib.sha256(
        f'20261105:workflow:{attempt}:{row["key"]}'.encode()
    ).hexdigest()
    selected, prior_parameters, dev_scenes = [], set(), set()
    for role, count, per_scene in specifications:
        if role == 'future':
            scenes = {14, 16, 25}
        elif role == 'dev':
            eligible = [
                scene
                for scene in known_scenes
                if all(
                    any(
                        row['scene'] == scene
                        and row['family'] == family
                        and row['parameter_digest'] not in prior_parameters
                        for row in candidates
                    )
                    for family in THERMAL_FAMILIES
                )
            ]
            eligible.sort(
                key=lambda scene: hashlib.sha256(
                    f'20261105:workflow-dev:{attempt}:{scene}'.encode()
                ).hexdigest()
            )
            dev_scenes = set(eligible[:3])
            if len(dev_scenes) != 3:
                raise ValueError('两类任务共同可用的开发场景不足')
            scenes = dev_scenes
        else:
            scenes = known_scenes - dev_scenes
        role_parameters = set()
        for family in THERMAL_FAMILIES:
            chosen, family_parameters, scene_counts = [], set(), Counter()
            rows = [
                row
                for row in candidates
                if row['family'] == family
                and row['scene'] in scenes
                and row['parameter_digest'] not in prior_parameters
            ]
            if role != 'train':
                try:
                    chosen = select_scene_strata(sorted(rows, key=rank), scenes, per_scene)
                except ValueError as error:
                    counts = {
                        scene: len(
                            {row['parameter_digest'] for row in rows if row['scene'] == scene}
                        )
                        for scene in scenes
                    }
                    raise ValueError(f'{role}/{family}可选参数数{counts}，场景匹配失败') from error
                selected.extend({**row, 'role': role} for row in chosen)
                role_parameters.update(row['parameter_digest'] for row in chosen)
                continue
            for row in sorted(rows, key=rank):
                identity = (row['scene'], row['parameter_digest'])
                if identity in family_parameters or scene_counts[row['scene']] >= per_scene:
                    continue
                chosen.append(row)
                family_parameters.add(identity)
                scene_counts[row['scene']] += 1
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'{role}/{family}来源不足，不能降低隔离要求补题')
            selected.extend({**row, 'role': role} for row in chosen)
            role_parameters.update(row['parameter_digest'] for row in chosen)
        prior_parameters.update(role_parameters)
    return sorted(
        selected, key=lambda row: (tuple(BASE_COUNTS).index(row['role']), row['family'], rank(row))
    )


def prepare(output):
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('新任务数据目录必须不存在')
    cold, legacy = read(COLD / 'manifest.json'), read(OLD / 'manifest.json')
    assert file_hash(COLD / 'manifest.json') == read(COLD / 'manifest-anchor.json')['sha256']
    assert file_hash(OLD / 'manifest.json') == read(OLD / 'manifest-anchor.json')['sha256']
    excluded = {task['source_key'] for task in cold['tasks'] + legacy['tasks']}
    for name in ('json.zip', 'games.zip'):
        assert file_hash(OLD / 'sources' / name) == CONFIG['sources'][name]['sha256']
    with (
        zipfile.ZipFile(OLD / 'sources/json.zip') as trajectories,
        zipfile.ZipFile(OLD / 'sources/games.zip') as games,
    ):
        ti, gi = archive_index(trajectories, 'traj_data.json'), archive_index(games, 'game.tw-pddl')
        candidates = []
        for key in sorted(set(ti) & set(gi)):
            if not key.startswith('train/') or key in excluded:
                continue
            metadata = json.loads(trajectories.read(ti[key]))
            parameters = metadata['pddl_params']
            if (
                metadata['task_type'] not in THERMAL_FAMILIES
                or parameters['object_target'] not in PUBLIC_OBJECTS
                or parameters.get('object_sliced', False)
            ):
                continue
            candidates.append(
                {
                    'key': key,
                    'family': metadata['task_type'],
                    'scene': metadata['scene']['scene_num'],
                    'parameter_digest': digest(parameters),
                }
            )
        failures = Counter()
        for selection_attempt in range(256):
            try:
                bases = select_bases(candidates, cold, selection_attempt)
                break
            except ValueError as error:
                failures[str(error)] += 1
        else:
            raise ValueError('256个固定元数据排序均不能满足既定隔离和数量要求')
        out.mkdir(parents=True)
        base_rows = []
        for index, row in enumerate(bases, 1):
            base_id = f'base_{row["role"]}_{index:02}'
            files = {}
            for leaf, archive, entry in [
                ('game.tw-pddl', games, gi[row['key']]),
                ('traj_data.json', trajectories, ti[row['key']]),
            ]:
                data = archive.read(entry)
                relative = f'games/{base_id}/{leaf}'
                files[relative] = hashlib.sha256(data).hexdigest()
                # 未来评价仅冻结来源摘要，准备入口没有这些实例的可执行文件。
                if row['role'] != 'future':
                    target = checked_target(out, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            base_rows.append(
                {
                    'base_id': base_id,
                    'role': row['role'],
                    'family': row['family'],
                    'scene': row['scene'],
                    'source_split': 'train',
                    'source_key': row['key'],
                    'parameter_digest': row['parameter_digest'],
                    'game_path': f'games/{base_id}/game.tw-pddl',
                    'files': files,
                }
            )
    tasks = []
    for phase, variant in enumerate(VARIANTS, 1):
        for row in base_rows:
            tasks.append(
                {
                    **row,
                    'task_id': f'workflow_{phase}_{row["base_id"]}',
                    'sequence': len(tasks) + 1,
                    'phase': phase,
                    'variant': variant,
                    'workflow': workflow_spec(variant, THERMAL_FAMILIES[row['family']]),
                }
            )
    contract = {
        'origin': 'teacher_public_program',
        'purpose': 'workflow_teacher_source',
        'public_observations': ['feedback', 'admissible_commands', 'current_workflow'],
        'teacher_policy_sha256': file_hash(REPO / 'src/serving_lab/workflow.py'),
        'old_records_allowed': False,
        'dev_and_future_trainable': False,
        'max_steps': 100,
        'stagnation_limit': 4,
        'history': 'verified successful train teacher records only; phase and sequence cutoff',
        'authorization': 'user authorized next research stage on 2026-11-05',
    }
    manifest = {
        'schema': 1,
        'name': 'q2-public-workflows-v1',
        'max_steps': 100,
        'selection_seed': 20261105,
        'selection_attempt': selection_attempt,
        'excluded_source_keys': sorted(excluded),
        'coldstart_manifest_sha256': file_hash(COLD / 'manifest.json'),
        'source_contract': contract,
        'source_contract_digest': digest(contract),
        'tasks': tasks,
        'sources': {name: CONFIG['sources'][name] for name in ('json.zip', 'games.zip')},
        'future_policy': 'source metadata and digest only; no preparation reset, memory or training',
        'unit_of_analysis': '42 base instances; repeated workflow revisions are paired, not independent instances',
    }
    validate_workflow_manifest(manifest)
    write_once(out / 'manifest.json', manifest)
    write_once(out / 'manifest-anchor.json', {'sha256': file_hash(out / 'manifest.json')})
    write_once(
        out / 'prepare-receipt.json',
        {
            'source_hashes': snapshot(out),
            'base_counts': BASE_COUNTS,
            'tasks': len(tasks),
            'extracted_base_instances': 30,
            'model_calls': 0,
            'selection_attempt': selection_attempt,
            'metadata_constraint_failures': dict(failures),
        },
    )
    return {'data': str(out), 'tasks': len(tasks), 'base_counts': BASE_COUNTS}


async def run_episode(task, data, output, condition, source_contract_digest):
    bridge, trace, initial, pending_action, failure = None, [], None, None, None
    try:
        bridge = await open_owned_bridge(task, data, output, IMAGE, 100, stagnation_limit=4)
        value = await asyncio.wait_for(asyncio.to_thread(bridge.read), 30)
        initial = value['public']
        assert value['workflow'] == task['workflow']
        chosen_workflow = value['workflow']
        if condition == 'opposite_order':
            opposite = 'preparation-v2' if task['variant'] == 'preparation-v1' else 'preparation-v1'
            chosen_workflow = workflow_spec(opposite, task['workflow']['thermal'])
        policy = PublicPolicy() if condition == 'base_policy' else WorkflowPolicy(chosen_workflow)
        while not value['terminal']:
            before = value['public']
            action = (
                policy.choose(before)
                if condition == 'base_policy'
                else policy.choose(before, trace)
            )
            pending_action = action
            value = await bridge.step(action)
            trace.append(
                {
                    'step': len(trace) + 1,
                    'before': before,
                    'action': action,
                    'after': value['public'],
                    'receipt': None,
                }
            )
            pending_action = None
        result = value['result']
        record = {
            'task_id': task['task_id'],
            'role': task['role'],
            'sequence': task['sequence'],
            'phase': task['phase'],
            'workflow': task['workflow'],
            'condition': condition,
            'initial': initial,
            'trace': trace,
            'result': result,
            'purpose': 'workflow_teacher_source',
            'origin': 'public_policy',
            'source_contract_digest': source_contract_digest,
            'eligible_for_training': condition == 'workflow_teacher'
            and task['role'] == 'train'
            and result['success'],
        }
        record['record_hash'] = digest(record)
        write_once(output / 'record.json', record)
        return record
    except BaseException as error:
        failure = error
        output.mkdir(parents=True, exist_ok=True)
        write_once(
            output / 'failure.json',
            {
                'task_id': task['task_id'],
                'condition': condition,
                'error': {'type': type(error).__name__, 'message': str(error)},
                'initial': initial,
                'trace': trace,
                'pending_action': pending_action,
            },
        )
        raise
    finally:
        if bridge is not None:
            try:
                await asyncio.to_thread(bridge.close)
            except Exception as error:
                write_once(
                    output / 'cleanup-failure.json',
                    {'type': type(error).__name__, 'message': str(error)},
                )
                if failure is None:
                    raise


def collection_plan(tasks):
    plan = [
        ('workflow_teacher', task)
        for role in ('dev', 'train')
        for task in tasks
        if task['role'] == role
    ]
    plan += [('base_policy', task) for task in tasks if task['role'] == 'dev']
    plan += [
        ('opposite_order', task)
        for task in tasks
        if task['role'] == 'dev' and task['workflow']['name'] == 'preparation'
    ]
    return plan


def reserve_cpu(path, task_id, condition):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len(rows) >= 160:
        raise ValueError('CPU准备累计已达160回合上限')
    with path.open('a', encoding='utf-8') as stream:
        stream.write(
            canonical(
                {
                    'episode': len(rows) + 1,
                    'task_id': task_id,
                    'condition': condition,
                    'max_actions': 100,
                }
            )
            + '\n'
        )


async def collect(data, output, budget):
    data, out = Path(data).resolve(), Path(output).resolve()
    manifest = read(data / 'manifest.json')
    tasks = list(validate_workflow_manifest(manifest).values())
    if (
        file_hash(REPO / 'src/serving_lab/workflow.py')
        != manifest['source_contract']['teacher_policy_sha256']
    ):
        raise ValueError('策略源码变化，需要新来源合同')
    out.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(out)
    plan = collection_plan(tasks)
    write_once(
        out / 'input.json',
        {
            'data': str(data),
            'manifest_sha256': file_hash(data / 'manifest.json'),
            'source_hashes': hashes,
            'image': IMAGE,
            'planned_episodes': len(plan),
            'budget_ledger': str(Path(budget).resolve()),
            'model_calls': 0,
        },
    )
    for condition, task in plan:
        reserve_cpu(budget, task['task_id'], condition)
        record = await run_episode(
            task,
            data,
            out / condition / task['task_id'],
            condition,
            manifest['source_contract_digest'],
        )
        print(
            canonical({'condition': condition, 'task': task['task_id'], **record['result']}),
            flush=True,
        )
        if (
            condition == 'workflow_teacher'
            and task['role'] == 'dev'
            and not record['result']['success']
        ):
            raise ValueError('开发可解性失败，停止扩展采集并保留原件')
    assert all(file_hash(REPO / name) == value for name, value in hashes.items())
    result = audit(data, out)
    write_once(out / 'summary.json', result)
    return result


def audit(data, root):
    data, root = Path(data).resolve(), Path(root).resolve()
    manifest, launch = read(data / 'manifest.json'), read(root / 'input.json')
    tasks = validate_workflow_manifest(manifest)
    assert (
        file_hash(data / 'manifest.json')
        == launch['manifest_sha256']
        == read(data / 'manifest-anchor.json')['sha256']
    )
    for name, value in launch['source_hashes'].items():
        assert file_hash(root / 'source' / name) == value
    assert (
        launch['source_hashes']['src/serving_lab/workflow.py']
        == manifest['source_contract']['teacher_policy_sha256']
    )
    paths = sorted(root.glob('*/*/record.json'))
    assert len(paths) == launch['planned_episodes'] == 120
    expected = {
        (condition, task['task_id']) for condition, task in collection_plan(list(tasks.values()))
    }
    seen, containers = set(), set()
    outcomes, actions, trainable = Counter(), 0, 0
    for path in paths:
        record, environment, created, exited = [
            read(path.with_name(name))
            for name in ('record.json', 'environment.json', 'launch.json', 'exit.json')
        ]
        task = tasks[record['task_id']]
        occurrence = (record['condition'], record['task_id'])
        assert occurrence in expected and occurrence not in seen
        seen.add(occurrence)
        assert path.parent.name == record['task_id'] == environment['task_id']
        assert path.parent.parent.name == record['condition']
        assert all(record[key] == task[key] for key in ('role', 'phase', 'sequence'))
        assert (
            record['purpose'] == 'workflow_teacher_source' and record['origin'] == 'public_policy'
        )
        assert created['container_id'] not in containers
        containers.add(created['container_id'])
        assert task['role'] != 'future'
        verify_files(task, data)
        assert record['record_hash'] == digest(
            {key: value for key, value in record.items() if key != 'record_hash'}
        )
        assert record['source_contract_digest'] == manifest['source_contract_digest']
        assert (
            record['initial'] == environment['initial']
            and record['result'] == environment['result']
        )
        assert record['workflow'] == environment['workflow'] == task['workflow']
        assert (
            not environment['reference_enabled']
            and exited['ExitCode'] == 0
            and not exited['Running']
        )
        assert created['image'] == launch['image'] == IMAGE
        assert len(record['trace']) == len(environment['trace']) == record['result']['steps'] <= 100
        for left, right in zip(record['trace'], environment['trace']):
            assert all(left[key] == right[key] for key in ('step', 'before', 'action', 'after'))
            assert set(left['before']) == set(left['after']) == {'feedback', 'admissible_commands'}
        recalculated = assess_workflow(
            record['trace'], task['workflow'], record['result']['base_success']
        )
        assert all(record['result'][key] == value for key, value in recalculated.items())
        eligible = (
            record['condition'] == 'workflow_teacher'
            and task['role'] == 'train'
            and record['result']['success']
        )
        assert record['eligible_for_training'] == eligible
        trainable += eligible
        outcomes[
            record['condition'], record['result']['base_success'], record['result']['success']
        ] += 1
        actions += len(record['trace'])
    assert seen == expected
    gate = outcomes == Counter(
        {
            ('workflow_teacher', True, True): 90,
            ('base_policy', True, False): 18,
            ('opposite_order', True, False): 12,
        }
    )
    return {
        'raw_verified': True,
        'data_gate_passed': gate,
        'episodes': 120,
        'actions': actions,
        'outcomes': [
            {
                'condition': key[0],
                'base_success': key[1],
                'workflow_success': key[2],
                'count': value,
            }
            for key, value in outcomes.items()
        ],
        'trainable_records': trainable if gate else 0,
        'future_episodes': 0,
        'model_calls': 0,
        'training_runs': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'collect', 'audit'), required=True)
    parser.add_argument('--data')
    parser.add_argument('--collection')
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget-ledger')
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.output)
    elif args.stage == 'collect':
        result = asyncio.run(collect(args.data, args.output, args.budget_ledger))
    else:
        result = audit(args.data, args.collection)
        write_once(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False))
