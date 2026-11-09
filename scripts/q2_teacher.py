"""新公开教师的选题、采集与动作监督导出；旧Q1数据与保护集保持不动。"""

import argparse
import asyncio
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from serving_lab.execution import write_once
from serving_lab.lab import file_hash, inspect_base
from serving_lab.task_protocol import FAMILIES, canonical, checked_target, digest, verify_files
from serving_lab.teacher_data import (
    action_samples,
    select_candidates,
    validate_teacher_manifest,
    validate_base_binding,
    FIXED_BASE_REVISION,
)
from q1_prepare import archive_index, CONFIG
from q2_run import docker, episode

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / 'artifacts/q1-data-20261102'
IMAGE = CONFIG['image_id']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot(root):
    names = [
        'scripts/q2_teacher.py',
        'scripts/q1_prepare.py',
        'scripts/q2_run.py',
        'scripts/q2_environment.py',
        'scripts/start_engine.py',
        'scripts/probe_serving.py',
    ]
    # 教师采集冻结自身执行路径；随后独立trainer的开发不改变CPU来源身份。
    names += [
        'src/serving_lab/' + p.name
        for p in (REPO / 'src/serving_lab').glob('*.py')
        if p.name != 'coldstart_train.py'
    ]
    hashes = {n: file_hash(REPO / n) for n in names}
    for name in names:
        target = root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    return hashes


def prepare(output):
    out = Path(output).resolve()
    if out.exists():
        raise ValueError('新来源目录必须不存在')
    old = read(OLD / 'manifest.json')
    assert file_hash(OLD / 'manifest.json') == read(OLD / 'manifest-anchor.json')['sha256']
    excluded_scenes = {t['scene'] for t in old['tasks'] if t['role'] != 'protected'}
    excluded_params = {t['parameter_digest'] for t in old['tasks'] if t['role'] != 'protected'}
    for name in ('json.zip', 'games.zip'):
        assert file_hash(OLD / 'sources' / name) == CONFIG['sources'][name]['sha256']
    candidates = []
    with (
        zipfile.ZipFile(OLD / 'sources/json.zip') as trajectories,
        zipfile.ZipFile(OLD / 'sources/games.zip') as games,
    ):
        ti, gi = archive_index(trajectories, 'traj_data.json'), archive_index(games, 'game.tw-pddl')
        for key in sorted(set(ti) & set(gi)):
            if not key.startswith('train/'):
                continue
            meta = json.loads(trajectories.read(ti[key]))
            family, scene, parameters = (
                meta['task_type'],
                meta['scene']['scene_num'],
                digest(meta['pddl_params']),
            )
            if family not in FAMILIES or scene in excluded_scenes or parameters in excluded_params:
                continue
            candidates.append(
                {'key': key, 'family': family, 'scene': scene, 'parameter_digest': parameters}
            )
        scenes = sorted(
            {r['scene'] for r in candidates},
            key=lambda s: hashlib.sha256(f'20261103:scene:{s}'.encode()).hexdigest(),
        )
        partitions = {s: 'dev' if i % 3 == 0 else 'train' for i, s in enumerate(scenes)}
        for r in candidates:
            r['scene_partition'] = partitions[r['scene']]
        selected = select_candidates(candidates, seed=20261103)
        out.mkdir(parents=True, exist_ok=False)
        tasks, counters = [], Counter()
        for sequence, row in enumerate(selected, 1):
            role, family = row['role'], row['family']
            counters[role, family] += 1
            task_id = f'cs_{role}_{FAMILIES.index(family) + 1:02}_{counters[role, family]:02}'
            files = {}
            for leaf, archive, entry in [
                ('game.tw-pddl', games, gi[row['key']]),
                ('traj_data.json', trajectories, ti[row['key']]),
            ]:
                data = archive.read(entry)
                relative = f'games/{task_id}/{leaf}'
                target = checked_target(out, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
                files[relative] = hashlib.sha256(data).hexdigest()
            tasks.append(
                {
                    'task_id': task_id,
                    'sequence': sequence,
                    'role': role,
                    'family': family,
                    'scene': row['scene'],
                    'parameter_digest': row['parameter_digest'],
                    'source_key': row['key'],
                    'source_split': 'train',
                    'game_path': f'games/{task_id}/game.tw-pddl',
                    'files': files,
                }
            )
    source_contract = {
        'name': 'teacher-public-program-coldstart-v1',
        'origin': 'teacher_public_program',
        'old_records_allowed': False,
        'teacher_policy_sha256': file_hash(REPO / 'src/serving_lab/public_tasks.py'),
        'observation': ['feedback', 'admissible_commands'],
        'max_steps': 100,
        'stagnation_limit': 4,
        'train_gate_per_family': 4,
        'dev_teacher_gate_per_family': 2,
        'supervision': 'successful train episodes only; legal action+EOS; exclude explicit rejection and Nothing happens.; full public causal prefixes',
        'authorization': 'user approved concrete next-step cold-start contract 2026-11-03',
    }
    manifest = {
        'schema': 1,
        'name': 'q2-public-teacher-coldstart-v1',
        'max_steps': 100,
        'selection_seed': 20261103,
        'old_manifest_sha256': file_hash(OLD / 'manifest.json'),
        'excluded_old_scenes': sorted(excluded_scenes),
        'excluded_old_parameter_digests': sorted(excluded_params),
        'scene_partitions': partitions,
        'source_contract': source_contract,
        'source_contract_digest': digest(source_contract),
        'sources': {n: CONFIG['sources'][n] for n in ('json.zip', 'games.zip')},
        'tasks': tasks,
        'protected_policy': 'old six valid_unseen identities remain unextracted; no new official validation tasks accessed',
    }
    validate_teacher_manifest(manifest)
    write_once(out / 'manifest.json', manifest)
    write_once(out / 'manifest-anchor.json', {'sha256': file_hash(out / 'manifest.json')})
    write_once(
        out / 'prepare-receipt.json',
        {
            'source_hashes': snapshot(out),
            'candidate_counts': dict(Counter(r['family'] for r in candidates)),
            'selected': 72,
            'train': 48,
            'dev': 24,
            'new_model_calls': 0,
        },
    )
    return {
        'data': str(out),
        'train': 48,
        'dev': 24,
        'source_contract_digest': manifest['source_contract_digest'],
    }


def rebind(data, output):
    """源码修复后仍使用完全相同72题；只创建新合同，不改A原件或挑成功题。"""
    data, out = Path(data).resolve(), Path(output).resolve()
    manifest = read(data / 'manifest.json')
    validate_teacher_manifest(manifest)
    assert file_hash(data / 'manifest.json') == read(data / 'manifest-anchor.json')['sha256']
    out.mkdir(parents=True, exist_ok=False)
    for task in manifest['tasks']:
        verify_files(task, data)
        for name in task['files']:
            target = checked_target(out, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(checked_target(data, name).read_bytes())
    manifest['source_contract']['teacher_policy_sha256'] = file_hash(
        REPO / 'src/serving_lab/public_tasks.py'
    )
    manifest['source_contract_digest'] = digest(manifest['source_contract'])
    manifest['rebind_from'] = {
        'path': str(data),
        'manifest_sha256': file_hash(data / 'manifest.json'),
        'meaning': 'same complete selection; public grammar/lamp-location source repair only',
    }
    validate_teacher_manifest(manifest)
    write_once(out / 'manifest.json', manifest)
    write_once(out / 'manifest-anchor.json', {'sha256': file_hash(out / 'manifest.json')})
    write_once(
        out / 'prepare-receipt.json',
        {
            'source_hashes': snapshot(out),
            'selected': 72,
            'train': 48,
            'dev': 24,
            'same_tasks_as_parent': manifest['tasks'] == read(data / 'manifest.json')['tasks'],
        },
    )
    return {
        'data': str(out),
        'source_contract_digest': manifest['source_contract_digest'],
        'same_72_tasks': True,
    }


async def collect(data, output):
    data, out = Path(data).resolve(), Path(output).resolve()
    manifest = read(data / 'manifest.json')
    tasks = validate_teacher_manifest(manifest)
    assert file_hash(data / 'manifest.json') == read(data / 'manifest-anchor.json')['sha256']
    assert manifest['source_contract']['teacher_policy_sha256'] == file_hash(
        REPO / 'src/serving_lab/public_tasks.py'
    )
    assert json.loads(docker('image', 'inspect', IMAGE))[0]['Id'] == IMAGE
    out.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(out)
    write_once(
        out / 'input.json',
        {
            'data': str(data),
            'manifest_sha256': file_hash(data / 'manifest.json'),
            'source_contract_digest': manifest['source_contract_digest'],
            'source_hashes': hashes,
            'image': IMAGE,
            'episode_cap': 72,
            'action_cap': 7200,
        },
    )
    for task in tasks.values():
        await episode(
            task,
            data,
            out / task['task_id'],
            IMAGE,
            condition='teacher_public_program',
            max_steps=100,
            stagnation_limit=4,
            purpose='teacher_coldstart_source',
            source_contract_digest=manifest['source_contract_digest'],
        )
    assert all(file_hash(REPO / n) == h for n, h in hashes.items())
    result = verify_collection(data, out)
    write_once(out / 'summary.json', result)
    return result


def verify_collection(data, root):
    """先核全部尝试，再筛成功；不能通过丢弃失败提高来源通过率。"""
    data, root = Path(data).resolve(), Path(root).resolve()
    manifest, initial = read(data / 'manifest.json'), read(root / 'input.json')
    tasks = validate_teacher_manifest(manifest)
    assert (
        file_hash(data / 'manifest.json')
        == initial['manifest_sha256']
        == read(data / 'manifest-anchor.json')['sha256']
    )
    assert initial['source_contract_digest'] == manifest['source_contract_digest']
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    assert (
        initial['source_hashes']['src/serving_lab/public_tasks.py']
        == manifest['source_contract']['teacher_policy_sha256']
    )
    paths = sorted(root.glob('cs_*/record.json'))
    assert len(paths) == 72 and {p.parent.name for p in paths} == set(tasks)
    counts, steps = Counter(), 0
    for path in paths:
        r, env, launch, exit_state = (
            read(path),
            read(path.parent / 'environment.json'),
            read(path.parent / 'launch.json'),
            read(path.parent / 'exit.json'),
        )
        task = tasks[path.parent.name]
        verify_files(task, data)
        assert (
            r['task_id'] == env['task_id'] == task['task_id']
            and r['role'] == task['role']
            and r['sequence'] == task['sequence']
        )
        assert r['record_hash'] == digest({k: v for k, v in r.items() if k != 'record_hash'})
        assert r['purpose'] == 'teacher_coldstart_source' and r['origin'] == 'public_policy'
        assert (
            r['source_contract_digest'] == manifest['source_contract_digest']
            and not r['eligible_for_primary_history']
        )
        assert r['eligible_for_training'] == (r['role'] == 'train' and r['result']['success'])
        assert launch['image'] == initial['image'] == IMAGE and launch['command'][-2:] == [
            IMAGE,
            '/runner.py',
        ]
        assert (
            exit_state['ExitCode'] == 0
            and not exit_state['Running']
            and not env['reference_enabled']
        )
        assert r['initial'] == env['initial'] and r['result'] == env['result']
        assert len(r['trace']) == len(env['trace']) == r['result']['steps'] <= 100
        for a, b in zip(r['trace'], env['trace']):
            assert all(a[k] == b[k] for k in ('step', 'before', 'action', 'after'))
            assert set(a['before']) == set(a['after']) == {'feedback', 'admissible_commands'}
            assert a['receipt'] is None
        steps += len(r['trace'])
        if r['result']['success']:
            counts[task['role'], task['family']] += 1
    gate = all(counts['train', f] >= 4 and counts['dev', f] >= 2 for f in FAMILIES)
    return {
        'raw_verified': True,
        'episodes': 72,
        'actions': steps,
        'data_gate_passed': gate,
        'successes': {role: {f: counts[role, f] for f in FAMILIES} for role in ('train', 'dev')},
        'failed_episodes': 72 - sum(counts.values()),
        'model_calls': 0,
        'training_runs': 0,
    }


def export(data, collection, output, model):
    data, collection, out = Path(data).resolve(), Path(collection).resolve(), Path(output).resolve()
    result = verify_collection(data, collection)
    if not result['data_gate_passed']:
        raise ValueError('教师数据门未过，保留全部失败，不补题或启动训练')
    manifest = read(data / 'manifest.json')
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    rows, sources = [], []
    for task in manifest['tasks']:
        record = read(collection / task['task_id'] / 'record.json')
        if task['role'] == 'train' and record['result']['success']:
            samples = action_samples(record, tokenize, manifest['source_contract_digest'])
            if not samples:
                raise ValueError('成功回合没有有效动作监督')
            rows.extend(samples)
            sources.append(
                {
                    'task_id': task['task_id'],
                    'record_hash': record['record_hash'],
                    'samples': len(samples),
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'train.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')
    receipt = {
        'name': 'q2-action-supervision-v1',
        'evidence_level': 'teacher_coldstart_dataset',
        'count': len(rows),
        'source_contract_digest': manifest['source_contract_digest'],
        'data_manifest_sha256': file_hash(data / 'manifest.json'),
        'data_sha256': file_hash(out / 'train.jsonl'),
        'sources': sources,
        'collection': str(collection),
        'data': str(data),
        'model': str(Path(model).resolve()),
        'tokenizer_config_sha256': file_hash(Path(model) / 'tokenizer_config.json'),
        'source_verification': result,
        'supervision': manifest['source_contract']['supervision'],
        'dev_samples': 0,
    }
    write_once(out / 'manifest.json', receipt)
    bind_base(out, model)
    return receipt


def bind_base(dataset, model):
    """附加资格回执保持既有数据快照不变；训练和评价都必须消费它。"""
    dataset = Path(dataset).resolve()
    manifest = read(dataset / 'manifest.json')
    base = inspect_base(model)
    if file_hash(dataset / 'train.jsonl') != manifest['data_sha256']:
        raise ValueError('数据快照变化')
    receipt = {
        'base': base,
        'base_artifact_digest': digest(base['files']),
        'base_revision': FIXED_BASE_REVISION,
        'dataset_manifest_sha256': file_hash(dataset / 'manifest.json'),
    }
    validate_base_binding(receipt, base, file_hash(dataset / 'manifest.json'))
    write_once(dataset / 'base-binding.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--stage',
        required=True,
        choices=['prepare', 'rebind', 'collect', 'export', 'bind', 'audit'],
    )
    p.add_argument('--data')
    p.add_argument('--collection')
    p.add_argument('--model')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.stage == 'prepare':
        result = prepare(a.output)
    elif a.stage == 'rebind':
        result = rebind(a.data, a.output)
    elif a.stage == 'collect':
        result = asyncio.run(collect(a.data, a.output))
    elif a.stage == 'export':
        result = export(a.data, a.collection, a.output, a.model)
    elif a.stage == 'bind':
        result = bind_base(a.data, a.model)
    else:
        result = verify_collection(a.data, a.collection)
        write_once(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
