"""固定公开文本来源并在观察运行结果前选择任务；不执行数据集代码。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from serving_lab.task_protocol import FAMILIES, canonical, digest, checked_target, validate_manifest

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / 'configs/evaluation/q1-sources.json').read_text(
        encoding='utf-8'
    )
)
REVISION = CONFIG['agentcl_revision']
ASSETS = CONFIG['sources']


def fetch_sources(root):
    cache = root / 'sources'
    cache.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, expected in ASSETS.items():
        url, size = expected['url'], expected['bytes']
        target = cache / name
        receipt = cache / (name + '.receipt.json')
        if target.exists():
            data = target.read_bytes()
            if len(data) != size:
                raise ValueError('缓存大小不符，保留原文件：' + name)
            if (
                receipt.exists()
                and json.loads(receipt.read_text(encoding='utf-8'))['sha256']
                != hashlib.sha256(data).hexdigest()
            ):
                raise ValueError('缓存摘要变化：' + name)
        else:
            print(json.dumps({'download': name, 'expected_bytes': size}), flush=True)
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError('下载大小与已核官方资产不符：' + name)
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('上游资产不再匹配冻结字节：' + name)
            with target.open('xb') as stream:
                stream.write(data)
        if hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise ValueError('缓存不匹配冻结上游字节：' + name)
        item = {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        if not receipt.exists():
            receipt.write_text(canonical(item) + '\n', encoding='utf-8')
        receipts[name] = item
    return receipts


def archive_index(archive, leaf):
    result = {}
    splits = {
        'train',
        'valid_seen',
        'valid_unseen',
        'eval_in_distribution',
        'eval_out_of_distribution',
    }
    for name in archive.namelist():
        parts = name.split('/')
        if parts[-1] != leaf:
            continue
        index = next((i for i, p in enumerate(parts) if p in splits), None)
        if index is not None:
            key = '/'.join(parts[index:-1])
            if key in result:
                raise ValueError('源归档身份重复：' + key)
            result[key] = name
    return result


def audit_codeeval(path):
    records = json.loads(path.read_text(encoding='utf-8'))
    by_key = {}
    for record in records:
        key = (record['category'], record['id'])
        if key in by_key:
            raise ValueError('CodeEval联合身份重复')
        by_key[key] = record
    injected = called = raw_checks = 0
    for (category, identifier), record in by_key.items():
        if category == 'raw':
            tree = ast.parse(record['test_code'])
            has_check = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'check'
                for node in tree.body
            )
            calls_check = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'check'
                for n in ast.walk(tree)
            )
            raw_checks += int(has_check and not calls_check)
            continue
        source = by_key[('raw', record['corresponding_raw_id'])]
        functions = [
            n.name
            for n in ast.parse(source['problem'] + source['solution']).body
            if isinstance(n, ast.FunctionDef)
        ]
        helper = functions[-1]
        hidden = [
            n.name for n in ast.parse(record['test_code']).body if isinstance(n, ast.FunctionDef)
        ]
        injected += helper in hidden
        calls = {
            n.func.id
            for n in ast.walk(ast.parse(record['problem'] + record['solution']))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        called += helper in calls
    return {
        'source_revision': REVISION,
        'records': len(records),
        'pairs': sum(k[0] == 'new' for k in by_key),
        'complex_tests_define_source_helper': injected,
        'complex_references_call_source_helper': called,
        'raw_checks_without_top_level_call': raw_checks,
        'scope': 'static source audit; no claim about unpublished AgentCL runner or paper result validity',
        'decision': 'use AgentCL task-relation principles; do not count injected library functions as skills internalized into weights',
    }


def prepare(destination, seed=CONFIG['selection_seed']):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise ValueError('任务清单已冻结，不覆盖；如需改选择须用新输出身份')
    receipts = fetch_sources(root)
    audit = audit_codeeval(root / 'sources/agentcl-humaneval-dependent.json')
    (root / 'codeeval-audit.json').write_text(canonical(audit) + '\n', encoding='utf-8')
    tasks = []
    with (
        zipfile.ZipFile(root / 'sources/json.zip') as trajectories,
        zipfile.ZipFile(root / 'sources/games.zip') as games,
    ):
        ti = archive_index(trajectories, 'traj_data.json')
        gi = archive_index(games, 'game.tw-pddl')
        common = set(ti) & set(gi)
        print(
            json.dumps(
                {
                    'archive_splits': sorted({k.split('/')[0] for k in common}),
                    'paired_files': len(common),
                }
            ),
            flush=True,
        )
        for family_index, family in enumerate(FAMILIES):
            ranked = sorted(
                (k for k in common if k.startswith('train/' + family + '-')),
                key=lambda k: hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest(),
            )
            selected = []
            for key in ranked:
                meta = json.loads(trajectories.read(ti[key]))
                if meta['task_type'] != family:
                    raise ValueError('路径与实际任务类型不符')
                scene = meta['scene']['scene_num']
                params = digest(meta['pddl_params'])
                if not selected or (scene != selected[0][2] and params != selected[0][3]):
                    selected.append((key, meta, scene, params))
                if len(selected) == 2:
                    break
            if len(selected) != 2:
                raise ValueError('没有符合预定场景/参数差异的任务：' + family)
            for number, (key, meta, scene, params) in enumerate(selected):
                role = 'support' if number == 0 else 'target'
                task_id = f'q1_{role}_{family_index + 1:02d}'
                game_bytes = games.read(gi[key])
                traj_bytes = trajectories.read(ti[key])
                files = {}
                for leaf, data in [('game.tw-pddl', game_bytes), ('traj_data.json', traj_bytes)]:
                    relative = f'games/{task_id}/{leaf}'
                    target = checked_target(root, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(data)
                    files[relative] = hashlib.sha256(data).hexdigest()
                tasks.append(
                    {
                        'task_id': task_id,
                        'role': role,
                        'sequence': family_index + 1 + (6 if number else 0),
                        'family': family,
                        'source_split': 'train',
                        'source_key': key,
                        'scene': scene,
                        'parameter_digest': params,
                        'game_path': f'games/{task_id}/game.tw-pddl',
                        'files': files,
                        'candidate_source_task_id': f'q1_support_{family_index + 1:02d}'
                        if number
                        else None,
                    }
                )
            protected = sorted(
                (
                    k
                    for k in common
                    if k.split('/')[0] in {'valid_unseen', 'eval_out_of_distribution'}
                    and k.split('/')[1].startswith(family + '-')
                ),
                key=lambda k: hashlib.sha256(f'{seed}:protected:{k}'.encode()).hexdigest(),
            )
            if not protected:
                raise ValueError('官方unseen中没有该任务类型：' + family)
            key = protected[0]
            tasks.append(
                {
                    'task_id': f'q1_protected_{family_index + 1:02d}',
                    'role': 'protected',
                    'sequence': 100 + family_index,
                    'family': family,
                    'source_split': key.split('/')[0],
                    'source_key': key,
                    'source_game_sha256': hashlib.sha256(games.read(gi[key])).hexdigest(),
                    'source_traj_sha256': hashlib.sha256(trajectories.read(ti[key])).hexdigest(),
                    'extracted': False,
                }
            )
    tasks.sort(key=lambda x: x['sequence'])
    manifest = {
        'schema': 1,
        'name': 'q1-alfworld-text-qualification-v1',
        'selection_seed': seed,
        'max_steps': CONFIG['max_steps'],
        'sources': receipts,
        'tasks': tasks,
        'codeeval_audit': audit,
        'relation_status': 'same-family/different-scene-and-parameters is a procedural-transfer hypothesis, not established learning benefit',
        'protected_policy': 'manifest/hash only; no extraction or reset in Q1',
        'licenses': {
            'AgentCL': 'CC-BY-NC-4.0; upstream terms retained',
            'ALFWorld_code': 'MIT; official release asset provenance retained',
        },
        'scope': 'CPU environment and evaluation preflight; no LLM, training, novel method, or enterprise transfer claim',
    }
    validate_manifest(manifest)
    (root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (root / 'manifest-anchor.json').write_text(
        canonical({'sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()})
        + '\n',
        encoding='utf-8',
    )
    return {
        'directory': str(root),
        'support': 6,
        'target': 6,
        'protected_unexecuted': 6,
        'source_audit': audit,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
