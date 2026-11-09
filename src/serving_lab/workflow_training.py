"""C1固定阶段一LoRA与旧技能回放：每次训练重新核原始来源。"""

import argparse
import json
from pathlib import Path, PureWindowsPath
import sys

from .coldstart_train import RECIPE, encode_example
from .execution import write_once
from .lab import inspect_base, file_hash
from .task_protocol import canonical, digest
from .teacher_data import action_samples
from .workflow import validate_workflow_manifest
from .workflow_study import workflow_samples, replay_indices

ROOT = Path(__file__).resolve().parents[2]
C0_DIGEST = '1ee83b8562404b56bdb208a675bb5d4e4c83b827156df8f6b1e73d30842989de'
N1_RECIPE = {
    **RECIPE,
    'seed': 20261105,
    'mix': 'new stage-one and old train approximately 1:1 supervised tokens; loss remains mean per example',
    'selection': 'one recipe, three fixed epochs, final checkpoint only',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_path(value, root=ROOT):
    """只重绑本仓库位置，不改任何冻结来源字节或允许任意跨仓库读取。"""
    if len(value) > 2 and value[1] == ':':
        relative = PureWindowsPath(value).relative_to(PureWindowsPath('D:/work/llm-serving-lab'))
        if '..' in relative.parts:
            raise ValueError('源路径越过仓库')
        return Path(root).joinpath(*relative.parts)
    path = Path(value).resolve()
    path.relative_to(Path(root).resolve())
    return path


def validate_parent(binding, base, manifest_digest, expected_parent=C0_DIGEST):
    if (
        digest(base['files']) != expected_parent
        or binding['base_artifact_digest'] != expected_parent
        or binding['base_files'] != base['files']
        or binding['dataset_manifest_sha256'] != manifest_digest
    ):
        raise ValueError('C1必须绑定冻结C0完整参数及确切训练数据，分词器相同不足以通过')


def reconstruct(sources, tokenizer, source_root=ROOT):
    """从原件重建，而非信任编译后JSONL自称为train。"""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from q2_workflow import audit as audit_workflow
    from q2_teacher import verify_collection

    workflow_data, collection = (
        source_path(sources['workflow_data'], source_root),
        source_path(sources['workflow_collection'], source_root),
    )
    old_data, old_collection = (
        source_path(sources['old_data'], source_root),
        source_path(sources['old_collection'], source_root),
    )
    checks = {
        'workflow': audit_workflow(workflow_data, collection),
        'old': verify_collection(old_data, old_collection),
    }
    if not all(item['data_gate_passed'] for item in checks.values()):
        raise ValueError('原始教师来源未通过')
    wf, old = read(workflow_data / 'manifest.json'), read(old_data / 'manifest.json')
    tasks = validate_workflow_manifest(wf)
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    new_rows, old_rows = [], []
    for task in tasks.values():
        if task['role'] == 'train' and task['phase'] == 1:
            record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
            new_rows.extend(
                workflow_samples(
                    record,
                    task,
                    wf['source_contract_digest'],
                    file_hash(workflow_data / 'manifest.json'),
                    tokenize,
                )
            )
    for task in old['tasks']:
        if task['role'] == 'train':
            record = read(old_collection / task['task_id'] / 'record.json')
            old_rows.extend(
                {**row, 'cohort': 'old_skill_replay'}
                for row in action_samples(record, tokenize, old['source_contract_digest'])
            )
    if not new_rows or not old_rows:
        raise ValueError('新技能或旧技能监督为空')
    counts = lambda rows: [sum(v != -100 for v in encode_example(r, tokenizer)[1]) for r in rows]
    new_counts, old_counts = counts(new_rows), counts(old_rows)
    indices = replay_indices(old_counts, sum(new_counts), N1_RECIPE['seed'])
    rows = new_rows + [old_rows[i] for i in indices]
    mix = {
        'new_samples': len(new_rows),
        'old_pool_samples': len(old_rows),
        'replay_samples': len(indices),
        'new_supervised_tokens': sum(new_counts),
        'replay_supervised_tokens': sum(old_counts[i] for i in indices),
        'replay_indices': indices,
        'future_rows': 0,
        'dev_rows': 0,
        'phase': 1,
    }
    return rows, mix, checks


def prepare(model, output):
    from transformers import AutoTokenizer

    model, output = Path(model).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('数据工件必须使用新目录')
    base = inspect_base(model)
    if digest(base['files']) != C0_DIGEST:
        raise ValueError('不是冻结C0工件')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    sources = {
        k: str(ROOT / 'artifacts' / v)
        for k, v in {
            'workflow_data': 'q2-workflow-data-20261105-a',
            'workflow_collection': 'q2-workflow-collection-20261105-a',
            'old_data': 'q2-teacher-data-20261103-b',
            'old_collection': 'q2-teacher-collection-20261103-b',
        }.items()
    }
    rows, mix, checks = reconstruct(sources, tokenizer)
    output.mkdir(parents=True)
    (output / 'train.jsonl').write_text(
        ''.join(canonical(row) + '\n' for row in rows), encoding='utf-8'
    )
    manifest = {
        'name': 'workflow-stage1-replay-v1',
        'sources': sources,
        'mix': mix,
        'source_verification': checks,
        'data_sha256': file_hash(output / 'train.jsonl'),
        'count': len(rows),
        'dev_samples': 0,
        'tokenizer_config_sha256': file_hash(model / 'tokenizer_config.json'),
        'recipe': N1_RECIPE,
        'source_manifest_hashes': {
            k: file_hash(Path(v) / 'manifest.json')
            for k, v in sources.items()
            if k.endswith('_data')
        },
    }
    write_once(output / 'manifest.json', manifest)
    write_once(
        output / 'base-binding.json',
        {
            'base_files': base['files'],
            'base_artifact_digest': digest(base['files']),
            'dataset_manifest_sha256': file_hash(output / 'manifest.json'),
        },
    )
    return {
        'samples': len(rows),
        'mix': {k: v for k, v in mix.items() if k != 'replay_indices'},
        'recipe': N1_RECIPE,
    }


def validated_examples(dataset, model, tokenizer, base, source_root=ROOT):
    dataset = Path(dataset)
    manifest = read(dataset / 'manifest.json')
    validate_parent(read(dataset / 'base-binding.json'), base, file_hash(dataset / 'manifest.json'))
    if (
        manifest.get('name') != 'workflow-stage1-replay-v1'
        or manifest['recipe'] != N1_RECIPE
        or manifest['dev_samples'] != 0
        or file_hash(dataset / 'train.jsonl') != manifest['data_sha256']
        or file_hash(Path(model) / 'tokenizer_config.json') != manifest['tokenizer_config_sha256']
    ):
        raise ValueError('C1固定配方或编译数据不符')
    for key, expected in manifest['source_manifest_hashes'].items():
        if (
            file_hash(source_path(manifest['sources'][key], source_root) / 'manifest.json')
            != expected
        ):
            raise ValueError('源任务合同已变')
    expected, mix, checks = reconstruct(manifest['sources'], tokenizer, source_root)
    rows = [
        json.loads(line)
        for line in (dataset / 'train.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    if (
        rows != expected
        or mix != manifest['mix']
        or checks != manifest['source_verification']
        or len(rows) != manifest['count']
    ):
        raise ValueError('监督不等于原始训练来源的因果前缀')
    return manifest, rows, [encode_example(row, tokenizer, N1_RECIPE['max_length']) for row in rows]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'train'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset')
    a = p.parse_args()
    try:
        if a.stage == 'prepare':
            result = prepare(a.model, a.output)
        else:
            from .coldstart_train import train_once

            result = train_once(a.dataset, a.model, a.output, mode='workflow_stage1')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        out = Path(a.output)
        if out.exists() and not (out / 'failure.json').exists():
            write_once(out / 'failure.json', {'error': repr(error), 'implicit_retry': False})
        raise
