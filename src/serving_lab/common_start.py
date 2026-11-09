"""N5一次共享4B基础适配；不修改原1.7B合同，也不消费流程/dev/future监督。"""

import argparse
import json
from pathlib import Path
import sys

from .coldstart_train import RECIPE, encode_example
from .execution import write_once
from .lab import inspect_base, file_hash, digest, canonical
from .parameters import parameter_identity
from .teacher_data import action_samples, validate_teacher_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE_ARTIFACT = 'b45fba1ab01819bd9540b73858cd32e109bfeefed07ac36b5ada9248650dbba2'
BASE_PARAMETER = '3136b44de9201303ec3dcfd0ed66595c94c15ff4f58862f2aff474ade48e788c'
BASE_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
PRIOR_PROTOCOL_SHA = '79270759d110a61f88e36d70e8248a5746e788ad89907bc0b073e57be64b991e'
COMMON_RECIPE = dict(RECIPE)
DATA = 'artifacts/q2-teacher-data-20261103-b'
COLLECTION = 'artifacts/q2-teacher-collection-20261103-b'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_parent(artifact, parameters):
    if artifact != BASE_ARTIFACT or parameters != BASE_PARAMETER:
        raise ValueError('共同基础适配仅接受冻结原4B完整工件和参数')


def validate_plan(plan, hashes=None):
    if (
        plan['name'] != 'n5-common-4b-basic-repair-v1'
        or plan['recipe'] != COMMON_RECIPE
        or plan['parent_artifact_digest'] != BASE_ARTIFACT
        or plan['parent_parameter_digest'] != BASE_PARAMETER
        or plan['parent_revision'] != BASE_REVISION
        or plan['source_data'] != DATA
        or plan['source_collection'] != COLLECTION
        or plan['recipes_allowed'] != 1
        or plan['final_checkpoint_only'] is not True
    ):
        raise ValueError('不是冻结的一次共享基础适配')
    if hashes is not None and hashes != {
        'manifest': plan['source_manifest_sha256'],
        'collection': plan['source_collection_sha256'],
    }:
        raise ValueError('实际原始来源不符合冻结修复卡')


def reconstruct(tokenizer, source_root=ROOT):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from q2_teacher import verify_collection

    data, collection = Path(source_root) / DATA, Path(source_root) / COLLECTION
    verification = verify_collection(data, collection)
    if not verification['data_gate_passed']:
        raise ValueError('基础教师来源未资格')
    manifest = read(data / 'manifest.json')
    tasks = validate_teacher_manifest(manifest)
    tokenize = lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    rows = []
    sources = []
    for task in tasks.values():
        if task['role'] != 'train':
            continue
        record = read(collection / task['task_id'] / 'record.json')
        samples = action_samples(record, tokenize, manifest['source_contract_digest'])
        rows.extend(samples)
        sources.append(
            {
                'task_id': task['task_id'],
                'record_hash': record['record_hash'],
                'samples': len(samples),
            }
        )
    if len(sources) != 48 or len(rows) != 1186:
        raise ValueError('共同基础样本不再是原48train/1186动作')
    return (
        rows,
        sources,
        verification,
        {
            'manifest': file_hash(data / 'manifest.json'),
            'collection': file_hash(collection / 'input.json'),
        },
    )


def prepare(observation, tokenizer_path, plan_path, output):
    from transformers import AutoTokenizer

    observation = read(observation)
    require_parent(observation['artifact_digest'], observation['parameter_digest'])
    if digest(observation['base']['files']) != BASE_ARTIFACT:
        raise ValueError('基座文件集合摘要不符')
    plan = read(plan_path)
    validate_plan(plan)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    if (
        file_hash(Path(tokenizer_path) / 'tokenizer_config.json')
        != observation['base']['files']['tokenizer_config.json']
    ):
        raise ValueError('分词器不同')
    rows, sources, verification, hashes = reconstruct(tokenizer)
    validate_plan(plan, hashes)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'repair-plan.json').write_bytes(Path(plan_path).read_bytes())
    with (out / 'train.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(canonical(row) + '\n')
    manifest = {
        'name': 'n5-common-basic-4b-v1',
        'count': len(rows),
        'sources': sources,
        'source_hashes': hashes,
        'source_verification': verification,
        'data_sha256': file_hash(out / 'train.jsonl'),
        'recipe': COMMON_RECIPE,
        'repair_plan_sha256': file_hash(plan_path),
        'tokenizer_config_sha256': file_hash(Path(tokenizer_path) / 'tokenizer_config.json'),
        'dev_samples': 0,
        'workflow_samples': 0,
        'future_samples': 0,
        'research_training': False,
    }
    write_once(out / 'manifest.json', manifest)
    write_once(
        out / 'base-binding.json',
        {
            'base': observation['base'],
            'base_artifact_digest': BASE_ARTIFACT,
            'base_parameter_digest': BASE_PARAMETER,
            'base_revision': BASE_REVISION,
            'dataset_manifest_sha256': file_hash(out / 'manifest.json'),
        },
    )
    return {
        'samples': len(rows),
        'sources': len(sources),
        'planned_steps': 447,
        'research_training': False,
    }


def validated_examples(dataset, model, tokenizer, base):
    dataset = Path(dataset)
    manifest = read(dataset / 'manifest.json')
    binding = read(dataset / 'base-binding.json')
    if file_hash(dataset / 'repair-plan.json') != manifest['repair_plan_sha256']:
        raise ValueError('基础修复计划身份变化')
    plan = read(dataset / 'repair-plan.json')
    validate_plan(plan)
    activation = read(dataset / 'activation.json')
    if activation['dataset_manifest_sha256'] != file_hash(dataset / 'manifest.json') or activation[
        'prerequisites'
    ] != check_prerequisites(dataset / 'prerequisite'):
        raise ValueError('缺少本数据对应的已审计C0和原4B失败前置')
    require_parent(digest(base['files']), parameter_identity(model)['digest'])
    if (
        binding['base']['files'] != base['files']
        or binding['dataset_manifest_sha256'] != file_hash(dataset / 'manifest.json')
        or binding['base_artifact_digest'] != BASE_ARTIFACT
        or binding['base_parameter_digest'] != BASE_PARAMETER
        or binding['base_revision'] != BASE_REVISION
    ):
        raise ValueError('基础训练模型绑定不同')
    if (
        manifest['name'] != 'n5-common-basic-4b-v1'
        or manifest['recipe'] != COMMON_RECIPE
        or manifest['dev_samples']
        or manifest['workflow_samples']
        or manifest['future_samples']
        or manifest['research_training']
        or file_hash(dataset / 'train.jsonl') != manifest['data_sha256']
        or file_hash(Path(model) / 'tokenizer_config.json') != manifest['tokenizer_config_sha256']
    ):
        raise ValueError('共同基础数据/配方发生变化')
    rows, sources, verification, hashes = reconstruct(tokenizer)
    validate_plan(plan, hashes)
    actual = [
        json.loads(s) for s in (dataset / 'train.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    if (
        actual != rows
        or manifest['sources'] != sources
        or manifest['source_verification'] != verification
        or manifest['source_hashes'] != hashes
        or manifest['count'] != len(rows)
    ):
        raise ValueError('基础监督不能从原始train因果前缀重建')
    return (
        manifest,
        rows,
        [encode_example(row, tokenizer, COMMON_RECIPE['max_length']) for row in rows],
    )


def check_prerequisite(directory, role='Qwen3-4B-fixed-original'):
    directory = Path(directory)
    summary = read(directory / 'summary.json')
    initial = read(directory / 'input.json')
    audit = read(directory / 'audit.json')
    from .continual_procedure import ACTOR_IDENTITIES

    if role not in {'C0-Qwen3-1.7B', 'Qwen3-4B-fixed-original'}:
        raise ValueError('错误前置角色')
    artifact, parameters = ACTOR_IDENTITIES[role]
    if (
        not audit['passed']
        or audit['qualified']
        or not summary['completed']
        or summary['qualified']
        or audit['summary_sha256'] != file_hash(directory / 'summary.json')
        or audit['input_sha256'] != file_hash(directory / 'input.json')
        or audit['actor_role'] != role
        or initial['actor_role'] != role
        or initial['plan_sha256'] != PRIOR_PROTOCOL_SHA
        or summary['artifact_digest'] != artifact
        or summary['parameter_digest'] != parameters
    ):
        raise ValueError('必须先有对应共同actor完整、已审计的资格失败')
    if (
        audit['episodes'] != 36
        or len(initial['cases']) != 36
        or summary['per_condition_denominator'] != 18
        or initial['worker_manifest']['artifact_digest'] != artifact
        or initial['worker_manifest']['parameter_digest'] != parameters
    ):
        raise ValueError('前置资格的完整覆盖或实际参数不符')
    return {
        'audit_sha256': file_hash(directory / 'audit.json'),
        'summary_sha256': file_hash(directory / 'summary.json'),
        'input_sha256': file_hash(directory / 'input.json'),
    }


def check_prerequisites(directory):
    directory = Path(directory)
    roles = {'c0': 'C0-Qwen3-1.7B', 'qwen4': 'Qwen3-4B-fixed-original'}
    if not all((directory / name / 'audit.json').is_file() for name in roles):
        raise ValueError('缺少C0或原4B的失败审计')
    return {name: check_prerequisite(directory / name, role) for name, role in roles.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--output', required=True)
    p.add_argument('--observation')
    p.add_argument('--tokenizer')
    p.add_argument('--plan')
    p.add_argument('--dataset')
    p.add_argument('--model')
    p.add_argument('--prerequisite')
    a = p.parse_args()
    if a.stage == 'prepare':
        result = prepare(a.observation, a.tokenizer, a.plan, a.output)
    else:
        check_prerequisites(a.prerequisite)
        from .coldstart_train import train_once

        result = train_once(a.dataset, a.model, a.output, mode='n5_common_basic')
    print(json.dumps(result, ensure_ascii=False))
