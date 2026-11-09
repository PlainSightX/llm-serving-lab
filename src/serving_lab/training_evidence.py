"""独立核验C1固定训练的来源、完整步序及实际最终参数。"""

import json
import math
from pathlib import Path
import random

from .lab import inspect_base, file_hash
from .task_protocol import digest
from .workflow_training import C0_DIGEST, N1_RECIPE, validated_examples


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_training_binding(initial, summary, manifest, manifest_hash, candidate_artifact):
    expected = math.ceil(manifest['count'] / N1_RECIPE['accumulation']) * N1_RECIPE['epochs']
    valid = (
        initial.get('mode') == 'workflow_stage1'
        and initial['recipe'] == manifest['recipe'] == N1_RECIPE
        and initial['data'] == manifest
        and initial['dataset_manifest_sha256'] == manifest_hash
        and initial['optimizer_steps'] == summary['optimizer_steps'] == expected == 675
        and summary['epochs'] == 3
        and summary['samples'] == manifest['count']
        and manifest['dev_samples']
        == manifest['mix']['dev_rows']
        == manifest['mix']['future_rows']
        == 0
        and manifest['mix']['phase'] == 1
        and summary['data_sha256'] == manifest['data_sha256']
        and summary['base_artifact_digest'] == C0_DIGEST
        and initial['base_binding_sha256'] == summary['base_binding_sha256']
        and digest(summary['merged_identity']['files']) == candidate_artifact
        and 0 <= summary['adapter_reload_max_difference'] <= 1e-3
        and 0 <= summary['merged_reload_max_difference'] <= 1e-3
    )
    # input中原先保存完整base，纯元数据测试可传等价摘要。
    parent = (
        digest(initial['base']['files'])
        if 'base' in initial
        else initial.get('base_artifact_digest')
    )
    if not valid or parent != C0_DIGEST:
        raise ValueError('候选未绑定本次完整固定C1训练')


def check_training_log(rows, encoded, recipe):
    expected = []
    seen = 0
    elapsed = -1.0
    for epoch in range(recipe['epochs']):
        order = list(range(len(encoded)))
        random.Random(recipe['seed'] + epoch).shuffle(order)
        expected.extend(
            (epoch + 1, order[offset : offset + recipe['accumulation']])
            for offset in range(0, len(order), recipe['accumulation'])
        )
    if len(rows) != len(expected):
        raise ValueError('训练步数缺失或多出')
    for update, (row, (epoch, batch)) in enumerate(zip(rows, expected), 1):
        seen += sum(encoded[index]['supervised_tokens'] for index in batch)
        if (
            row['update'] != update
            or row['epoch'] != epoch
            or row['samples'] != batch
            or row['supervised_tokens_seen'] != seen
            or not math.isfinite(row['mean_loss'])
            or row['elapsed_seconds'] < elapsed
        ):
            raise ValueError('训练顺序、监督token或数值证据不符')
        elapsed = row['elapsed_seconds']
    return seen


def verify_training(training, dataset, candidate_model, parent_model):
    """只有完整训练通过后才返回可进入评价/发布的工件绑定。"""
    from transformers import AutoTokenizer
    from .parameters import parameter_identity

    training, dataset, candidate_model, parent_model = map(
        lambda p: Path(p).resolve(), (training, dataset, candidate_model, parent_model)
    )
    for p in [
        training / 'failure.json',
        training / 'interruption.json',
        training.parent / 'bootstrap-failure.json',
    ]:
        if p.exists():
            raise ValueError('训练失败或中断，不能作为完整C1')
    if not (training / 'summary.json').is_file():
        raise ValueError('训练尚未完成')
    initial, summary, manifest = (
        read(training / 'input.json'),
        read(training / 'summary.json'),
        read(dataset / 'manifest.json'),
    )
    owner = dataset.parent
    bundle = read(owner / 'training-bundle-manifest.json')
    bootstrap = read(training.parent / 'bootstrap.json')
    if (
        bootstrap['bundle_sha256'] != file_hash(owner / 'training-bundle-manifest.json')
        or bootstrap['launcher_sha256'] != file_hash(owner / 'cloud_training_entry.py')
        or initial['trainer_sha256'] != bundle['src/serving_lab/coldstart_train.py']
        or initial['dataset_manifest_sha256']
        != bundle['artifacts/n1-20261105-a/training-data/manifest.json']
    ):
        raise ValueError('训练未绑定启动前冻结的源码、依赖与数据包')
    artifact = inspect_base(candidate_model)
    check_training_binding(
        initial, summary, manifest, file_hash(dataset / 'manifest.json'), digest(artifact['files'])
    )
    if inspect_base(training / 'merged')['files'] != artifact['files']:
        raise ValueError('评价没有使用训练最终合并产物')
    if file_hash(training / 'trainer.py') != initial['trainer_sha256']:
        raise ValueError('训练器源码不符')
    if file_hash(dataset / 'base-binding.json') != initial['base_binding_sha256']:
        raise ValueError('训练数据/父参数绑定不符')
    parent = inspect_base(parent_model)
    if digest(parent['files']) != C0_DIGEST or initial['base']['files'] != parent['files']:
        raise ValueError('训练父参数不是固定C0')
    tokenizer = AutoTokenizer.from_pretrained(
        parent_model, local_files_only=True, trust_remote_code=False
    )
    source_root = dataset.parents[2]
    checked_manifest, source_rows, examples = validated_examples(
        dataset, parent_model, tokenizer, parent, source_root=source_root
    )
    if checked_manifest != manifest:
        raise ValueError('训练来源改变')
    encoded = [
        {
            'task_id': row['task_id'],
            'step': row['step'],
            'tokens': len(ids),
            'supervised_tokens': sum(v != -100 for v in labels),
            'ids_sha256': digest(ids),
            'labels_sha256': digest(labels),
        }
        for row, (ids, labels, _) in zip(source_rows, examples)
    ]
    if encoded != initial['encoded_examples']:
        raise ValueError('实际训练编码与因果前缀不同')
    logs = [
        json.loads(s)
        for s in (training / 'training.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    seen = check_training_log(logs, encoded, initial['recipe'])
    if summary['supervised_tokens_seen'] != seen:
        raise ValueError('训练总token不符')
    before, after = (
        file_hash(training / name / 'adapter_model.safetensors')
        for name in ('initial-adapter', 'adapter')
    )
    if before != summary['before_sha256'] or after != summary['adapter_sha256'] or before == after:
        raise ValueError('没有对应真实adapter变化')
    source = read(candidate_model / 'source.json')
    if (
        source.get('kind') != 'workflow_stage1_replay_candidate'
        or source['parent_artifact_digest'] != C0_DIGEST
        or source['adapter_sha256'] != after
        or source['dataset_sha256'] != manifest['data_sha256']
        or source['recipe'] != N1_RECIPE
    ):
        raise ValueError('合并权重来源不符')
    parameter = parameter_identity(candidate_model)
    if parameter['digest'] == '040bf1295cf8a6198df7888534b313ae3bc049b8c2c715bf81bf4579669e37bd':
        raise ValueError('实际参数仍为C0，不能凑第二版本')
    return {
        'training': str(training),
        'dataset': str(dataset),
        'parent_model': str(parent_model),
        'source_root': str(source_root),
        'bootstrap_sha256': file_hash(training.parent / 'bootstrap.json'),
        'bundle_manifest_sha256': file_hash(owner / 'training-bundle-manifest.json'),
        'input_sha256': file_hash(training / 'input.json'),
        'summary_sha256': file_hash(training / 'summary.json'),
        'log_sha256': file_hash(training / 'training.jsonl'),
        'dataset_manifest_sha256': file_hash(dataset / 'manifest.json'),
        'artifact_digest': digest(artifact['files']),
        'parameter_digest': parameter['digest'],
        'parent_artifact_digest': C0_DIGEST,
        'optimizer_steps': len(logs),
        'supervised_tokens': seen,
        'adapter_sha256': after,
        'complete': True,
    }
