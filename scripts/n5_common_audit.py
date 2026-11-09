"""独立核共享4B基础适配：实际源码、原train重建、完整更新序列及真实数值变化。"""

import argparse
import gc
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(args):
    source = Path(args.source).resolve()
    training = Path(args.training).resolve()
    dataset = Path(args.dataset).resolve()
    sys.path[:0] = [str(source / 'src'), str(source / 'scripts')]
    from serving_lab.common_start import (
        validated_examples,
        COMMON_RECIPE,
        BASE_ARTIFACT,
        BASE_PARAMETER,
        check_prerequisites,
    )
    from serving_lab.execution import write_once
    from serving_lab.lab import file_hash, inspect_base, digest
    from serving_lab.parameters import parameter_identity
    from serving_lab.training_evidence import check_training_log
    from n4_compare import tensor_metrics
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    torch.set_num_threads(4)
    initial = read(training / 'input.json')
    summary = read(training / 'summary.json')
    bootstrap = read(training.parent / 'bootstrap.json')
    if (
        any(
            (training.parent / name).exists() for name in ['failure.json', 'bootstrap-failure.json']
        )
        or (training / 'failure.json').exists()
    ):
        raise ValueError('有失败标记，不能作为完整共同起点')
    bundle = read(args.bundle)
    if bootstrap['bundle_sha256'] != file_hash(args.bundle) or bootstrap[
        'launcher_sha256'
    ] != file_hash(args.launcher):
        raise ValueError('启动身份未绑定实际包')
    for name, sha in bundle.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or file_hash(path) != sha:
            raise ValueError('冻结执行包字节不同')
    if (
        bootstrap['launcher_sha256']
        != bundle['artifacts/n5-20261106-a/common-bootstrap-a/entry.py']
    ):
        raise ValueError('实际启动器不等于包内副本')
    if (
        initial['trainer_sha256'] != file_hash(training / 'trainer.py')
        or initial['trainer_sha256'] != bundle['src/serving_lab/coldstart_train.py']
    ):
        raise ValueError('实际训练器不符')
    if (
        initial['mode'] != 'n5_common_basic'
        or initial['recipe'] != COMMON_RECIPE
        or initial['optimizer_steps'] != 447
        or summary['purpose'] != 'n5_shared_basic_4b_candidate'
        or summary['optimizer_steps'] != 447
        or summary['epochs'] != 3
        or summary['samples'] != 1186
        or initial['versions'] != summary['versions']
    ):
        raise ValueError('共享基础配方/完成步数不符')
    parent = inspect_base(args.parent)
    if (
        digest(parent['files']) != BASE_ARTIFACT
        or initial['base']['files'] != parent['files']
        or summary['base_artifact_digest'] != BASE_ARTIFACT
    ):
        raise ValueError('训练父参数不是原4B')
    tokenizer = AutoTokenizer.from_pretrained(
        args.parent, local_files_only=True, trust_remote_code=False
    )
    manifest, rows, examples = validated_examples(dataset, args.parent, tokenizer, parent)
    if initial['data'] != manifest or initial['dataset_manifest_sha256'] != file_hash(
        dataset / 'manifest.json'
    ):
        raise ValueError('实际消费的数据合同不同')
    if initial['base_binding_sha256'] != summary['base_binding_sha256'] or initial[
        'base_binding_sha256'
    ] != file_hash(dataset / 'base-binding.json'):
        raise ValueError('模型/数据绑定不同')
    encoded = [
        {
            'task_id': row['task_id'],
            'step': row['step'],
            'tokens': len(ids),
            'supervised_tokens': sum(v != -100 for v in labels),
            'ids_sha256': digest(ids),
            'labels_sha256': digest(labels),
        }
        for row, (ids, labels, _) in zip(rows, examples)
    ]
    if encoded != initial['encoded_examples']:
        raise ValueError('实际输入或监督mask不同')
    logs = [
        json.loads(line)
        for line in (training / 'training.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    seen = check_training_log(logs, encoded, COMMON_RECIPE)
    if (
        summary['supervised_tokens_seen'] != seen
        or summary['first_loss'] != logs[0]['mean_loss']
        or summary['last_loss'] != logs[-1]['mean_loss']
    ):
        raise ValueError('完整训练日志与摘要不同')
    for key in ['adapter_reload_max_difference', 'merged_reload_max_difference']:
        if not math.isfinite(summary[key]) or not 0 <= summary[key] <= 1e-3:
            raise ValueError('保存重载检查未通过')
    before = file_hash(training / 'initial-adapter/adapter_model.safetensors')
    after = file_hash(training / 'adapter/adapter_model.safetensors')
    if before == after or before != summary['before_sha256'] or after != summary['adapter_sha256']:
        raise ValueError('adapter更新未匹配')
    from safetensors.torch import load_file

    adapter_before = load_file(training / 'initial-adapter/adapter_model.safetensors')
    adapter_after = load_file(training / 'adapter/adapter_model.safetensors')
    adapter_numeric = tensor_metrics(adapter_before, adapter_after)
    if not adapter_numeric['changed_tensors'] or adapter_numeric['relative_l2'] == 0:
        raise ValueError('adapter实际张量没有更新')
    del adapter_before, adapter_after
    candidate = inspect_base(training / 'merged')
    parameter = parameter_identity(training / 'merged')
    if candidate['files'] != summary['merged_identity']['files']:
        raise ValueError('最终消费工件不是本次输出')
    provenance = read(training / 'merged/source.json')
    if (
        provenance['kind'] != 'n5_shared_basic_4b_candidate'
        or provenance['parent_artifact_digest'] != BASE_ARTIFACT
        or provenance['adapter_sha256'] != after
        or provenance['dataset_sha256'] != manifest['data_sha256']
        or provenance['recipe'] != COMMON_RECIPE
    ):
        raise ValueError('合并来源不符')
    # 使用相同HF结构加载两者，避免把tied权重重新存储或dtype元数据变化当成学习。
    kwargs = {
        'local_files_only': True,
        'trust_remote_code': False,
        'dtype': torch.bfloat16,
        'attn_implementation': 'sdpa',
    }
    parent_model = AutoModelForCausalLM.from_pretrained(args.parent, **kwargs)
    candidate_model = AutoModelForCausalLM.from_pretrained(training / 'merged', **kwargs)
    reference = parent_model.state_dict()
    actual = candidate_model.state_dict()
    numeric = tensor_metrics(reference, actual)
    if not numeric['changed_tensors'] or numeric['relative_l2'] == 0:
        raise ValueError('实际完整参数没有数值更新')
    for name, value in reference.items():
        if not any(
            name.endswith('.' + target + '.weight') for target in COMMON_RECIPE['target_modules']
        ):
            if not torch.equal(value, actual[name]):
                raise ValueError('非LoRA目标参数发生改变: ' + name)
    del reference, parent_model
    gc.collect()
    # 按训练器原BF16/GPU路径，从本次adapter重建合并结果；先冻结逐bit一致门，不事后放宽。
    if not torch.cuda.is_available():
        raise ValueError('实际合并路径重建需要同类CUDA设备')
    from peft import PeftModel

    loaded = (
        PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.parent, **kwargs),
            training / 'adapter',
            local_files_only=True,
        )
        .to('cuda')
        .eval()
    )
    rebuilt = loaded.merge_and_unload(safe_merge=True).eval()
    rebuilt_state = {name: tensor.detach().cpu() for name, tensor in rebuilt.state_dict().items()}
    merge_equality = tensor_metrics(actual, rebuilt_state)
    if merge_equality['max_abs'] != 0:
        raise ValueError('最终工件不等于本次adapter按原路径的实际合并结果')
    if bootstrap['prerequisites'] != check_prerequisites(dataset / 'prerequisite'):
        raise ValueError('训练启动没有绑定两套原失败')
    result = {
        'passed': True,
        'artifact_digest': digest(candidate['files']),
        'parameter_digest': parameter['digest'],
        'parent_artifact_digest': BASE_ARTIFACT,
        'parent_parameter_digest': BASE_PARAMETER,
        'optimizer_steps': 447,
        'supervised_tokens': seen,
        'training_tasks': 48,
        'samples': 1186,
        'epochs': 3,
        'numeric_delta': numeric,
        'adapter_numeric_delta': adapter_numeric,
        'reconstructed_merge': merge_equality,
        'merge_tolerance': 'exact original BF16 CUDA merge path',
        'non_target_parameters_unchanged': True,
        'unique_parameters': sum(p.numel() for p in candidate_model.parameters()),
        'elapsed_seconds': summary['elapsed_seconds'],
        'peak_allocated_bytes': summary['peak_allocated_bytes'],
        'summary_sha256': file_hash(training / 'summary.json'),
        'input_sha256': file_hash(training / 'input.json'),
        'dataset_manifest_sha256': file_hash(dataset / 'manifest.json'),
        'bundle_sha256': file_hash(args.bundle),
        'source_sha256': file_hash(__file__),
        'research_training': False,
        'quality_status': 'not_evaluated',
        'scope': 'one shared basic-action initialization; no workflow/dev/future supervision or continual-learning benefit claim',
    }
    write_once(args.output, result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['source', 'training', 'dataset', 'parent', 'bundle', 'launcher', 'output']:
        p.add_argument('--' + name, required=True)
    audit(p.parse_args())
