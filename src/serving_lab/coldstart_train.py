"""一次固定配方的公开教师LoRA冷启动；仅训练合格动作，产物不自动发布。"""

import argparse
import gc
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

from .execution import write_once
from .lab import canonical, digest, file_hash, inspect_base

RECIPE = {
    'seed': 20261103,
    'epochs': 3,
    'micro_batch': 1,
    'accumulation': 8,
    'learning_rate': 2e-4,
    'weight_decay': 0.0,
    'warmup_ratio': 0.05,
    'scheduler': 'cosine',
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'max_length': 3564,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'precision': 'bfloat16',
    'gradient_checkpointing': True,
    'gradient_clip_norm': 1.0,
    'projection': 'supervised action tail plus preceding context token; full transformer context retained',
    'selection': 'final epoch only; no development-driven checkpoint selection',
}


def encode_example(row, tokenizer, max_length=3564):
    messages = row['messages']
    if len(messages) < 2 or messages[-1]['role'] != 'assistant':
        raise ValueError('缺少动作监督')
    action = messages[-1]['content']
    if (
        not isinstance(action, str)
        or not action
        or len(action) > 256
        or '\n' in action
        or '\r' in action
    ):
        raise ValueError('监督不是单个动作')
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    target = tokenizer.encode(action, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(target) > max_length or len(target) > 64:
        raise ValueError('监督超预算，禁止隐式截断')
    return prefix + target, [-100] * len(prefix) + target, prefix


def supervised_tail(labels):
    """保留预测首个动作的前缀末位；只省去loss恒为零的词表投影。"""
    first = next((i for i, v in enumerate(labels) if v != -100), None)
    if first is None or first == 0 or any(v == -100 for v in labels[first:]):
        raise ValueError('要求非空前缀及连续动作/EOS监督')
    keep = len(labels) - first + 1
    return keep, labels[-keep:]


def validated_examples(dataset, model, tokenizer, base):
    """训练前重建全部监督，拒绝仅自称合格的JSONL或把dev混入的快照。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from q2_teacher import verify_collection, read
    from .teacher_data import action_samples, validate_base_binding

    dataset = Path(dataset).resolve()
    manifest = read(dataset / 'manifest.json')
    validate_base_binding(
        read(dataset / 'base-binding.json'), base, file_hash(dataset / 'manifest.json')
    )
    if (
        manifest.get('name') != 'q2-action-supervision-v1'
        or manifest.get('evidence_level') != 'teacher_coldstart_dataset'
        or file_hash(dataset / 'train.jsonl') != manifest['data_sha256']
        or manifest['dev_samples'] != 0
        or file_hash(Path(model) / 'tokenizer_config.json') != manifest['tokenizer_config_sha256']
    ):
        raise ValueError('不是合格的教师冷启动数据')
    data, collection = Path(manifest['data']), Path(manifest['collection'])
    verification = verify_collection(data, collection)
    if not verification['data_gate_passed'] or verification != manifest['source_verification']:
        raise ValueError('原始教师资格变化或未通过')
    raw_manifest = read(data / 'manifest.json')
    if (
        file_hash(data / 'manifest.json') != manifest['data_manifest_sha256']
        or raw_manifest['source_contract_digest'] != manifest['source_contract_digest']
    ):
        raise ValueError('来源合同变化')
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    expected = []
    for task in raw_manifest['tasks']:
        record = read(collection / task['task_id'] / 'record.json')
        if task['role'] == 'train' and record['result']['success']:
            expected.extend(action_samples(record, tokenize, manifest['source_contract_digest']))
    rows = [
        json.loads(line)
        for line in (dataset / 'train.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    if not rows or rows != expected or len(rows) != manifest['count']:
        raise ValueError('监督与原始因果前缀不一致')
    return manifest, rows, [encode_example(row, tokenizer, RECIPE['max_length']) for row in rows]


def train_once(dataset, model_path, output, *, mode='coldstart'):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        set_seed,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, PeftModel, get_peft_model

    recipe, validate = RECIPE, validated_examples
    source_kind = 'teacher_coldstart_merged_candidate'
    purpose = 'teacher_coldstart_candidate'
    if mode == 'workflow_stage1':
        from .workflow_training import N1_RECIPE, validated_examples as validate

        recipe = N1_RECIPE
        source_kind = purpose = 'workflow_stage1_replay_candidate'
    elif mode == 'n5_common_basic':
        from .common_start import COMMON_RECIPE, validated_examples as validate

        recipe = COMMON_RECIPE
        source_kind = purpose = 'n5_shared_basic_4b_candidate'
    elif mode != 'coldstart':
        raise ValueError('未知训练合同')
    model_path, dataset, output = (
        Path(model_path).resolve(),
        Path(dataset).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise ValueError('训练输出必须为新身份，禁止覆盖或隐式恢复')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        raise ValueError('本地CUDA或8GiB空闲显存资格未过，不回退CPU')
    versions = {
        n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate')
    }
    if versions != {
        'torch': '2.9.0+cu128',
        'transformers': '4.57.1',
        'peft': '0.17.1',
        'accelerate': '1.10.1',
    }:
        raise ValueError('使用既有锁定trainer环境')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    base = inspect_base(model_path)
    manifest, rows, examples = validate(dataset, model_path, tokenizer, base)
    output.mkdir(parents=True)
    total_steps = math.ceil(len(examples) / recipe['accumulation']) * recipe['epochs']
    write_once(
        output / 'input.json',
        {
            'recipe': recipe,
            'mode': mode,
            'base': base,
            'data': manifest,
            'dataset_manifest_sha256': file_hash(dataset / 'manifest.json'),
            'base_binding_sha256': file_hash(dataset / 'base-binding.json'),
            'optimizer_steps': total_steps,
            'versions': versions,
            'trainer_sha256': file_hash(Path(__file__)),
            'encoded_examples': [
                {
                    'task_id': r['task_id'],
                    'step': r['step'],
                    'tokens': len(ids),
                    'supervised_tokens': sum(v != -100 for v in labels),
                    'ids_sha256': digest(ids),
                    'labels_sha256': digest(labels),
                }
                for r, (ids, labels, _) in zip(rows, examples)
            ],
        },
    )
    (output / 'trainer.py').write_bytes(Path(__file__).read_bytes())
    set_seed(recipe['seed'])
    started = time.monotonic()
    kwargs = {
        'local_files_only': True,
        'trust_remote_code': False,
        'dtype': torch.bfloat16,
        'attn_implementation': 'sdpa',
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to('cuda')
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type='CAUSAL_LM',
            target_modules=recipe['target_modules'],
        ),
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.save_pretrained(output / 'initial-adapter')
    before = file_hash(output / 'initial-adapter/adapter_model.safetensors')
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=recipe['learning_rate'],
        weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, math.ceil(0.05 * total_steps), total_steps
    )
    torch.cuda.reset_peak_memory_stats()
    update = 0
    seen_tokens = 0
    losses = []
    with (output / 'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(recipe['epochs']):
            order = list(range(len(examples)))
            random.Random(recipe['seed'] + epoch).shuffle(order)
            model.train()
            for offset in range(0, len(order), recipe['accumulation']):
                batch = order[offset : offset + recipe['accumulation']]
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for index in batch:
                    ids, labels, _ = examples[index]
                    keep, target_labels = supervised_tail(labels)
                    x, y = (
                        torch.tensor([ids], device='cuda'),
                        torch.tensor([target_labels], device='cuda'),
                    )
                    loss = model(
                        input_ids=x,
                        attention_mask=torch.ones_like(x),
                        labels=y,
                        logits_to_keep=keep,
                        use_cache=False,
                    ).loss
                    if not torch.isfinite(loss):
                        raise ValueError('非有限loss，停止且保留原件')
                    (loss / len(batch)).backward()
                    total_loss += float(loss.detach())
                    seen_tokens += sum(v != -100 for v in labels)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError('非有限梯度，停止')
                optimizer.step()
                scheduler.step()
                update += 1
                row = {
                    'update': update,
                    'epoch': epoch + 1,
                    'samples': batch,
                    'mean_loss': total_loss / len(batch),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'supervised_tokens_seen': seen_tokens,
                    'elapsed_seconds': time.monotonic() - started,
                }
                losses.append(row['mean_loss'])
                log.write(canonical(row) + '\n')
                log.flush()
                if update == 1 or update % 10 == 0 or update == total_steps:
                    print(canonical({k: v for k, v in row.items() if k != 'samples'}), flush=True)
    assert update == total_steps
    model.gradient_checkpointing_disable()
    model.eval()
    probe = torch.tensor([examples[0][2]], device='cuda')
    with torch.no_grad():
        reference = (
            model(input_ids=probe, attention_mask=torch.ones_like(probe), logits_to_keep=1)
            .logits[0, -1]
            .float()
            .cpu()
        )
    model.save_pretrained(output / 'adapter')
    after = file_hash(output / 'adapter/adapter_model.safetensors')
    if before == after:
        raise ValueError('未观察到参数字节更新')
    peak = torch.cuda.max_memory_allocated()
    del model, optimizer, scheduler, x, y, loss
    gc.collect()
    torch.cuda.empty_cache()
    loaded = (
        PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model_path, **kwargs),
            output / 'adapter',
            local_files_only=True,
        )
        .to('cuda')
        .eval()
    )
    with torch.no_grad():
        actual = (
            loaded(input_ids=probe, attention_mask=torch.ones_like(probe), logits_to_keep=1)
            .logits[0, -1]
            .float()
            .cpu()
        )
    reload_error = float((reference - actual).abs().max())
    if not torch.allclose(reference, actual, rtol=1e-3, atol=1e-3):
        raise ValueError('adapter重载不一致')
    merged = loaded.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        merged_ref = (
            merged(input_ids=probe, attention_mask=torch.ones_like(probe), logits_to_keep=1)
            .logits[0, -1]
            .float()
            .cpu()
        )
    merge_delta = float((actual - merged_ref).abs().max())
    # BF16合并有舍入差；实际任务评价使用这个固定合并产物，不冒称动态LoRA等价。
    merged.save_pretrained(output / 'merged', safe_serialization=True)
    tokenizer.save_pretrained(output / 'merged')
    parent_source = json.loads((model_path / 'source.json').read_text())
    write_once(
        output / 'merged/source.json',
        {
            'kind': source_kind,
            'base_revision': parent_source.get('revision', parent_source.get('base_revision')),
            'parent_artifact_digest': digest(base['files']),
            'adapter_sha256': after,
            'dataset_sha256': manifest['data_sha256'],
            'recipe': recipe,
        },
    )
    del loaded, merged
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(output / 'merged', **kwargs).to('cuda').eval()
    with torch.no_grad():
        merged_actual = (
            reloaded(input_ids=probe, attention_mask=torch.ones_like(probe), logits_to_keep=1)
            .logits[0, -1]
            .float()
            .cpu()
        )
    if not torch.allclose(merged_ref, merged_actual, rtol=1e-3, atol=1e-3):
        raise ValueError('实际合并产物保存重载不一致')
    result = {
        'purpose': purpose,
        'optimizer_steps': update,
        'epochs': recipe['epochs'],
        'samples': len(examples),
        'supervised_tokens_seen': seen_tokens,
        'before_sha256': before,
        'adapter_sha256': after,
        'adapter_reload_max_difference': reload_error,
        'merge_max_difference': merge_delta,
        'merged_reload_max_difference': float((merged_ref - merged_actual).abs().max()),
        'merged_identity': inspect_base(output / 'merged'),
        'data_sha256': manifest['data_sha256'],
        'base_artifact_digest': digest(base['files']),
        'base_binding_sha256': file_hash(dataset / 'base-binding.json'),
        'elapsed_seconds': time.monotonic() - started,
        'peak_allocated_bytes': peak,
        'versions': versions,
        'first_loss': losses[0],
        'last_loss': losses[-1],
        'quality_status': 'not_evaluated',
        'publication_status': 'not_published',
    }
    write_once(output / 'summary.json', result)
    del reloaded, probe
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    try:
        print(json.dumps(train_once(a.dataset, a.model, a.output), ensure_ascii=False, indent=2))
    except BaseException as exc:
        out = Path(a.output)
        if out.exists():
            write_once(out / 'failure.json', {'error': repr(exc), 'implicit_retry': False})
        raise
