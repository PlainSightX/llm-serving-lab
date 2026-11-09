"""真实LoRA训练资格入口；依赖与预训练模型必须显式准备，不进行伪后端回退。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from .lab import LabError, canonical, file_hash, inspect_base


def validate_export(directory, purpose):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    data = directory / 'train.jsonl'
    if file_hash(data) != manifest.get('data_sha256'):
        raise LabError('训练数据与快照不符')
    if purpose == 'research' and manifest.get('evidence_level') != 'research_dataset':
        raise LabError('控制面或模型资格数据不可作为正式研究训练材料')
    rows = [json.loads(s) for s in data.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows or len(rows) != manifest.get('count'):
        raise LabError('训练数据为空或数量不符')
    for row in rows:
        m = row.get('messages', [])
        if len(m) < 2 or m[-1].get('role') != 'assistant' or not m[-1].get('content'):
            raise LabError('没有有效assistant监督')
    return manifest, rows


def train(
    model_path,
    export_path,
    output,
    *,
    steps=8,
    max_length=512,
    purpose='model_smoke',
    device='cuda',
):
    if purpose not in {'model_smoke', 'research'} or steps <= 0:
        raise LabError('必须声明用途及正数优化步数')
    output = Path(output).resolve()
    if output.exists():
        raise LabError('训练输出已存在；不可覆盖历史工件')
    model_path = Path(model_path).resolve()
    if not (model_path / 'config.json').is_file():
        raise LabError('需要已准备的本地预训练模型目录')
    manifest, rows = validate_export(export_path, purpose)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
        from peft import LoraConfig, get_peft_model, PeftModel
    except ImportError as exc:
        raise LabError('请在environments/trainer独立锁环境运行训练') from exc
    if device == 'cuda' and not torch.cuda.is_available():
        raise LabError('CUDA不可用；不自动回退CPU')
    if device == 'cuda' and torch.cuda.mem_get_info()[0] < 4 * 1024**3:
        raise LabError('可用显存不足4GiB，停止本次资格运行')
    set_seed(42)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    examples = []
    for row in rows:
        prefix = tokenizer.apply_chat_template(
            row['messages'][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        target = tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(prefix) + len(target) > max_length:
            raise LabError('样本超出长度上限；禁止暗中截断监督')
        examples.append((prefix + target, [-100] * len(prefix) + target, prefix))
    if not any(any(x != -100 for x in y) for _, y, _ in examples):
        raise LabError('没有有效监督token')
    output.mkdir(parents=True)
    started = time.time()
    base_files = inspect_base(model_path)['files']
    base_kwargs = {
        'local_files_only': True,
        'trust_remote_code': False,
        'dtype': dtype,
        'attn_implementation': 'sdpa',
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **base_kwargs).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            task_type='CAUSAL_LM',
            target_modules=['q_proj', 'v_proj'],
        ),
    )
    model.save_pretrained(output / 'initial')
    before = file_hash(output / 'initial/adapter_model.safetensors')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    losses = []
    supervised = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(steps):
        ids, labels, _ = examples[step % len(examples)]
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=x, labels=y, use_cache=False).loss
        if not torch.isfinite(loss):
            raise LabError('训练loss非有限，停止')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        supervised += sum(v != -100 for v in labels)
        print(json.dumps({'step': step + 1, 'loss': losses[-1]}, ensure_ascii=False), flush=True)
    model.eval()
    probe = torch.tensor([examples[0][2]], device=device)
    with torch.no_grad():
        reference = (
            model(input_ids=probe, attention_mask=torch.ones_like(probe))
            .logits[0, -1]
            .float()
            .cpu()
        )
    adapter = output / 'adapter'
    model.save_pretrained(adapter)
    after = file_hash(adapter / 'adapter_model.safetensors')
    if before == after:
        raise LabError('未观察到参数字节更新')
    peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    del model, optimizer, x, y, loss
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    loaded = (
        PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model_path, **base_kwargs),
            adapter,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad():
        actual = (
            loaded(input_ids=probe, attention_mask=torch.ones_like(probe))
            .logits[0, -1]
            .float()
            .cpu()
        )
        generated = loaded.generate(
            input_ids=probe,
            attention_mask=torch.ones_like(probe),
            use_model_defaults=False,
            generation_config=GenerationConfig(
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ),
        )
    error = float((reference - actual).abs().max())
    if not torch.allclose(reference, actual, rtol=1e-3, atol=1e-3):
        raise LabError(f'重载logits不一致: {error}')
    receipt = {
        'producer': 'serving_lab.train.v1',
        'purpose': purpose,
        'steps': steps,
        'supervised_tokens': supervised,
        'base_path': str(model_path),
        'base_files': base_files,
        'data_sha256': manifest['data_sha256'],
        'data_evidence_level': manifest['evidence_level'],
        'before_sha256': before,
        'adapter_sha256': after,
        'reload_verified': True,
        'reload_max_logit_difference': error,
        'losses': losses,
        'generated_smoke_text': tokenizer.decode(
            generated[0, probe.shape[1] :], skip_special_tokens=True
        ),
        'device': device,
        'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
        'peak_allocated_bytes': peak,
        'elapsed_seconds': time.time() - started,
        'seed': 42,
        'versions': {
            name: importlib.metadata.version(name)
            for name in ('torch', 'transformers', 'peft', 'accelerate')
        },
        'trainer_source_sha256': file_hash(Path(__file__)),
        'limits': '有限步真实参数/保存重载资格，不是业务泛化、旧技能保持或持续学习效果证明。',
    }
    (adapter / 'training-receipt.json').write_text(canonical(receipt) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return receipt


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='真实PEFT权重更新与重载资格验证')
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--purpose', choices=['model_smoke', 'research'], default='model_smoke')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    try:
        result = train(
            a.model,
            a.dataset,
            a.output,
            steps=a.steps,
            max_length=a.max_length,
            purpose=a.purpose,
            device=a.device,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LabError, OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
