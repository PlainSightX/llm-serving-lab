"""单次 AWQ 校准：只消费冻结训练面板，候选是否合格由独立评价决定。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash, inspect_base


def calibration_rows(plan_path, calibration_path):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if file_hash(calibration_path) != plan['files']['calibration']:
        raise ValueError('校准文件偏离冻结面板')
    rows = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    if len(rows) != 128 or [r['id'] for r in rows] != [f'train-{i:04}' for i in range(128)]:
        raise ValueError('只接受冻结的128条train校准记录')
    return rows


def run(args):
    rows = calibration_rows(args.plan, args.calibration)
    base = inspect_base(args.model)
    if digest(base['files']) != args.expected_base:
        raise ValueError('校准基座与已评价的BF16身份不同')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__)
    (output / 'runner.py').write_bytes(source.read_bytes())
    from datasets import Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    data = []
    tokens = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        tokens.append({'id': row['id'], 'full_length': len(ids), 'input_ids': ids[:512]})
        data.append({'input_ids': ids[:512], 'attention_mask': [1] * min(len(ids), 512)})
    write_once(output / 'calibration-tokens.json', tokens)
    recipe = AWQModifier(
        ignore=['lm_head'], scheme='W4A16_ASYM', targets=['Linear'], duo_scaling='both'
    )
    write_once(
        output / 'input.json',
        {
            'base': base,
            'base_digest': args.expected_base,
            'plan_sha256': file_hash(args.plan),
            'calibration_sha256': file_hash(args.calibration),
            'runner_sha256': file_hash(source),
            'recipe': recipe.model_dump(mode='json'),
            'count': 128,
            'max_seq_length': 512,
            'seed': 0,
            'truncated_calibration': sum(t['full_length'] > 512 for t in tokens),
            'versions': {
                n: importlib.metadata.version(n)
                for n in (
                    'torch',
                    'transformers',
                    'llmcompressor',
                    'compressed-tensors',
                    'datasets',
                )
            },
            'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0),
            'evaluation_accessed': False,
        },
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=Dataset.from_list(data),
        recipe=[recipe],
        max_seq_length=512,
        num_calibration_samples=128,
        shuffle_calibration_samples=False,
        pad_to_max_length=False,
        concatenate_data=False,
    )
    destination = output / 'model'
    model.save_pretrained(destination, save_compressed=True)
    tokenizer.save_pretrained(destination)
    write_once(
        destination / 'source.json',
        {
            'parent_artifact_digest': args.expected_base,
            'quantization_input_sha256': file_hash(output / 'input.json'),
            'calibration_tokens_sha256': file_hash(output / 'calibration-tokens.json'),
        },
    )
    config = json.loads((destination / 'config.json').read_text(encoding='utf-8'))
    quant = config['quantization_config']
    if quant['quant_method'] != 'compressed-tensors':
        raise ValueError('未生成预期压缩工件')
    groups = quant['config_groups']
    if not groups or any(
        g['weights']['num_bits'] != 4 or g['weights']['group_size'] != 128 for g in groups.values()
    ):
        raise ValueError('实际量化不是4bit/group128')
    candidate = inspect_base(destination)
    result = {
        'completed': True,
        'artifact_digest': digest(candidate['files']),
        'files': candidate['files'],
        'quantization_config': quant,
        'elapsed_seconds': time.monotonic() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'quality_status': 'not_evaluated',
    }
    write_once(output / 'result.json', result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ('files', 'quantization_config')}),
        flush=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'plan', 'calibration', 'expected-base', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
