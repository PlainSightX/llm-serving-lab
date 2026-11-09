"""固定公开训练输入与0.6B工件；N4不消费GSM8K test或N5未来任务。"""

import argparse
import json
from pathlib import Path
from serving_lab.execution import write_once
from serving_lab.lab import inspect_base, digest, file_hash


def main(args):
    from transformers import AutoTokenizer

    model = Path(args.model).resolve()
    source = Path(args.train).resolve()
    out = Path(args.output).resolve()
    if file_hash(source) != '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465':
        raise ValueError('不是固定GSM8K train文件')
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 7473
    base = inspect_base(model)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    records = []
    for index, row in enumerate(rows[128:256], 128):
        text = 'Question: ' + row['question'] + '\nAnswer: ' + row['answer']
        ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(ids) < 8:
            raise ValueError('样本过短')
        records.append(
            {
                'id': f'gsm8k-train-{index:04}',
                'source_index': index,
                'text_sha256': digest(text),
                'full_tokens': len(ids),
                'input_ids': ids[:512],
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'samples.json', records)
    plan = {
        'name': 'n4-distributed-v2',
        'source': 'GSM8K train commit3101c7d5072418e28b9008a6636bde82a006892c',
        'source_sha256': file_hash(source),
        'samples_sha256': file_hash(out / 'samples.json'),
        'samples': len(records),
        'model_artifact_digest': digest(base['files']),
        'model_files': base['files'],
        'model_source': json.loads((model / 'source.json').read_text()),
        'tokenizer_eos': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'seed': 20261106,
        'global_batch': 4,
        'micro_batch': 1,
        'dropout': 0,
        'gradient_checkpointing': False,
        'correctness': {
            'precision': 'fp32',
            'optimizer': 'sgd',
            'lr': 1e-3,
            'sequence_length': 128,
            'steps': 8,
            'loss_relative_tolerance': 1e-5,
            'final_parameter_relative_l2_tolerance': 1e-6,
            'final_parameter_max_abs_tolerance': 1e-6,
        },
        'performance': {
            'precision': 'bf16_autocast_with_fp32_parameters',
            'optimizer': 'adamw',
            'lr': 1e-5,
            'betas': [0.9, 0.95],
            'eps': 1e-8,
            'weight_decay': 0.01,
            'sequence_length': 512,
            'warmup_steps': 4,
            'measured_steps': 16,
        },
        'weak_scaling': {
            'global_batch_per_gpu': 4,
            'warmup_steps': 4,
            'measured_steps': 12,
            'meaning': 'separate fixed work per GPU; never substitute for strong-scaling result',
        },
        'recovery': {
            'precision': 'bf16_autocast_with_fp32_parameters',
            'optimizer': 'adamw',
            'steps': 12,
            'checkpoint_step': 4,
            'fault_after_step': 6,
            'loss_relative_tolerance': 1e-5,
            'final_parameter_relative_l2_tolerance': 1e-6,
            'final_parameter_max_abs_tolerance': 1e-6,
        },
        'communication': {
            'dtype': 'float32',
            'bytes': [4096, 1048576, 16777216, 67108864, 134217728],
            'warmup': 5,
            'iterations': 20,
        },
        'invariants': [
            'same samples and global valid-token denominator',
            'rank-exclusive actual GPU identities',
            'new run directories',
            'default sharding, gradient synchronization only on last microstep',
            'no unqualified application release',
        ],
        'topologies': ['single_gpu', 'two_node', 'single_node'],
        'future_evaluation_accessed': False,
    }
    write_once(out / 'plan.json', plan)
    print(
        json.dumps(
            {
                'samples': len(records),
                'model_digest': plan['model_artifact_digest'],
                'plan_sha256': file_hash(out / 'plan.json'),
            }
        )
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('model', 'train', 'output'):
        p.add_argument('--' + n, required=True)
    main(p.parse_args())
