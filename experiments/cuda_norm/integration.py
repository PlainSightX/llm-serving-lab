"""真实Qwen前向中的默认、成熟CUDA与实验CUDA；生成原件与插桩窗口分开。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from evidence import check_full, sha


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def execute(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    model = Path(args.model)
    from serving_lab.lab import inspect_base, digest

    identity = inspect_base(model)
    if (
        digest(identity['files'])
        != 'b45fba1ab01819bd9540b73858cd32e109bfeefed07ac36b5ada9248650dbba2'
    ):
        raise ValueError('N3只接受已核BF16原始工件')
    if args.variant == 'custom':
        check_full(args.library, args.build_receipt, args.verification)
        build = json.loads(Path(args.build_receipt).read_text(encoding='utf-8'))
        if hashlib.sha256(Path(args.library).read_bytes()).hexdigest() != build['library_sha256']:
            raise ValueError('编译产物身份变化')
        os.environ.update(
            LSL_N3_ENABLE='1',
            LSL_N3_LIBRARY=args.library,
            LSL_N3_LIBRARY_SHA256=build['library_sha256'],
            LSL_N3_AUDIT_DIR=str(out / 'plugin'),
        )
        os.environ['VLLM_PLUGINS'] = 'serving_lab_n3'
    else:
        os.environ['VLLM_PLUGINS'] = ''
    if args.profile:
        os.environ['VLLM_TORCH_CUDA_PROFILE'] = '1'
    from vllm import LLM, SamplingParams

    config = {
        'model': str(model),
        'dtype': 'bfloat16',
        'max_model_len': 4096,
        'max_num_seqs': 8,
        'max_num_batched_tokens': 4096,
        'kv_cache_memory_bytes': 4294967296,
        'gpu_memory_utilization': 0.85,
        'enable_prefix_caching': False,
        'enforce_eager': False,
        'seed': 0,
        'trust_remote_code': False,
        'compilation_config': {
            'custom_ops': ['none'] + (['+rms_norm'] if args.variant != 'default' else [])
        },
    }
    write(
        out / 'input.json',
        {
            'variant': args.variant,
            'engine_args': config,
            'model_artifact_digest': digest(identity['files']),
            'plan_sha256': hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'profile_after_measurement': args.profile,
            'warmup_generations': 4,
            'measured_generations': 16,
            'additional_profile_generations': 16 if args.profile else 0,
        },
    )
    llm = LLM(**config)
    groups = {
        name: [s for s in plan['samples'] if s['shape'] == name]
        for name in ('short', 'prefill', 'decode')
    }
    batches = {name: groups[name][:4] for name in groups}
    batches['heldout_mixed'] = [
        groups['prefill'][4],
        groups['decode'][4],
        groups['prefill'][5],
        groups['decode'][5],
    ]

    def one(label, items):
        params = [
            SamplingParams(
                temperature=0, top_p=1, seed=0, max_tokens=s['body']['max_tokens'], ignore_eos=True
            )
            for s in items
        ]
        start = time.monotonic()
        results = llm.generate(
            [{'prompt_token_ids': s['input_token_ids']} for s in items], params, use_tqdm=False
        )
        elapsed = time.monotonic() - start
        rows = []
        for sample, result in zip(items, results):
            item = result.outputs[0]
            assert (
                item.finish_reason == 'length'
                and len(item.token_ids) == sample['body']['max_tokens']
            )
            rows.append(
                {
                    'sample': sample['id'],
                    'prompt_token_ids': result.prompt_token_ids,
                    'output_token_ids': list(item.token_ids),
                    'text': item.text,
                    'finish_reason': item.finish_reason,
                }
            )
        write(out / (label + '.json'), {'rows': rows, 'seconds': elapsed})
        print(json.dumps({'label': label, 'seconds': elapsed, 'cases': len(rows)}), flush=True)

    one('warmup', groups['short'][:4])
    for label, items in batches.items():
        one(label, items)
    if args.profile:
        llm.start_profile()
        try:
            for label, items in batches.items():
                one('profile-' + label, items)
        finally:
            llm.stop_profile()
    write(
        out / 'completed.json',
        {
            'completed': True,
            'actual_generations': 36 if args.profile else 20,
            'variant': args.variant,
            'system_gain_claim': False,
        },
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['default', 'mature', 'custom'], required=True)
    for n in ('model', 'plan', 'output'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--library')
    p.add_argument('--build-receipt')
    p.add_argument('--verification')
    p.add_argument('--profile', action='store_true')
    execute(p.parse_args())
