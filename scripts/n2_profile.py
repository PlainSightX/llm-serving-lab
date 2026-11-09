"""分离诊断窗口：线上 PyTorch 或独立 Nsight，不计入无剖析器性能窗口。"""

import argparse
import asyncio
import json
from pathlib import Path
import time

from serving_lab.execution import write_once
from serving_lab.lab import file_hash, digest, inspect_base


def batches(plan):
    pure = {
        shape: [s for s in plan['samples'] if s['shape'] == shape][:4]
        for shape in ('short', 'prefill', 'decode')
    }
    return {
        **pure,
        'mixed': [pure['prefill'][0], pure['decode'][0], pure['prefill'][1], pure['decode'][1]],
    }


async def online(plan, out):
    import httpx

    async with httpx.AsyncClient(
        base_url='http://127.0.0.1:8000', timeout=180, trust_env=False
    ) as client:

        async def one(sample, label, index):
            body = {**sample['native_body'], 'stream': False}
            body.pop('stream_options', None)
            rid = f"n2-profile-{plan['variant']}-{out.name}-{label}-{index}"
            started = time.monotonic()
            response = await client.post(
                '/v1/chat/completions', json=body, headers={'X-Request-Id': rid}
            )
            response.raise_for_status()
            raw = response.json()
            write_once(
                out / (label + f'-{index}.json'),
                {
                    'request': body,
                    'request_id': rid,
                    'response': raw,
                    'seconds': time.monotonic() - started,
                },
            )
            assert raw['id'] == 'chatcmpl-' + rid and raw['model'] == sample['version']
            assert (
                raw['usage']['completion_tokens'] == body['max_tokens']
                and raw['choices'][0]['finish_reason'] == 'length'
            )

        selected = batches(plan)
        await asyncio.gather(*(one(s, 'warmup', i) for i, s in enumerate(selected['short'])))
        (await client.post('/start_profile')).raise_for_status()
        try:
            for label, items in selected.items():
                await asyncio.gather(*(one(s, label, i) for i, s in enumerate(items)))
        finally:
            (await client.post('/stop_profile')).raise_for_status()


def offline(plan, model, out):
    from vllm import LLM, SamplingParams

    identity = inspect_base(model)
    if digest(identity['files']) != plan['worker_manifest']['artifact_digest']:
        raise ValueError('Nsight加载工件偏离已评价参数')
    args = {
        'model': model,
        'dtype': 'bfloat16',
        'max_model_len': 4096,
        'max_num_seqs': 8,
        'max_num_batched_tokens': 4096,
        'kv_cache_memory_bytes': 4294967296,
        'gpu_memory_utilization': 0.85,
        'enable_prefix_caching': False,
        'enforce_eager': False,
        'trust_remote_code': False,
        'seed': 0,
    }
    write_once(out / 'engine-args.json', args)
    llm = LLM(**args)

    def one(label, items):
        prompts = [{'prompt_token_ids': s['input_token_ids']} for s in items]
        params = [
            SamplingParams(
                temperature=0, top_p=1, seed=0, max_tokens=s['body']['max_tokens'], ignore_eos=True
            )
            for s in items
        ]
        start = time.monotonic()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        rows = []
        for sample, value in zip(items, outputs):
            result = value.outputs[0]
            assert (
                len(result.token_ids) == sample['body']['max_tokens']
                and result.finish_reason == 'length'
            )
            rows.append(
                {
                    'sample': sample['id'],
                    'prompt_token_ids': value.prompt_token_ids,
                    'output_token_ids': list(result.token_ids),
                    'finish_reason': result.finish_reason,
                    'text': result.text,
                }
            )
        write_once(out / (label + '.json'), {'seconds': time.monotonic() - start, 'rows': rows})

    selected = batches(plan)
    one('warmup', selected['short'])
    llm.start_profile()
    try:
        for label, items in selected.items():
            one(label, items)
    finally:
        llm.stop_profile()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['torch-online', 'nsight-offline'], required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--model')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding='utf-8'))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    write_once(
        out / 'input.json',
        {
            'mode': a.mode,
            'plan_sha256': file_hash(a.plan),
            'runner_sha256': file_hash(__file__),
            'warmup_calls': 4,
            'profiled_calls': 16,
            'scope': 'diagnostic overhead included; excluded from performance comparison',
        },
    )
    if a.mode == 'torch-online':
        asyncio.run(online(plan, out))
    else:
        offline(plan, a.model, out)
    write_once(
        out / 'completed.json',
        {'completed': True, 'actual_generations': 20, 'profiled_generations': 16},
    )
