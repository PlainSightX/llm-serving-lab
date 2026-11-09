"""N2固定token负载：独立发生器上的并发试点和三入口open-loop比较。"""

import argparse
import asyncio
import json
from pathlib import Path
import time
import httpx
from serving_lab.benchmark import StreamObservation, window
from serving_lab.inference_study import summarize_fixed, require_quality_proof
from serving_lab.execution import write_once
from serving_lab.lab import file_hash, digest
from serving_lab.profile import ServingProfile
from serving_lab.tls import client_context


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(tokenizer_path, engine_config, quality, output, *, diagnostic_rejected=False):
    from transformers import AutoTokenizer

    config = read(engine_config)
    quality_result = read(quality)
    require_quality_proof(
        quality_result, config['manifest'], diagnostic_rejected=diagnostic_rejected
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    profile = ServingProfile(config['manifest']['profile'], config['manifest']['version_id'])
    samples = []
    for shape, target, budget in [
        ('short', 256, 128),
        ('prefill', 3072, 128),
        ('decode', 512, 512),
    ]:
        for i in range(8):
            count = target
            for _ in range(8):
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {
                        'role': 'user',
                        'content': f'Request {i}: Context notes:'
                        + ' data' * count
                        + '\nExplain how database indexes work in detail. Keep elaborating.',
                    },
                ]
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                if len(ids) == target:
                    break
                count += target - len(ids)
            if len(ids) != target:
                raise ValueError('无法取得固定输入长度')
            body = {
                'model': 'active',
                'messages': messages,
                'max_tokens': budget,
                'stream': True,
                'ignore_eos': True,
            }
            samples.append(
                {
                    'id': shape + '-' + str(i),
                    'shape': shape,
                    'body': body,
                    'native_body': profile.payload(body),
                    'input_token_ids': ids,
                    'version': config['manifest']['version_id'],
                    'profile_digest': profile.digest,
                }
            )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        'name': 'n2-fixed-tokens-v1',
        'variant': quality_result['variant'],
        'worker_manifest': config['manifest'],
        'quality_gate_passed': quality_result['gate_passed'],
        'claim_scope': 'diagnostic_rejected_candidate'
        if diagnostic_rejected
        else 'qualified_candidate',
        'quality_sha256': file_hash(quality),
        'quality_score': quality_result['score'],
        'samples': samples,
        'shapes': ['short', 'prefill', 'decode', 'mixed'],
        'client': {'client_timeout_seconds': 90, 'late_drop_seconds': 0.25},
        'endpoints': {
            'native': 'https://worker-b.lsl-r5.svc.cluster.local:30446',
            'gateway': 'https://gateway.lsl.internal:30443',
            'upstream': 'https://gateway.lsl.internal:30444',
        },
        'slo': {
            'short_decode_ttft': 1,
            'prefill_ttft': 3,
            'max_tpot': 0.08,
            'e2e': '3 + .08 * output_budget',
            'required_fraction': 0.99,
        },
        'formal_count': 24,
        'repeats': 3,
        'lag_fraction': 0.1,
        'prefix_caching': False,
        'interpretation': 'synthetic fixed generated-token work; not natural useful-output throughput',
    }
    write_once(out / 'plan.json', plan)
    return {'samples': len(samples), 'variant': plan['variant']}


def samples_for(plan, shape, count):
    groups = {
        s: [r for r in plan['samples'] if r['shape'] == s] for s in ('short', 'prefill', 'decode')
    }
    if shape == 'mixed':
        return [groups['prefill' if i % 2 == 0 else 'decode'][(i // 2) % 8] for i in range(count)]
    return [groups[shape][i % 8] for i in range(count)]


def request_row(plan, path, sample, index, label):
    body = sample['body'] if path == 'gateway' else sample['native_body']
    rid = f"n2-{plan['variant']}-{plan['run_id']}-{label}-{index:03}"
    return {
        'id': rid,
        'shape': sample['id'],
        'path': path,
        'endpoint': plan['endpoints'][path],
        'expected_version': sample['version'],
        'expected_profile': sample['profile_digest'],
        'input_tokens': len(sample['input_token_ids']),
        'output_budget': body['max_tokens'],
        'body': body,
        'headers': {'X-Request-Id': rid, 'Idempotency-Key': rid},
    }


def settings(plan, certificates):
    root = Path(certificates)
    return {
        **plan['client'],
        'tls': {
            'ca': str(root / 'ca.pem'),
            'cert': str(root / 'loadgen.pem'),
            'key': str(root / 'loadgen.key'),
        },
    }


async def closed_loop(plan, shape, concurrency, certificates):
    requests = [
        request_row(plan, 'native', s, i, f'pilot-{shape}-c{concurrency}')
        for i, s in enumerate(samples_for(plan, shape, 8))
    ]
    config = settings(plan, certificates)
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    async with httpx.AsyncClient(
        verify=client_context(config['tls']),
        trust_env=False,
        timeout=90,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
    ) as client:

        async def one(item):
            async with semaphore:
                begin = time.monotonic()
                obs = StreamObservation(
                    item['expected_version'], item['input_tokens'], item['output_budget']
                )
                row = {
                    k: item[k]
                    for k in ('id', 'shape', 'input_tokens', 'output_budget', 'expected_version')
                }
                row.update(sent=True, status=None, send_lag=0)
                try:
                    async with client.stream(
                        'POST',
                        item['endpoint'] + '/v1/chat/completions',
                        json=item['body'],
                        headers=item['headers'],
                    ) as response:
                        row['status'] = response.status_code
                        row['response_headers'] = dict(response.headers)
                        if response.status_code != 200:
                            row['response_error'] = (await response.aread()).decode()
                        else:
                            async for line in response.aiter_lines():
                                if line.startswith('data:'):
                                    obs.feed(line[5:].strip(), time.monotonic() - begin)
                except Exception as error:
                    row['error'] = repr(error)
                row.update(
                    protocol_valid=obs.valid and not row.get('error'),
                    events=obs.events,
                    content=obs.content,
                    finish_reason=obs.finish,
                    usage=obs.usage,
                    ttft_sent=obs.first_content,
                    ttft_scheduled=obs.first_content,
                    e2e_sent=obs.done,
                    e2e_scheduled=obs.done,
                )
                return row

        rows = await asyncio.gather(*(one(r) for r in requests))
    return {'requests': requests, 'rows': rows, 'duration_seconds': time.monotonic() - started}


def reserve(root, label, count):
    path = Path(root) / 'budget.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    total = sum(r['count'] for r in rows)
    if total + count > 5000:
        raise ValueError('性能负载超过5000次预约')
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'label': label, 'count': count, 'cumulative': total + count}) + '\n')


async def pilot(plan_path, certificates, output):
    plan = read(plan_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    plan['run_id'] = digest(str(out.resolve()))[:10]
    write_once(
        out / 'input.json',
        {
            'plan_sha256': file_hash(plan_path),
            'run_id': plan['run_id'],
            'count_per_cell': 8,
            'concurrency': [1, 4, 8],
            'script_sha256': file_hash(__file__),
        },
    )
    for shape in plan['shapes']:
        for concurrency in (1, 4, 8):
            label = f'{shape}-c{concurrency}'
            reserve(out, label, 8)
            result = await closed_loop(plan, shape, concurrency, certificates)
            write_once(out / (label + '-raw.json'), result)
            summary = summarize_fixed(result['rows'], result['duration_seconds'])
            summary.update(shape=shape, concurrency=concurrency)
            write_once(out / (label + '-summary.json'), summary)
            results.append(summary)
            print(json.dumps(summary), flush=True)
            if summary['fixed_length_valid'] != 8:
                raise ValueError('试点长度/协议不合格')
    write_once(
        out / 'summary.json',
        {
            'variant': plan['variant'],
            'cells': results,
            'generations': 96,
            'quality_gate_passed': plan.get('quality_gate_passed', True),
            'claim_scope': plan.get('claim_scope', 'qualified_candidate'),
        },
    )


def freeze(pilot_path, output):
    pilot = read(pilot_path)
    rates = {}
    for shape in ('short', 'prefill', 'decode', 'mixed'):
        cell = next(c for c in pilot['cells'] if c['shape'] == shape and c['concurrency'] == 8)
        rates[shape] = round(max(0.1, 0.6 * cell['completed_rps']), 4)
    write_once(
        output,
        {
            'rates': rates,
            'selection': '0.6 times BF16 concurrency-8 observed rate, frozen before candidate quantization',
            'pilot_sha256': file_hash(pilot_path),
            'count_per_window': 24,
            'repeats': 3,
            'quality_gate': 'BF16 >=70%; AWQ decrease <=2pp',
            'profiling_windows_separate': True,
        },
    )
    return rates


async def formal(plan_path, rates_path, certificates, output):
    plan, rates = read(plan_path), read(rates_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    plan['run_id'] = digest(str(out.resolve()))[:10]
    write_once(
        out / 'input.json',
        {
            'plan_sha256': file_hash(plan_path),
            'run_id': plan['run_id'],
            'rates_sha256': file_hash(rates_path),
            'script_sha256': file_hash(__file__),
            'settings': settings(plan, certificates),
        },
    )
    for repeat in range(3):
        paths = ['native', 'gateway', 'upstream']
        paths = paths[repeat:] + paths[:repeat]
        for shape in plan['shapes']:
            for path in paths:
                label = f'r{repeat}-{shape}-{path}'
                directory = out / label
                directory.mkdir()
                items = [
                    request_row(plan, path, s, i, label)
                    for i, s in enumerate(samples_for(plan, shape, 24))
                ]
                rate = rates['rates'][shape]
                offsets = [i / rate for i in range(len(items))]
                reserve(out, label, len(items))
                write_once(
                    directory / 'input.json', {'requests': items, 'offsets': offsets, 'rate': rate}
                )
                result = await window(
                    items[0]['endpoint'], items, offsets, settings(plan, certificates)
                )
                write_once(directory / 'raw.json', result)
                summary = summarize_fixed(result['rows'], result['elapsed_seconds'], rate)
                summary.update(path=path, shape=shape, repeat=repeat, rate=rate)
                write_once(directory / 'summary.json', summary)
                results.append(summary)
                print(json.dumps(summary), flush=True)
    write_once(
        out / 'summary.json',
        {
            'variant': plan['variant'],
            'windows': results,
            'requests': 864,
            'raw_audit_status': 'pending',
            'quality_gate_passed': plan.get('quality_gate_passed', True),
            'claim_scope': plan.get('claim_scope', 'qualified_candidate'),
        },
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'pilot', 'freeze', 'formal'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokenizer')
    p.add_argument('--engine-config')
    p.add_argument('--quality')
    p.add_argument('--plan')
    p.add_argument('--certificates')
    p.add_argument('--pilot')
    p.add_argument('--rates')
    p.add_argument('--diagnostic-rejected', action='store_true')
    a = p.parse_args()
    if a.stage == 'prepare':
        print(
            json.dumps(
                prepare(
                    a.tokenizer,
                    a.engine_config,
                    a.quality,
                    a.output,
                    diagnostic_rejected=a.diagnostic_rejected,
                )
            )
        )
    elif a.stage == 'freeze':
        print(json.dumps(freeze(a.pilot, a.output)))
    elif a.stage == 'pilot':
        asyncio.run(pilot(a.plan, a.certificates, a.output))
    else:
        asyncio.run(formal(a.plan, a.rates, a.certificates, a.output))
