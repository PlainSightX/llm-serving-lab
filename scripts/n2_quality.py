"""固定GSM8K面板的单次自然停止评价；超预算、错误和截断均保留。"""

import argparse
import asyncio
import json
from pathlib import Path
import time

from serving_lab.execution import write_once
from serving_lab.inference_study import (
    quality_messages,
    quality_pass,
    numeric_answer,
    require_variant,
    require_baseline,
)
from serving_lab.lab import file_hash
from serving_lab.research_action import check_readiness, generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.worker_protocol import RemoteEngine

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    train = [json.loads(s) for s in (data / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    test = [json.loads(s) for s in (data / 'test.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(train) != 7473 or len(test) != 1319:
        raise ValueError('不是固定GSM8K原始面板')
    calibration = [
        {
            'id': f'train-{i:04}',
            'messages': quality_messages(row) + [{'role': 'assistant', 'content': row['answer']}],
        }
        for i, row in enumerate(train[:128])
    ]
    quality = [
        {'id': f'test-{i:04}', 'messages': quality_messages(row), 'answer': row['answer']}
        for i, row in enumerate(test[:256])
    ]
    if any(numeric_answer(r['answer']) is None for r in quality):
        raise ValueError('参考答案格式不符')
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'calibration.json', calibration)
    write_once(output / 'quality.json', quality)
    write_once(
        output / 'smoke.json',
        [
            {'id': f'train-{i:04}', 'messages': quality_messages(row), 'answer': row['answer']}
            for i, row in enumerate(train[:4])
        ],
    )
    plan = {
        'name': 'n2-gsm8k-v1',
        'source_revision': '3101c7d5072418e28b9008a6636bde82a006892c',
        'source_hashes': {n: file_hash(data / (n + '.jsonl')) for n in ('train', 'test')},
        'calibration_count': 128,
        'quality_count': 256,
        'max_input_tokens': 3500,
        'max_output_tokens': 512,
        'concurrency': 4,
        'baseline_minimum': 0.70,
        'maximum_quantized_drop': 0.02,
        'natural_stop_required': True,
        'files': {
            n: file_hash(output / (n + '.json')) for n in ('calibration', 'quality', 'smoke')
        },
    }
    write_once(output / 'plan.json', plan)
    return plan


async def run(data, engine_config, tokenizer_path, output, variant, smoke=False, baseline=None):
    from transformers import AutoTokenizer

    data, out = Path(data).resolve(), Path(output).resolve()
    plan = read(data / 'plan.json')
    config = read(engine_config)
    if config['manifest']['profile']['name'] != 'n2_length_controlled_v1':
        raise ValueError('不是N2冻结profile')
    if config.get('expected_artifact_digest') != config['manifest']['artifact_digest']:
        raise ValueError('worker与独立来源/量化工件身份不符')
    require_variant(config['manifest'], variant)
    name = 'smoke' if smoke else 'quality'
    if file_hash(data / (name + '.json')) != plan['files'][name]:
        raise ValueError('面板变化')
    cases = read(data / (name + '.json'))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    prior = None
    if variant == 'awq' and not smoke:
        prior = read(baseline)
        require_baseline(prior, file_hash(data / 'plan.json'))
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in list((ROOT / 'src/serving_lab').glob('*.py')) + [Path(__file__)]:
        source_name = path.relative_to(ROOT).as_posix()
        target = out / 'source' / source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[source_name] = file_hash(path)
    write_once(
        out / 'input.json',
        {
            'plan': str(data / 'plan.json'),
            'plan_sha256': file_hash(data / 'plan.json'),
            'cases_sha256': file_hash(data / (name + '.json')),
            'case_file': str(data / (name + '.json')),
            'tokenizer': str(Path(tokenizer_path).resolve()),
            'worker_manifest': config['manifest'],
            'variant': variant,
            'smoke': smoke,
            'baseline': str(Path(baseline).resolve()) if baseline else None,
            'baseline_sha256': file_hash(baseline) if baseline else None,
            'concurrency': 4,
            'script_sha256': file_hash(__file__),
            'grader_sha256': file_hash(ROOT / 'src/serving_lab/inference_study.py'),
            'source_hashes': hashes,
            'expected_artifact_digest': config['expected_artifact_digest'],
        },
    )
    engine = RemoteEngine(
        config['endpoint'],
        Path(config['token_file']).read_text().strip(),
        config['manifest'],
        tls=config['tls'],
    )
    rows = []
    reserved = []
    started = time.monotonic()
    semaphore = asyncio.Semaphore(4)
    try:
        ready, checks = await check_readiness(engine, 2)
        write_once(out / 'readiness.json', {'ready': ready, 'checks': checks})
        if not ready:
            raise RuntimeError('worker_not_ready')
        with RuntimeStore(out / 'runtime') as store:

            async def one(case):
                async with semaphore:
                    ids = tokenizer.apply_chat_template(
                        case['messages'],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if len(ids) > 3500:
                        raise ValueError('输入超预算，不截断质量题')
                    reserved.append(case['id'])
                    with (out / 'reservations.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'number': len(reserved), 'case': case['id']}) + '\n')
                    receipt = await generate_action(
                        engine,
                        store,
                        out / 'calls' / case['id'],
                        case['messages'],
                        ids,
                        {'purpose': 'n2_quality', 'case': case['id'], 'variant': variant},
                        readiness_attempts=2,
                        max_tokens=512,
                        ignore_eos=False,
                    )
                    row = {
                        'id': case['id'],
                        'correct': quality_pass(
                            receipt['content'], case['answer'], receipt['finish_reason']
                        ),
                        'finish_reason': receipt['finish_reason'],
                        'arrival_id': receipt['arrival_id'],
                        'usage': receipt['usage'],
                    }
                    rows.append(row)
                    if len(rows) % 32 == 0:
                        print(
                            json.dumps(
                                {
                                    'finished': len(rows),
                                    'of': len(cases),
                                    'correct': sum(r['correct'] for r in rows),
                                }
                            ),
                            flush=True,
                        )

            tasks = [asyncio.create_task(one(case)) for case in cases]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            if store.unresolved():
                raise RuntimeError('质量评价仍有未解除租约')
        correct = sum(row['correct'] for row in rows)
        score = correct / len(cases)
        passed = score >= 0.70 and (prior is None or score >= prior['score'] - 0.02)
        summary = {
            'variant': variant,
            'smoke': smoke,
            'cases': len(cases),
            'correct': correct,
            'score': score,
            'gate_passed': passed,
            'plan_sha256': file_hash(data / 'plan.json'),
            'artifact_digest': config['manifest']['artifact_digest'],
            'parameter_digest': config['manifest']['parameter_digest'],
            'elapsed_seconds': time.monotonic() - started,
            'truncated': sum(r['finish_reason'] == 'length' for r in rows),
            'rows': sorted(rows, key=lambda r: r['id']),
            'raw_audit_status': 'pending',
        }
        write_once(out / 'summary.json', summary)
        return {k: v for k, v in summary.items() if k != 'rows'}
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {
                'error': repr(error),
                'reservations': len(reserved),
                'completed': len(rows),
                'implicit_retry': False,
            },
        )
        raise
    finally:
        await engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'run'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--engine-config')
    p.add_argument('--tokenizer')
    p.add_argument('--variant', choices=['bf16', 'awq'], default='bf16')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--baseline')
    a = p.parse_args()
    result = (
        prepare(a.data, a.output)
        if a.stage == 'prepare'
        else asyncio.run(
            run(a.data, a.engine_config, a.tokenizer, a.output, a.variant, a.smoke, a.baseline)
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
