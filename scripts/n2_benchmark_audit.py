"""从原始 SSE、预定到达与原生释放重算性能，不把汇总布尔值当证据。"""

import argparse
import json
from pathlib import Path
import sqlite3

from serving_lab.benchmark import StreamObservation
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile
from serving_lab.inference_study import summarize_fixed, require_quality_proof


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expected_samples(shape, count=24):
    if shape not in {'short', 'prefill', 'decode', 'mixed'}:
        raise ValueError('未知窗口形状')
    return [
        f"{'prefill' if i % 2 == 0 else 'decode'}-{(i // 2) % 8}"
        if shape == 'mixed'
        else f'{shape}-{i % 8}'
        for i in range(count)
    ]


def verify_duration(raw, rate, count=24):
    planned = count / rate
    duration = raw['elapsed_seconds']
    observed = max(r['end_offset'] for r in raw['rows'])
    closed = raw['ended_monotonic'] - raw['started_monotonic']
    if duration + 1e-8 < max(planned, observed) or duration > max(planned, closed) + 1e-8:
        raise ValueError('窗口分母未覆盖实际完成或超过结束时钟')
    for row in raw['rows']:
        if row['end_offset'] > closed + 1e-8:
            raise ValueError('请求在窗口关闭后结束')
        if (
            row.get('e2e_sent') is not None
            and row['end_offset'] + 1e-8 < row['send_offset'] + row['e2e_sent']
        ):
            raise ValueError('请求结束早于原始DONE')


def replay_stream(row, request):
    observation = StreamObservation(
        request['expected_version'], request['input_tokens'], request['output_budget']
    )
    previous = -1
    ids = set()
    done = 0
    for event in row.get('events', []):
        elapsed = event['elapsed']
        if elapsed < previous:
            raise ValueError('流式时间发生倒退')
        if done:
            raise ValueError('DONE后仍有事件')
        previous = elapsed
        if event['data'] == '[DONE]':
            done += 1
        else:
            value = json.loads(event['data'])
            ids.add(value['id'])
            if value['model'] != request['expected_version']:
                raise ValueError('流式模型不符')
        observation.feed(event['data'], elapsed)
    if len(ids) > 1:
        raise ValueError('同一请求混入不同响应ID')
    valid = observation.valid and not row.get('error')
    if valid:
        if done != 1 or len(ids) != 1:
            raise ValueError('完整响应缺少唯一ID或DONE')
        for name, value in [
            ('content', observation.content),
            ('usage', observation.usage),
            ('finish_reason', observation.finish),
            ('ttft_sent', observation.first_content),
            ('e2e_sent', observation.done),
        ]:
            if row[name] != value:
                raise ValueError('记录与原始SSE不一致: ' + name)
        lag = row['send_lag']
        if (
            row['ttft_scheduled'] != lag + observation.first_content
            or row['e2e_scheduled'] != lag + observation.done
        ):
            raise ValueError('预定到达时延未包含迟发')
    if bool(row['protocol_valid']) != bool(valid):
        raise ValueError('协议汇总标记与原件不同')
    return next(iter(ids)) if ids else None


def audit(
    run,
    plan_path,
    rates_path,
    releases,
    gateway_db,
    output,
    *,
    tokenizer_path,
    quality_proof,
    payloads,
    source_manifest,
):
    from transformers import AutoTokenizer

    root = Path(run)
    initial = read(root / 'input.json')
    plan = read(plan_path)
    rates = read(rates_path)
    summary = read(root / 'summary.json')
    assert (
        file_hash(plan_path) == initial['plan_sha256']
        and file_hash(rates_path) == initial['rates_sha256']
    )
    assert summary['variant'] == plan['variant'] and summary['requests'] == 864
    manifest = plan['worker_manifest']
    profile = ServingProfile(manifest['profile'], manifest['version_id'])
    assert profile.digest == manifest['profile_digest']
    diagnostic = plan.get('claim_scope') == 'diagnostic_rejected_candidate'
    proof = read(quality_proof)
    require_quality_proof(proof, manifest, diagnostic_rejected=diagnostic)
    assert plan.get('quality_gate_passed', True) == proof['gate_passed']
    if diagnostic:
        assert (
            summary['claim_scope'] == 'diagnostic_rejected_candidate'
            and summary['quality_gate_passed'] is False
        )
    assert proof['score'] == plan['quality_score'] and proof['variant'] == plan['variant']
    assert read(source_manifest)['scripts/n2_benchmark.py'] == initial['script_sha256']
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    # 入口可能改model别名；实际到达worker的采样、模板与长度必须相同。
    samples = {r['id']: r for r in plan['samples']}
    for sample in samples.values():
        assert sample['native_body'] == profile.payload(sample['body'])
        assert (
            sample['version'] == manifest['version_id']
            and sample['profile_digest'] == profile.digest
        )
        expected = {'short': (256, 128), 'prefill': (3072, 128), 'decode': (512, 512)}[
            sample['shape']
        ]
        assert (
            len(sample['input_token_ids']),
            sample['body']['max_tokens'],
        ) == expected and sample['body']['ignore_eos'] is True
        assert (
            tokenizer.apply_chat_template(
                sample['body']['messages'],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            == sample['input_token_ids']
        )
    native = {}
    for line in Path(releases).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') == 'request_released':
            assert event['request_id'] not in native
            native[event['request_id']] = event
    with sqlite3.connect(f'file:{Path(gateway_db).resolve().as_posix()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        arrivals = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    requests_seen = set()
    releases_seen = set()
    arrivals_seen = set()
    results = []
    for saved in summary['windows']:
        label = f"r{saved['repeat']}-{saved['shape']}-{saved['path']}"
        cell = root / label
        parameters = read(cell / 'input.json')
        raw = read(cell / 'raw.json')
        rows = raw['rows']
        requests = parameters['requests']
        rate = rates['rates'][saved['shape']]
        assert len(rows) == len(requests) == 24 and parameters['rate'] == rate
        assert parameters['offsets'] == [i / rate for i in range(24)]
        verify_duration(raw, rate)
        assert [r['shape'] for r in requests] == expected_samples(saved['shape'])
        for index, (row, request) in enumerate(zip(rows, requests)):
            sample = samples[request['shape']]
            assert row['id'] == request['id'] and row['id'] not in requests_seen
            requests_seen.add(row['id'])
            assert row['id'] == f"n2-{plan['variant']}-{initial['run_id']}-{label}-{index:03}"
            assert (
                request['body'] == sample['body' if saved['path'] == 'gateway' else 'native_body']
            )
            assert (
                request['endpoint'] == plan['endpoints'][saved['path']]
                and request['path'] == saved['path']
            )
            for key in ('shape', 'expected_version', 'input_tokens', 'output_budget'):
                assert row[key] == request[key]
            assert row['scheduled_offset'] == parameters['offsets'][index]
            assert abs(row['send_offset'] - row['scheduled_offset'] - row['send_lag']) < 1e-8
            response_id = replay_stream(row, request)
            if not row['protocol_valid']:
                continue
            candidates = [k for k in (response_id, response_id + '-0') if k in native]
            assert len(candidates) == 1
            key = candidates[0]
            release = native[key]
            assert key not in releases_seen
            releases_seen.add(key)
            assert release['worker_instance_id'] == manifest['identity']['worker_instance_id']
            assert release['startup_generation'] == manifest['identity']['startup_generation']
            assert release['request_absent'] and not any(release['after_blocks'])
            assert release['status'] == 'FINISHED_LENGTH_CAPPED'
            if saved['path'] == 'gateway':
                headers = row['response_headers']
                aid = headers['x-arrival-id']
                arrival = arrivals[aid]
                assert aid not in arrivals_seen
                arrivals_seen.add(aid)
                assert (
                    headers['x-model-version'] == manifest['version_id']
                    and headers['x-serving-profile'] == profile.digest
                )
                assert headers['x-worker-instance'] == manifest['identity']['worker_instance_id']
                assert headers['x-worker-generation'] == manifest['identity']['startup_generation']
                assert (
                    arrival['state'] == 'completed'
                    and arrival['lease_released'] == 1
                    and arrival['cleanup'] == 'confirmed'
                )
                assert json.loads(arrival['worker']) == manifest['identity']
                assert json.loads(arrival['cleanup_evidence']) == release
                assert arrival['input_hash'] == digest(
                    {**request['body'], 'model': manifest['version_id']}
                )
                assert response_id == 'chatcmpl-' + arrival['attempt_id']
                assert read(Path(payloads) / 'requests' / (aid + '.json')) == {
                    **request['body'],
                    'model': manifest['version_id'],
                }
                assert read(Path(payloads) / 'wire-requests' / (aid + '.json')) == {
                    'payload': sample['native_body'],
                    'profile_digest': profile.digest,
                    'input_token_ids': sample['input_token_ids'],
                }
            else:
                assert response_id == 'chatcmpl-' + request['id']
        rebuilt = summarize_fixed(rows, raw['elapsed_seconds'], rate)
        rebuilt.update(path=saved['path'], shape=saved['shape'], repeat=saved['repeat'], rate=rate)
        assert rebuilt == saved == read(cell / 'summary.json')
        results.append(rebuilt)
    assert len(results) == 36 and len(requests_seen) == 864
    assert {(r['repeat'], r['shape'], r['path']) for r in results} == {
        (r, s, p)
        for r in range(3)
        for s in ('short', 'prefill', 'decode', 'mixed')
        for p in ('native', 'gateway', 'upstream')
    }
    budget = [json.loads(s) for s in (root / 'budget.jsonl').read_text().splitlines()]
    assert sum(r['count'] for r in budget) == 864 and budget[-1]['cumulative'] == 864
    result = {
        'raw_verified': True,
        'variant': plan['variant'],
        'requests': 864,
        'windows': results,
        'quality_gate_passed': proof['gate_passed'],
        'adoptable': proof['gate_passed'],
        'claim_scope': 'diagnostic_rejected_candidate' if diagnostic else 'qualified_candidate',
        'fixed_length_valid': sum(r['fixed_length_valid'] for r in results),
        'matched_native_releases': len(releases_seen),
        'gateway_arrivals_verified': len(arrivals_seen),
        'schedule_valid_windows': sum(r['schedule_valid'] for r in results),
        'slo_met': sum(r['slo_met'] for r in results),
        'artifact_digest': manifest['artifact_digest'],
        'parameter_digest': manifest['parameter_digest'],
        'profile_digest': profile.digest,
        'quality_proof_sha256': file_hash(quality_proof),
        'source_manifest_sha256': file_hash(source_manifest),
        'plan_sha256': file_hash(plan_path),
        'rates_sha256': file_hash(rates_path),
        'scope': 'finite synthetic fixed-token windows; TPOT is per-request mean, not per-token ITL',
    }
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'windows'}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in (
        'run',
        'plan',
        'rates',
        'releases',
        'gateway-db',
        'output',
        'tokenizer',
        'quality-proof',
        'payloads',
        'source-manifest',
    ):
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    audit(
        a.run,
        a.plan,
        a.rates,
        a.releases,
        a.gateway_db,
        a.output,
        tokenizer_path=a.tokenizer,
        quality_proof=a.quality_proof,
        payloads=a.payloads,
        source_manifest=a.source_manifest,
    )
