"""逐题从固定消息、原始响应、账本和原生释放重算N2质量。"""

import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile
from serving_lab.inference_study import require_variant, require_baseline, QUALITY_PLAN_SHA256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(run, releases, output):
    from transformers import AutoTokenizer

    root = Path(run).resolve()
    initial = read(root / 'input.json')
    plan = read(initial['plan'])
    summary = read(root / 'summary.json')
    assert (
        file_hash(initial['plan']) == initial['plan_sha256']
        and file_hash(initial['case_file']) == initial['cases_sha256']
    )
    assert initial['plan_sha256'] == QUALITY_PLAN_SHA256
    cases = read(initial['case_file'])
    name = 'smoke' if initial['smoke'] else 'quality'
    assert initial['cases_sha256'] == plan['files'][name] and len(cases) == (
        4 if initial['smoke'] else 256
    )
    for path, expected in initial['source_hashes'].items():
        assert file_hash(root / 'source' / path) == expected
    grader_path = root / 'source/src/serving_lab/inference_study.py'
    spec = importlib.util.spec_from_file_location('frozen_n2_grader', grader_path)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    assert file_hash(grader_path) == initial['grader_sha256']
    manifest = initial['worker_manifest']
    assert manifest['artifact_digest'] == initial['expected_artifact_digest']
    require_variant(manifest, initial['variant'])
    profile = ServingProfile(manifest['profile'], manifest['version_id'])
    assert profile.digest == manifest['profile_digest']
    tokenizer = AutoTokenizer.from_pretrained(
        initial['tokenizer'], local_files_only=True, trust_remote_code=False
    )
    native = {}
    for line in Path(releases).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') == 'request_released':
            native[event['request_id']] = event
    with sqlite3.connect(root / 'runtime/service.sqlite') as db:
        db.row_factory = sqlite3.Row
        rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    results = []
    seen = set()
    used = set()
    for case in cases:
        directory = root / 'calls' / case['id']
        reservation = read(directory / 'reservation.json')
        receipt = read(directory / 'receipt.json')
        raw = read(directory / 'response.raw')
        ids = tokenizer.apply_chat_template(
            case['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        body = {
            'model': manifest['version_id'],
            'messages': case['messages'],
            'max_tokens': 512,
            'stream': False,
            'ignore_eos': False,
        }
        assert (
            reservation['wire'] == profile.payload(body) and reservation['input_token_ids'] == ids
        )
        a, t = receipt['arrival_id'], receipt['attempt_id']
        assert a not in seen
        seen.add(a)
        assert (
            reservation['arrival_id'] == a
            and reservation['attempt_id'] == t
            and raw['id'] == 'chatcmpl-' + t
            and raw['model'] == manifest['version_id']
        )
        assert reservation['metadata'] == {
            'purpose': 'n2_quality',
            'case': case['id'],
            'variant': initial['variant'],
        }
        row = rows[a]
        assert (
            row['state'] == 'completed'
            and row['lease_released'] == 1
            and row['cleanup'] == 'confirmed'
            and json.loads(row['observation']) == receipt
        )
        assert (
            row['input_hash'] == digest(body)
            and read(root / 'runtime/requests' / (a + '.json')) == body
        )
        assert (
            json.loads(row['worker']) == manifest['identity']
            and reservation['worker'] == manifest['identity']
        )
        choice = raw['choices'][0]
        assert (
            len(raw['choices']) == 1
            and choice['message']['content'] == receipt['content']
            and choice['finish_reason'] == receipt['finish_reason']
        )
        assert (
            raw['usage'] == receipt['usage']
            and raw['usage']['prompt_tokens'] == len(ids)
            and 0 < raw['usage']['completion_tokens'] <= 512
        )
        assert raw['usage']['total_tokens'] == len(ids) + raw['usage']['completion_tokens']
        release = receipt['release']
        assert (
            release['request_id'] in {'chatcmpl-' + t, 'chatcmpl-' + t + '-0'}
            and release['request_id'] not in used
        )
        assert native[release['request_id']] == release == json.loads(row['cleanup_evidence'])
        used.add(release['request_id'])
        assert (
            release['worker_instance_id'] == manifest['identity']['worker_instance_id']
            and release['startup_generation'] == manifest['identity']['startup_generation']
        )
        assert release['request_absent'] and not any(release['after_blocks'])
        assert (
            release['status']
            == {'stop': 'FINISHED_STOPPED', 'length': 'FINISHED_LENGTH_CAPPED'}[
                receipt['finish_reason']
            ]
        )
        results.append(
            {
                'id': case['id'],
                'correct': grader.quality_pass(
                    receipt['content'], case['answer'], receipt['finish_reason']
                ),
                'finish_reason': receipt['finish_reason'],
                'arrival_id': a,
                'usage': receipt['usage'],
            }
        )
    assert seen == set(rows) and len(list((root / 'calls').glob('*/receipt.json'))) == len(cases)
    reserved = [json.loads(s) for s in (root / 'reservations.jsonl').read_text().splitlines()]
    assert sorted(r['case'] for r in reserved) == sorted(c['id'] for c in cases) and [
        r['number'] for r in reserved
    ] == list(range(1, len(cases) + 1))
    assert sorted(results, key=lambda r: r['id']) == summary['rows']
    correct = sum(r['correct'] for r in results)
    score = correct / len(cases)
    passed = score >= 0.70
    if initial['variant'] == 'awq' and not initial['smoke']:
        assert file_hash(initial['baseline']) == initial['baseline_sha256']
        base = read(initial['baseline'])
        require_baseline(base, initial['plan_sha256'])
        passed = passed and score >= base['score'] - 0.02
    assert (
        summary['score'] == score
        and summary['correct'] == correct
        and summary['gate_passed'] == passed
    )
    result = {
        'raw_verified': True,
        'variant': initial['variant'],
        'smoke': initial['smoke'],
        'cases': len(cases),
        'correct': correct,
        'score': score,
        'gate_passed': passed,
        'truncated': sum(r['finish_reason'] == 'length' for r in results),
        'actual_generations': len(rows),
        'matched_releases': len(used),
        'unreleased_leases': 0,
        'artifact_digest': manifest['artifact_digest'],
        'parameter_digest': manifest['parameter_digest'],
        'profile_digest': manifest['profile_digest'],
        'plan_sha256': initial['plan_sha256'],
        'input_tokens': sum(r['usage']['prompt_tokens'] for r in results),
        'output_tokens': sum(r['usage']['completion_tokens'] for r in results),
        'scope': 'fixed public arithmetic quality sentinel; not general enterprise capability',
    }
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--releases', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.run, a.releases, a.output)
