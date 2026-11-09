"""重算恢复模型的真实工程资格，并关联全部原生释放与关闭记录。"""

import argparse
import json
from pathlib import Path
import sqlite3

from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(root):
    root = Path(root)
    state = root / 'serving-state-a'
    run = state / read(state / 'current.json')['run'].split('/')[-1]
    manifest = read(run / 'manifest.json')
    export = read(root / 'export-a/receipt.json')
    if (
        manifest['artifact_digest'] != export['artifact_digest']
        or manifest['parameter_digest'] != export['parameter_digest']
    ):
        raise ValueError('服务实例和恢复产物不符')
    native = [
        json.loads(s)
        for s in (run / 'native/release-events.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    released = {r['request_id']: r for r in native if r['event'] == 'request_released'}
    if len(released) != 6:
        raise ValueError('实际生成数不是2次bootstrap加4项资格')
    matched = []
    folder = state / 'transport-a/qualification'
    initial = read(folder / 'input.json')
    receipt = read(folder / 'receipt.json')
    if (
        digest(initial['panel']) != receipt['panel_digest']
        or digest(initial['profile']) != manifest['profile_digest']
    ):
        raise ValueError('资格面板或实际profile不符')
    profile = ServingProfile(initial['profile'], receipt['version_id'])
    for name, sha in initial['source_hashes'].items():
        if file_hash(folder / name) != sha:
            raise ValueError('资格执行源码不符')
    with sqlite3.connect(
        (state / 'transport-a/runtime/service.sqlite').as_uri() + '?mode=ro', uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        arrivals = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    if len(arrivals) != 4 or any(
        not r['lease_released'] or r['cleanup'] != 'confirmed' for r in arrivals.values()
    ):
        raise ValueError('请求没有完整释放')
    for case in initial['panel']['cases']:
        row = read(folder / (case['id'] + '.json'))
        attempt = row['attempt_id']
        release = row['release']
        body = {
            'model': receipt['version_id'],
            'messages': case['messages'],
            'max_tokens': case['max_tokens'],
            'stream': case['stream'],
        }
        if (
            row['wire_payload'] != profile.payload(body)
            or row['input_token_ids'] != row['engine_token_ids']
        ):
            raise ValueError('请求或输入token不符')
        if case['stream']:
            if row['events'][-1] != '[DONE]':
                raise ValueError('流未终结')
            raw = [json.loads(e) for e in row['events'][:-1]]
            content = ''
            finish = None
            usage = None
            for event in raw:
                if event['id'] != 'chatcmpl-' + attempt:
                    raise ValueError('原始流响应ID不符')
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    content += choice.get('delta', {}).get('content') or ''
                    finish = choice.get('finish_reason') or finish
        else:
            raw = json.loads(row['raw_response'])
            if raw != row['response'] or raw['id'] != 'chatcmpl-' + attempt:
                raise ValueError('原始响应ID不符')
            content = raw['choices'][0]['message']['content']
            finish = raw['choices'][0]['finish_reason']
            usage = raw['usage']
        if (
            not content
            or content != row['content']
            or finish != case['expected_finish']
            or usage != row['usage']
            or usage['prompt_tokens'] != len(row['input_token_ids'])
            or not 0 < usage['completion_tokens'] <= case['max_tokens']
            or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']
            or finish == 'length'
            and usage['completion_tokens'] != case['max_tokens']
        ):
            raise ValueError('响应内容/终止或usage合同不符')
        if (
            release['request_id'] != 'chatcmpl-' + attempt
            or released.get(release['request_id']) != release
            or not release['request_absent']
            or any(release['after_blocks'])
        ):
            raise ValueError('未找到相同请求的原生KV释放')
        if json.loads(arrivals[row['arrival_id']]['observation']) != row:
            raise ValueError('账本与原件不同')
        matched.append(release['request_id'])
    for label in ['plain', 'stream']:
        row = read(run / ('bootstrap/' + label + '-response.json'))
        release = row['release']
        if (
            released.get(release['request_id']) != release
            or not release['request_absent']
            or any(release['after_blocks'])
        ):
            raise ValueError('bootstrap释放不完整')
        matched.append(release['request_id'])
    if set(matched) != set(released):
        raise ValueError('存在未归属的实际生成')
    for event in released.values():
        for key in ['worker_instance_id', 'startup_generation']:
            if event[key] != manifest['identity'][key]:
                raise ValueError('混入其他worker代际')
    cleanup = read(run / 'cleanup.json')
    if not cleanup['engine']['stopped'] or cleanup['engine']['remaining']:
        raise ValueError('原监督未确认停止')
    return {
        'passed': True,
        'actual_generations': 6,
        'bootstrap': 2,
        'transport_cases': 4,
        'leases_released': 4,
        'artifact_digest': export['artifact_digest'],
        'parameter_digest': export['parameter_digest'],
        'profile_digest': manifest['profile_digest'],
        'application_qualified': False,
        'published': False,
        'supervisor_cleanup_verified': True,
        'scope': 'recovered model consumes existing real engineering qualification; no application or speed claim',
        'source_sha256': file_hash(__file__),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = audit(Path(a.root).resolve())
    write_once(a.output, result)
    print(json.dumps(result))
