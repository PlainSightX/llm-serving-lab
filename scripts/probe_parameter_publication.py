"""在单一新控制者内验证两个合格参数版本的灰度、排空及热回滚。"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time
import uuid

import httpx
from serving_lab.application_release import load_policy, POOL_SCOPE
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.tls import client_context
from serving_lab.worker_protocol import RemoteEngine


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_response(receipt, payload, specs, expected=None):
    version = receipt['version']
    if (
        receipt['status'] != 200
        or version not in specs
        or (expected is not None and version != expected)
    ):
        raise ValueError('响应参数版本不符')
    spec = specs[version]
    usage = receipt['usage']
    if (
        receipt['profile'] != spec['profile_digest']
        or receipt['evidence'] != spec['evidence_digest']
        or receipt['content'] not in payload['action_choices']
        or receipt['finish_reason'] != 'stop'
        or usage['prompt_tokens'] != len(payload['input_token_ids'])
        or not 0 < usage['completion_tokens'] <= 64
        or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']
    ):
        raise ValueError('响应profile、应用证据、动作或token不符')


def queue_snapshot(runtime):
    with sqlite3.connect(
        'file:' + (Path(runtime) / 'service.sqlite').as_posix() + '?mode=ro', uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in db.execute(
                'SELECT id,client_key,version,route,state,dispatched,lease_released FROM arrivals WHERE client_key IS NOT NULL'
            )
        ]


def queued_cutover_proof(events, arrivals, old_version, new_version, generation):
    """用同一控制者的提交事件顺序证明排队跨越切换，客户端早先快照不能替代。"""
    changes = [
        e
        for e in events
        if e['kind'] == 'pool_route_changed'
        and json.loads(e['body'])['current']['generation'] == generation
    ]
    if len(changes) != 1:
        raise ValueError('缺少唯一切换提交事件')
    change = changes[0]
    body = json.loads(change['body'])
    if (
        body['operation'] != 'promote'
        or body['previous']['active'] != old_version
        or body['previous']['canary'] is not None
        or body['current']['active'] != new_version
    ):
        raise ValueError('切换没有来自确定的C0路由')
    proof = []
    for arrival in sorted(arrivals):
        own = [e for e in events if e['arrival_id'] == arrival]
        bindings = [e for e in own if e['kind'] == 'binding']
        dispatch = [e for e in own if e['kind'] == 'dispatched']
        terminal = [e for e in own if e['kind'] == 'terminal']
        if len(bindings) != 1 or len(dispatch) != 1 or len(terminal) != 1:
            raise ValueError('请求生命周期事件不完整或重复')
        bound = json.loads(bindings[0]['body'])
        if (
            bound['version_id'] == old_version
            and bound['route_generation'] == body['previous']['generation']
            and bindings[0]['id'] < change['id'] < dispatch[0]['id'] < terminal[0]['id']
            and json.loads(terminal[0]['body'])['state'] == 'completed'
        ):
            proof.append(
                {
                    'arrival_id': arrival,
                    'binding_event': bindings[0]['id'],
                    'cutover_event': change['id'],
                    'dispatched_event': dispatch[0]['id'],
                    'terminal_event': terminal[0]['id'],
                    'version': old_version,
                }
            )
    return proof


def read_events(runtime):
    with sqlite3.connect(
        'file:' + (Path(runtime) / 'service.sqlite').as_posix() + '?mode=ro', uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(r) for r in db.execute('SELECT id,arrival_id,kind,body FROM events ORDER BY id')
        ]


async def run(config_path, input_path, output):
    config, source = read(config_path), read(input_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'management').mkdir()
    policy = load_policy(
        config['application_release']['bundle'], config['application_release']['sha256']
    )
    assert source['bundle_sha256'] == config['application_release']['sha256']
    versions = source['versions']
    payloads = source['payloads']
    specs = {}
    engines = {}
    manifests = {}
    if source['queue_requirement'] != {
        'batch': 16,
        'minimum_before_switch': 8,
        'minimum_dispatched_after_commit': 4,
        'version': 'c0',
        'proof': 'controller SQLite event order',
    }:
        raise ValueError('冻结的排队判定与执行程序不一致')
    for payload in payloads:
        policy.validate_request(
            {
                'model': 'active',
                'messages': payload['messages'],
                'action_choices': payload['action_choices'],
                'max_tokens': 64,
                'stream': False,
            }
        )
    for item in config['replicas']:
        manifest = read(item['manifest'])
        engine = RemoteEngine(
            item['endpoint'],
            Path(item['token_file']).read_text().strip(),
            manifest,
            tls=item['tls'],
        )
        engines[engine.version_id] = engine
        manifests[engine.version_id] = manifest
        cert = policy.bundle['certificates'][manifest['artifact_digest']]
        decision = policy.summary(cert)
        if not decision['passed']:
            raise ValueError('只允许两个同应用合格参数版本')
        specs[engine.version_id] = {
            'profile_digest': manifest['profile_digest'],
            'evidence_digest': decision['evidence_digest'],
        }
    if (
        set(engines) != set(versions.values())
        or len({m['parameter_digest'] for m in manifests.values()}) != 2
    ):
        raise ValueError('不是两个不同真实参数')
    admin = {'Authorization': 'Bearer ' + Path(config['admin_token_file']).read_text().strip()}
    endpoint = 'https://127.0.0.1:' + str(config['port'])
    write_once(
        out / 'input.json',
        {
            'config_sha256': file_hash(config_path),
            'input_sha256': file_hash(input_path),
            'script_sha256': file_hash(__file__),
            'source': source,
            'worker_manifests': manifests,
            'runtime': config['runtime'],
            'generation_cap': 256,
            'prior_n1_generations': source['prior_n1_generations'],
            'claim': 'hot rollback; workers retain resident weights',
        },
    )
    requests = []
    management = []
    direct = []
    reserved = 0
    sequence = 0
    pending = []

    def reserve(label, count=1):
        nonlocal reserved
        if reserved + count > 256 or source['prior_n1_generations'] + reserved + count > 11000:
            raise ValueError('发布调用超过冻结上限')
        reserved += count
        with (out / 'reservations.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'label': label, 'count': count, 'cumulative': reserved}) + '\n')

    async with httpx.AsyncClient(
        verify=client_context(config['replicas'][0]['tls']), trust_env=False, timeout=120
    ) as client:

        async def manage(path, body=None, expected=200):
            if path.endswith('/qualify'):
                reserve(path, 8)
            response = await client.request(
                'GET' if body is None else 'POST', endpoint + path, headers=admin, json=body
            )
            row = {
                'path': path,
                'body': body,
                'status': response.status_code,
                'response': response.json(),
                'monotonic': time.monotonic(),
            }
            management.append(row)
            write_once(out / 'management' / f'{len(management):03}.json', row)
            if response.status_code != expected:
                raise RuntimeError('管理动作结果不符: ' + path)
            return row['response']

        async def settled():
            deadline = time.monotonic() + 15
            while True:
                status = await manage('/admin/releases')
                if all(row['pending'] == 0 for row in status['versions']):
                    return status
                if time.monotonic() >= deadline:
                    raise RuntimeError('请求未在清理宽限内释放')
                await asyncio.sleep(0.05)

        async def call(
            label,
            payload,
            expected_version=None,
            *,
            stream=False,
            status=200,
            model='active',
            key=None,
        ):
            nonlocal sequence
            reserve(label)
            sequence += 1
            directory = out / 'requests' / f'{sequence:03}-{label}'
            directory.mkdir(parents=True)
            key = key or uuid.uuid4().hex
            body = {
                'model': model,
                'messages': payload['messages'],
                'action_choices': payload['action_choices'],
                'max_tokens': 64,
                'stream': stream,
            }
            write_once(
                directory / 'request.json',
                {
                    'body': body,
                    'input_token_ids': payload['input_token_ids'],
                    'client_key': key,
                    'expected_version': expected_version,
                    'expected_status': status,
                },
            )
            raw = b''
            try:
                async with client.stream(
                    'POST',
                    endpoint + '/v1/chat/completions',
                    json=body,
                    headers={'Idempotency-Key': key},
                ) as response:
                    async for chunk in response.aiter_bytes():
                        raw += chunk
                    headers = dict(response.headers)
                    code = response.status_code
                (directory / 'response.raw').write_bytes(raw)
                content = ''
                finish = None
                usage = None
                version = headers.get('x-model-version')
                if code == 200 and stream:
                    events = [
                        s[5:].strip() for s in raw.decode().splitlines() if s.startswith('data:')
                    ]
                    if not events or events[-1] != '[DONE]':
                        raise ValueError('流没有完整结束')
                    for item in events[:-1]:
                        value = json.loads(item)
                        if value.get('model') != version or 'error' in value:
                            raise ValueError('流中混入其他版本或错误')
                        if value.get('usage'):
                            usage = value['usage']
                        for choice in value.get('choices', []):
                            content += choice.get('delta', {}).get('content') or ''
                            if choice.get('finish_reason'):
                                finish = choice['finish_reason']
                elif code == 200:
                    value = json.loads(raw)
                    if value.get('model') != version:
                        raise ValueError('响应正文与版本头不符')
                    content = value['choices'][0]['message']['content']
                    finish = value['choices'][0]['finish_reason']
                    usage = value['usage']
                receipt = {
                    'label': label,
                    'status': code,
                    'version': version,
                    'profile': headers.get('x-serving-profile'),
                    'evidence': headers.get('x-application-evidence'),
                    'route_generation': headers.get('x-route-generation'),
                    'content': content,
                    'finish_reason': finish,
                    'usage': usage,
                    'headers': headers,
                    'client_key': key,
                    'folder': str(directory),
                }
                write_once(directory / 'receipt.json', receipt)
                requests.append(receipt)
                if code != status:
                    raise ValueError('调用状态不符: ' + label)
                if code == 200:
                    if headers.get('x-qualification-scope') != POOL_SCOPE:
                        raise ValueError('不是应用发布范围')
                    validate_response(receipt, payload, specs, expected_version)
                return receipt
            except BaseException as error:
                (directory / 'response.raw').write_bytes(raw)
                write_once(
                    directory / 'failure.json', {'error': repr(error), 'received_bytes': len(raw)}
                )
                raise

        try:
            initial = await manage('/admin/releases')
            assert initial['route']['generation'] == 0
            from serving_lab.releases import ReleaseError

            rejected = []
            for cert in policy.bundle['certificates'].values():
                if not policy.summary(cert)['passed']:
                    try:
                        policy.require(cert)
                    except ReleaseError as error:
                        assert error.code == 'application_quality_rejected'
                        rejected.append(
                            {
                                'artifact_digest': cert['artifact_digest'],
                                'decision': policy.summary(cert),
                                'layer': 'application adoption predicate; not an HTTP worker call',
                                'error': error.code,
                            }
                        )
                    else:
                        raise ValueError('真实不合格证书未被拒绝')
            if not rejected:
                raise ValueError('缺少真实不合格参数反例')
            write_once(out / 'unqualified-parameter-rejection.json', rejected)
            for rid in ('a', 'b'):
                assert (await manage('/admin/replicas/' + rid + '/qualify', {}))['passed']
            await call('before-publication', payloads[0], status=503, model=versions['c1'])
            route = await manage(
                '/admin/route',
                {'operation': 'promote', 'target': versions['c0'], 'expected_generation': 0},
            )
            pairs = []
            with RuntimeStore(out / 'direct-runtime') as store:
                for index, payload in enumerate(payloads[:2]):
                    reserve('direct-c0')
                    ref = await generate_action(
                        engines[versions['c0']],
                        store,
                        out / 'direct' / f'c0-{index}',
                        payload['messages'],
                        payload['input_token_ids'],
                        {'purpose': 'publication_pair', 'payload': index},
                        choices=payload['action_choices'],
                        readiness_attempts=2,
                    )
                    got = await call(
                        'paired-c0-' + str(index), payload, versions['c0'], stream=bool(index % 2)
                    )
                    assert (
                        ref['content'],
                        ref['finish_reason'],
                        ref['usage']['completion_tokens'],
                    ) == (got['content'], got['finish_reason'], got['usage']['completion_tokens'])
                    direct.append(ref)
                    pairs.append(
                        {
                            'version': versions['c0'],
                            'payload': index,
                            'client_key': got['client_key'],
                            'direct_attempt': ref['attempt_id'],
                        }
                    )
                route = await manage(
                    '/admin/route',
                    {
                        'operation': 'canary',
                        'target': versions['c1'],
                        'expected_generation': route['generation'],
                        'percentage': 50,
                    },
                )
                canaries = [
                    await call('canary-' + str(i), payloads[i % len(payloads)]) for i in range(24)
                ]
                assert {r['version'] for r in canaries} == set(versions.values())
                await settled()
                route = await manage(
                    '/admin/route',
                    {
                        'operation': 'promote',
                        'target': versions['c0'],
                        'expected_generation': route['generation'],
                    },
                )
                keys = [uuid.uuid4().hex for _ in range(16)]
                long_payload = max(payloads, key=lambda p: len(p['input_token_ids']))
                pending = [
                    asyncio.create_task(
                        call('queued-' + str(i), long_payload, versions['c0'], key=key)
                    )
                    for i, key in enumerate(keys)
                ]
                snapshot = []
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    snapshot = [
                        r
                        for r in queue_snapshot(config['runtime'])
                        if r['client_key'] in keys and not r['lease_released']
                    ]
                    if sum(not r['dispatched'] for r in snapshot) >= 8:
                        break
                    await asyncio.sleep(0.005)
                if sum(not r['dispatched'] for r in snapshot) < 8:
                    raise ValueError('未取得足够排队窗口，停止切换试验')
                write_once(out / 'queued-before-switch.json', snapshot)
                route = await manage(
                    '/admin/route',
                    {
                        'operation': 'promote',
                        'target': versions['c1'],
                        'expected_generation': route['generation'],
                    },
                )
                completed = await asyncio.gather(*pending)
                pending = []
                by_key = {r['client_key']: r for r in completed}
                for old in snapshot:
                    assert by_key[old['client_key']]['version'] == old['version']
                    assert (
                        int(by_key[old['client_key']]['route_generation'])
                        == json.loads(old['route'])['route_generation']
                    )
                arrivals = {
                    r['id'] for r in queue_snapshot(config['runtime']) if r['client_key'] in keys
                }
                cutover = queued_cutover_proof(
                    read_events(config['runtime']),
                    arrivals,
                    versions['c0'],
                    versions['c1'],
                    route['generation'],
                )
                write_once(out / 'queued-at-cutover-proof.json', cutover)
                if len(cutover) < 4:
                    raise ValueError('不足四个C0请求在真实切换后才派发，不算在途切换通过')
                await call('after-promotion', payloads[0], versions['c1'])
                for index, payload in enumerate(payloads[:2]):
                    reserve('direct-c1')
                    ref = await generate_action(
                        engines[versions['c1']],
                        store,
                        out / 'direct' / f'c1-{index}',
                        payload['messages'],
                        payload['input_token_ids'],
                        {'purpose': 'publication_pair', 'payload': index},
                        choices=payload['action_choices'],
                        readiness_attempts=2,
                    )
                    got = await call(
                        'paired-c1-' + str(index), payload, versions['c1'], stream=bool(index % 2)
                    )
                    assert (
                        ref['content'],
                        ref['finish_reason'],
                        ref['usage']['completion_tokens'],
                    ) == (got['content'], got['finish_reason'], got['usage']['completion_tokens'])
                    direct.append(ref)
                    pairs.append(
                        {
                            'version': versions['c1'],
                            'payload': index,
                            'client_key': got['client_key'],
                            'direct_attempt': ref['attempt_id'],
                        }
                    )
                assert not store.unresolved()
            assert (await manage('/admin/replicas/a/drain', {}))['drained']
            retired = await manage('/admin/replicas/a/retire', {})
            assert not retired['worker_stop_observed']
            await manage(
                '/admin/route',
                {
                    'operation': 'rollback',
                    'target': versions['c0'],
                    'expected_generation': route['generation'],
                },
                expected=503,
            )
            await manage('/admin/replicas/a/reload', {})
            assert (await manage('/admin/replicas/a/qualify', {}))['passed']
            route = await manage(
                '/admin/route',
                {
                    'operation': 'rollback',
                    'target': versions['c0'],
                    'expected_generation': route['generation'],
                },
            )
            await call('after-rollback', payloads[0], versions['c0'], stream=True)
            final = await settled()
            result = {
                'passed': True,
                'generation_reservations': reserved,
                'client_arrivals': len(requests),
                'direct_calls': len(direct),
                'canary_counts': dict(Counter(r['version'] for r in canaries)),
                'queued_bindings_preserved': len(cutover),
                'pairs': pairs,
                'final_route': final['route'],
                'cold_worker_reload': False,
                'cross_parameter_hot_rollback': True,
                'unqualified_parameter_rejected': True,
                'raw_audit_status': 'pending',
                'scope': 'qualified C0/C1 controlled cloud publication, not controller HA',
            }
            write_once(out / 'summary.json', result)
            return result
        except BaseException as error:
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            write_once(
                out / 'failure.json',
                {
                    'error': repr(error),
                    'reserved': reserved,
                    'client_receipts': len(requests),
                    'implicit_retry': False,
                },
            )
            raise
        finally:
            for engine in engines.values():
                await engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(asyncio.run(run(a.config, a.input, a.output)), ensure_ascii=False, indent=2))
