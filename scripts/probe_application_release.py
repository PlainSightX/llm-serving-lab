"""本机真实C0应用发布：质量拒绝、在途撤回、网关闭环与同参数实例恢复。"""

import argparse
import asyncio
import json
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from serving_lab.application_release import load_policy, POOL_SCOPE, validate_transport_panel
from serving_lab.engine import check_port_available, process_identity
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.measurement_runtime import export_arrivals
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.standard_actor import StandardActor
from serving_lab.task_protocol import validate_manifest, verify_files
from serving_lab.worker_protocol import RemoteEngine
from q2_run import open_owned_bridge, docker
from q2_teacher import IMAGE, read
from start_engine import start, stop
from probe_serving import probe

REPO = Path(__file__).resolve().parents[1]
CAP = 680


def same_pair_output(left, right):
    """vLLM流式省略空的prompt明细；保留原件，比较全部实质计量和输出。"""

    def usage(value):
        return {k: v for k, v in value.items() if not (k == 'prompt_tokens_details' and v is None)}

    return (
        left['content'] == right['content']
        and left.get('finish_reason') == right.get('finish_reason')
        and usage(left['usage']) == usage(right['usage'])
    )


def reservation_count(root):
    """预约先于调用写入；失败与拒绝占用的预算也不退还。"""
    total = 0
    path = Path(root) / 'reservations.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        count = row['reservation']
        if type(count) is not int or count <= 0 or row['total'] != total + count:
            raise ValueError('预约账本计数不连续')
        total += count
    return total


def prior_failures(paths, evidence_sha256):
    """修复重跑显式携带所有失败前轮；离线审计复核同一组原件摘要。"""
    roots = [Path(p).resolve() for p in paths]
    if len(set(roots)) != len(roots):
        raise ValueError('失败前轮不能重复计数')
    records = []
    for root in roots:
        if not (root / 'failure.json').exists() or (root / 'summary.json').exists():
            raise ValueError('只允许携带已失败且未成功收口的前轮')
        initial, failure, cleanup = (
            read(root / name) for name in ('input.json', 'failure.json', 'cleanup.json')
        )
        if initial['evidence_sha256'] != evidence_sha256 or initial['generation_cap'] != CAP:
            raise ValueError('失败前轮不是同一证据与有界工作包')
        if any(
            Path(r['path']).resolve() not in roots or Path(r['path']).resolve() == root
            for r in initial.get('prior_failed_runs', [])
        ):
            raise ValueError('遗漏了失败前轮的既有成本来源')
        if (
            cleanup['errors']
            or any(
                not w['result']['stopped'] or w['result'].get('remaining')
                for w in cleanup['workers']
            )
            or any(p['returncode'] is None for p in cleanup['processes'])
        ):
            raise ValueError('前轮自有清理未完成')
        count = reservation_count(root)
        if count != failure['generation_reservations']:
            raise ValueError('失败摘要漏计预约')
        names = ['input.json', 'failure.json', 'cleanup.json']
        names += [
            n
            for n in ('reservations.jsonl', 'startup-failure-cleanup-verification.json')
            if (root / n).exists()
        ]
        records.append(
            {
                'path': str(root),
                'reservations': count,
                'files': {n: file_hash(root / n) for n in names},
            }
        )
    if sum(r['reservations'] for r in records) >= CAP:
        raise ValueError('工作包预算已耗尽')
    return records


class GatewayClient:
    """服务端拥有arrival/attempt；客户端保留原始请求与响应，不冒充后端ID。"""

    def __init__(
        self, http, endpoint, output, reserve, version, profile_digest, evidence_digest, tokenize
    ):
        self.http, self.endpoint, self.output, self.reserve = http, endpoint, output, reserve
        self.version, self.profile_digest, self.evidence_digest, self.tokenize = (
            version,
            profile_digest,
            evidence_digest,
            tokenize,
        )
        self.sequence = 0

    async def call(self, body, label, *, expected=200, key=None):
        self.reserve(1)
        self.sequence += 1
        out = self.output / f'{self.sequence:04}-{label}'
        out.mkdir(parents=True, exist_ok=False)
        key = key or uuid.uuid4().hex
        ids = self.tokenize(body['messages'])
        write_once(
            out / 'request.json',
            {'body': body, 'client_key': key, 'input_token_ids': ids, 'expected_status': expected},
        )
        started = time.monotonic()
        async with self.http.stream(
            'POST',
            self.endpoint + '/v1/chat/completions',
            json=body,
            headers={'Idempotency-Key': key},
        ) as response:
            raw = await response.aread()
            (out / 'response.raw').write_bytes(raw)
            headers = dict(response.headers)
            value = None
            content = ''
            finish = None
            usage = None
            if body.get('stream') and response.status_code == 200:
                events = [
                    line[5:].strip()
                    for line in raw.decode().splitlines()
                    if line.startswith('data:')
                ]
                if not events or events[-1] != '[DONE]':
                    raise ValueError('流不完整')
                for event in events[:-1]:
                    parsed = json.loads(event)
                    if parsed.get('model') != self.version or 'error' in parsed:
                        raise ValueError('流模型身份/协议不符')
                    if parsed.get('usage'):
                        usage = parsed['usage']
                    for choice in parsed.get('choices', []):
                        content += choice.get('delta', {}).get('content') or ''
                        if choice.get('finish_reason'):
                            finish = choice['finish_reason']
                value = {'events': events}
            else:
                value = json.loads(raw)
                if response.status_code == 200:
                    if value.get('model') != self.version:
                        raise ValueError('响应模型身份不符')
                    content = value['choices'][0]['message']['content']
                    finish = value['choices'][0]['finish_reason']
                    usage = value['usage']
            receipt = {
                'status': response.status_code,
                'headers': headers,
                'value': value,
                'content': content,
                'finish_reason': finish,
                'usage': usage,
                'elapsed_seconds': time.monotonic() - started,
                'client_key': key,
                'folder': str(out),
            }
            write_once(out / 'receipt.json', receipt)
        accepted = [expected] if isinstance(expected, int) else list(expected)
        if receipt['status'] not in accepted:
            raise ValueError(f'{label}: unexpected HTTP {receipt["status"]}')
        if receipt['status'] == 200:
            if (
                headers.get('x-model-version') != self.version
                or headers.get('x-serving-profile') != self.profile_digest
                or headers.get('x-qualification-scope') != POOL_SCOPE
                or headers.get('x-application-evidence') != self.evidence_digest
                or not headers.get('x-replica-id')
                or not headers.get('x-worker-generation')
            ):
                raise ValueError('网关实际版本/应用证据绑定不符')
            StandardActor().action(receipt, body['action_choices'])
            if (
                usage['prompt_tokens'] != len(ids)
                or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']
            ):
                raise ValueError('网关输入token不符')
        return receipt


async def application_episode(task, data, output, client):
    if task['role'] != 'support':
        raise ValueError('本包仅运行旧公开support集成任务')
    bridge = None
    trace = []
    started = time.monotonic()
    try:
        bridge = await open_owned_bridge(task, data, output, IMAGE, 100, 4)
        value = await asyncio.wait_for(asyncio.to_thread(bridge.read), 30)
        initial = view = value['public']
        while not value['terminal']:
            prepared = StandardActor().prepare(initial, view, trace, client.tokenize)
            body = {
                'model': 'active',
                'messages': prepared['messages'],
                'action_choices': prepared['choices'],
                'max_tokens': 64,
                'stream': False,
            }
            receipt = await client.call(body, task['task_id'] + '-' + str(len(trace) + 1))
            action = StandardActor().action(receipt, prepared['choices'])
            before = view
            value = await bridge.step(action)
            view = value['public']
            trace.append(
                {
                    'step': len(trace) + 1,
                    'before': before,
                    'action': action,
                    'after': view,
                    'receipt': receipt,
                    'context': prepared['context'],
                }
            )
        record = {
            'producer': 'serving_lab.gateway_episode.v1',
            'purpose': 'publication_integration',
            'task_id': task['task_id'],
            'role': task['role'],
            'initial': initial,
            'trace': trace,
            'result': value['result'],
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
            'elapsed_seconds': time.monotonic() - started,
        }
        record['record_hash'] = digest(record)
        write_once(output / 'record.json', record)
        print(
            json.dumps(
                {'task_id': task['task_id'], 'result': record['result']}, ensure_ascii=False
            ),
            flush=True,
        )
        return record
    except BaseException as exc:
        if output.exists():
            write_once(
                output / 'failure.json', {'error': repr(exc), 'steps': len(trace), 'trace': trace}
            )
        raise
    finally:
        if bridge:
            await asyncio.to_thread(bridge.close)


async def run(output, evidence, previous=()):
    evidence = Path(evidence).resolve()
    imported = read(evidence.parent / 'import-receipt.json')
    if Path(imported['bundle']).resolve() != evidence:
        raise ValueError('证据包不是指定导入回执的对象')
    policy = load_policy(evidence, imported['sha256'])
    previous = prior_failures(previous, imported['sha256'])
    prior_count = sum(r['reservations'] for r in previous)
    certified = next(
        c for c in policy.bundle['certificates'].values() if policy.summary(c)['passed']
    )
    quality_input = Path(certified['source']['run']) / 'input.json'
    if file_hash(quality_input) != certified['source']['input_sha256']:
        raise ValueError('原评价输入变化')
    parity_data = REPO / 'artifacts/q2-coldstart-dataset-20261103-a/train.jsonl'
    parity_hash = file_hash(parity_data)
    if parity_hash != read(quality_input)['controls']['data_sha256']:
        raise ValueError('配对输入不再是原训练来源')
    data = REPO / 'artifacts/q1-data-20261102'
    support_manifest = read(data / 'manifest.json')
    support_hash = file_hash(data / 'manifest.json')
    if support_hash != read(data / 'manifest-anchor.json')['sha256']:
        raise ValueError('Q1支持任务清单与原anchor不符')
    tasks = validate_manifest(support_manifest)
    for i in range(1, 7):
        verify_files(tasks[f'q1_support_{i:02}'], data)
    panel = read(REPO / 'fixtures/serving/r2-transport-v2.json')
    validate_transport_panel(panel)
    if json.loads(docker('image', 'inspect', IMAGE))[0]['Id'] != IMAGE:
        raise ValueError('冻结环境镜像不可用')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'support-manifest.json').write_bytes((data / 'manifest.json').read_bytes())
    (out / 'transport-panel.json').write_bytes(
        (REPO / 'fixtures/serving/r2-transport-v2.json').read_bytes()
    )
    bundle = policy.bundle
    handles = []
    bindings = []
    remotes = []
    stages = []
    reservations = []
    client_tasks = []
    sources = [
        'scripts/' + n
        for n in (
            'probe_application_release.py',
            'serve_replicas.py',
            'serve_worker.py',
            'q2_run.py',
            'q2_environment.py',
            'start_engine.py',
            'probe_serving.py',
        )
    ]
    sources += ['src/serving_lab/' + p.name for p in (REPO / 'src/serving_lab').glob('*.py')]
    sources += [
        'fixtures/serving/r2-transport-v2.json',
        'configs/serving/r3-large.json',
        'configs/evaluation/q1-sources.json',
    ]
    hashes = {n: file_hash(REPO / n) for n in sources}
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'source_hashes': hashes,
            'application_evidence': str(evidence),
            'evidence_sha256': imported['sha256'],
            'import_receipt_sha256': file_hash(evidence.parent / 'import-receipt.json'),
            'bundle_digest': bundle['bundle_digest'],
            'generation_cap': CAP,
            'scope': 'local application publication; no cloud or new learning score',
            'prior_failed_runs': previous,
            'prior_generation_reservations': prior_count,
            'supports': [f'q1_support_{i:02}' for i in range(1, 7)],
            'transport_panel': panel,
            'support_data': str(data),
            'support_manifest_sha256': support_hash,
            'environment_image': IMAGE,
            'parity_data': str(parity_data),
            'parity_data_sha256': parity_hash,
            'quality_result_not_remeasured': 'C0 23/24 and base0/24 original dev receipts',
        },
    )

    def reserve(count):
        if prior_count + sum(reservations) + count > CAP:
            raise RuntimeError('publication generation reservation exhausted')
        reservations.append(count)
        with (out / 'reservations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'reservation': count, 'total': sum(reservations)}) + '\n')

    def spawn(label, args):
        with (out / (label + '.log')).open('xb') as log:
            process = subprocess.Popen(
                [sys.executable, *map(str, args)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        handles.append((label, process))
        write_once(out / (label + '-process.json'), process_identity(process.pid))
        return process

    def stop_child(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    completed = None
    async with httpx.AsyncClient(timeout=120, trust_env=False) as http:

        async def wait_url(url, process, headers=None):
            for _ in range(400):
                if process.poll() is not None:
                    raise RuntimeError('owned service startup failed')
                try:
                    response = await http.get(url, headers=headers, timeout=2)
                    if response.status_code == 200:
                        return response.json()
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
            raise RuntimeError('owned service startup timeout')

        try:
            for port in (19080, 19082, 19180, 19182, 19183):
                check_port_available('127.0.0.1', port)
            worker_token = secrets.token_hex(24)
            worker_file = out / 'worker-token'
            worker_file.write_text(worker_token)
            worker_file.chmod(0o600)
            admin_token = secrets.token_hex(24)
            admin_file = out / 'admin-token'
            admin_file.write_text(admin_token)
            admin_file.chmod(0o600)
            model_configs = {}
            manifests = {}
            bridges = {}
            endpoints = {'base': 19180, 'c0': 19182}
            # 两个不同真实参数同时健康，才能观察质量门的拒绝。
            for role, engine_port in [('base', 19080), ('c0', 19082)]:
                cfg = read(REPO / 'configs/serving/r3-large.json')
                cfg.update(
                    port=engine_port,
                    gpu_memory_utilization=0.40,
                    kv_cache_memory_bytes=768 * 1024**2,
                )
                if role == 'c0':
                    cfg.update(
                        version_id='qwen3-17b-coldstart-candidate',
                        model_path='artifacts/q2-coldstart-training-20261103-b/merged',
                        model_revision=read(
                            REPO / 'artifacts/q2-coldstart-training-20261103-b/summary.json'
                        )['adapter_sha256'],
                    )
                config = out / (role + '-config.json')
                write_once(config, cfg)
                model_configs[role] = config
                folder = out / (role + '-worker')
                binding = folder / 'binding.json'
                try:
                    start(config, folder)
                finally:
                    if binding.exists():
                        bindings.append(binding)
                reserve(2)
                probe(binding, out / (role + '-bootstrap'))
                manifest = out / (role + '-manifest.json')
                manifests[role] = manifest
                bridge = spawn(
                    role + '-bridge',
                    [
                        REPO / 'scripts/serve_worker.py',
                        '--binding',
                        binding,
                        '--token-file',
                        worker_file,
                        '--port',
                        endpoints[role],
                        '--manifest-output',
                        manifest,
                        '--profile-policy',
                        'q2_public_choice_v1',
                    ],
                )
                bridges[role] = bridge
                identity = await wait_url(
                    f'http://127.0.0.1:{endpoints[role]}/identity',
                    bridge,
                    {'Authorization': 'Bearer ' + worker_token},
                )
                if not identity['ready']:
                    raise ValueError('worker not ready')
            by_role = {role: read(path) for role, path in manifests.items()}
            versions = {role: m['version_id'] for role, m in by_role.items()}
            c0_cert = bundle['certificates'][by_role['c0']['artifact_digest']]
            if (
                not policy.summary(c0_cert)['passed']
                or policy.summary(bundle['certificates'][by_role['base']['artifact_digest']])[
                    'passed'
                ]
            ):
                raise ValueError('证据不再对应唯一合格C0和不合格基座')
            config = {
                'replicas': [
                    {
                        'id': role,
                        'endpoint': f'http://127.0.0.1:{endpoints[role]}',
                        'manifest': str(manifests[role]),
                        'token_file': str(worker_file),
                        'capacity': {'max_inflight': 1, 'max_queue': 2},
                    }
                    for role in ('base', 'c0')
                ],
                'panel': str(out / 'transport-panel.json'),
                'runtime': str(out / 'runtime'),
                'port': 19183,
                'admin_token_file': str(admin_file),
                'application_release': {'bundle': str(evidence), 'sha256': imported['sha256']},
                'limits': {
                    'max_inflight': 2,
                    'max_queue': 4,
                    'max_input_tokens': 3500,
                    'max_tokens': 64,
                    'context_tokens': 4096,
                    'deadline_seconds': 90,
                    'cancel_grace_seconds': 10,
                },
            }
            cfg_path = out / 'gateway.json'
            write_once(cfg_path, config)
            gateway = spawn('gateway', [REPO / 'scripts/serve_replicas.py', '--config', cfg_path])
            url = 'http://127.0.0.1:19183'
            auth = {'Authorization': 'Bearer ' + admin_token}
            await wait_url(url + '/admin/releases', gateway, auth)

            async def state():
                response = await http.get(url + '/admin/releases', headers=auth)
                response.raise_for_status()
                return response.json()

            async def manage(path, body=None, expected=200):
                response = await http.post(url + path, headers=auth, json=body or {})
                value = response.json()
                entry = {
                    'path': path,
                    'body': body,
                    'status': response.status_code,
                    'response': value,
                }
                stages.append(entry)
                with (out / 'stages.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry) + '\n')
                if response.status_code != expected:
                    raise ValueError(f'{path}: {response.status_code} {value}')
                return value

            async def drained():
                for _ in range(300):
                    value = await state()
                    if all(r['pending'] == 0 for r in value['replicas']):
                        return value
                    await asyncio.sleep(0.05)
                raise ValueError('request leases did not drain')

            for role in ('base', 'c0'):
                reserve(8)
                q = await manage('/admin/replicas/' + role + '/qualify')
                if not q['passed']:
                    raise ValueError('fresh worker qualification failed')
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                REPO / 'artifacts/q2-coldstart-training-20261103-b/merged',
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenize = lambda ms: tokenizer.apply_chat_template(
                ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            client = GatewayClient(
                http,
                url,
                out / 'calls',
                reserve,
                versions['c0'],
                by_role['c0']['profile_digest'],
                digest(c0_cert),
                tokenize,
            )
            samples = [json.loads(line) for line in parity_data.read_text().splitlines()]
            selected = []
            task_ids = set()
            for row in sorted(
                samples,
                key=lambda r: (
                    -len(tokenizer.encode(r['messages'][-1]['content'])),
                    r['task_id'],
                    r['step'],
                ),
            ):
                if row['task_id'] in task_ids:
                    continue
                selected.append(row)
                task_ids.add(row['task_id'])
                if len(selected) == 3:
                    break
            write_once(out / 'probe-inputs.json', selected)

            def public_body(row, stream=False):
                return {
                    'model': 'active',
                    'messages': row['messages'][:-1],
                    'action_choices': json.loads(row['messages'][-2]['content'])[
                        'CURRENT_COMMANDS'
                    ],
                    'max_tokens': 64,
                    'stream': stream,
                }

            body = public_body(selected[0])
            await client.call(body, 'before-publish', expected=503)
            for operation in ('promote', 'canary', 'rollback'):
                bad = await manage(
                    '/admin/route',
                    {
                        'operation': operation,
                        'target': versions['base'],
                        'expected_generation': 0,
                        'percentage': 10,
                    },
                    expected=409,
                )
                if bad['error']['code'] != 'application_quality_rejected':
                    raise ValueError('quality gate was not the rejection reason')
            bad = await client.call(
                {**body, 'model': versions['base']}, 'explicit-bad-base', expected=409
            )
            if bad['value']['error']['code'] != 'application_quality_rejected':
                raise ValueError('explicit request bypassed application gate')
            await manage(
                '/admin/route',
                {'operation': 'promote', 'target': versions['c0'], 'expected_generation': 0},
            )
            await client.call({**body, 'action_choices': ['look']}, 'wrong-choices', expected=422)
            await client.call({**body, 'temperature': 0.0}, 'override-profile', expected=422)
            remote = RemoteEngine(
                f'http://127.0.0.1:{endpoints["c0"]}', worker_token, by_role['c0']
            )
            remotes.append(remote)
            pairs = []
            with RuntimeStore(out / 'direct-runtime') as store:
                for i, row in enumerate(selected, 1):
                    b = public_body(row)
                    ids = tokenize(b['messages'])
                    reserve(1)
                    direct = await generate_action(
                        remote,
                        store,
                        out / 'direct' / str(i),
                        b['messages'],
                        ids,
                        {'purpose': 'publication_interface_pair', 'pair': i},
                        choices=b['action_choices'],
                    )
                    routed = await client.call(public_body(row, stream=i == 3), f'paired-{i}')
                    if not same_pair_output(direct, routed):
                        raise ValueError('direct/gateway pair differs')
                    pairs.append({'pair': i, 'direct': direct, 'gateway': routed})
            write_once(out / 'pairs.json', pairs)
            records = []
            for i in range(1, 7):
                task = tasks[f'q1_support_{i:02}']
                records.append(
                    await application_episode(task, data, out / 'support' / task['task_id'], client)
                )
            await drained()
            # 同一批三个请求产生可观察的已绑定队列；撤回不要求请求仍占GPU计算槽。
            withdrawn = None
            in_flight = []
            for attempt in range(1, 4):
                batch = [
                    asyncio.create_task(
                        client.call(
                            body,
                            f'withdraw-{attempt}-{i}',
                            key=f'withdraw-{attempt}-{i}',
                            expected=[200, 503],
                        )
                    )
                    for i in range(3)
                ]
                client_tasks.extend(batch)
                for _ in range(300):
                    observed = await state()
                    if next(r for r in observed['replicas'] if r['id'] == 'c0')['pending'] >= 2:
                        break
                    if all(t.done() for t in batch):
                        break
                    await asyncio.sleep(0.005)
                observed = await state()
                if next(r for r in observed['replicas'] if r['id'] == 'c0')['pending']:
                    withdrawn = await manage(
                        '/admin/route',
                        {
                            'operation': 'withdraw',
                            'target': versions['c0'],
                            'expected_generation': observed['route']['generation'],
                        },
                    )
                results = await asyncio.gather(*batch)
                in_flight.extend(results)
                for result in results:
                    if (
                        result['status'] == 503
                        and result['value']['error']['code'] != 'no_active_version'
                    ):
                        raise ValueError('withdraw batch failed for an unrelated reason')
                if withdrawn:
                    bound = set(withdrawn['bound_arrivals'])
                    if bound:
                        if not bound <= {
                            r['headers']['x-arrival-id'] for r in results if r['status'] == 200
                        }:
                            raise ValueError('bound request failed or changed during withdrawal')
                        dedup_key = next(
                            r['client_key']
                            for r in results
                            if r['headers']['x-arrival-id'] in bound
                        )
                        break
                    # 时序未捕获在途绑定，只算一次保留的时序尝试；重新显式发布再做下一批。
                    await manage(
                        '/admin/route',
                        {
                            'operation': 'promote',
                            'target': versions['c0'],
                            'expected_generation': withdrawn['generation'],
                        },
                    )
                    withdrawn = None
            if not withdrawn:
                raise ValueError('bounded in-flight withdraw observation unavailable')
            write_once(out / 'withdraw-requests.json', in_flight)
            await client.call(body, 'withdrawn-active', expected=503)
            await client.call({**body, 'model': versions['c0']}, 'withdrawn-explicit', expected=503)
            bad = await manage(
                '/admin/route',
                {
                    'operation': 'rollback',
                    'target': versions['base'],
                    'expected_generation': withdrawn['generation'],
                },
                expected=409,
            )
            if bad['error']['code'] != 'application_quality_rejected':
                raise ValueError('invalid fallback accepted')
            await drained()
            restored = await manage(
                '/admin/route',
                {
                    'operation': 'promote',
                    'target': versions['c0'],
                    'expected_generation': withdrawn['generation'],
                },
            )
            await client.call(body, 'republished')
            await client.call(body, 'dedup-after-restore', key=dedup_key, expected=409)
            stop_child(gateway)
            gateway = spawn(
                'gateway-restarted', [REPO / 'scripts/serve_replicas.py', '--config', cfg_path]
            )
            recovered = await wait_url(url + '/admin/releases', gateway, auth)
            if recovered['route'] != restored:
                raise ValueError('controller route did not persist')
            write_once(
                out / 'controller-recovery.json',
                {'expected_route': restored, 'observed': recovered},
            )
            await client.call(body, 'controller-restarted')
            await drained()
            # 实例在无在途请求时停止；不把这次故障称为生成中kill或跨参数回滚。
            stopped = stop(bindings[1])
            write_once(out / 'c0-worker-stop.json', stopped)
            if not stopped['stopped']:
                raise ValueError('old C0 generation still alive')
            offline = await client.call(body, 'c0-worker-offline', expected=503)
            alternative = (
                await http.get(
                    f'http://127.0.0.1:{endpoints["base"]}/identity',
                    headers={'Authorization': 'Bearer ' + worker_token},
                )
            ).json()
            if not alternative['ready'] or alternative['manifest'] != by_role['base']:
                raise ValueError('rejected alternative was not a live original base worker')
            write_once(
                out / 'outage.json', {'gateway_rejection': offline, 'base_still_ready': alternative}
            )
            bad = await manage(
                '/admin/route',
                {
                    'operation': 'rollback',
                    'target': versions['base'],
                    'expected_generation': restored['generation'],
                },
                expected=409,
            )
            if bad['error']['code'] != 'application_quality_rejected':
                raise ValueError('quality gate failed during outage')
            stop_child(gateway)
            stop_child(bridges['c0'])
            replacement = out / 'c0-worker-rebound/binding.json'
            try:
                start(model_configs['c0'], replacement.parent)
            finally:
                if replacement.exists():
                    bindings.append(replacement)
            reserve(2)
            probe(replacement, out / 'c0-rebound-bootstrap')
            new_manifest = out / 'c0-rebound-manifest.json'
            bridge = spawn(
                'c0-bridge-rebound',
                [
                    REPO / 'scripts/serve_worker.py',
                    '--binding',
                    replacement,
                    '--token-file',
                    worker_file,
                    '--port',
                    endpoints['c0'],
                    '--manifest-output',
                    new_manifest,
                    '--profile-policy',
                    'q2_public_choice_v1',
                ],
            )
            new_identity = await wait_url(
                f'http://127.0.0.1:{endpoints["c0"]}/identity',
                bridge,
                {'Authorization': 'Bearer ' + worker_token},
            )
            if new_identity['manifest']['identity'] == by_role['c0']['identity']:
                raise ValueError('replacement reused old identity')
            if not new_identity['ready'] or new_identity['manifest'] != read(new_manifest):
                raise ValueError('replacement manifest not ready')
            config['replicas'][1]['manifest'] = str(new_manifest)
            config['allow_rebind'] = ['c0']
            rebound_config = out / 'gateway-rebound.json'
            write_once(rebound_config, config)
            gateway = spawn(
                'gateway-rebound', [REPO / 'scripts/serve_replicas.py', '--config', rebound_config]
            )
            value = await wait_url(url + '/admin/releases', gateway, auth)
            if next(r for r in value['replicas'] if r['id'] == 'c0')['state'] != 'loading':
                raise ValueError('new generation inherited readiness')
            await client.call(body, 'before-requalification', expected=503)
            reserve(8)
            await manage('/admin/replicas/c0/qualify')
            final = await client.call(body, 'rebound-serving')
            if (
                final['headers']['x-worker-generation']
                != new_identity['manifest']['identity']['startup_generation']
            ):
                raise ValueError('wrong replacement served')
            await drained()
            export_arrivals(out / 'runtime/service.sqlite', out / 'arrivals.json')
            if not all(file_hash(REPO / n) == h for n, h in hashes.items()):
                raise ValueError('source changed during execution')
            completed = {
                'verified_execution': True,
                'generation_reservations': sum(reservations),
                'generation_cap': CAP,
                'prior_generation_reservations': prior_count,
                'package_generation_reservations': prior_count + sum(reservations),
                'support_integration': {
                    'episodes': 6,
                    'successes': sum(r['result']['success'] for r in records),
                    'actions': sum(len(r['trace']) for r in records),
                },
                'paired_requests': 3,
                'inflight_withdraw_pending': withdrawn['bound_requests_at_withdraw'],
                'application_quality_rejection': True,
                'controller_restart': True,
                'same_parameter_generation_recovery': True,
                'cross_parameter_application_rollback': False,
                'scope': 'local published application runtime exercised and stopped; no cluster, production or new learning claim',
            }
        except BaseException as exc:
            write_once(
                out / 'failure.json',
                {
                    'error': repr(exc),
                    'generation_reservations': sum(reservations),
                    'prior_generation_reservations': prior_count,
                    'package_generation_reservations': prior_count + sum(reservations),
                },
            )
            raise
        finally:
            errors = []
            for task in client_tasks:
                if not task.done():
                    task.cancel()
            if client_tasks:
                await asyncio.gather(*client_tasks, return_exceptions=True)
            for e in remotes:
                try:
                    await e.close()
                except Exception as exc:
                    errors.append({'remote_close': repr(exc)})
            for label, process in reversed(handles):
                try:
                    stop_child(process)
                except Exception as exc:
                    errors.append({'process': label, 'error': repr(exc)})
            workers = []
            for path in bindings:
                try:
                    workers.append({'binding': str(path), 'result': stop(path)})
                except Exception as exc:
                    errors.append({'binding': str(path), 'error': repr(exc)})
            cleanup = {
                'errors': errors,
                'workers': workers,
                'processes': [{'label': label, 'returncode': p.returncode} for label, p in handles],
            }
            write_once(out / 'cleanup.json', cleanup)
            if errors or any(
                not w['result']['stopped'] or w['result'].get('remaining') for w in workers
            ):
                raise ValueError('owned cleanup incomplete')
    if completed:
        write_once(out / 'summary.json', completed)
    return completed


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--prior-run', action='append', default=[])
    a = p.parse_args()
    print(
        json.dumps(
            asyncio.run(run(a.output, a.evidence, a.prior_run)), ensure_ascii=False, indent=2
        )
    )
