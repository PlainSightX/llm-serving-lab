"""有界公开信息试点：自有CPU容器经stdio交互，模型在独立WSL环境。"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.profile import application_history
from serving_lab.public_tasks import (
    PublicPolicy,
    action_context,
    verified_history,
    matched_histories,
)
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.task_protocol import canonical, digest, validate_manifest
from serving_lab.transport import qualify_transport
from start_engine import start, stop
from probe_serving import probe

DOCKER = 'docker.exe'
REPO = Path(__file__).resolve().parents[1]
BRIDGE_TIMEOUT = 30


def windows(path):
    resolved = str(Path(path).resolve())
    if sys.platform == 'win32':
        return resolved
    return subprocess.check_output(['wslpath', '-w', resolved], text=True).strip()


def docker(*args):
    return subprocess.check_output([DOCKER, *args], text=True, timeout=60)


class Bridge:
    def __init__(self, task, data, out, image, max_steps, stagnation_limit=None):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.process = None
        self.cid = None
        self.run_id = uuid.uuid4().hex
        self.name = 'lsl-q2-' + self.run_id[:12]
        self.log = (out / 'bridge.stderr').open('xb')
        args = [
            'create',
            '-i',
            '--name',
            self.name,
            '--label',
            'serving_lab.owner=q2',
            '--label',
            'serving_lab.run_id=' + self.run_id,
            '--network',
            'none',
            '--read-only',
            '--cap-drop',
            'ALL',
            '--security-opt',
            'no-new-privileges',
            '--pids-limit',
            '128',
            '--memory',
            '2g',
            '--cpus',
            '2',
            '--tmpfs',
            '/tmp:rw,exec,nosuid,nodev,size=256m',
            '--env',
            'ALFWORLD_DATA=/tmp/alfworld',
            '--env',
            'PYTHONPATH=/code',
            '--user',
            '65534:65534',
        ]
        for source, target, ro in [
            (REPO / 'src', '/code', True),
            (REPO / 'scripts/q2_environment.py', '/runner.py', True),
            (data, '/data', True),
            (out, '/output', False),
        ]:
            args += [
                '--mount',
                f'type=bind,source={windows(source)},target={target}' + (',readonly' if ro else ''),
            ]
        args += [image, '/runner.py']
        self.cid = docker(*args).strip()
        try:
            write_once(
                out / 'launch.json',
                {
                    'container_id': self.cid,
                    'name': self.name,
                    'run_id': self.run_id,
                    'command': args,
                    'image': image,
                },
            )
            self.process = subprocess.Popen(
                [DOCKER, 'start', '--attach', '--interactive', self.cid],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
            )
            self.send(
                {
                    'task_id': task['task_id'],
                    'max_steps': max_steps,
                    'stagnation_limit': stagnation_limit,
                }
            )
        except BaseException:
            self.close()
            raise

    def send(self, value):
        self.process.stdin.write((canonical(value) + '\n').encode())
        self.process.stdin.flush()

    def read(self):
        # 私有环境输出只能是单行JSON。阻塞读取放在线程并有明确外部deadline。
        raw = self.process.stdout.readline()
        if not raw:
            raise RuntimeError('环境桥异常结束；见stderr')
        return json.loads(raw)

    async def step(self, action):
        self.send({'action': action})
        self.value = await asyncio.wait_for(asyncio.to_thread(self.read), 30)
        return self.value

    def close(self):
        if not self.cid:
            return
        info = json.loads(docker('inspect', self.cid))[0]
        labels = info['Config']['Labels']
        if (
            info['Name'] != '/' + self.name
            or labels.get('serving_lab.owner') != 'q2'
            or labels.get('serving_lab.run_id') != self.run_id
        ):
            raise RuntimeError('拒绝清理非自有容器')
        if self.process:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                docker('stop', '--time', '2', self.cid)
                self.process.wait(timeout=8)
        info = json.loads(docker('inspect', self.cid))[0]
        if info['State']['Running']:
            docker('stop', '--time', '2', self.cid)
            info = json.loads(docker('inspect', self.cid))[0]
        try:
            write_once(self.out / 'exit.json', info['State'])
        finally:
            try:
                docker('rm', self.cid)
                self.cid = None
            finally:
                self.log.close()
        if info['State'].get('ExitCode') != 0:
            raise RuntimeError('环境容器非零退出；即使曾发送terminal也不得接纳回合')


async def open_owned_bridge(task, data, out, image, max_steps, stagnation_limit=None):
    """即使调用者在构造期间取消，也取得确切容器所有权后清理。"""
    options = {'stagnation_limit': stagnation_limit} if stagnation_limit is not None else {}
    construction = asyncio.create_task(
        asyncio.to_thread(Bridge, task, data, out, image, max_steps, **options)
    )
    try:
        return await asyncio.shield(construction)
    except asyncio.CancelledError:
        bridge = await construction
        await asyncio.to_thread(bridge.close)
        raise


async def episode(
    task,
    data,
    out,
    image,
    engine=None,
    store=None,
    history=None,
    condition='public_policy',
    max_steps=100,
    history_pair=None,
    *,
    stagnation_limit=None,
    instruction=None,
    purpose=None,
    action_parser=None,
    actor=None,
    source_contract_digest=None,
):
    if purpose == 'teacher_coldstart_source':
        if (
            engine is not None
            or task['role'] not in {'train', 'dev'}
            or not isinstance(source_contract_digest, str)
            or len(source_contract_digest) != 64
            or any(c not in '0123456789abcdef' for c in source_contract_digest)
        ):
            raise ValueError('公开教师只允许新来源合同和程序策略，禁止模型或旧任务重标')
    elif source_contract_digest is not None:
        raise ValueError('来源合同不适用于该用途')
    if actor is not None and (
        engine is None
        or history is not None
        or history_pair is not None
        or instruction is not None
        or action_parser is not None
        or purpose != 'actor_diagnostic'
    ):
        raise ValueError('标准actor只用于显式开发诊断；禁止静默忽略历史或覆盖提示')
    bridge = None
    trace = []
    started = time.monotonic()
    try:
        bridge = await open_owned_bridge(task, data, out, image, max_steps, stagnation_limit)
        bridge.value = await asyncio.wait_for(asyncio.to_thread(bridge.read), BRIDGE_TIMEOUT)
        initial = bridge.value['public']
        view = initial
        policy = PublicPolicy() if engine is None else None
        if history_pair:
            matched_histories(*history_pair, engine.tokenizer, initial['feedback'], initial)
        while not bridge.value['terminal']:
            receipt = None
            context = None
            if engine:
                tokenizer = lambda ms: engine.tokenizer.apply_chat_template(
                    ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                choices = None
                if actor is not None:
                    prepared = actor.prepare(initial, view, trace, tokenizer)
                    messages, ids, context, choices = (
                        prepared[k] for k in ('messages', 'input_token_ids', 'context', 'choices')
                    )
                else:
                    options = {'instruction': instruction} if instruction is not None else {}
                    messages, ids, context = action_context(
                        initial['feedback'], view, trace, history, tokenizer, 3000, **options
                    )
                generation_options = {'choices': choices} if choices is not None else {}
                receipt = await generate_action(
                    engine,
                    store,
                    out / f'action-{len(trace) + 1:03}',
                    messages,
                    ids,
                    {
                        'task_id': task['task_id'],
                        'step': len(trace) + 1,
                        'condition': condition,
                        'public_digest': digest(view),
                        'context': context,
                    },
                    **generation_options,
                )
                action = (
                    actor.action(receipt, choices)
                    if actor is not None
                    else (
                        action_parser(receipt['content'])
                        if action_parser
                        else receipt['content'].strip()
                    )
                )
            else:
                action = policy.choose(view)
            before = view
            value = await bridge.step(action)
            view = value['public']
            step = {
                'step': len(trace) + 1,
                'before': before,
                'action': action,
                'after': view,
                'receipt': receipt,
                'context': context,
            }
            trace.append(step)
        result = bridge.value['result']
        record = {
            'producer': 'serving_lab.public_episode.v1',
            'origin': 'model_public' if engine else 'public_policy',
            'task_id': task['task_id'],
            'role': task['role'],
            'sequence': task['sequence'],
            'condition': condition,
            'initial': initial,
            'trace': trace,
            'result': result,
            'elapsed_seconds': time.monotonic() - started,
        }
        if purpose:
            record.update(
                purpose=purpose,
                eligible_for_training=False,
                eligible_for_primary_history=False,
                stagnation_limit=stagnation_limit,
            )
        if purpose == 'teacher_coldstart_source':
            record.update(
                source_contract_digest=source_contract_digest,
                eligible_for_training=task['role'] == 'train' and result['success'],
            )
        record['record_hash'] = digest(record)
        write_once(out / 'record.json', record)
        print(
            canonical({'task_id': task['task_id'], 'condition': condition, 'result': result}),
            flush=True,
        )
        return record
    except BaseException as exc:
        if out.exists():
            write_once(
                out / 'failure.json', {'error': repr(exc), 'steps': len(trace), 'trace': trace}
            )
        raise
    finally:
        if bridge:
            await asyncio.to_thread(bridge.close)


async def run(out, mode, cpu_budget=1200):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = REPO / 'artifacts/q1-data-20261102'
    manifest = json.loads((data / 'manifest.json').read_text())
    validate_manifest(manifest)
    assert (
        file_hash(data / 'manifest.json')
        == json.loads((data / 'manifest-anchor.json').read_text())['sha256']
    )
    image = 'sha256:d3938e3d3b2d4d7a2aa15fac50c9a42a6a65f0603ba025339f7e2c30a1118ae4'
    assert json.loads(docker('image', 'inspect', image))[0]['Id'] == image
    sources = [
        'scripts/q2_run.py',
        'scripts/q2_environment.py',
        'scripts/start_engine.py',
        'scripts/probe_serving.py',
    ]
    sources += ['src/serving_lab/' + p.name for p in (REPO / 'src/serving_lab').glob('*.py')]
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'mode': mode,
            'manifest_sha256': file_hash(data / 'manifest.json'),
            'image_id': image,
            'max_steps': 100,
            'model_action_cap': 2400,
            'source_hashes': {n: file_hash(REPO / n) for n in sources},
            'policy': {'name': 'q2_public_non_thinking_v1', 'seed': 20261103},
            'max_input_tokens': 3000,
            'max_output_tokens': 64,
            'history_tokens': 768,
            'cpu_budget': cpu_budget,
            'history_control': 'next available success cyclic by support sequence; not assumed unrelated',
            'clipping': 'fixed initial/current/history; remove oldest full own action-feedback pairs',
            'no_implicit_retry': True,
        },
    )
    if mode == 'cpu':
        records = []
        for task in manifest['tasks']:
            remaining = cpu_budget - sum(r['result']['steps'] for r in records)
            if task['role'] != 'protected' and remaining > 0:
                records.append(
                    await episode(
                        task, data, out / task['task_id'], image, max_steps=min(100, remaining)
                    )
                )
        summary = {
            'cpu_episodes': len(records),
            'successes': sum(r['result']['success'] for r in records),
            'steps': sum(r['result']['steps'] for r in records),
            'protected_executed': 0,
            'model_calls': 0,
        }
        write_once(out / 'summary.json', summary)
        return summary
    engine = None
    binding_path = out / 'worker/binding.json'
    records = []
    try:
        start(REPO / 'configs/serving/r3-large.json', out / 'worker')
        probe(binding_path, out / 'direct-bootstrap')
        binding = EngineBinding.load(binding_path)
        engine = HttpEngine(
            binding,
            profile=True,
            profile_policy={'name': 'q2_public_non_thinking_v1', 'seed': 20261103},
        )
        write_once(
            out / 'profile.json',
            {'document': engine.profile.document, 'digest': engine.profile.digest},
        )
        application = application_history(
            REPO / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json', binding
        )
        with RuntimeStore(out / 'qualification-runtime') as store:
            qualification = await qualify_transport(
                engine,
                json.loads((REPO / 'fixtures/serving/r2-transport-v2.json').read_text()),
                out / 'qualification',
                store=store,
                application_result=application,
            )
        if not qualification['passed']:
            raise RuntimeError('新研究profile预资格失败')
        with RuntimeStore(out / 'action-runtime') as store:
            supports = {}
            for task in manifest['tasks']:
                if task['role'] == 'support':
                    r = await episode(
                        task,
                        data,
                        out / 'support' / task['task_id'],
                        image,
                        engine,
                        store,
                        condition='none',
                    )
                    supports[task['task_id']] = r
                    records.append(r)
            histories = {
                k: verified_history(r, cutoff=7)
                for k, r in supports.items()
                if r['result']['success']
            }
            write_once(
                out / 'histories.json',
                {
                    'sources': histories,
                    'source_hashes': {k: supports[k]['record_hash'] for k in histories},
                },
            )
            pairings = []
            keys = list(histories)
            for task in manifest['tasks']:
                if task['role'] != 'target':
                    continue
                key = task['candidate_source_task_id']
                ordered = [t['task_id'] for t in manifest['tasks'] if t['role'] == 'support']
                index = ordered.index(key)
                cyclic = ordered[index + 1 :] + ordered[:index]
                other = next((k for k in cyclic if k in histories), None)
                ready = key in histories and other is not None
                pairings.append(
                    {'task_id': task['task_id'], 'related': key, 'control': other, 'ready': ready}
                )
            # 一个完整目标组的三条件必须全部可容纳；未就绪目标保留在6对分母。
            write_once(out / 'pairings.json', pairings)
            completed_pairs = 0
            for pair in pairings:
                if not pair['ready']:
                    continue
                task = next(t for t in manifest['tasks'] if t['task_id'] == pair['task_id'])
                texts = [histories[pair['related']], histories[pair['control']]]
                encoded = [engine.tokenizer.encode(t, add_special_tokens=False) for t in texts]
                n = min(768, *map(len, encoded))
                texts = [
                    engine.tokenizer.decode(ids[:n], skip_special_tokens=False) for ids in encoded
                ]
                write_once(
                    out / (pair['task_id'] + '-history-budget.json'),
                    {'tokens': n, 'texts': texts, 'pairing': pair},
                )
                # 首次真实目标观察产生后，先检查完整模板。失败时整个目标组停止，不切换到无历史凑结果。
                for condition, history in [
                    ('none', None),
                    ('public_related', texts[0]),
                    ('token_matched_control', texts[1]),
                ]:
                    records.append(
                        await episode(
                            task,
                            data,
                            out / 'target' / task['task_id'] / condition,
                            image,
                            engine,
                            store,
                            history,
                            condition,
                            history_pair=texts,
                        )
                    )
                completed_pairs += 1
        actions = sum(len(r['trace']) for r in records)
        assert actions <= 2400
        summary = {
            'model_episodes': len(records),
            'model_actions': actions,
            'support_successes': len(histories),
            'target_pairs_source_ready': sum(p['ready'] for p in pairings),
            'target_pairs_ready': completed_pairs,
            'protected_executed': 0,
            'training_runs': 0,
            'scope': 'public action/history feasibility only',
            'rows': [
                {'task_id': r['task_id'], 'condition': r['condition'], 'result': r['result']}
                for r in records
            ],
        }
        write_once(out / 'summary.json', summary)
        return summary
    finally:
        if engine:
            await engine.close()
        if binding_path.exists():
            write_once(out / 'worker-cleanup.json', stop(binding_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['cpu', 'model'], required=True)
    parser.add_argument('--cpu-budget', type=int, default=1200)
    args = parser.parse_args()
    print(canonical(asyncio.run(run(args.output, args.mode, args.cpu_budget))))
