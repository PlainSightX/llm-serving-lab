"""在完整开发面板实测公告协议强记忆；未资格时不打开未来评价或训练。"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from serving_lab.continual_procedure import (
    ProcedureActor,
    validate_memory_snapshot,
    validate_qualification_cases,
    validate_actor_binding,
)
from serving_lab.execution import write_once
from serving_lab.lab import file_hash, inspect_base, digest
from serving_lab.research_action import check_readiness, generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.worker_protocol import RemoteEngine
from serving_lab.workflow import validate_workflow_manifest
from q2_run import open_owned_bridge
from q2_workflow import IMAGE

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reserve(owner, run, case, step, cap):
    path = owner / 'qualification-reservations.jsonl'
    count = sum(1 for _ in path.open(encoding='utf-8')) if path.exists() else 0
    if count >= cap:
        raise ValueError('达到开发资格总生成上限')
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'index': count + 1, 'run': run, 'case': case, 'step': step}) + '\n')


async def episode(
    engine, store, tokenizer, owner, case, task, data, memory, out, plan, *, basic=False
):
    bridge = None
    trace = []
    initial = None
    pending = None
    q = plan['qualification']
    from serving_lab.coldstart_actor import ColdStartActor

    actor = (
        ColdStartActor()
        if basic
        else ProcedureActor(
            task['workflow'],
            memory['prompt'] if memory else None,
            explicit=case['condition'] == 'explicit',
        )
    )
    purpose = 'n5_basic_actor_readiness' if basic else 'n5_qualification_development'
    tokenize = lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    try:
        bridge = await open_owned_bridge(
            task,
            data,
            out / 'environment',
            IMAGE,
            q['max_steps'],
            stagnation_limit=q['stagnation_limit'],
        )
        value = await asyncio.wait_for(asyncio.to_thread(bridge.read), 30)
        initial = value['public']
        if memory and digest(initial) != memory['initial_digest']:
            raise ValueError('开发资格实例与准备时不同')
        while not value['terminal']:
            before = value['public']
            step = len(trace) + 1
            prepared = actor.prepare(initial, before, trace, tokenize, q['input_tokens'])
            reserve(owner, str(out.parents[1]), case['id'], step, q['generation_cap'])
            receipt = await generate_action(
                engine,
                store,
                out / 'calls' / f'{step:03}',
                prepared['messages'],
                prepared['input_token_ids'],
                {
                    'purpose': purpose,
                    'case': case['id'],
                    'step': step,
                    'actor_context': prepared['context'],
                    'plan_sha256': file_hash(owner / 'plan.json'),
                },
                choices=prepared['choices'],
                readiness_attempts=2,
            )
            action = actor.action(receipt, prepared['choices'])
            pending = {'action': action, 'receipt': receipt}
            value = await bridge.step(action)
            trace.append(
                {
                    'step': step,
                    'before': before,
                    'action': action,
                    'after': value['public'],
                    'receipt': receipt,
                    'context': prepared['context'],
                }
            )
            pending = None
        record = {
            'case': case,
            'source_task_id': task['task_id'],
            'initial': initial,
            'trace': trace,
            'result': value['result'],
            'workflow': task.get('workflow'),
            'worker': engine.identity,
            'profile_digest': engine.profile.digest,
            'memory_receipt': memory,
            'purpose': purpose,
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
        }
        record['record_hash'] = digest(record)
        write_once(out / 'record.json', record)
        return record
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {
                'case': case,
                'error': repr(error),
                'initial': initial,
                'trace': trace,
                'pending': pending,
            },
        )
        raise
    finally:
        if bridge is not None:
            await asyncio.to_thread(bridge.close)


async def main(args):
    from transformers import AutoTokenizer

    owner = Path(args.owner).resolve()
    prepared = Path(args.prepared).resolve()
    out = Path(args.output).resolve()
    plan = read(owner / 'plan.json')
    input_ = read(prepared / 'input.json')
    config = read(args.engine)
    if config['manifest']['profile']['name'] != 'q2_public_choice_v1':
        raise ValueError('共同actor必须使用既有公开命令choice profile')
    bindings = read(owner / 'actor-bindings.json')
    role = args.actor_role
    if (
        file_hash(owner / 'actor-bindings.json')
        != read(owner / 'actor-bindings-anchor.json')['sha256']
    ):
        raise ValueError('actor执行绑定变化')
    if bindings['plan_sha256'] != file_hash(owner / 'plan.json'):
        raise ValueError('actor执行绑定属于其他协议')
    validate_actor_binding(role, config['manifest'], bindings['actors'][role])
    if role == 'Qwen3-4B-shared-basic':
        if plan['name'] != 'n5-announced-procedure-v2-shared-basic' or not args.prior:
            raise ValueError('共同基础模型必须先经过v2基础门')
        prior_dir = Path(args.prior)
        prior = read(prior_dir / 'summary.json')
        prior_input = read(prior_dir / 'input.json')
        prior_audit = read(prior_dir / 'audit.json')
        if (
            not prior_audit['passed']
            or not prior_audit['qualified']
            or not prior['completed']
            or not prior['qualified']
            or prior_audit['scope'] != 'complete basic readiness audited'
            or prior['cases'] != 24
            or prior['passed_cases'] < plan['basic_gate']['minimum_success']
            or prior_audit['summary_sha256'] != file_hash(prior_dir / 'summary.json')
            or prior_audit['input_sha256'] != file_hash(prior_dir / 'input.json')
            or prior_input['plan_sha256'] != file_hash(owner / 'plan.json')
            or prior_input['actor_role'] != role
            or prior['artifact_digest'] != config['manifest']['artifact_digest']
            or prior['parameter_digest'] != config['manifest']['parameter_digest']
        ):
            raise ValueError('没有同一共同模型对应的完整基础资格')
        validate_actor_binding(role, prior_input['worker_manifest'], bindings['actors'][role])
    if role == 'Qwen3-4B-fixed-original':
        if not args.prior:
            raise ValueError('4B升级必须先有完整C0资格失败')
        prior = read(Path(args.prior) / 'summary.json')
        prior_input = read(Path(args.prior) / 'input.json')
        prior_audit = read(Path(args.prior) / 'audit.json')
        if (
            not prior_audit['passed']
            or prior_audit['summary_sha256'] != file_hash(Path(args.prior) / 'summary.json')
            or prior_audit['input_sha256'] != file_hash(Path(args.prior) / 'input.json')
            or prior_input['actor_role'] != 'C0-Qwen3-1.7B'
            or prior_audit['actor_role'] != 'C0-Qwen3-1.7B'
        ):
            raise ValueError('C0先行结果没有对应原始记录审计')
        validate_actor_binding(
            'C0-Qwen3-1.7B', prior_input['worker_manifest'], bindings['actors']['C0-Qwen3-1.7B']
        )
        if (
            not prior['completed']
            or prior['qualified']
            or prior['artifact_digest'] != bindings['actors']['C0-Qwen3-1.7B']['artifact_digest']
        ):
            raise ValueError('C0未完整判为未资格，不能候选单独升级')
        if prior_input['plan_sha256'] != file_hash(owner / 'plan.json'):
            raise ValueError('C0先行资格不属于当前协议')
    if file_hash(owner / 'plan.json') != read(owner / 'plan-anchor.json')['sha256'] or input_[
        'plan_sha256'
    ] != file_hash(owner / 'plan.json'):
        raise ValueError('协议身份不同')
    if file_hash(prepared / 'memories.json') != input_['memories_sha256']:
        raise ValueError('准备记忆变化')
    if config['expected_artifact_digest'] != config['manifest']['artifact_digest']:
        raise ValueError('实际worker不是当前共同参数候选')
    if (
        list(Path(args.model).glob('*.safetensors'))
        and digest(inspect_base(args.model)['files']) != config['manifest']['artifact_digest']
    ):
        raise ValueError('本地完整参数与worker不符')
    if (
        file_hash(Path(args.model) / 'tokenizer_config.json')
        != bindings['actors'][role]['tokenizer_config_sha256']
    ):
        raise ValueError('分词器不是该共同actor的冻结工件')
    if file_hash(Path(args.model) / 'tokenizer_config.json') != input_['tokenizer_config_sha256']:
        raise ValueError('tokenizer准备身份不同')
    data = Path(input_['data'])
    manifest = read(data / 'manifest.json')
    tasks = validate_workflow_manifest(manifest)
    if file_hash(data / 'manifest.json') != input_['data_manifest_sha256']:
        raise ValueError('源数据变化')
    memories = read(prepared / 'memories.json')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    from n5_prepare import reconstruct_memories

    reconstructed, source_audit = reconstruct_memories(
        data, Path(input_['collection']), tokenizer, plan
    )
    if reconstructed != memories or source_audit != input_['source_audit']:
        raise ValueError('M不等于原始train轨迹的完整重建，拒绝派发')
    validate_qualification_cases(input_['cases'], tasks)
    for case in input_['cases']:
        task = tasks[case['source_task_id']]
        if task['role'] != 'dev':
            raise ValueError('资格只能使用dev')
        validate_memory_snapshot(memories[task['task_id']], task['workflow'], task['phase'])
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in list((ROOT / 'src/serving_lab').glob('*.py')) + list(
        (ROOT / 'scripts').glob('*.py')
    ):
        relative = path.relative_to(ROOT)
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[relative.as_posix()] = file_hash(path)
    write_once(
        out / 'input.json',
        {
            'preparation': input_,
            'worker_manifest': config['manifest'],
            'source_hashes': sources,
            'plan_sha256': file_hash(owner / 'plan.json'),
            'actor_role': role,
            'actor_bindings_sha256': file_hash(owner / 'actor-bindings.json'),
            'cases': input_['cases'],
            'memory_file_sha256': file_hash(prepared / 'memories.json'),
            'purpose': 'common actor/strong memory development qualification only; no future access',
        },
    )
    engine = RemoteEngine(
        config['endpoint'],
        Path(config['token_file']).read_text().strip(),
        config['manifest'],
        tls=config['tls'],
    )
    store = RuntimeStore(out / 'runtime')
    records = []
    try:
        ready, checks = await check_readiness(engine, 2)
        write_once(out / 'readiness.json', {'ready': ready, 'checks': checks})
        if not ready:
            raise ValueError('worker未就绪')
        for i, command in enumerate(['look', 'inventory'], 1):
            messages = [{'role': 'user', 'content': 'Return the single permitted command.'}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            reserve(
                owner,
                str(out),
                'choice-canary-' + str(i),
                0,
                plan['qualification']['generation_cap'],
            )
            canary = await generate_action(
                engine,
                store,
                out / 'canaries' / str(i),
                messages,
                ids,
                {'purpose': 'n5_interface_canary', 'case': 'canary-' + str(i), 'step': 0},
                choices=[command],
                readiness_attempts=2,
            )
            if canary['content'] != command or canary['finish_reason'] != 'stop':
                raise ValueError('真实choice接口未通过')
        for case in input_['cases']:
            task = tasks[case['source_task_id']]
            memory = memories[task['task_id']] if case['condition'] == 'memory' else None
            row = await episode(
                engine,
                store,
                tokenizer,
                owner,
                case,
                task,
                data,
                memory,
                out / 'episodes' / case['id'],
                plan,
            )
            records.append(row)
            print(
                json.dumps(
                    {
                        'case': case['id'],
                        'success': row['result']['success'],
                        'base_success': row['result']['base_success'],
                        'actions': len(row['trace']),
                    }
                ),
                flush=True,
            )
        scores = Counter()
        coverage = Counter()
        thermal = Counter()
        for row in records:
            c = row['case']
            task = tasks[row['source_task_id']]
            success = int(row['result']['success'])
            scores[c['condition']] += success
            coverage[c['condition'], task['phase']] += success
            thermal[c['condition'], task['phase'], task['workflow']['thermal']] += success
        q = plan['qualification']
        qualified = (
            scores['explicit'] >= q['min_control_success']
            and scores['memory']
            >= max(q['min_memory_success'], scores['explicit'] - q['max_memory_gap'])
            and all(
                coverage[c, s] >= q['min_each_phase_success']
                and all(thermal[c, s, t] >= q['min_each_thermal_success'] for t in ['heat', 'cool'])
                for c in q['conditions']
                for s in [1, 2, 3]
            )
        )
        result = {
            'completed': True,
            'qualified': qualified,
            'scores': dict(scores),
            'per_condition_denominator': 18,
            'per_phase': {c: {str(s): coverage[c, s] for s in [1, 2, 3]} for c in q['conditions']},
            'actual_generations': sum(len(r['trace']) for r in records) + 2,
            'study_actions': sum(len(r['trace']) for r in records),
            'interface_canaries': 2,
            'future_episodes': 0,
            'training_runs': 0,
            'artifact_digest': config['manifest']['artifact_digest'],
            'parameter_digest': config['manifest']['parameter_digest'],
            'scope': 'development common actor and strong-memory eligibility, not continual-learning result',
        }
        write_once(out / 'summary.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {'error': repr(error), 'completed_episodes': len(records), 'implicit_retry': False},
        )
        raise
    finally:
        await engine.close()
        store.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['owner', 'prepared', 'output', 'model', 'engine']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--actor-role', required=True)
    p.add_argument('--prior')
    asyncio.run(main(p.parse_args()))
