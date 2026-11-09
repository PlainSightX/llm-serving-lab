"""N1固定状态诊断、完整流程与C1原应用评价；每个输出目录只能运行一次。"""

import argparse
import asyncio
import json
from pathlib import Path
import re

from serving_lab.coldstart_actor import ColdStartActor
from serving_lab.execution import write_once
from serving_lab.lab import file_hash, inspect_base
from serving_lab.parameters import parameter_identity
from serving_lab.research_action import check_readiness, generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.task_protocol import canonical, digest
from serving_lab.worker_protocol import RemoteEngine
from serving_lab.workflow import WorkflowActor, validate_workflow_manifest
from serving_lab.workflow_memory import memory_entry
from serving_lab.workflow_study import retrieve_procedural, decision_state
from q2_run import open_owned_bridge
from q2_workflow import IMAGE, audit as audit_teacher, read

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / 'artifacts/n1-20261105-a'
GENERATION_CAP = 11000


def tokens(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def reserve(run, case, kind, step):
    path = OWNER / 'reservations.jsonl'
    rows = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    if len(rows) >= GENERATION_CAP:
        raise ValueError('达到N1总动作预约上限')
    with path.open('a', encoding='utf-8') as f:
        f.write(
            canonical(
                {'number': len(rows) + 1, 'run': str(run), 'case': case, 'kind': kind, 'step': step}
            )
            + '\n'
        )


def prepare(output):
    from transformers import AutoTokenizer

    out = Path(output).resolve()
    data = ROOT / 'artifacts/q2-workflow-data-20261105-a'
    collection = ROOT / 'artifacts/q2-workflow-collection-20261105-a'
    qualification = audit_teacher(data, collection)
    if not qualification['data_gate_passed']:
        raise ValueError('教师来源未资格')
    manifest = read(data / 'manifest.json')
    tasks = validate_workflow_manifest(manifest)
    model = ROOT / 'artifacts/q2-coldstart-training-20261103-b/merged'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    entries = [
        memory_entry(
            read(collection / 'workflow_teacher' / t['task_id'] / 'record.json'),
            t,
            manifest['source_contract_digest'],
            file_hash(data / 'manifest.json'),
        )
        for t in tasks.values()
        if t['role'] == 'train'
    ]
    development = [t for t in tasks.values() if t['role'] == 'dev']
    bases = {
        family: min(t['base_id'] for t in development if t['family'] == family)
        for family in {t['family'] for t in development}
    }
    selected = [t for t in development if t['base_id'] in bases.values()]
    retrievals, states = {}, {}
    for task in development:
        record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
        goal = re.search(r'Your task is to:\s*([^\n]+)', record['initial']['feedback'])[1].strip()
        for representation in ('script', 'combined'):
            value = retrieve_procedural(
                entries,
                goal,
                task['workflow'],
                cutoff=task['sequence'],
                tokenize=lambda text: tokenizer.encode(text, add_special_tokens=False),
                representation=representation,
            )
            if not value['selected']:
                raise ValueError('完整记忆不能容纳，停止比较')
            retrievals[task['task_id'] + '/' + representation] = {
                **value,
                'initial_digest': digest(record['initial']),
            }
        if task in selected:
            states[task['task_id']] = decision_state(record)
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'memory.json', {'entries': entries, 'source_qualification': qualification})
    write_once(out / 'retrievals.json', retrievals)
    write_once(out / 'states.json', states)
    cases = [
        {'task_id': t['task_id'], 'condition': condition}
        for t in selected
        for condition in ('workflow', 'script', 'combined')
    ]
    plan = {
        'name': 'n1-fixed-state-and-rollout-v1',
        'data': str(data),
        'collection': str(collection),
        'data_sha256': file_hash(data / 'manifest.json'),
        'cases': cases,
        'max_steps': 100,
        'stagnation_limit': 4,
        'files': {
            name: file_hash(out / name)
            for name in ('memory.json', 'retrievals.json', 'states.json')
        },
        'baseline_episodes_per_model': 18,
        'decision_queries_per_model': 18,
        'models': ['base', 'c0'],
        'purpose': 'development_only',
        'future_episodes': 0,
        'memory_builder': 'deterministic_public_rule_projection_v1',
    }
    write_once(out / 'plan.json', plan)
    return {
        'cases_per_model': len(cases),
        'states': len(states),
        'qualified_train_sources': len(entries),
        'memory_tokens': {k: v['prompt_tokens'] for k, v in retrievals.items()},
    }


def freeze_sources(out):
    names = [
        p.relative_to(ROOT).as_posix()
        for folder in ('src/serving_lab', 'scripts')
        for p in (ROOT / folder).glob('*.py')
    ]
    names.append('configs/evaluation/q1-sources.json')
    hashes = {n: file_hash(ROOT / n) for n in names}
    for name in names:
        p = out / 'source' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    return hashes


def memory_for(case, task, retrievals):
    condition = case['condition']
    if condition in ('script', 'combined'):
        return retrievals[task['task_id'] + '/' + condition]
    return None


async def query(
    engine, store, out, run, case, kind, step, actor, initial, view, trace, tokenizer, memory
):
    prepared = actor.prepare(initial, view, trace, lambda ms: tokens(tokenizer, ms))
    reserve(run, case, kind, step)
    metadata = {
        'purpose': 'n1_development',
        'case': case,
        'kind': kind,
        'step': step,
        'actor_context': prepared['context'],
        'memory_digest': digest(memory) if memory else None,
    }
    receipt = await generate_action(
        engine,
        store,
        out,
        prepared['messages'],
        prepared['input_token_ids'],
        metadata,
        choices=prepared['choices'],
        readiness_attempts=2,
    )
    return actor.action(receipt, prepared['choices']), receipt, prepared['context']


async def episode(engine, store, tokenizer, task, data, case, memory, out, run):
    bridge = None
    trace = []
    initial = None
    pending = None
    actor = (
        ColdStartActor()
        if case['condition'] == 'old_application'
        else WorkflowActor(task['workflow'], memory['prompt'] if memory else [])
    )
    try:
        bridge = await open_owned_bridge(task, data, out, IMAGE, 100, stagnation_limit=4)
        value = await asyncio.wait_for(asyncio.to_thread(bridge.read), 30)
        initial = value['public']
        if memory and digest(initial) != memory['initial_digest']:
            raise ValueError('当前环境不等于冻结记忆上下文')
        while not value['terminal']:
            before = value['public']
            step = len(trace) + 1
            action, receipt, context = await query(
                engine,
                store,
                out / 'calls' / f'{step:03}',
                run,
                case,
                'episode',
                step,
                actor,
                initial,
                before,
                trace,
                tokenizer,
                memory,
            )
            pending = {'action': action, 'receipt': receipt, 'context': context}
            value = await bridge.step(action)
            trace.append(
                {
                    'step': step,
                    'before': before,
                    'action': action,
                    'after': value['public'],
                    'receipt': receipt,
                    'context': context,
                }
            )
            pending = None
        record = {
            'task_id': task['task_id'],
            'role': task['role'],
            'phase': task.get('phase'),
            'family': task['family'],
            'condition': case['condition'],
            'initial': initial,
            'trace': trace,
            'result': value['result'],
            'workflow': task.get('workflow'),
            'worker': engine.identity,
            'profile_digest': engine.profile.digest,
            'memory_receipt': memory,
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
            'purpose': 'actor_diagnostic',
        }
        record['record_hash'] = digest(record)
        write_once(out / 'record.json', record)
        return record
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
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


async def run(
    preparation, model, engine_config, output, mode='baseline', memory='script', training=None
):
    from transformers import AutoTokenizer

    preparation, model, out = (
        Path(preparation).resolve(),
        Path(model).resolve(),
        Path(output).resolve(),
    )
    plan, config = read(preparation / 'plan.json'), read(engine_config)
    for name, expected in plan['files'].items():
        if file_hash(preparation / name) != expected:
            raise ValueError('准备工件发生变化')
    data = Path(plan['data'])
    if file_hash(data / 'manifest.json') != plan['data_sha256']:
        raise ValueError('任务身份变化')
    tasks = validate_workflow_manifest(read(data / 'manifest.json'))
    retrievals = read(preparation / 'retrievals.json')
    base = inspect_base(model)
    parameters = parameter_identity(model)
    if (
        config['manifest']['artifact_digest'] != digest(base['files'])
        or config['manifest']['parameter_digest'] != parameters['digest']
    ):
        raise ValueError('实际参数与本轮参考模型不符')
    from serving_lab.workflow_training import C0_DIGEST
    from serving_lab.teacher_data import FIXED_BASE_DIGEST

    artifact_digest = digest(base['files'])
    candidate_training = None
    if mode == 'baseline' and artifact_digest not in {C0_DIGEST, FIXED_BASE_DIGEST}:
        raise ValueError('基线矩阵只允许已冻结的原基座与C0')
    if mode == 'application_control' and artifact_digest != C0_DIGEST:
        raise ValueError('当前部署应用对照只允许冻结C0')
    if mode == 'application' or (mode == 'stage1' and artifact_digest != C0_DIGEST):
        if not training:
            raise ValueError('C1评价缺少本轮完整训练证明')
        from serving_lab.training_evidence import verify_training

        candidate_training = verify_training(
            training,
            OWNER / 'training-data',
            model,
            ROOT / 'artifacts/q2-coldstart-training-20261103-b/merged',
        )
        if candidate_training['parameter_digest'] != parameters['digest']:
            raise ValueError('C1实际参数与训练不符')
    if mode == 'baseline':
        cases = plan['cases']
    elif mode == 'stage1':
        cases = [
            {'task_id': t['task_id'], 'condition': c}
            for t in tasks.values()
            if t['role'] == 'dev' and t['phase'] == 1
            for c in ('workflow', memory)
        ]
    elif mode in {'application', 'application_control'}:
        data = ROOT / 'artifacts/q2-teacher-data-20261103-b'
        from serving_lab.teacher_data import validate_teacher_manifest

        tasks = validate_teacher_manifest(read(data / 'manifest.json'))
        cases = [
            {'task_id': t['task_id'], 'condition': 'old_application'}
            for t in tasks.values()
            if t['role'] == 'dev'
        ]
    else:
        raise ValueError('未知N1模式')
    out.mkdir(parents=True, exist_ok=False)
    input_record = {
        'name': 'n1-model-run-v1',
        'preparation': str(preparation),
        'plan_sha256': file_hash(preparation / 'plan.json'),
        'model': str(model),
        'data': str(data),
        'data_sha256': file_hash(data / 'manifest.json'),
        'mode': mode,
        'cases': cases,
        'worker_manifest': config['manifest'],
        'source_hashes': freeze_sources(out),
        'model_identity': {
            'artifact_digest': digest(base['files']),
            'parameter_digest': parameters['digest'],
        },
        'reference_model': base,
        'readiness_attempts': 2,
        'max_steps': 100,
        'stagnation_limit': 4,
        'memory_choice': memory,
        'environment_image': IMAGE,
        'environment_contract_recorded': True,
        'candidate_training': candidate_training,
    }
    write_once(out / 'input.json', input_record)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    engine = RemoteEngine(
        config['endpoint'],
        Path(config['token_file']).read_text().strip(),
        config['manifest'],
        tls=config['tls'],
    )
    records, decisions = [], []
    try:
        ready, checks = await check_readiness(engine, 2)
        write_once(out / 'readiness.json', {'ready': ready, 'checks': checks})
        if not ready:
            raise RuntimeError('worker_not_ready')
        with RuntimeStore(out / 'runtime') as store:
            if mode == 'baseline':
                states = read(preparation / 'states.json')
                for case in cases:
                    task = tasks[case['task_id']]
                    m = memory_for(case, task, retrievals)
                    state = states[task['task_id']]
                    actor = WorkflowActor(task['workflow'], m['prompt'] if m else [])
                    public = state['public']
                    directory = out / 'decisions' / case['condition'] / task['task_id']
                    action, receipt, context = await query(
                        engine,
                        store,
                        directory / 'call',
                        out,
                        case,
                        'decision',
                        state['step'],
                        actor,
                        public['initial'],
                        public['view'],
                        public['trace'],
                        tokenizer,
                        m,
                    )
                    value = {
                        'case': case,
                        'public_digest': state['public_digest'],
                        'source_record_hash': state['source_record_hash'],
                        'action': action,
                        'expected_action': state['expected_action'],
                        'matches_teacher_action': action == state['expected_action'],
                        'receipt': receipt,
                        'context': context,
                        'memory_receipt': m,
                    }
                    write_once(directory / 'result.json', value)
                    decisions.append(value)
            for case in cases:
                task = tasks[case['task_id']]
                m = memory_for(case, task, retrievals)
                record = await episode(
                    engine,
                    store,
                    tokenizer,
                    task,
                    data,
                    case,
                    m,
                    out / 'episodes' / case['condition'] / task['task_id'],
                    out,
                )
                records.append(record)
                print(
                    canonical(
                        {
                            'completed': len(records),
                            'total': len(cases),
                            'case': case,
                            'result': record['result'],
                        }
                    ),
                    flush=True,
                )
            if store.unresolved():
                raise RuntimeError('存在未解除租约')
        summary = {
            'mode': mode,
            'episodes': len(records),
            'decisions': len(decisions),
            'future_episodes': 0,
            'conditions': {
                c: {
                    'episodes': sum(r['condition'] == c for r in records),
                    'physical_success': sum(
                        bool(r['result'].get('base_success', r['result']['success']))
                        for r in records
                        if r['condition'] == c
                    ),
                    'workflow_success': sum(
                        bool(r['result']['success']) for r in records if r['condition'] == c
                    ),
                }
                for c in sorted({r['condition'] for r in records})
            },
            'decision_matches': {
                c: sum(
                    r['matches_teacher_action'] for r in decisions if r['case']['condition'] == c
                )
                for c in ('workflow', 'script', 'combined')
            },
            'raw_audit_status': 'pending',
            'training_data_exported': False,
        }
        write_once(out / 'summary.json', summary)
        return summary
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {
                'error': repr(error),
                'completed_episodes': len(records),
                'completed_decisions': len(decisions),
                'implicit_retry': False,
            },
        )
        raise
    finally:
        await engine.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare', 'run'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--preparation')
    p.add_argument('--model')
    p.add_argument('--engine-config')
    p.add_argument('--training')
    p.add_argument(
        '--mode',
        choices=['baseline', 'stage1', 'application', 'application_control'],
        default='baseline',
    )
    p.add_argument('--memory', choices=['script', 'combined'], default='script')
    a = p.parse_args()
    value = (
        prepare(a.output)
        if a.stage == 'prepare'
        else asyncio.run(
            run(a.preparation, a.model, a.engine_config, a.output, a.mode, a.memory, a.training)
        )
    )
    print(json.dumps(value, ensure_ascii=False, indent=2))
