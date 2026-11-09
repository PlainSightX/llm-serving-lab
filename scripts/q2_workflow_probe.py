"""在独立研究账本中检验C0基础能力、公开流程与冻结轨迹记忆。"""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re

from serving_lab.execution import write_once
from serving_lab.lab import file_hash, inspect_base
from serving_lab.parameters import parameter_identity
from serving_lab.research_action import check_readiness, generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.standard_actor import StandardActor
from serving_lab.task_protocol import canonical, digest
from serving_lab.worker_protocol import RemoteEngine
from serving_lab.workflow import WorkflowActor, validate_workflow_manifest
from serving_lab.workflow_memory import memory_entry, retrieve
from q2_run import open_owned_bridge
from q2_workflow import IMAGE, audit as audit_teacher, read, snapshot

REPO = Path(__file__).resolve().parents[1]


def tokenizer_at(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)


def message_tokens(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def probe_cases(tasks, mode):
    development = [task for task in tasks if task['role'] == 'dev']
    if mode == 'instruction_check':
        bases = {
            family: min(task['base_id'] for task in development if task['family'] == family)
            for family in {task['family'] for task in development}
        }
        development = [task for task in development if task['base_id'] in bases.values()]
        cases = []
    elif mode == 'full':
        cases = [
            {'condition': 'plain', 'task_id': task['task_id']}
            for task in development
            if task['phase'] == 1
        ]
    else:
        raise ValueError('未知开发试点模式')
    for index, task in enumerate(development):
        conditions = (
            ('frozen_workflow', 'trajectory_memory')
            if index % 2 == 0
            else ('trajectory_memory', 'frozen_workflow')
        )
        cases += [{'condition': condition, 'task_id': task['task_id']} for condition in conditions]
    return cases


def validate_reference_model(plan, manifest):
    reference = plan.get('reference_model')
    keys = ('artifact_digest', 'parameter_digest')
    if (
        not isinstance(reference, dict)
        or any(
            not isinstance(reference.get(key), str)
            or not re.fullmatch(r'[a-f0-9]{64}', reference[key])
            for key in keys
        )
        or any(reference[key] != manifest.get(key) for key in keys)
    ):
        raise ValueError('模型参数身份与冻结计划不符，旧无绑定计划须重新准备')


def prepare(data, collection, model, output, mode='full', prior_audit=None):
    data, collection, model, out = map(
        lambda value: Path(value).resolve(), (data, collection, model, output)
    )
    qualification = audit_teacher(data, collection)
    if not qualification['data_gate_passed']:
        raise ValueError('公开教师来源尚未取得资格')
    manifest = read(data / 'manifest.json')
    tasks = validate_workflow_manifest(manifest)
    artifact = inspect_base(model)
    parameters = parameter_identity(model)
    if inspect_base(model) != artifact:
        raise ValueError('冻结参数身份期间本地工件发生变化')
    reference = {
        'artifact_digest': digest(artifact['files']),
        'parameter_digest': parameters['digest'],
        'tensor_count': parameters['tensor_count'],
    }
    tokenizer = tokenizer_at(model)
    tokenize_text = lambda text: tokenizer.encode(text, add_special_tokens=False)
    tokenize_messages = lambda messages: message_tokens(tokenizer, messages)
    entries = []
    for task in tasks.values():
        if task['role'] == 'train':
            record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
            entries.append(
                memory_entry(
                    record,
                    task,
                    manifest['source_contract_digest'],
                    file_hash(data / 'manifest.json'),
                )
            )
    assert len(entries) == 72
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'memory.json', {'entries': entries, 'source_qualification': qualification})
    memories = {}
    development = [task for task in tasks.values() if task['role'] == 'dev']
    for task in development:
        initial = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')['initial']
        goal = re.search(r'Your task is to:\s*([^\n]+)', initial['feedback'])[1].strip()
        memory = retrieve(
            entries, goal, task['workflow'], cutoff=task['sequence'], tokenize=tokenize_text
        )
        prepared = WorkflowActor(task['workflow'], memory['prompt']).prepare(
            initial, initial, [], tokenize_messages
        )
        memories[task['task_id']] = {
            **memory,
            'initial_input_tokens': len(prepared['input_token_ids']),
            'initial_digest': digest(initial),
        }
    write_once(out / 'retrieval.json', memories)
    if not all(memory['selected'] for memory in memories.values()):
        raise ValueError('部分开发任务没有可完整容纳的相关记忆，停止模型比较')
    cases = probe_cases(list(tasks.values()), mode)
    prior = None
    if mode == 'instruction_check':
        previous = read(prior_audit)
        prior_input_path = Path(prior_audit).resolve().parent / 'input.json'
        prior_input = read(prior_input_path)
        prior_plan = read(Path(prior_input['preparation']) / 'plan.json')
        if (
            not previous['raw_verified']
            or previous['episodes'] != 42
            or previous['conditions']['plain']['episodes'] != 6
            or previous['conditions']['plain']['base_success'] < 4
            or previous['artifact_digest'] != reference['artifact_digest']
            or previous['parameter_digest'] != reference['parameter_digest']
            or prior_plan['data_manifest_sha256'] != file_hash(data / 'manifest.json')
        ):
            raise ValueError('缺少同数据、同参数的已核验基础门')
        if previous['model_reservations'] + len(cases) * 100 > 4200:
            raise ValueError('澄清试点的最大预约会超过累计4200上限')
        prior = {
            'audit': str(Path(prior_audit).resolve()),
            'sha256': file_hash(prior_audit),
            'input': str(prior_input_path),
            'input_sha256': file_hash(prior_input_path),
            'worker_manifest': prior_input['worker_manifest'],
            'previous_reservations': previous['model_reservations'],
        }
    hashes = snapshot(out)
    name = 'scripts/q2_workflow_probe.py'
    hashes[name] = file_hash(REPO / name)
    target = out / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPO / name).read_bytes())
    plan = {
        'name': 'workflow-c0-readiness-v2',
        'mode': mode,
        'data': str(data),
        'collection': str(collection),
        'model': str(model),
        'reference_model': reference,
        'prior_audit': prior,
        'data_manifest_sha256': file_hash(data / 'manifest.json'),
        'source_hashes': hashes,
        'memory_sha256': file_hash(out / 'memory.json'),
        'retrieval_sha256': file_hash(out / 'retrieval.json'),
        'tokenizer_files': {
            name: file_hash(model / name)
            for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
        },
        'cases': cases,
        'episode_cap': len(cases),
        'generation_cap': len(cases) * 100,
        'max_steps': 100,
        'stagnation_limit': 4,
        'readiness_attempts': 2,
        'base_gate': {'required_successes': 4, 'episodes': 6, 'each_family_minimum': 2},
        'purpose': 'actor_diagnostic',
        'eligible_for_training': False,
        'eligible_for_primary_history': False,
        'scope': 'development readiness only; no future evaluation or parameter update',
    }
    assert len(cases) == (42 if mode == 'full' else 12)
    write_once(out / 'plan.json', plan)
    return {
        'cases': len(cases),
        'memory_entries': len(entries),
        'retrieval_coverage': len(memories),
        'initial_prompt_tokens': [row['initial_input_tokens'] for row in memories.values()],
        'model_calls': 0,
    }


def reserve(output, task, condition, step, generation_cap):
    path = output / 'reservations.jsonl'
    entries = path.read_text().splitlines() if path.exists() else []
    if len(entries) >= generation_cap:
        raise ValueError('模型试点达到冻结预约上限')
    with path.open('a', encoding='utf-8') as stream:
        stream.write(
            canonical(
                {
                    'call': len(entries) + 1,
                    'task_id': task['task_id'],
                    'condition': condition,
                    'step': step,
                }
            )
            + '\n'
        )


async def model_episode(
    task,
    condition,
    data,
    output,
    engine,
    store,
    tokenizer,
    memory,
    reservation_root,
    generation_cap,
    readiness_attempts,
):
    bridge, initial, trace, failure, pending = None, None, [], None, None
    try:
        bridge = await open_owned_bridge(task, data, output, IMAGE, 100, stagnation_limit=4)
        value = await asyncio.wait_for(asyncio.to_thread(bridge.read), 30)
        initial = value['public']
        assert value['workflow'] == task['workflow'] and digest(initial) == memory['initial_digest']
        selected = memory['prompt'] if condition == 'trajectory_memory' else []
        actor = (
            StandardActor() if condition == 'plain' else WorkflowActor(task['workflow'], selected)
        )
        tokenize = lambda messages: message_tokens(tokenizer, messages)
        while not value['terminal']:
            before = value['public']
            prepared = actor.prepare(initial, before, trace, tokenize)
            step = len(trace) + 1
            reserve(reservation_root, task, condition, step, generation_cap)
            metadata = {
                'purpose': 'workflow_readiness_diagnostic',
                'task_id': task['task_id'],
                'condition': condition,
                'step': step,
                'actor_context': prepared['context'],
                'workflow_digest': digest(task['workflow']),
                'memory_receipt_digest': digest(memory) if selected else None,
                'readiness_attempts': readiness_attempts,
            }
            receipt = await generate_action(
                engine,
                store,
                output / 'calls' / f'{step:03}',
                prepared['messages'],
                prepared['input_token_ids'],
                metadata,
                choices=prepared['choices'],
                readiness_attempts=readiness_attempts,
            )
            action = actor.action(receipt, prepared['choices'])
            pending = {'action': action, 'receipt': receipt, 'context': prepared['context']}
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
            'task_id': task['task_id'],
            'role': task['role'],
            'phase': task['phase'],
            'sequence': task['sequence'],
            'condition': condition,
            'workflow': task['workflow'],
            'initial': initial,
            'trace': trace,
            'result': value['result'],
            'worker': engine.identity,
            'profile_digest': engine.profile.digest,
            'purpose': 'actor_diagnostic',
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
            'memory_receipt': memory if selected else None,
        }
        record['record_hash'] = digest(record)
        write_once(output / 'record.json', record)
        return record
    except BaseException as error:
        failure = error
        output.mkdir(parents=True, exist_ok=True)
        write_once(
            output / 'failure.json',
            {
                'task_id': task['task_id'],
                'condition': condition,
                'error': {'type': type(error).__name__, 'message': str(error)},
                'initial': initial,
                'trace': trace,
                'pending': pending,
            },
        )
        raise
    finally:
        if bridge is not None:
            try:
                await asyncio.to_thread(bridge.close)
            except Exception as error:
                write_once(
                    output / 'cleanup-failure.json',
                    {'type': type(error).__name__, 'message': str(error)},
                )
                if failure is None:
                    raise


async def run(preparation, engine_config, output):
    preparation, out = Path(preparation).resolve(), Path(output).resolve()
    plan, config = read(preparation / 'plan.json'), read(engine_config)
    validate_reference_model(plan, config['manifest'])
    for name, expected in plan['source_hashes'].items():
        assert file_hash(REPO / name) == expected
    data, model = Path(plan['data']), Path(plan['model'])
    assert file_hash(data / 'manifest.json') == plan['data_manifest_sha256']
    assert file_hash(preparation / 'memory.json') == plan['memory_sha256']
    assert file_hash(preparation / 'retrieval.json') == plan['retrieval_sha256']
    for name, expected in plan['tokenizer_files'].items():
        assert file_hash(model / name) == expected
    tasks = validate_workflow_manifest(read(data / 'manifest.json'))
    mode = plan.get('mode', 'full')
    assert plan['cases'] == probe_cases(list(tasks.values()), mode)
    assert plan['generation_cap'] == len(plan['cases']) * 100
    assert plan['readiness_attempts'] == 2
    if mode == 'instruction_check':
        previous = plan['prior_audit']
        assert (
            file_hash(previous['audit']) == previous['sha256']
            and file_hash(previous['input']) == previous['input_sha256']
        )
        assert config['manifest'] == previous['worker_manifest']
    memories = read(preparation / 'retrieval.json')
    tokenizer = tokenizer_at(model)
    engine = RemoteEngine(
        config['endpoint'],
        Path(config['token_file']).read_text().strip(),
        config['manifest'],
        tls=config['tls'],
    )
    out.mkdir(parents=True, exist_ok=False)
    write_once(
        out / 'input.json',
        {
            'plan_sha256': file_hash(preparation / 'plan.json'),
            'preparation': str(preparation),
            'worker_manifest': config['manifest'],
            'endpoint': config['endpoint'],
            'started_purpose': plan['purpose'],
            'generation_cap': plan['generation_cap'],
        },
    )
    records = []
    try:
        ready, checks = await check_readiness(engine, plan['readiness_attempts'])
        write_once(out / 'preflight-readiness.json', {'ready': ready, 'checks': checks})
        if not ready:
            raise ValueError('冻结的研究worker不就绪或已换代')
        with RuntimeStore(out / 'runtime') as store:
            for index, case in enumerate(plan['cases']):
                task = tasks[case['task_id']]
                assert task['role'] == 'dev'
                record = await model_episode(
                    task,
                    case['condition'],
                    data,
                    out / case['condition'] / task['task_id'],
                    engine,
                    store,
                    tokenizer,
                    memories[task['task_id']],
                    out,
                    plan['generation_cap'],
                    plan['readiness_attempts'],
                )
                records.append(record)
                print(
                    canonical(
                        {
                            'case': index + 1,
                            'condition': case['condition'],
                            'task': task['task_id'],
                            **record['result'],
                        }
                    ),
                    flush=True,
                )
                if mode == 'full' and index == 5:
                    successes = sum(record['result']['base_success'] for record in records)
                    families = Counter(
                        tasks[record['task_id']]['family']
                        for record in records
                        if record['result']['base_success']
                    )
                    if successes < 4 or any(
                        families[family] < 2
                        for family in set(task['family'] for task in tasks.values())
                    ):
                        write_once(
                            out / 'base-gate.json',
                            {'passed': False, 'successes': successes, 'by_family': dict(families)},
                        )
                        raise ValueError('C0新实例基础能力门未过，停止流程和记忆比较')
                    write_once(
                        out / 'base-gate.json',
                        {'passed': True, 'successes': successes, 'by_family': dict(families)},
                    )
            assert (
                store.db.execute('SELECT COUNT(*) FROM arrivals WHERE lease_released=0').fetchone()[
                    0
                ]
                == 0
            )
        assert all(
            file_hash(REPO / name) == expected for name, expected in plan['source_hashes'].items()
        )
        summary = {}
        for condition in ('plain', 'frozen_workflow', 'trajectory_memory'):
            selected = [record for record in records if record['condition'] == condition]
            if not selected:
                continue
            summary[condition] = {
                'episodes': len(selected),
                'base_success': sum(record['result']['base_success'] for record in selected),
                'workflow_success': sum(record['result']['success'] for record in selected),
                'actions': sum(len(record['trace']) for record in selected),
            }
        write_once(
            out / 'summary.json',
            {
                'status': 'executed_pending_independent_audit',
                'conditions': summary,
                'model_reservations': len((out / 'reservations.jsonl').read_text().splitlines()),
                'future_episodes': 0,
                'training_runs': 0,
            },
        )
        return summary
    finally:
        await engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    parser.add_argument('--data')
    parser.add_argument('--collection')
    parser.add_argument('--model')
    parser.add_argument('--preparation')
    parser.add_argument('--engine-config')
    parser.add_argument('--mode', choices=('full', 'instruction_check'), default='full')
    parser.add_argument('--prior-audit')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(
            args.data, args.collection, args.model, args.output, args.mode, args.prior_audit
        )
    else:
        result = asyncio.run(run(args.preparation, args.engine_config, args.output))
    print(json.dumps(result, ensure_ascii=False))
