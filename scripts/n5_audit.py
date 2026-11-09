"""从原始模型响应、环境、训练来源及原生释放重新核N5开发资格。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(args):
    root = Path(args.run).resolve()
    owner = Path(args.owner).resolve()
    initial = read(root / 'input.json')
    summary = read(root / 'summary.json')
    plan = read(owner / 'plan.json')
    if (
        initial['plan_sha256'] != sha(owner / 'plan.json')
        or sha(owner / 'actor-bindings.json') != initial['actor_bindings_sha256']
    ):
        raise ValueError('运行协议或actor绑定变化')
    for name, expected in initial['source_hashes'].items():
        if sha(root / 'source' / name) != expected:
            raise ValueError('执行源码快照变化')
    sys.path.insert(0, str(root / 'source/src'))
    from serving_lab.continual_procedure import (
        ProcedureActor,
        validate_actor_binding,
        validate_qualification_cases,
    )
    from serving_lab.execution import write_once
    from serving_lab.profile import ServingProfile
    from serving_lab.task_protocol import digest, verify_files
    from serving_lab.workflow import validate_workflow_manifest, assess_workflow
    from serving_lab.teacher_data import validate_teacher_manifest
    from serving_lab.coldstart_actor import ColdStartActor
    from transformers import AutoTokenizer
    from n1_audit import check_environment_contract, check_response_identity
    from n5_prepare import reconstruct_memories

    binding = initial['worker_manifest']
    role = initial['actor_role']
    actors = read(owner / 'actor-bindings.json')['actors']
    validate_actor_binding(role, binding, actors[role])
    profile = ServingProfile(binding['profile'], binding['version_id'])
    if (
        profile.digest != binding['profile_digest']
        or sha(Path(args.model) / 'tokenizer_config.json')
        != actors[role]['tokenizer_config_sha256']
    ):
        raise ValueError('分词器或profile不符')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    tokenize = lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    basic = args.basic
    if basic:
        data = Path(initial['data'])
        collection = owner.parents[1] / 'artifacts/q2-teacher-collection-20261103-b'
        if (
            sha(data / 'manifest.json') != initial['data_manifest_sha256']
            or sha(data / 'manifest.json') != plan['basic_gate']['data_manifest_sha256']
        ):
            raise ValueError('基础面板来源变化')
        tasks = validate_teacher_manifest(read(data / 'manifest.json'))
        expected = [
            {
                'id': 'n5-basic-' + t['task_id'],
                'source_task_id': t['task_id'],
                'condition': 'basic',
                'role': 'dev',
                'family': t['family'],
            }
            for t in tasks.values()
            if t['role'] == 'dev'
        ]
        if len(expected) != 24 or initial['cases'] != expected:
            raise ValueError('基础门必须完整覆盖实际24个dev身份')
        if not args.training_evidence:
            raise ValueError('缺少共同训练证明')
        evidence = read(args.training_evidence)
        if (
            not evidence['passed']
            or sha(args.training_evidence) != initial['training_evidence_sha256']
            or evidence['artifact_digest'] != binding['artifact_digest']
            or evidence['parameter_digest'] != binding['parameter_digest']
        ):
            raise ValueError('基础评价没有绑定实际训练')
        memories = {}
    else:
        preparation = initial['preparation']
        data = Path(preparation['data'])
        collection = Path(preparation['collection'])
        if sha(data / 'manifest.json') != preparation['data_manifest_sha256']:
            raise ValueError('任务来源变化')
        tasks = validate_workflow_manifest(read(data / 'manifest.json'))
        validate_qualification_cases(initial['cases'], tasks)
        memories, source_audit = reconstruct_memories(data, collection, tokenizer, plan)
        if source_audit != preparation['source_audit']:
            raise ValueError('训练历史资格变化')
    # 辅助脚本需要原仓库ROOT定位原始数据；实际加载字节必须与执行快照完全相同。
    # 不把当前工作树的同名文件或已核但未执行的快照，当作实际审计依赖。
    loaded_scripts = {}
    script_root = Path(__file__).resolve().parent
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.parent == script_root and path.name != 'n5_audit.py':
            name = 'scripts/' + path.name
            if name not in initial['source_hashes'] or sha(path) != initial['source_hashes'][name]:
                raise ValueError('实际辅助脚本不等于已核执行快照: ' + name)
            loaded_scripts[name] = {'path': str(path), 'sha256': sha(path)}
    native = {}
    for line in Path(args.releases).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') == 'request_released':
            key = (event['worker_instance_id'], event['startup_generation'], event['request_id'])
            if key in native:
                raise ValueError('重复原生释放')
            native[key] = event
    with sqlite3.connect((root / 'runtime/service.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        arrivals = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    seen = set()
    matched = set()
    cost = Counter()

    def call(directory, messages, ids, choices, metadata, receipt, action):
        reserved = read(directory / 'reservation.json')
        raw = read(directory / 'response.raw')
        body = {
            'model': binding['version_id'],
            'messages': messages,
            'max_tokens': 64,
            'stream': False,
            'action_choices': choices,
        }
        if (
            reserved['wire'] != profile.payload(body)
            or reserved['input_token_ids'] != ids
            or reserved['metadata'] != metadata
        ):
            raise ValueError('实际输入或wire不同')
        if (
            reserved['worker'] != binding['identity']
            or reserved['profile_digest'] != profile.digest
        ):
            raise ValueError('请求worker/profile漂移')
        aid, attempt = receipt['arrival_id'], receipt['attempt_id']
        check_response_identity(raw, attempt)
        if (
            aid in seen
            or read(directory / 'receipt.json') != receipt
            or reserved['arrival_id'] != aid
            or reserved['attempt_id'] != attempt
        ):
            raise ValueError('请求身份不唯一或对应错误')
        seen.add(aid)
        row = arrivals[aid]
        if (
            row['state'] != 'completed'
            or row['cleanup'] != 'confirmed'
            or row['lease_released'] != 1
            or row['attempt_id'] != attempt
            or json.loads(row['observation']) != receipt
            or json.loads(row['route']) != metadata
            or json.loads(row['worker']) != binding['identity']
        ):
            raise ValueError('账本未匹配实际完成/释放')
        if read(root / 'runtime/requests' / (aid + '.json')) != body:
            raise ValueError('账本保存的请求不同')
        wire = read(root / 'runtime/wire-requests' / (aid + '.json'))
        if wire['payload'] != reserved['wire'] or wire['input_token_ids'] != ids:
            raise ValueError('账本wire不符')
        usage = receipt['usage']
        content = raw['choices'][0]['message']['content']
        if (
            raw['model'] != binding['version_id']
            or raw['usage'] != usage
            or content != receipt['content']
            or content != action
            or action not in choices
            or raw['choices'][0]['finish_reason'] != receipt['finish_reason']
            or receipt['finish_reason'] != 'stop'
            or not receipt['protocol_valid']
            or usage['prompt_tokens'] != len(ids)
            or not 0 < usage['completion_tokens'] <= 64
            or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']
        ):
            raise ValueError('原响应/动作/token不符')
        release = receipt['release']
        key = (release['worker_instance_id'], release['startup_generation'], release['request_id'])
        if (
            key in matched
            or native.get(key) != release
            or key[:2]
            != (
                binding['identity']['worker_instance_id'],
                binding['identity']['startup_generation'],
            )
            or key[2] not in {'chatcmpl-' + attempt, 'chatcmpl-' + attempt + '-0'}
            or release['status'] != 'FINISHED_STOPPED'
            or not release['request_absent']
            or any(release['after_blocks'])
        ):
            raise ValueError('没有精确原生KV释放')
        matched.add(key)
        cost['input_tokens'] += usage['prompt_tokens']
        cost['output_tokens'] += usage['completion_tokens']

    for i, command in enumerate(['look', 'inventory'], 1):
        folder = root / 'canaries' / str(i)
        messages = [{'role': 'user', 'content': 'Return the single permitted command.'}]
        call(
            folder,
            messages,
            tokenize(messages),
            [command],
            {'purpose': 'n5_interface_canary', 'case': 'canary-' + str(i), 'step': 0},
            read(folder / 'receipt.json'),
            command,
        )
    records = {p.parent.name: p for p in (root / 'episodes').glob('*/record.json')}
    if set(records) != {c['id'] for c in initial['cases']}:
        raise ValueError('开发回合缺失或多出')
    scores = Counter()
    coverage = Counter()
    thermal = Counter()
    containers = set()
    actions = 0
    image = read(collection / 'input.json')['image']
    for case in initial['cases']:
        path = records[case['id']]
        record = read(path)
        task = tasks[case['source_task_id']]
        verify_files(task, data)
        if (
            record['record_hash'] != digest({k: v for k, v in record.items() if k != 'record_hash'})
            or record['case'] != case
            or record['source_task_id'] != task['task_id']
        ):
            raise ValueError('回合身份不同')
        if (
            record['worker'] != binding['identity']
            or record['profile_digest'] != profile.digest
            or record['eligible_for_training']
            or record['eligible_for_primary_history']
        ):
            raise ValueError('开发结果错误晋级或模型不符')
        env, launch, exited = [
            read(path.parent / 'environment' / n)
            for n in ['environment.json', 'launch.json', 'exit.json']
        ]
        check_environment_contract(env, launch, image)
        if (
            exited['ExitCode'] != 0
            or exited['Running']
            or env['reference_enabled']
            or launch['container_id'] in containers
        ):
            raise ValueError('环境资格或清理失败')
        containers.add(launch['container_id'])
        contract = env['execution_contract']
        names = [
            'scripts/q2_environment.py',
            'src/serving_lab/public_tasks.py',
            'src/serving_lab/task_protocol.py',
        ]
        if not basic:
            names.append('src/serving_lab/workflow.py')
        if (
            contract['sources'] != {n: initial['source_hashes'][n] for n in names}
            or contract['stagnation_limit'] != 4
        ):
            raise ValueError('实际环境源码/停止规则不同')
        if (
            record['initial'] != env['initial']
            or record['result'] != env['result']
            or env['task_id'] != task['task_id']
            or record['workflow'] != env.get('workflow')
            or record['workflow'] != task.get('workflow')
            or len(record['trace']) != len(env['trace'])
        ):
            raise ValueError('环境原件不一致')
        assessed = (
            {'success': env['result']['success']}
            if basic
            else assess_workflow(record['trace'], task['workflow'], env['result']['base_success'])
        )
        if any(record['result'][k] != v for k, v in assessed.items()):
            raise ValueError('流程成绩重算不同')
        memory = memories[task['task_id']] if case['condition'] == 'memory' else None
        if record['memory_receipt'] != memory:
            raise ValueError('实际M不是train重建记忆')
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
        if record['purpose'] != purpose:
            raise ValueError('回合用途与评价模式不同')
        for i, (row, actual) in enumerate(zip(record['trace'], env['trace']), 1):
            if any(row[k] != actual[k] for k in ['step', 'before', 'action', 'after']):
                raise ValueError('模型动作与实际环境动作不同')
            prepared = actor.prepare(
                record['initial'],
                row['before'],
                record['trace'][: i - 1],
                tokenize,
                plan['qualification']['input_tokens'],
            )
            if prepared['context'] != row['context']:
                raise ValueError('实际actor上下文不同')
            metadata = {
                'purpose': purpose,
                'case': case['id'],
                'step': i,
                'actor_context': row['context'],
                'plan_sha256': sha(owner / 'plan.json'),
            }
            call(
                path.parent / 'calls' / f'{i:03}',
                prepared['messages'],
                prepared['input_token_ids'],
                prepared['choices'],
                metadata,
                row['receipt'],
                row['action'],
            )
        actions += len(record['trace'])
        value = int(assessed['success'])
        condition = case['condition']
        scores[condition] += value
        if not basic:
            coverage[condition, task['phase']] += value
            thermal[condition, task['phase'], task['workflow']['thermal']] += value
    if set(arrivals) != seen or len(seen) != actions + 2:
        raise ValueError('存在未归属请求或缺失调用')
    bootstrap_keys = set()
    bootstrap_tokens = Counter()
    for label in ['plain', 'stream']:
        row = read(Path(args.bootstrap) / (label + '-response.json'))
        release = row['release']
        key = (release['worker_instance_id'], release['startup_generation'], release['request_id'])
        if (
            key in matched
            or key in bootstrap_keys
            or native.get(key) != release
            or not release['request_absent']
            or any(release['after_blocks'])
        ):
            raise ValueError('bootstrap与原生释放不符')
        if key[:2] != (
            binding['identity']['worker_instance_id'],
            binding['identity']['startup_generation'],
        ):
            raise ValueError('bootstrap属于其他worker')
        bootstrap_keys.add(key)
        bootstrap_tokens['input_tokens'] += row['usage']['prompt_tokens']
        bootstrap_tokens['output_tokens'] += row['usage']['completion_tokens']
    if set(native) != matched | bootstrap_keys:
        raise ValueError('存在未归属的worker实际生成，不能忽略成本')
    q = plan['qualification']
    if basic:
        qualified = scores['basic'] >= plan['basic_gate']['minimum_success']
        phases = {}
        expected_summary = {
            'completed': True,
            'qualified': qualified,
            'passed_cases': scores['basic'],
            'cases': 24,
            'actual_generations': len(seen),
            'study_actions': actions,
            'interface_canaries': 2,
            'future_episodes': 0,
            'artifact_digest': binding['artifact_digest'],
            'parameter_digest': binding['parameter_digest'],
            'scope': 'shared basic actor development readiness, no workflow or continual-learning claim',
        }
    else:
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
        phases = {c: {str(s): coverage[c, s] for s in [1, 2, 3]} for c in q['conditions']}
        expected_summary = {
            'completed': True,
            'qualified': qualified,
            'scores': dict(scores),
            'per_condition_denominator': 18,
            'per_phase': phases,
            'actual_generations': len(seen),
            'study_actions': actions,
            'interface_canaries': 2,
            'future_episodes': 0,
            'training_runs': 0,
            'artifact_digest': binding['artifact_digest'],
            'parameter_digest': binding['parameter_digest'],
            'scope': 'development common actor and strong-memory eligibility, not continual-learning result',
        }
    if summary != expected_summary:
        raise ValueError('摘要身份、固定计数或结果不能被原件重算支持')
    for name, value in loaded_scripts.items():
        if sha(value['path']) != value['sha256']:
            raise ValueError('审计期间实际辅助脚本变化')
    result = {
        'passed': True,
        'qualified': qualified,
        'summary_sha256': sha(root / 'summary.json'),
        'input_sha256': sha(root / 'input.json'),
        'source_sha256': sha(__file__),
        'actual_generations': len(seen),
        'environment_actions': actions,
        'episodes': len(records),
        'matched_native_releases': len(matched),
        'scores': dict(scores),
        'per_phase': phases,
        'token_cost': dict(cost),
        'bootstrap_generations': len(bootstrap_keys),
        'total_worker_generations': len(native),
        'bootstrap_token_cost': dict(bootstrap_tokens),
        'actual_script_dependencies': loaded_scripts,
        'actor_role': role,
        'artifact_digest': binding['artifact_digest'],
        'parameter_digest': binding['parameter_digest'],
        'future_episodes': 0,
        'training_runs': 0,
        'scope': 'complete basic readiness audited'
        if basic
        else 'complete development qualification audited, not research effect',
    }
    write_once(args.output or root / 'audit.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['owner', 'run', 'model', 'releases', 'bootstrap']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--basic', action='store_true')
    p.add_argument('--training-evidence')
    p.add_argument('--output')
    audit(p.parse_args())
