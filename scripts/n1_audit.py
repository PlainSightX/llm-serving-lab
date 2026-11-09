"""N1独立原件审计：冻结源码/参数、实际提示、环境及原生释放逐次对应。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_response_identity(raw, attempt):
    if raw.get('id') != 'chatcmpl-' + attempt or ('choices' in raw and len(raw['choices']) != 1):
        raise ValueError('原始响应不属于本次请求或响应分支数不符')


def check_environment_contract(env, launch, expected_image):
    if launch['image'] != expected_image or launch['command'][-2:] != [
        expected_image,
        '/runner.py',
    ]:
        raise ValueError('环境image或runner不符')
    result = env['result']
    trace = env['trace']
    if result['max_steps'] != 100 or not 0 < len(trace) <= 100 or result['steps'] != len(trace):
        raise ValueError('不是原100步合同')
    streak, last = 0, None
    for i, row in enumerate(trace, 1):
        before, after, action = row['before'], row['after'], row['action']
        valid = (
            isinstance(action, str)
            and bool(action.strip())
            and len(action) <= 256
            and '\n' not in action
            and '\r' not in action
        )
        admissible = valid and (
            action in before['admissible_commands'] or action in {'look', 'inventory'}
        )
        if (
            row['step'] != i
            or row['format_valid'] != valid
            or row['admissible'] != admissible
            or not admissible
        ):
            raise ValueError('动作合法性或步号不符')
        same = before['feedback'] == after['feedback'] and sorted(
            before['admissible_commands']
        ) == sorted(after['admissible_commands'])
        streak = (streak + 1 if action == last else 1) if same else 0
        last = action
        if i < len(trace) and streak >= 4:
            raise ValueError('停滞终止后仍继续执行')
    physical = result.get('base_success', result['success'])
    termination = result.get('base_termination', result['termination'])
    if result['format_valid_rate'] != 1 or result['admissible_rate'] != 1:
        raise ValueError('动作分母不符')
    if termination not in {'won', 'environment_done', 'step_budget', 'stagnation_terminated'}:
        raise ValueError('未知终态')
    if (termination == 'won') != physical:
        raise ValueError('物理成功与终态冲突')
    if termination == 'stagnation_terminated' and (streak != 4 or physical):
        raise ValueError('停滞终止不符')
    if termination == 'step_budget' and (len(trace) != 100 or streak >= 4 or physical):
        raise ValueError('步数终止不符')


def check_coverage(cases, present):
    expected = [(row['condition'], row['task_id']) for row in cases]
    if (
        len(expected) != len(set(expected))
        or len(present) != len(set(present))
        or set(expected) != set(present)
    ):
        raise ValueError('案例缺失、重复或超出冻结集合')


def audit(root, releases, output):
    root = Path(root).resolve()
    initial = read(root / 'input.json')
    preparation = Path(initial['preparation'])
    plan = read(preparation / 'plan.json')
    assert sha(preparation / 'plan.json') == initial['plan_sha256']
    for name, expected in initial['source_hashes'].items():
        assert sha(root / 'source' / name) == expected
    sys.path.insert(0, str(root / 'source/src'))
    from serving_lab import workflow_study

    assert Path(workflow_study.__file__).resolve().is_relative_to(root / 'source/src')
    from serving_lab.lab import inspect_base, digest
    from serving_lab.parameters import parameter_identity
    from serving_lab.profile import ServingProfile
    from serving_lab.coldstart_actor import ColdStartActor
    from serving_lab.workflow import validate_workflow_manifest, WorkflowActor, assess_workflow
    from serving_lab.workflow_memory import memory_entry
    from serving_lab.teacher_data import validate_teacher_manifest
    from serving_lab.task_protocol import verify_files
    from transformers import AutoTokenizer

    model = Path(initial['model'])
    artifact = inspect_base(model)
    parameters = parameter_identity(model)
    binding = initial['worker_manifest']
    assert (
        digest(artifact['files'])
        == binding['artifact_digest']
        == initial['model_identity']['artifact_digest']
    )
    assert (
        parameters['digest']
        == binding['parameter_digest']
        == initial['model_identity']['parameter_digest']
    )
    c0_digest = '1ee83b8562404b56bdb208a675bb5d4e4c83b827156df8f6b1e73d30842989de'
    raw_digest = '2d5f13caf0959c19912d4ed0ac556e6c6949e2fe855028e97233c5b35fa7f013'
    if initial['mode'] == 'baseline':
        assert binding['artifact_digest'] in {c0_digest, raw_digest}
    if initial['mode'] == 'application_control':
        assert (
            binding['artifact_digest'] == c0_digest
            and initial.get('environment_contract_recorded') is True
        )
    candidate_training = initial.get('candidate_training')
    if initial['mode'] == 'application' or (
        initial['mode'] == 'stage1' and binding['artifact_digest'] != c0_digest
    ):
        assert candidate_training and initial.get('environment_contract_recorded') is True
        from serving_lab.training_evidence import verify_training

        assert candidate_training == verify_training(
            candidate_training['training'],
            candidate_training['dataset'],
            model,
            candidate_training['parent_model'],
        )
    profile = ServingProfile(binding['profile'], binding['version_id'])
    assert profile.digest == binding['profile_digest']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    for name, expected in plan['files'].items():
        assert sha(preparation / name) == expected
    wfdata = Path(plan['data'])
    manifest = read(wfdata / 'manifest.json')
    assert sha(wfdata / 'manifest.json') == plan['data_sha256']
    workflow_tasks = validate_workflow_manifest(manifest)
    collection = Path(plan['collection'])
    # 已执行早期运行未另存config；其固定环境沿用已资格教师collection的原始image。
    expected_image = read(collection / 'input.json')['image']
    environment_anchor = 'qualified_teacher_collection'
    if initial.get('environment_contract_recorded'):
        assert (
            read(root / 'source/configs/evaluation/q1-sources.json')['image_id']
            == initial['environment_image']
            == expected_image
        )
        environment_anchor = 'frozen_environment_config_and_runtime_source_hashes'
    expected_entries = [
        memory_entry(
            read(collection / 'workflow_teacher' / t['task_id'] / 'record.json'),
            t,
            manifest['source_contract_digest'],
            plan['data_sha256'],
        )
        for t in workflow_tasks.values()
        if t['role'] == 'train'
    ]
    entries = read(preparation / 'memory.json')['entries']
    assert entries == expected_entries and len(entries) == 72
    memories = read(preparation / 'retrievals.json')
    states = read(preparation / 'states.json')
    for task in workflow_tasks.values():
        if task['role'] != 'dev':
            continue
        record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
        goal = re.search(r'Your task is to:\s*([^\n]+)', record['initial']['feedback'])[1].strip()
        for kind in ('script', 'combined'):
            expected = workflow_study.retrieve_procedural(
                entries,
                goal,
                task['workflow'],
                cutoff=task['sequence'],
                tokenize=lambda text: tokenizer.encode(text, add_special_tokens=False),
                representation=kind,
            )
            assert memories[task['task_id'] + '/' + kind] == {
                **expected,
                'initial_digest': digest(record['initial']),
            }
        if task['task_id'] in states:
            assert states[task['task_id']] == workflow_study.decision_state(record)
    data = Path(initial['data'])
    assert sha(data / 'manifest.json') == initial['data_sha256']
    mode = initial['mode']
    if mode in {'application', 'application_control'}:
        tasks = validate_teacher_manifest(read(data / 'manifest.json'))
        expected = [
            {'task_id': t['task_id'], 'condition': 'old_application'}
            for t in tasks.values()
            if t['role'] == 'dev'
        ]
    elif mode == 'stage1':
        tasks = workflow_tasks
        expected = [
            {'task_id': t['task_id'], 'condition': c}
            for t in tasks.values()
            if t['role'] == 'dev' and t['phase'] == 1
            for c in ('workflow', initial['memory_choice'])
        ]
    else:
        assert mode == 'baseline'
        tasks = workflow_tasks
        dev = [t for t in tasks.values() if t['role'] == 'dev']
        bases = {
            family: min(t['base_id'] for t in dev if t['family'] == family)
            for family in {t['family'] for t in dev}
        }
        expected = [
            {'task_id': t['task_id'], 'condition': c}
            for t in dev
            if t['base_id'] in bases.values()
            for c in ('workflow', 'script', 'combined')
        ]
        assert expected == plan['cases']
    assert expected == initial['cases']
    records = sorted((root / 'episodes').glob('*/*/record.json'))
    decisions = sorted((root / 'decisions').glob('*/*/result.json'))
    check_coverage(expected, [(p.parent.parent.name, p.parent.name) for p in records])
    check_coverage(
        expected if mode == 'baseline' else [],
        [(p.parent.parent.name, p.parent.name) for p in decisions],
    )
    native = {}
    for line in Path(releases).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') == 'request_released':
            key = (event['worker_instance_id'], event['startup_generation'], event['request_id'])
            assert key not in native
            native[key] = event
    db = sqlite3.connect(
        'file:' + (root / 'runtime/service.sqlite').as_posix() + '?mode=ro', uri=True
    )
    db.row_factory = sqlite3.Row
    arrivals = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    seen = set()
    attempts = set()
    matched = set()
    calls = []

    def verify_call(directory, case, kind, step, actor, public, mem, receipt, context, action):
        prepared = actor.prepare(public['initial'], public['view'], public['trace'], tokenize)
        reservation, response = (
            read(directory / 'reservation.json'),
            read(directory / 'response.raw'),
        )
        assert read(directory / 'receipt.json') == receipt and prepared['context'] == context
        assert reservation['input_token_ids'] == prepared['input_token_ids']
        body = {
            'model': binding['version_id'],
            'messages': prepared['messages'],
            'max_tokens': 64,
            'stream': False,
            'action_choices': prepared['choices'],
        }
        metadata = {
            'purpose': 'n1_development',
            'case': case,
            'kind': kind,
            'step': step,
            'actor_context': context,
            'memory_digest': digest(mem) if mem else None,
        }
        assert reservation['wire'] == profile.payload(body) and reservation['metadata'] == metadata
        assert (
            reservation['profile_digest'] == profile.digest
            and reservation['worker'] == binding['identity']
        )
        a, t = receipt['arrival_id'], receipt['attempt_id']
        assert a not in seen and t not in attempts
        check_response_identity(response, t)
        seen.add(a)
        attempts.add(t)
        row = arrivals[a]
        assert (
            reservation['arrival_id'] == a
            and reservation['attempt_id'] == t
            and row['attempt_id'] == t
        )
        assert (
            row['state'] == 'completed'
            and row['cleanup'] == 'confirmed'
            and row['lease_released'] == 1
        )
        assert json.loads(row['observation']) == receipt and json.loads(row['route']) == metadata
        assert json.loads(row['worker']) == binding['identity'] and row['input_hash'] == digest(
            body
        )
        assert read(root / 'runtime/requests' / (a + '.json')) == body
        wire = read(root / 'runtime/wire-requests' / (a + '.json'))
        assert (
            wire['payload'] == reservation['wire']
            and wire['input_token_ids'] == prepared['input_token_ids']
        )
        assert response['model'] == binding['version_id'] and response['usage'] == receipt['usage']
        assert response['choices'][0]['message']['content'] == receipt['content'] == action
        assert response['choices'][0]['finish_reason'] == receipt['finish_reason'] == 'stop'
        assert (
            actor.action(receipt, prepared['choices']) == action
            and receipt['protocol_valid'] is True
        )
        usage = receipt['usage']
        assert (
            usage['prompt_tokens'] == len(prepared['input_token_ids'])
            and 0 < usage['completion_tokens'] <= 64
        )
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        release = receipt['release']
        key = (release['worker_instance_id'], release['startup_generation'], release['request_id'])
        assert (
            key not in matched
            and native[key] == release
            and key[:2]
            == (
                binding['identity']['worker_instance_id'],
                binding['identity']['startup_generation'],
            )
        )
        assert (
            key[2] in {'chatcmpl-' + t, 'chatcmpl-' + t + '-0'}
            and release['status'] == 'FINISHED_STOPPED'
        )
        assert release['request_absent'] and not any(release['after_blocks'])
        matched.add(key)
        checks = receipt['readiness_checks']
        assert 1 <= len(checks) <= 2 and checks[-1]['ready']
        calls.append(
            {
                'case': case,
                'kind': kind,
                'step': step,
                'input_tokens': usage['prompt_tokens'],
                'output_tokens': usage['completion_tokens'],
            }
        )

    decision_results = []
    episode_results = []
    containers = set()
    for path in decisions:
        result = read(path)
        case = result['case']
        task = tasks[case['task_id']]
        kind = case['condition']
        mem = memories[task['task_id'] + '/' + kind] if kind in ('script', 'combined') else None
        state = states[task['task_id']]
        assert (
            result['public_digest'] == state['public_digest']
            and result['source_record_hash'] == state['source_record_hash']
        )
        assert (
            result['memory_receipt'] == mem
            and result['expected_action'] == state['expected_action']
        )
        assert result['matches_teacher_action'] == (result['action'] == state['expected_action'])
        actor = WorkflowActor(task['workflow'], mem['prompt'] if mem else [])
        verify_call(
            path.parent / 'call',
            case,
            'decision',
            state['step'],
            actor,
            state['public'],
            mem,
            result['receipt'],
            result['context'],
            result['action'],
        )
        decision_results.append({'case': case, 'matches': result['matches_teacher_action']})
    for path in records:
        record = read(path)
        task = tasks[record['task_id']]
        condition = record['condition']
        case = {'task_id': task['task_id'], 'condition': condition}
        assert task['role'] == record['role'] == 'dev' and task['family'] == record['family']
        verify_files(task, data)
        assert record['record_hash'] == digest(
            {k: v for k, v in record.items() if k != 'record_hash'}
        )
        assert (
            record['worker'] == binding['identity'] and record['profile_digest'] == profile.digest
        )
        assert (
            record['eligible_for_training'] is False
            and record['eligible_for_primary_history'] is False
            and record['purpose'] == 'actor_diagnostic'
        )
        env, launch, exited = [
            read(path.with_name(n)) for n in ('environment.json', 'launch.json', 'exit.json')
        ]
        check_environment_contract(env, launch, expected_image)
        if initial.get('environment_contract_recorded'):
            contract = env['execution_contract']
            assert contract['max_steps'] == 100 and contract['stagnation_limit'] == 4
            expected_sources = [
                'scripts/q2_environment.py',
                'src/serving_lab/public_tasks.py',
                'src/serving_lab/task_protocol.py',
            ]
            if mode not in {'application', 'application_control'}:
                expected_sources.append('src/serving_lab/workflow.py')
            assert contract['sources'] == {
                name: initial['source_hashes'][name] for name in expected_sources
            }
        assert (
            record['task_id'] == env['task_id'] == path.parent.name
            and condition == path.parent.parent.name
        )
        assert not env['reference_enabled'] and exited['ExitCode'] == 0 and not exited['Running']
        assert launch['container_id'] not in containers
        containers.add(launch['container_id'])
        assert record['initial'] == env['initial'] and record['result'] == env['result']
        assert len(record['trace']) == len(env['trace']) == record['result']['steps'] <= 100
        if mode not in {'application', 'application_control'}:
            assert (
                record['phase'] == task['phase']
                and record['workflow'] == env['workflow'] == task['workflow']
            )
            assessed = assess_workflow(
                record['trace'], task['workflow'], record['result']['base_success']
            )
            assert all(record['result'][k] == v for k, v in assessed.items())
        mem = (
            memories[task['task_id'] + '/' + condition]
            if condition in ('script', 'combined')
            else None
        )
        assert record['memory_receipt'] == mem
        actor = (
            ColdStartActor()
            if mode in {'application', 'application_control'}
            else WorkflowActor(task['workflow'], mem['prompt'] if mem else [])
        )
        prefix = []
        for i, (row, envrow) in enumerate(zip(record['trace'], env['trace']), 1):
            assert all(row[k] == envrow[k] for k in ('step', 'before', 'after', 'action'))
            verify_call(
                path.parent / 'calls' / f'{i:03}',
                case,
                'episode',
                i,
                actor,
                {'initial': record['initial'], 'view': row['before'], 'trace': prefix},
                mem,
                row['receipt'],
                row['context'],
                row['action'],
            )
            prefix.append(row)
        assert len(list((path.parent / 'calls').glob('*/receipt.json'))) == len(prefix)
        episode_results.append(
            {
                'task_id': task['task_id'],
                'condition': condition,
                'family': task['family'],
                'base_id': task.get('base_id'),
                'phase': task.get('phase'),
                'steps': len(prefix),
                'physical_success': bool(
                    record['result'].get('base_success', record['result']['success'])
                ),
                'success': bool(record['result']['success']),
            }
        )
    assert seen == set(arrivals) and len(list(root.rglob('receipt.json'))) == len(seen)
    reserved = [
        json.loads(s)
        for s in (root.parent / 'reservations.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    assert [r['number'] for r in reserved] == list(range(1, len(reserved) + 1)) and len(
        reserved
    ) <= 11000
    own = [r for r in reserved if r['run'] == str(root)]
    key = lambda r: (r['case']['task_id'], r['case']['condition'], r['kind'], r['step'])
    assert Counter(map(key, own)) == Counter(map(key, calls)) and len(own) == len(seen)
    summary = {
        'raw_verified': True,
        'complete_run': True,
        'mode': mode,
        'episodes': len(episode_results),
        'decisions': len(decision_results),
        'real_generations': len(calls),
        'matched_native_releases': len(matched),
        'unreleased_leases': 0,
        'artifact_digest': binding['artifact_digest'],
        'parameter_digest': parameters['digest'],
        'profile_digest': profile.digest,
        'input_tokens': sum(r['input_tokens'] for r in calls),
        'output_tokens': sum(r['output_tokens'] for r in calls),
        'future_episodes': 0,
        'case_results': episode_results,
        'decision_results': decision_results,
        'environment_anchor': environment_anchor,
        'response_ids_verified': True,
        'candidate_training': candidate_training,
        'auditor_sha256': sha(__file__),
        'scope': 'fixed development diagnosis; not future generalization',
    }
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ('case_results', 'decision_results')},
            ensure_ascii=False,
        )
    )
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--releases', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.run, a.releases, a.output)
