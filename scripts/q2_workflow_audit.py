"""使用执行时冻结源码，逐动作复核流程试点、记忆、实际参数和原生释放。"""

import argparse
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


def audit(root, releases_path, output, nondispatched_interruption=False):
    root = Path(root).resolve()
    initial = read(root / 'input.json')
    preparation = Path(initial['preparation'])
    plan = read(preparation / 'plan.json')
    assert sha(preparation / 'plan.json') == initial['plan_sha256']
    for name, expected in plan['source_hashes'].items():
        assert sha(preparation / 'source' / name) == expected
    source = preparation / 'source/src'
    sys.path.insert(0, str(source))
    from serving_lab import workflow

    if not Path(workflow.__file__).resolve().is_relative_to(source):
        raise ValueError('请在独立Python进程运行审计，避免导入当前源码覆盖冻结版本')
    from serving_lab.lab import inspect_base, digest
    from serving_lab.parameters import parameter_identity
    from serving_lab.profile import ServingProfile
    from serving_lab.standard_actor import StandardActor
    from serving_lab.task_protocol import verify_files
    from serving_lab.workflow_memory import memory_entry, retrieve
    from transformers import AutoTokenizer

    model = Path(plan['model'])
    binding = initial['worker_manifest']
    artifact = inspect_base(model)
    parameters = parameter_identity(model)
    assert binding['artifact_digest'] == digest(artifact['files'])
    assert binding['parameter_digest'] == parameters['digest']
    assert digest(inspect_base(model)['files']) == binding['artifact_digest']
    profile = ServingProfile(binding['profile'], binding['version_id'])
    assert profile.digest == binding['profile_digest']
    if 'reference_model' in plan:
        assert plan['reference_model']['artifact_digest'] == binding['artifact_digest']
        assert plan['reference_model']['parameter_digest'] == parameters['digest']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    for name, expected in plan['tokenizer_files'].items():
        assert sha(model / name) == expected
    tokenize = lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    tokenize_text = lambda text: tokenizer.encode(text, add_special_tokens=False)
    data, collection = Path(plan['data']), Path(plan['collection'])
    manifest = read(data / 'manifest.json')
    assert sha(data / 'manifest.json') == plan['data_manifest_sha256']
    tasks = workflow.validate_workflow_manifest(manifest)
    assert sha(preparation / 'memory.json') == plan['memory_sha256']
    assert sha(preparation / 'retrieval.json') == plan['retrieval_sha256']
    entries = read(preparation / 'memory.json')['entries']
    expected_sources = {task['task_id'] for task in tasks.values() if task['role'] == 'train'}
    assert len(entries) == len(expected_sources) == 72
    assert {entry['source_task_id'] for entry in entries} == expected_sources
    for entry in entries:
        task = tasks[entry['source_task_id']]
        record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
        assert entry == memory_entry(
            record, task, manifest['source_contract_digest'], plan['data_manifest_sha256']
        )
    retrievals = read(preparation / 'retrieval.json')
    development = [task for task in tasks.values() if task['role'] == 'dev']
    for task in development:
        original = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
        goal = re.search(r'Your task is to:\s*([^\n]+)', original['initial']['feedback'])[1].strip()
        recomputed = retrieve(
            entries, goal, task['workflow'], cutoff=task['sequence'], tokenize=tokenize_text
        )
        assert all(retrievals[task['task_id']][key] == value for key, value in recomputed.items())
        assert retrievals[task['task_id']]['initial_digest'] == digest(original['initial'])
    mode = plan.get('mode', 'full')
    if mode == 'instruction_check':
        bases = {
            family: min(task['base_id'] for task in development if task['family'] == family)
            for family in {task['family'] for task in development}
        }
        selected_development = [task for task in development if task['base_id'] in bases.values()]
        expected_cases = []
    else:
        assert mode == 'full'
        selected_development = development
        expected_cases = [
            {'condition': 'plain', 'task_id': task['task_id']}
            for task in development
            if task['phase'] == 1
        ]
    for index, task in enumerate(selected_development):
        conditions = (
            ('frozen_workflow', 'trajectory_memory')
            if index % 2 == 0
            else ('trajectory_memory', 'frozen_workflow')
        )
        expected_cases += [
            {'condition': condition, 'task_id': task['task_id']} for condition in conditions
        ]
    assert plan['cases'] == expected_cases and len(expected_cases) == (42 if mode == 'full' else 12)
    paths = sorted(root.glob('*/*/record.json'))
    present = {(path.parent.parent.name, path.parent.name) for path in paths}
    planned_cases = expected_cases
    failed_case = None
    if nondispatched_interruption:
        assert 0 < len(paths) < len(planned_cases)
        expected_cases = planned_cases[: len(paths)]
        failed_case = planned_cases[len(paths)]
    assert present == {(case['condition'], case['task_id']) for case in expected_cases}
    releases = {}
    for line in Path(releases_path).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') == 'request_released':
            key = (event['worker_instance_id'], event['startup_generation'], event['request_id'])
            assert key not in releases
            releases[key] = event
    connection = sqlite3.connect(
        'file:' + (root / 'runtime/service.sqlite').as_posix() + '?mode=ro', uri=True
    )
    connection.row_factory = sqlite3.Row
    arrivals = {row['id']: dict(row) for row in connection.execute('SELECT * FROM arrivals')}
    seen_arrivals, seen_attempts, containers, matched = set(), set(), set(), set()
    results = []
    for case in expected_cases:
        task, condition = tasks[case['task_id']], case['condition']
        directory = root / condition / task['task_id']
        record, environment, launch, exit_state = [
            read(directory / name)
            for name in ('record.json', 'environment.json', 'launch.json', 'exit.json')
        ]
        verify_files(task, data)
        assert (
            record['task_id'] == environment['task_id'] == task['task_id']
            and record['condition'] == condition
        )
        assert record['role'] == task['role'] == 'dev'
        assert all(record[key] == task[key] for key in ('phase', 'sequence', 'workflow'))
        assert (
            record['worker'] == binding['identity'] and record['profile_digest'] == profile.digest
        )
        assert record['record_hash'] == digest(
            {key: value for key, value in record.items() if key != 'record_hash'}
        )
        assert (
            record['purpose'] == 'actor_diagnostic'
            and record['eligible_for_training'] is False
            and record['eligible_for_primary_history'] is False
        )
        assert (
            record['initial'] == environment['initial']
            and record['result'] == environment['result']
        )
        assert digest(record['initial']) == retrievals[task['task_id']]['initial_digest']
        assert environment['workflow'] == task['workflow']
        assert (
            not environment['reference_enabled']
            and exit_state['ExitCode'] == 0
            and not exit_state['Running']
        )
        assert launch['container_id'] not in containers
        containers.add(launch['container_id'])
        memory = retrievals[task['task_id']] if condition == 'trajectory_memory' else None
        assert record['memory_receipt'] == memory
        actor = (
            StandardActor()
            if condition == 'plain'
            else workflow.WorkflowActor(task['workflow'], memory['prompt'] if memory else [])
        )
        prefix = []
        input_tokens, output_tokens = 0, 0
        assert len(record['trace']) == len(environment['trace']) == record['result']['steps']
        assert len(list((directory / 'calls').glob('*/receipt.json'))) == len(record['trace'])
        for index, (step, env_step) in enumerate(zip(record['trace'], environment['trace']), 1):
            assert all(step[key] == env_step[key] for key in ('step', 'before', 'action', 'after'))
            assert set(step['before']) == set(step['after']) == {'feedback', 'admissible_commands'}
            prepared = actor.prepare(record['initial'], step['before'], prefix, tokenize)
            call = directory / 'calls' / f'{index:03}'
            reservation, receipt, response = [
                read(call / name) for name in ('reservation.json', 'receipt.json', 'response.raw')
            ]
            assert receipt == step['receipt'] and prepared['context'] == step['context']
            assert reservation['input_token_ids'] == prepared['input_token_ids']
            body = {
                'model': binding['version_id'],
                'messages': prepared['messages'],
                'max_tokens': 64,
                'stream': False,
                'action_choices': prepared['choices'],
            }
            assert reservation['wire'] == profile.payload(body)
            arrival, attempt = receipt['arrival_id'], receipt['attempt_id']
            assert reservation['arrival_id'] == arrival and reservation['attempt_id'] == attempt
            metadata = {
                'purpose': 'workflow_readiness_diagnostic',
                'task_id': task['task_id'],
                'condition': condition,
                'step': index,
                'actor_context': prepared['context'],
                'workflow_digest': digest(task['workflow']),
                'memory_receipt_digest': digest(memory) if memory else None,
            }
            if 'readiness_attempts' in plan:
                metadata['readiness_attempts'] = plan['readiness_attempts']
                checks = receipt['readiness_checks']
                assert 1 <= len(checks) <= plan['readiness_attempts'] and checks[-1]['ready']
                assert all(not check['ready'] for check in checks[:-1])
                assert all(
                    check.get('same_manifest') is not False
                    and check.get('http_status') not in (401, 403)
                    for check in checks
                )
            assert reservation['metadata'] == metadata
            assert (
                reservation['worker'] == binding['identity']
                and reservation['profile_digest'] == profile.digest
            )
            assert arrival not in seen_arrivals and attempt not in seen_attempts
            seen_arrivals.add(arrival)
            seen_attempts.add(attempt)
            row = arrivals[arrival]
            assert row['attempt_id'] == attempt and json.loads(row['worker']) == binding['identity']
            assert json.loads(row['route']) == metadata and row['input_hash'] == digest(body)
            assert (
                row['state'] == 'completed'
                and row['cleanup'] == 'confirmed'
                and row['lease_released'] == 1
            )
            assert json.loads(row['observation']) == receipt
            assert read(root / 'runtime/requests' / (arrival + '.json')) == body
            wire = read(root / 'runtime/wire-requests' / (arrival + '.json'))
            assert (
                wire['payload'] == reservation['wire']
                and wire['input_token_ids'] == prepared['input_token_ids']
            )
            assert (
                response['model'] == binding['version_id'] and response['usage'] == receipt['usage']
            )
            assert (
                response['choices'][0]['message']['content'] == receipt['content'] == step['action']
            )
            assert receipt['usage']['prompt_tokens'] == len(prepared['input_token_ids'])
            assert 0 < receipt['usage']['completion_tokens'] <= 64
            assert (
                receipt['usage']['total_tokens']
                == receipt['usage']['prompt_tokens'] + receipt['usage']['completion_tokens']
            )
            assert actor.action(receipt, prepared['choices']) == step['action']
            release = receipt['release']
            key = (
                release['worker_instance_id'],
                release['startup_generation'],
                release['request_id'],
            )
            assert key not in matched and releases[key] == release
            assert release['request_id'] in {'chatcmpl-' + attempt, 'chatcmpl-' + attempt + '-0'}
            assert release['request_absent'] and not any(release['after_blocks'])
            matched.add(key)
            input_tokens += receipt['usage']['prompt_tokens']
            output_tokens += receipt['usage']['completion_tokens']
            prefix.append(step)
        assessment = workflow.assess_workflow(
            record['trace'], task['workflow'], record['result']['base_success']
        )
        assert all(record['result'][key] == value for key, value in assessment.items())
        results.append(
            {
                'task_id': task['task_id'],
                'base_id': task['base_id'],
                'family': task['family'],
                'condition': condition,
                'variant': task['variant'],
                'base_success': assessment['base_success'],
                'workflow_success': assessment['success'],
                'obsolete_order_observed': assessment['obsolete_order_observed'],
                'steps': len(prefix),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            }
        )
    failed_reservations = []
    if failed_case is not None:
        task, condition = tasks[failed_case['task_id']], failed_case['condition']
        directory = root / condition / task['task_id']
        failure, environment, launch, exit_state = [
            read(directory / name)
            for name in ('failure.json', 'environment.json', 'launch.json', 'exit.json')
        ]
        verify_files(task, data)
        assert (
            failure['task_id'] == environment['task_id'] == task['task_id']
            and failure['condition'] == condition
        )
        assert (
            not failure['trace']
            and failure['pending'] is None
            and not environment['trace']
            and environment['result'] is None
        )
        assert (
            failure['initial'] == environment['initial']
            and exit_state['ExitCode'] == 0
            and not exit_state['Running']
        )
        assert (
            digest(failure['initial']) == retrievals[task['task_id']]['initial_digest']
            and environment['workflow'] == task['workflow']
        )
        assert launch['container_id'] not in containers
        assert len(list((directory / 'calls').glob('*/receipt.json'))) == 1
        reservation, receipt = [
            read(directory / 'calls/001' / name) for name in ('reservation.json', 'receipt.json')
        ]
        assert (
            not receipt['protocol_valid']
            and receipt['error']['message'] == 'worker_not_ready'
            and receipt['release'] is None
        )
        arrival, attempt = receipt['arrival_id'], receipt['attempt_id']
        assert reservation['arrival_id'] == arrival and reservation['attempt_id'] == attempt
        assert arrival not in seen_arrivals and attempt not in seen_attempts
        seen_arrivals.add(arrival)
        seen_attempts.add(attempt)
        row = arrivals[arrival]
        assert (
            row['state'] == 'failed'
            and row['dispatched'] == 0
            and row['lease_released'] == 1
            and row['cleanup'] == 'none'
        )
        assert json.loads(row['observation']) == receipt
        assert not any(
            key[2] in {'chatcmpl-' + attempt, 'chatcmpl-' + attempt + '-0'} for key in releases
        )
        memory = retrievals[task['task_id']] if condition == 'trajectory_memory' else None
        actor = workflow.WorkflowActor(task['workflow'], memory['prompt'] if memory else [])
        prepared = actor.prepare(failure['initial'], failure['initial'], [], tokenize)
        body = {
            'model': binding['version_id'],
            'messages': prepared['messages'],
            'max_tokens': 64,
            'stream': False,
            'action_choices': prepared['choices'],
        }
        assert (
            reservation['wire'] == profile.payload(body)
            and reservation['input_token_ids'] == prepared['input_token_ids']
        )
        assert read(root / 'runtime/requests' / (arrival + '.json')) == body
        assert (
            row['input_hash'] == digest(body) and json.loads(row['worker']) == binding['identity']
        )
        assert (
            reservation['metadata']['task_id'] == task['task_id']
            and reservation['metadata']['condition'] == condition
        )
        assert reservation['metadata']['step'] == 1 and reservation['worker'] == binding['identity']
        failed_reservations.append((task['task_id'], condition, 1))
        for case in planned_cases[len(expected_cases) + 1 :]:
            assert not (root / case['condition'] / case['task_id']).exists()
    assert seen_arrivals == set(arrivals)
    reserved = [json.loads(line) for line in (root / 'reservations.jsonl').read_text().splitlines()]
    assert len(reserved) == len(seen_arrivals) <= plan['generation_cap']
    for index, row in enumerate(reserved, 1):
        assert row['call'] == index
    expected_reservations = [
        (row['task_id'], row['condition'], step)
        for row in results
        for step in range(1, row['steps'] + 1)
    ]
    expected_reservations.extend(failed_reservations)
    assert [
        (row['task_id'], row['condition'], row['step']) for row in reserved
    ] == expected_reservations
    plain = [row for row in results if row['condition'] == 'plain']
    if mode == 'full':
        assert sum(row['base_success'] for row in plain) >= 4
        assert all(
            sum(row['base_success'] for row in plain if row['family'] == family) >= 2
            for family in {row['family'] for row in plain}
        )
    else:
        prior = plan['prior_audit']
        assert (
            sha(prior['audit']) == prior['sha256'] and sha(prior['input']) == prior['input_sha256']
        )
        previous = read(prior['audit'])
        assert previous['raw_verified'] and previous['conditions']['plain']['base_success'] >= 4
        assert prior['worker_manifest'] == binding
        assert previous['model_reservations'] + len(reserved) <= 4200
    result = {
        'raw_verified': True,
        'episodes': len(results),
        'base_instances': len({row['base_id'] for row in results}),
        'planned_episodes': len(planned_cases),
        'complete_run': failed_case is None,
        'failed_before_dispatch': len(failed_reservations),
        'unexecuted_episodes': len(planned_cases) - len(results) - len(failed_reservations),
        'model_reservations': len(reserved),
        'matched_releases': len(matched),
        'unreleased_leases': 0,
        'parameter_digest': parameters['digest'],
        'artifact_digest': binding['artifact_digest'],
        'input_tokens': sum(row['input_tokens'] for row in results),
        'output_tokens': sum(row['output_tokens'] for row in results),
        'future_episodes': 0,
        'training_runs': 0,
        'episodes_detail': results,
        'mode': mode,
        'conditions': {
            condition: {
                'episodes': len([row for row in results if row['condition'] == condition]),
                'base_success': sum(
                    row['base_success'] for row in results if row['condition'] == condition
                ),
                'workflow_success': sum(
                    row['workflow_success'] for row in results if row['condition'] == condition
                ),
            }
            for condition in ('plain', 'frozen_workflow', 'trajectory_memory')
            if any(row['condition'] == condition for row in results)
        },
        'scope': 'paired development readiness; not parameter learning or future generalization',
        'auditor_sha256': sha(__file__),
    }
    with Path(output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != 'episodes_detail'},
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--releases', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nondispatched-interruption', action='store_true')
    args = parser.parse_args()
    audit(args.run, args.releases, args.output, args.nondispatched_interruption)
