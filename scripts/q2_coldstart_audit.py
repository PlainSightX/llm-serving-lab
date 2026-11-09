"""从冷启动评价原件重建命令、上下文、分母和成本，不执行任务或模型。"""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from serving_lab.coldstart_actor import ColdStartActor
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile
from serving_lab.teacher_data import validate_teacher_manifest
from compare_runtime_baseline import copied_database
from q2_audit import check_action_identity
from q2_coldstart_eval import summarize, validate_controls
from q2_rescue import reserved_generations
from q2_standard import CANARIES, POLICY, validate_profile_binding
from q2_teacher import IMAGE, read


def audit(root, data, output):
    root, data = Path(root).resolve(), Path(data).resolve()
    initial, summary = read(root / 'input.json'), read(root / 'summary.json')
    control = initial['controls']
    manifest = read(data / 'manifest.json')
    tasks = validate_teacher_manifest(manifest)
    validate_controls(control)
    assert (
        file_hash(data / 'manifest.json')
        == initial['manifest_sha256']
        == control['manifest_sha256']
    )
    assert (
        initial['conditions'] in (['base', 'fixed_demo'], ['trained'])
        and control['policy'] == POLICY
    )
    assert initial['source_contract_digest'] == manifest['source_contract_digest']
    assert digest(control['examples_text']) == control['examples_digest']
    assert all(tasks[s['task_id']]['role'] == 'train' for s in control['sources'])
    repo = Path(__file__).resolve().parents[1]
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    for name in ('coldstart_actor.py', 'standard_actor.py', 'task_protocol.py'):
        assert (
            file_hash(repo / 'src/serving_lab' / name)
            == initial['source_hashes']['src/serving_lab/' + name]
        ), 'actor重建需要原版本'
    binding, profile, launch = (
        read(root / 'worker/binding.json'),
        read(root / 'profile.json'),
        read(root / 'worker/launch.json'),
    )
    assert (
        launch['config'] == read(root / 'engine-config.json')
        and digest(launch['base_identity']['files']) == binding['artifact_digest']
    )
    validate_profile_binding({**initial, 'policy': control['policy']}, profile, binding)
    if initial['training']:
        trained = read(Path(initial['training']) / 'summary.json')
        assert (
            trained['data_sha256'] == control['data_sha256']
            and digest(trained['merged_identity']['files']) == binding['artifact_digest']
        )
        assert (
            trained['base_artifact_digest'] == control['base_artifact_digest']
            and trained['base_binding_sha256'] == control['base_binding_sha256']
        )
    else:
        assert binding['artifact_digest'] == control['base_artifact_digest']
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        binding['model_path'], local_files_only=True, trust_remote_code=False
    )
    token = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    serving_profile = ServingProfile(profile['document'], binding['served_model'])
    release_rows = [
        r
        for line in Path(binding['events_path']).read_text().splitlines()
        if (r := json.loads(line)).get('event') == 'request_released'
    ]
    releases = {r['request_id']: r for r in release_rows}
    assert len(releases) == len(release_rows)
    ledgers = {}
    for path in root.glob('*runtime/service.sqlite'):
        with copied_database(path) as db:
            db.row_factory = sqlite3.Row
            ledgers[path.parent.name] = {
                r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')
            }
    usage, seen = Counter(), set()

    def check(folder, runtime, choices):
        res, receipt, raw = (
            read(folder / 'reservation.json'),
            read(folder / 'receipt.json'),
            read(folder / 'response.raw'),
        )
        row = ledgers[runtime][res['arrival_id']]
        check_action_identity(res, receipt, raw, row)
        assert (
            row['state'] == 'completed'
            and row['cleanup'] == 'confirmed'
            and row['lease_released'] == 1
        )
        assert (
            json.loads(row['worker']) == res['worker'] and json.loads(row['observation']) == receipt
        )
        assert (
            json.loads(row['route']) == res['metadata']
            and json.loads(row['cleanup_evidence']) == receipt['release']
        )
        assert (
            receipt['release'] == releases[receipt['release']['request_id']]
            and receipt['release']['startup_generation'] == binding['startup_generation']
        )
        assert (
            receipt['protocol_valid']
            and receipt['finish_reason'] == raw['choices'][0]['finish_reason'] == 'stop'
        )
        assert (
            receipt['content'] == raw['choices'][0]['message']['content']
            and receipt['content'] in choices
        )
        assert raw['model'] == binding['version_id'] and receipt['usage'] == raw['usage']
        assert (
            res['profile_digest'] == profile['digest']
            and token(res['wire']['messages']) == res['input_token_ids']
        )
        assert (
            serving_profile.payload(
                {
                    'model': binding['version_id'],
                    'messages': res['wire']['messages'],
                    'max_tokens': 64,
                    'stream': False,
                    'action_choices': choices,
                }
            )
            == res['wire']
        )
        assert read(root / runtime / 'wire-requests' / (row['id'] + '.json')) == {
            'payload': res['wire'],
            'profile_digest': profile['digest'],
            'input_token_ids': res['input_token_ids'],
        }
        assert receipt['usage']['prompt_tokens'] == len(res['input_token_ids']) <= 3500
        assert 0 < receipt['usage']['completion_tokens'] <= 64
        assert (
            receipt['usage']['total_tokens']
            == receipt['usage']['prompt_tokens'] + receipt['usage']['completion_tokens']
        )
        assert (
            receipt['release']['status'] == 'FINISHED_STOPPED' and receipt['attempt_id'] not in seen
        )
        seen.add(receipt['attempt_id'])
        usage.update(
            {k: receipt['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
        return res, receipt

    for i, choices in enumerate(CANARIES, 1):
        check(root / 'canaries' / str(i), 'canary-runtime', choices)
    records = []
    for path in sorted(root.glob('dev/*/*/record.json')):
        r, env, exited, launch_env = (
            read(path),
            read(path.parent / 'environment.json'),
            read(path.parent / 'exit.json'),
            read(path.parent / 'launch.json'),
        )
        t = tasks[r['task_id']]
        assert (
            t['role'] == r['role'] == 'dev'
            and r['task_id'] == env['task_id'] == path.parent.parent.name
        )
        assert (
            r['condition'] == path.parent.name
            and r['condition'] in initial['conditions']
            and r['sequence'] == t['sequence']
        )
        assert r['record_hash'] == digest({k: v for k, v in r.items() if k != 'record_hash'})
        assert (
            r['purpose'] == 'actor_diagnostic'
            and not r['eligible_for_training']
            and not r['eligible_for_primary_history']
        )
        assert (
            r['initial'] == env['initial']
            and r['result'] == env['result']
            and not env['reference_enabled']
        )
        assert launch_env['image'] == IMAGE and launch_env['command'][-2:] == [IMAGE, '/runner.py']
        assert not exited['Running'] and exited['ExitCode'] == 0
        assert (
            len(r['trace']) == len(env['trace']) == r['result']['steps'] <= 100
            and r['result']['max_steps'] == 100
        )
        actor = ColdStartActor(control['examples_text'] if r['condition'] == 'fixed_demo' else None)
        streak, last = 0, None
        for i, (step, raw_step) in enumerate(zip(r['trace'], env['trace'])):
            assert all(step[k] == raw_step[k] for k in ('step', 'before', 'after', 'action'))
            prepared = actor.prepare(r['initial'], step['before'], r['trace'][:i], token)
            res, receipt = check(
                path.parent / f'action-{i + 1:03}', 'action-runtime', prepared['choices']
            )
            assert step['receipt'] == receipt and step['action'] == receipt['content']
            assert (
                res['wire']['messages'] == prepared['messages']
                and res['input_token_ids'] == prepared['input_token_ids']
            )
            assert step['context'] == prepared['context'] == res['metadata']['context']
            assert (
                res['metadata']['task_id'] == r['task_id']
                and res['metadata']['step'] == i + 1
                and res['metadata']['condition'] == r['condition']
            )
            assert (
                res['metadata']['public_digest'] == digest(step['before'])
                and raw_step['format_valid']
                and raw_step['admissible']
            )
            same = step['before']['feedback'] == step['after']['feedback'] and sorted(
                step['before']['admissible_commands']
            ) == sorted(step['after']['admissible_commands'])
            streak = (streak + 1 if last == step['action'] else 1) if same else 0
            last = step['action']
            if i < len(r['trace']) - 1:
                assert streak < 4
        assert r['result']['format_valid_rate'] == r['result']['admissible_rate'] == 1
        if r['result']['termination'] == 'stagnation_terminated':
            assert streak == 4 and not r['result']['success']
        if r['result']['termination'] == 'step_budget':
            assert len(r['trace']) == 100 and not r['result']['success']
        records.append(r)
    calculated = summarize(records, tasks, initial['conditions'])
    assert calculated == summary['conditions']
    assert (
        len(ledgers['action-runtime'])
        == sum(len(r['trace']) for r in records)
        == summary['task_calls']
    )
    assert len(ledgers['canary-runtime']) == 4
    qualification = read(root / 'qualification/receipt.json')
    assert (
        qualification['passed']
        and qualification['cases']
        == qualification['passed_cases']
        == len(ledgers['qualification-runtime'])
        == 4
    )
    assert qualification['profile_digest'] == profile['digest']
    for row in ledgers['qualification-runtime'].values():
        receipt = json.loads(row['observation'])
        assert (
            row['state'] == 'completed'
            and row['cleanup'] == 'confirmed'
            and row['lease_released'] == 1
        )
        assert receipt['passed'] and receipt['release'] == releases[
            receipt['release']['request_id']
        ] == json.loads(row['cleanup_evidence'])
        assert receipt['attempt_id'] == row['attempt_id']
        usage.update(
            {k: receipt['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
    for label in ('plain', 'stream'):
        r = read(root / 'bootstrap' / (label + '-response.json'))
        assert r['release'] == releases[r['release']['request_id']] and r['token_counts_match']
        usage.update(
            {k: r['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
    total = summary['task_calls'] + 10
    assert (
        total
        == summary['total_generations']
        == len(releases)
        == len(reserved_generations(root))
        <= initial['generation_cap']
    )
    cleanup = read(root / 'cleanup.json')
    assert (
        cleanup['stopped'] and not cleanup.get('remaining') and not cleanup['client_close_errors']
    )
    result = {
        'verified': True,
        'conditions': calculated,
        'total_generations': total,
        'usage': dict(usage),
        'source_files': len(initial['source_hashes']),
        'scope': 'cold-start development results; no continual-learning or protected-test claim',
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.run, a.data, a.output), ensure_ascii=False, indent=2))
