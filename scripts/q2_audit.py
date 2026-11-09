"""离线消费逐动作原件；不进行任务reset或模型调用。"""

import argparse
import json
import sqlite3
import ast
from collections import Counter
from pathlib import Path
from serving_lab.lab import file_hash
from serving_lab.task_protocol import digest, validate_manifest
from serving_lab.public_tasks import action_context, verified_history
from serving_lab.execution import write_once
from compare_runtime_baseline import copied_database


def check_action_identity(reservation, receipt, raw, ledger):
    assert reservation['arrival_id'] == receipt['arrival_id'] == ledger['id']
    assert reservation['attempt_id'] == receipt['attempt_id'] == ledger['attempt_id']
    expected = 'chatcmpl-' + reservation['attempt_id']
    assert raw['id'] == expected
    assert receipt['release']['request_id'] in {expected, expected + '-0'}
    assert receipt['release']['worker_instance_id'] == reservation['worker']['worker_instance_id']
    assert receipt['release']['startup_generation'] == reservation['worker']['startup_generation']


def audit(root, output):
    root = Path(root).resolve()
    config = json.loads((root / 'input.json').read_text())
    repo = Path(__file__).resolve().parents[1]
    for n, h in config['source_hashes'].items():
        assert file_hash(root / 'source' / n) == h
    manifest = json.loads((repo / 'artifacts/q1-data-20261102/manifest.json').read_text())
    tasks = validate_manifest(manifest)
    assert file_hash(repo / 'artifacts/q1-data-20261102/manifest.json') == config['manifest_sha256']
    records = []
    calls = 0
    steps = 0
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    lengths = 0
    repeats = 0
    tokenizer = None
    if config['mode'] == 'model':
        from transformers import AutoTokenizer

        binding = json.loads((root / 'worker/binding.json').read_text())
        profile = json.loads((root / 'profile.json').read_text())
        released = {
            r['request_id']: r
            for line in Path(binding['events_path']).read_text().splitlines()
            if (r := json.loads(line)).get('event') == 'request_released'
        }
        tokenizer = AutoTokenizer.from_pretrained(
            binding['model_path'], local_files_only=True, trust_remote_code=False
        )
        with sqlite3.connect(root / 'action-runtime/service.sqlite') as db:
            db.row_factory = sqlite3.Row
            ledger = {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
        token = lambda ms: tokenizer.apply_chat_template(
            ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        budget = {
            p.name.split('-history-budget.json')[0]: json.loads(p.read_text())
            for p in root.glob('*-history-budget.json')
        }
    for path in sorted(root.glob('**/record.json')):
        record = json.loads(path.read_text())
        assert record['record_hash'] == digest(
            {k: v for k, v in record.items() if k != 'record_hash'}
        )
        task = tasks[record['task_id']]
        assert task['role'] != 'protected'
        env = json.loads((path.parent / 'environment.json').read_text())
        assert env['reference_enabled'] is False
        assert (
            env['initial'] == record['initial']
            and env['result'] == record['result']
            and len(env['trace']) == len(record['trace'])
        )
        seen = set()
        for i, (s, e) in enumerate(zip(record['trace'], env['trace']), 1):
            for key in ('step', 'before', 'action', 'after'):
                assert s[key] == e[key]
            for view in (s['before'], s['after']):
                assert set(view) == {'feedback', 'admissible_commands'}
            assert e['format_valid'] == (
                isinstance(s['action'], str)
                and bool(s['action'].strip())
                and len(s['action']) <= 256
                and '\n' not in s['action']
                and '\r' not in s['action']
            )
            assert e['admissible'] == (
                e['format_valid']
                and (
                    s['action'] in s['before']['admissible_commands']
                    or s['action'] in {'look', 'inventory'}
                )
            )
            repeats += s['action'] in seen
            seen.add(s['action'])
            steps += 1
            if record['origin'] == 'model_public':
                folder = path.parent / f'action-{i:03}'
                res = json.loads((folder / 'reservation.json').read_text())
                receipt = json.loads((folder / 'receipt.json').read_text())
                raw = json.loads((folder / 'response.raw').read_bytes())
                assert receipt == s['receipt'] and receipt['protocol_valid']
                check_action_identity(res, receipt, raw, ledger[res['arrival_id']])
                assert released[receipt['release']['request_id']] == receipt['release']
                entry = ledger[res['arrival_id']]
                wire = json.loads(
                    (
                        root / 'action-runtime/wire-requests' / (res['arrival_id'] + '.json')
                    ).read_text()
                )
                assert wire == {
                    'payload': res['wire'],
                    'profile_digest': res['profile_digest'],
                    'input_token_ids': res['input_token_ids'],
                }
                assert (
                    json.loads(entry['worker']) == res['worker']
                    and json.loads(entry['cleanup_evidence']) == receipt['release']
                )
                assert (
                    json.loads(entry['observation']) == receipt
                    and json.loads(entry['route']) == res['metadata']
                )
                assert (
                    res['profile_digest'] == profile['digest']
                    and res['worker']['startup_generation'] == binding['startup_generation']
                )
                assert (
                    res['metadata']['public_digest'] == digest(s['before'])
                    and res['metadata']['step'] == i
                )
                assert (
                    res['wire']['temperature'] == 0.7
                    and res['wire']['top_p'] == 0.8
                    and res['wire']['seed'] == 20261103
                )
                assert (
                    raw['model'] == binding['version_id']
                    and raw['choices'][0]['message']['content'].strip() == s['action']
                )
                assert raw['usage'] == receipt['usage']
                calls += 1
                for k in usage:
                    usage[k] += raw['usage'][k]
                lengths += receipt['finish_reason'] == 'length'
                history = None
                if record['condition'] != 'none':
                    history = budget[record['task_id']]['texts'][
                        0 if record['condition'] == 'public_related' else 1
                    ]
                ms, ids, meta = action_context(
                    record['initial']['feedback'],
                    s['before'],
                    record['trace'][: i - 1],
                    history,
                    token,
                    3000,
                )
                assert (
                    ms == res['wire']['messages']
                    and ids == res['input_token_ids']
                    and meta == s['context']
                )
                assert len(ids) == raw['usage']['prompt_tokens'] and len(ids) <= 3000
                assert (
                    receipt['release']['worker_instance_id'] == binding['worker_instance_id']
                    and receipt['release']['startup_generation'] == binding['startup_generation']
                )
        assert record['result']['steps'] == len(record['trace'])
        records.append(record)
    if config['mode'] == 'model':
        with sqlite3.connect(root / 'action-runtime/service.sqlite') as db:
            rows = db.execute('SELECT state,cleanup,lease_released FROM arrivals').fetchall()
            assert len(rows) == calls and all(r == ('completed', 'confirmed', 1) for r in rows)
        histories = json.loads((root / 'histories.json').read_text())
        for k, value in histories['sources'].items():
            record = next(r for r in records if r['task_id'] == k)
            assert value == verified_history(record, cutoff=7)
        assert calls <= 2400
    result = {
        'verified': True,
        'records': len(records),
        'steps': steps,
        'model_action_calls': calls,
        'usage': usage,
        'budget_truncations': lengths,
        'repeated_actions': repeats,
        'protected_executed': 0,
        'scope': 'raw protocol and artifact validation only, not learning effectiveness',
        'rows': [
            {'task_id': r['task_id'], 'condition': r['condition'], 'result': r['result']}
            for r in records
        ],
    }
    write_once(output, result)
    return result


def audit_rescue(root, output):
    """独立核原始身份、公开上下文、预定标签及停止门；不改旧判分。"""
    from serving_lab.actor_diagnostic import (
        paired_contexts,
        score_action,
        score_json_action,
        parse_json_action,
        parse_action_envelope,
        score_envelope,
        closed_loop_gate,
    )

    root = Path(root).resolve()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    initial = read(root / 'input.json')
    summary = read(root / 'summary.json')
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    cases = {c['id']: c for c in initial['panel']['cases']}
    repo = Path(__file__).resolve().parents[1]
    for c in cases.values():
        source = repo / c['source']['path']
        assert file_hash(source) == c['source']['sha256']
        original = read(source)
        step = c['source_step']
        assert c['view'] == (
            original['initial'] if not step else original['trace'][step - 1]['before']
        )
        assert c['prefix'] == [
            {'action': s['action'], 'after': s['after']}
            for s in original['trace'][: max(0, step - 1)]
        ]
        assert c['source']['role'] == 'support' and not c['eligible_for_training']

    def literal(name, path):
        for node in ast.parse(path.read_text(encoding='utf-8')).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                return ast.literal_eval(node.value)
        return None

    instructions = {
        'legacy': literal('TOOL_CONTRACT', root / 'source/src/serving_lab/task_protocol.py'),
        'goal_action': literal(
            'GOAL_ACTION_CONTRACT', root / 'source/src/serving_lab/actor_diagnostic.py'
        ),
        'tool_json': literal(
            'TOOL_JSON_CONTRACT', root / 'source/src/serving_lab/actor_diagnostic.py'
        ),
    }
    instructions['tool_envelope'] = instructions['tool_json']
    from transformers import AutoTokenizer

    binding = read(root / 'worker/binding.json')
    tokenizer = AutoTokenizer.from_pretrained(
        binding['model_path'], local_files_only=True, trust_remote_code=False
    )
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    profiles = read(root / 'profiles.json')
    repair = (root / 'repair-input.json').exists()
    adapter = (root / 'adapter-input.json').exists()
    prepared = read(root / 'prepared-contexts.json') if not adapter else {}
    released = {
        r['request_id']: r
        for line in Path(binding['events_path']).read_text().splitlines()
        if (r := json.loads(line)).get('event') == 'request_released'
    }
    ledgers = {}
    for path in root.glob('*runtime*/service.sqlite'):
        with copied_database(path) as db:
            db.row_factory = sqlite3.Row
            ledgers[path.parent.name] = {
                r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')
            }
    usage = Counter()
    seen = set()
    model_calls = 0

    def check(folder, runtime):
        nonlocal model_calls
        reservation = read(folder / 'reservation.json')
        receipt = read(folder / 'receipt.json')
        raw = read(folder / 'response.raw')
        entry = ledgers[runtime][reservation['arrival_id']]
        check_action_identity(reservation, receipt, raw, entry)
        assert (
            receipt['protocol_valid']
            and receipt['release'] == released[receipt['release']['request_id']]
        )
        assert (
            entry['state'] == 'completed'
            and entry['cleanup'] == 'confirmed'
            and entry['lease_released'] == 1
        )
        assert (
            json.loads(entry['worker']) == reservation['worker']
            and json.loads(entry['cleanup_evidence']) == receipt['release']
            and json.loads(entry['observation']) == receipt
        )
        wire = read(root / runtime / 'wire-requests' / (entry['id'] + '.json'))
        assert wire == {
            'payload': reservation['wire'],
            'profile_digest': reservation['profile_digest'],
            'input_token_ids': reservation['input_token_ids'],
        }
        assert tokenize(reservation['wire']['messages']) == reservation['input_token_ids']
        assert (
            raw['usage'] == receipt['usage']
            and raw['usage']['prompt_tokens'] == len(reservation['input_token_ids']) <= 3000
        )
        assert (
            raw['model'] == binding['version_id']
            and raw['choices'][0]['message']['content'] == receipt['content']
        )
        seed = str(reservation['wire']['seed'])
        profile = profiles[seed]
        assert digest(profile['document']) == profile['digest'] == reservation['profile_digest']
        assert (
            reservation['wire']['temperature'] == 0.7
            and reservation['wire']['top_p'] == 0.8
            and reservation['wire']['top_k'] == 20
        )
        assert receipt['attempt_id'] not in seen
        seen.add(receipt['attempt_id'])
        usage.update(
            {k: receipt['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
        model_calls += 1
        return reservation, receipt

    rows = read(root / 'probe-results.json') if not adapter else []
    conditions = [] if adapter else (['tool_json'] if repair else ['legacy', 'goal_action'])
    expected = {
        (sid, condition, seed)
        for sid in cases
        for condition in conditions
        for seed in initial['panel']['seeds']
    }
    assert (
        len(rows) == len(expected)
        and {(r['state_id'], r['condition'], r['seed']) for r in rows} == expected
    )
    for sid, c in cases.items():
        if adapter:
            continue
        if repair:
            ms, ids, meta = action_context(
                c['initial']['feedback'],
                c['view'],
                c['prefix'],
                None,
                tokenize,
                3000,
                instruction=instructions['tool_json'],
            )
            assert prepared[sid] == {
                'tool_json': {'messages': ms, 'input_token_ids': ids, 'context': meta}
            }
        else:
            expected_context = paired_contexts(c, tokenize, 3000)
            assert (
                prepared[sid] == expected_context
                and prepared[sid]['legacy']['messages'][0]['content'] == instructions['legacy']
                and prepared[sid]['goal_action']['messages'][0]['content']
                == instructions['goal_action']
            )
    for row in rows:
        case = cases[row['state_id']]
        folder = root / 'probes' / row['state_id'] / str(row['seed']) / row['condition']
        reservation, receipt = check(folder, 'probe-runtime')
        assert (
            receipt == row['receipt']
            and reservation['metadata']['source_view_hash']
            == digest(case['view'])
            == row['source_view_hash']
        )
        assert (
            reservation['wire']['messages']
            == prepared[row['state_id']][row['condition']]['messages']
        )
        assert row['score'] == (score_json_action if repair else score_action)(
            receipt['content'], case
        )
    if adapter:
        info = read(root / 'adapter-input.json')
        source = Path(info['source'])
        assert file_hash(source / 'probe-results.json') == info['source_probe_results_sha256']
        old = read(source / 'probe-results.json')
        replay = read(root / 'adapter-replay.json')
        assert len(old) == len(replay) == 36
        for r, derived in zip(old, replay):
            assert (
                derived['source_attempt_id'] == r['receipt']['attempt_id']
                and derived['state_id'] == r['state_id']
                and derived['seed'] == r['seed']
            )
            assert derived['score'] == score_envelope(r['receipt']['content'], cases[r['state_id']])
        gate = closed_loop_gate(replay)
    else:
        gate = closed_loop_gate(
            [{**r, 'condition': 'goal_action'} for r in rows] if repair else rows
        )
    assert gate == read(root / 'closed-loop-gate.json')['passed'] == summary['closed_loop_gate']
    closed = []
    for path in root.glob('closed/**/record.json'):
        r = read(path)
        env = read(path.parent / 'environment.json')
        exit = read(path.parent / 'exit.json')
        assert (
            r['record_hash'] == digest({k: v for k, v in r.items() if k != 'record_hash'})
            and r['result'] == env['result']
            and r['initial'] == env['initial']
        )
        assert (
            r['purpose'] == 'actor_diagnostic'
            and r['eligible_for_training'] is False
            and r['eligible_for_primary_history'] is False
        )
        assert exit['ExitCode'] == 0 and not exit['Running'] and not env['reference_enabled']
        assert len(r['trace']) == len(env['trace']) == r['result']['steps'] <= 40
        streak = 0
        last = None
        for index, (step, e) in enumerate(zip(r['trace'], env['trace'])):
            assert all(step[k] == e[k] for k in ('step', 'before', 'after', 'action'))
            reservation, receipt = check(path.parent / f'action-{index + 1:03}', 'closed-runtime')
            assert step['receipt'] == receipt and reservation['wire']['seed'] == 20261103
            action = (
                parse_action_envelope(receipt['content'])
                if adapter
                else (
                    parse_json_action(receipt['content']) if repair else receipt['content'].strip()
                )
            )
            assert action == step['action']
            ms, ids, meta = action_context(
                r['initial']['feedback'],
                step['before'],
                r['trace'][:index],
                None,
                tokenize,
                3000,
                instruction=instructions[r['condition']],
            )
            assert (
                ms == reservation['wire']['messages']
                and ids == reservation['input_token_ids']
                and meta == step['context']
            )
            same = step['before']['feedback'] == step['after']['feedback'] and sorted(
                step['before']['admissible_commands']
            ) == sorted(step['after']['admissible_commands'])
            streak = (streak + 1 if action == last else 1) if same else 0
            last = action
            if index < len(r['trace']) - 1:
                assert streak < 4
        if r['result']['termination'] == 'stagnation_terminated':
            assert streak == 4 and not r['result']['success']
        closed.append({'task_id': r['task_id'], 'condition': r['condition'], 'result': r['result']})
    assert len(closed) == (2 if repair else 4) if gate else len(closed) == 0
    qualification_calls = 0
    for seed in [20261103] if adapter else initial['panel']['seeds']:
        receipt = read(root / f'qualification-{seed}/receipt.json')
        assert (
            receipt['passed']
            and receipt['cases'] == receipt['passed_cases'] == 4
            and receipt['profile_digest'] == profiles[str(seed)]['digest']
        )
        rows_q = ledgers[f'qualification-runtime-{seed}']
        assert len(rows_q) == 4
        for r in rows_q.values():
            assert (
                r['state'] == 'completed'
                and r['cleanup'] == 'confirmed'
                and r['lease_released'] == 1
            )
            evidence = json.loads(r['cleanup_evidence'])
            assert evidence == released[evidence['request_id']]
        qualification_calls += 4
    for name in ('plain', 'stream'):
        response = read(root / 'bootstrap' / (name + '-response.json'))
        assert response['release'] == released[response['release']['request_id']]
    total = model_calls + qualification_calls + 2
    assert total == summary['total_generation_calls'] == len(released)
    assert summary['probe_calls'] == len(rows) and summary['closed_loop_calls'] == sum(
        x['result']['steps'] for x in closed
    )
    actual_conditions = {
        condition: {
            'format_valid': sum(
                r['score']['format_valid'] for r in rows if r['condition'] == condition
            ),
            'admissible': sum(
                r['score']['admissible'] for r in rows if r['condition'] == condition
            ),
            'key_action_hits': sum(
                r['score']['acceptable_key_action'] is True
                for r in rows
                if r['condition'] == condition
            ),
            'key_action_total': 18,
            'total': 36,
        }
        for condition in conditions
    }
    if not adapter:
        assert actual_conditions == summary['conditions']
    else:
        actual_conditions = {
            'offline_envelope_replay': {
                'admissible': sum(r['score']['admissible'] for r in replay),
                'key_action_hits': sum(r['score']['acceptable_key_action'] is True for r in replay),
                'total': 36,
                'key_action_total': 18,
            }
        }
        assert actual_conditions['offline_envelope_replay'] == summary['replay_conditions']
    result = {
        'verified': True,
        'probe_calls': len(rows),
        'conditions': actual_conditions,
        'closed_loop_gate': gate,
        'closed': closed,
        'total_generations': total,
        'probe_and_closed_usage': dict(usage),
        'source_files': len(initial['source_hashes']),
        'training_runs': 0,
        'primary_history_sources': 0,
        'scope': 'actor development diagnosis, not learning/transfer evidence',
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--rescue', action='store_true')
    a = p.parse_args()
    print(
        json.dumps(
            (audit_rescue if a.rescue else audit)(a.run, a.output), ensure_ascii=False, indent=2
        )
    )
