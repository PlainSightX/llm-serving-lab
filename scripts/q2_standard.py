"""固定公开命令选择基线；运行与离线复核共用入口，不提供训练出口。"""

import argparse
import asyncio
import importlib.util
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.profile import ServingProfile, application_history
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.standard_actor import StandardActor, public_working_state
from serving_lab.task_protocol import validate_manifest
from serving_lab.transport import qualify_transport
from compare_runtime_baseline import copied_database
from probe_serving import probe
from q2_audit import check_action_identity
from q2_rescue import DATA, IMAGE, close_models, reserved_generations, validate_cpu_baseline
from q2_run import docker, episode
from start_engine import start

REPO = Path(__file__).resolve().parents[1]
SUPPORTS = [f'q1_support_{i:02}' for i in range(1, 7)]
POLICY = {'name': 'q2_public_choice_v1', 'seed': 20261103}
CANARIES = [
    ['inventory', 'look'],
    ['look'],
    ['take apple 1 from table 1', 'take apple 2 from table 1'],
    ['go to cabinet 1', 'open cabinet 1'],
]


def validate_profile_binding(initial, profile, binding):
    """自洽不等于符合冻结条件；独立核实际参数与完整解码值。"""
    document = profile['document']
    expected = {
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'min_p': 0.0,
        'seed': 20261103,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
        'repetition_penalty': 1.0,
        'stop': [],
        'stop_token_ids': [],
        'ignore_eos': False,
        'skip_special_tokens': True,
        'include_stop_str_in_output': False,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    if (
        initial['policy'] != POLICY
        or digest(document) != profile['digest']
        or document['name'] != POLICY['name']
        or document['decoding'] != expected
        or document['artifact_digest'] != binding['artifact_digest']
        or document['max_tokens'] != 64
        or document['thinking'] is not False
        or document['context_tokens'] != 4096
        or document['engine_version'] != '0.11.2'
        or any(
            h != initial['source_hashes']['src/serving_lab/' + n]
            for n, h in document['implementation'].items()
        )
    ):
        raise ValueError('实际profile不符合冻结actor条件或参数身份')


def validate_episode_binding(record, env, launch, task, folder, image):
    """环境原件、控制侧标签、目录与冻结身份必须指向同一任务。"""
    if (
        record['task_id'] != env['task_id']
        or record['task_id'] != task['task_id']
        or record['task_id'] != folder
        or record['role'] != task['role']
        or record['sequence'] != task['sequence']
        or task['role'] != 'support'
        or launch['image'] != image
        or image != IMAGE
        or launch['command'][-2:] != [image, '/runner.py']
    ):
        raise ValueError('实际环境任务或启动镜像未绑定冻结身份')


def summarize_records(records):
    """完整六题是报告单位；失败或重复记录不能被成功子集掩盖。"""
    if len(records) != 6 or {r['task_id'] for r in records} != set(SUPPORTS):
        raise ValueError('需要六个不同support的完整结果，包括失败')
    rows = []
    for record in sorted(records, key=lambda r: r['task_id']):
        trace = record['trace']
        goal = re.search(r'Your task is to:\s*([^\n]+)', record['initial']['feedback'])[1]
        match = re.search(
            r'^(?:put|examine|clean|heat|cool|find)\s+(?:a|an|the|two|some)\s+(?:(?:clean|hot|cool|cold)\s+)?([a-z]+)',
            goal.lower(),
        )
        obj = match[1] if match else None
        takes = [
            s['action']
            for s in trace
            if isinstance(s['action'], str) and s['action'].startswith('take ')
        ]
        state = public_working_state(trace)
        counts = Counter(s['action'] for s in trace)
        rows.append(
            {
                'task_id': record['task_id'],
                'goal': goal,
                'result': record['result'],
                'goal_object_parsed_for_report_only': obj,
                'goal_object_take_attempts': sum(a.startswith('take ' + obj + ' ') for a in takes)
                if obj
                else None,
                'other_object_take_attempts': sum(
                    not a.startswith('take ' + obj + ' ') for a in takes
                )
                if obj
                else None,
                'visited_places': [x['place'] for x in state['visited_places']],
                'distinct_actions': len(counts),
                'repeated_action_count': sum(n - 1 for n in counts.values()),
                'action_counts': dict(counts),
                'max_input_tokens': max(
                    (s.get('context', {}).get('input_tokens', 0) for s in trace), default=0
                ),
                'steps_with_clipping': sum(
                    bool(s.get('context', {}).get('dropped_history_steps'))
                    or any(s.get('context', {}).get('dropped_state_entries', {}).values())
                    for s in trace
                ),
            }
        )
    successes = sum(r['result']['success'] for r in records)
    return {
        'episodes': 6,
        'successes': successes,
        'all_families_ready': successes == 6,
        'task_action_calls': sum(len(r['trace']) for r in records),
        'tasks': rows,
    }


async def run(output, cpu_run):
    baseline = validate_cpu_baseline(Path(cpu_run).resolve())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    tasks = validate_manifest(manifest)
    assert set(SUPPORTS) == {k for k, v in tasks.items() if v['role'] == 'support'}
    assert (
        file_hash(DATA / 'manifest.json')
        == json.loads((DATA / 'manifest-anchor.json').read_text())['sha256']
    )
    assert json.loads(docker('image', 'inspect', IMAGE))[0]['Id'] == IMAGE
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = [
        'scripts/' + n
        for n in (
            'q2_standard.py',
            'q2_run.py',
            'q2_rescue.py',
            'q2_audit.py',
            'q2_environment.py',
            'start_engine.py',
            'probe_serving.py',
            'compare_runtime_baseline.py',
        )
    ]
    names += ['src/serving_lab/' + p.name for p in (REPO / 'src/serving_lab').glob('*.py')]
    names += ['configs/serving/r3-large.json', 'fixtures/serving/r2-transport-v2.json']
    hashes = {n: file_hash(REPO / n) for n in names}
    for name in names:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'purpose': 'actor_diagnostic',
            'policy': POLICY,
            'supports': SUPPORTS,
            'canaries': CANARIES,
            'max_steps': 100,
            'stagnation_limit': 4,
            'model_generation_cap': 610,
            'max_input_tokens': 3500,
            'max_output_tokens': 64,
            'manifest_sha256': file_hash(DATA / 'manifest.json'),
            'image': IMAGE,
            'source_hashes': hashes,
            'cpu_source': str(Path(cpu_run).resolve()),
            'cpu_verification': baseline,
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
            'protected_executed': 0,
        },
    )
    binding_path = out / 'worker/binding.json'
    engines, records = {}, []
    try:
        start(REPO / 'configs/serving/r3-large.json', out / 'worker')
        probe(binding_path, out / 'bootstrap')
        binding = EngineBinding.load(binding_path)
        engine = HttpEngine(binding, profile=True, profile_policy=POLICY)
        engines['actor'] = engine
        write_once(
            out / 'profile.json',
            {'document': engine.profile.document, 'digest': engine.profile.digest},
        )
        application = application_history(
            REPO / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json', binding
        )
        with RuntimeStore(out / 'qualification-runtime') as store:
            q = await qualify_transport(
                engine,
                json.loads((REPO / 'fixtures/serving/r2-transport-v2.json').read_text()),
                out / 'qualification',
                store=store,
                application_result=application,
            )
        if not q['passed']:
            raise RuntimeError('transport资格失败，停止')
        tokenize = lambda ms: engine.tokenizer.apply_chat_template(
            ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        with RuntimeStore(out / 'canary-runtime') as store:
            for i, choices in enumerate(CANARIES, 1):
                messages = [
                    {
                        'role': 'user',
                        'content': 'Select exactly one available command: ' + json.dumps(choices),
                    }
                ]
                receipt = await generate_action(
                    engine,
                    store,
                    out / 'canaries' / str(i),
                    messages,
                    tokenize(messages),
                    {'purpose': 'structured_interface_canary', 'case': i},
                    choices=choices,
                )
                StandardActor().action(receipt, choices)
        print(json.dumps({'structured_canaries_passed': 4}), flush=True)
        with RuntimeStore(out / 'action-runtime') as store:
            for task_id in SUPPORTS:
                # 每题一次、最多100步；早期业务失败不跳过其他family。
                record = await episode(
                    tasks[task_id],
                    DATA,
                    out / 'support' / task_id,
                    IMAGE,
                    engine,
                    store,
                    condition='standard_public_choice_v1',
                    max_steps=100,
                    stagnation_limit=4,
                    purpose='actor_diagnostic',
                    actor=StandardActor(),
                )
                records.append(record)
        assert all(file_hash(REPO / n) == h for n, h in hashes.items())
        summary = summarize_records(records)
        total = summary['task_action_calls'] + 10
        assert total == len(reserved_generations(out)) <= 610
        summary.update(
            total_generation_calls=total,
            bootstrap_calls=2,
            transport_calls=4,
            structured_canary_calls=4,
            training_runs=0,
            primary_history_sources=0,
            purpose='actor_diagnostic',
            scope='six seen support tasks, one seed; not learning effect or generalization',
        )
    except BaseException as exc:
        write_once(
            out / 'failure.json',
            {
                'error': repr(exc),
                'completed_tasks': [r['task_id'] for r in records],
                'reserved_generations': len(reserved_generations(out)),
            },
        )
        raise
    finally:
        await close_models(engines, binding_path, out)
    write_once(out / 'summary.json', summary)
    return summary


def audit(root, output):
    """从原始wire、环境与引擎释放重算；不reset、不调用模型、不原地读写SQLite。"""
    root = Path(root).resolve()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    initial, summary = read(root / 'input.json'), read(root / 'summary.json')
    tasks = validate_manifest(read(DATA / 'manifest.json'))
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    assert file_hash(DATA / 'manifest.json') == initial['manifest_sha256']
    assert (
        initial['supports'] == SUPPORTS
        and initial['policy'] == POLICY
        and initial['canaries'] == CANARIES
    )
    assert (
        initial['max_steps'] == 100
        and initial['stagnation_limit'] == 4
        and initial['model_generation_cap'] == 610
    )
    # 使用该次快照中的actor重建输入，避免后来提示改动污染旧运行解释。
    spec = importlib.util.spec_from_file_location(
        'serving_lab._recorded_actor', root / 'source/src/serving_lab/standard_actor.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = module.StandardActor()
    from transformers import AutoTokenizer

    binding, profile = read(root / 'worker/binding.json'), read(root / 'profile.json')
    validate_profile_binding(initial, profile, binding)
    tokenizer = AutoTokenizer.from_pretrained(
        binding['model_path'], local_files_only=True, trust_remote_code=False
    )
    tokenize = lambda ms: tokenizer.apply_chat_template(
        ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    assert digest(profile['document']) == profile['digest']
    serving_profile = ServingProfile(profile['document'], binding['served_model'])
    released = {
        r['request_id']: r
        for line in Path(binding['events_path']).read_text().splitlines()
        if (r := json.loads(line)).get('event') == 'request_released'
    }
    ledgers = {}
    for path in root.glob('*runtime/service.sqlite'):
        with copied_database(path) as db:
            db.row_factory = sqlite3.Row
            ledgers[path.parent.name] = {
                r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')
            }
    usage, seen = Counter(), set()

    def check(folder, runtime, choices):
        res, rec, raw = (
            read(folder / 'reservation.json'),
            read(folder / 'receipt.json'),
            read(folder / 'response.raw'),
        )
        entry = ledgers[runtime][res['arrival_id']]
        check_action_identity(res, rec, raw, entry)
        assert (
            entry['state'] == 'completed'
            and entry['cleanup'] == 'confirmed'
            and entry['lease_released'] == 1
        )
        assert (
            json.loads(entry['worker']) == res['worker']
            and json.loads(entry['cleanup_evidence']) == rec['release']
        )
        assert (
            json.loads(entry['observation']) == rec
            and json.loads(entry['route']) == res['metadata']
        )
        assert rec['protocol_valid'] and rec['release'] == released[rec['release']['request_id']]
        assert res['worker']['startup_generation'] == binding['startup_generation']
        assert (
            res['profile_digest'] == profile['digest']
            and tokenize(res['wire']['messages']) == res['input_token_ids']
        )
        body = {
            'model': binding['version_id'],
            'messages': res['wire']['messages'],
            'max_tokens': 64,
            'stream': False,
            'action_choices': choices,
        }
        assert serving_profile.payload(body) == res['wire']
        assert read(root / runtime / 'wire-requests' / (entry['id'] + '.json')) == {
            'payload': res['wire'],
            'profile_digest': profile['digest'],
            'input_token_ids': res['input_token_ids'],
        }
        assert (
            raw['model'] == binding['version_id']
            and raw['choices'][0]['message']['content'] == rec['content']
        )
        assert raw['choices'][0]['finish_reason'] == rec['finish_reason'] == 'stop'
        assert (
            raw['usage'] == rec['usage']
            and raw['usage']['prompt_tokens'] == len(res['input_token_ids']) <= 3500
        )
        assert (
            rec['usage']['total_tokens']
            == rec['usage']['prompt_tokens'] + rec['usage']['completion_tokens']
        )
        assert (
            rec['release']['status'] == 'FINISHED_STOPPED'
            and 0 < rec['usage']['completion_tokens'] <= 64
        )
        actor.action(rec, choices)
        assert rec['attempt_id'] not in seen
        seen.add(rec['attempt_id'])
        usage.update(
            {k: rec['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
        return res, rec

    for i, choices in enumerate(initial['canaries'], 1):
        res, _ = check(root / 'canaries' / str(i), 'canary-runtime', choices)
        assert res['wire']['messages'] == [
            {
                'role': 'user',
                'content': 'Select exactly one available command: ' + json.dumps(choices),
            }
        ]
    records = []
    for path in sorted(root.glob('support/*/record.json')):
        record, env, exit_state = (
            read(path),
            read(path.parent / 'environment.json'),
            read(path.parent / 'exit.json'),
        )
        validate_episode_binding(
            record,
            env,
            read(path.parent / 'launch.json'),
            tasks[record['task_id']],
            path.parent.name,
            initial['image'],
        )
        assert record['record_hash'] == digest(
            {k: v for k, v in record.items() if k != 'record_hash'}
        )
        assert record['task_id'] in SUPPORTS and record['role'] == 'support'
        assert (
            record['purpose'] == 'actor_diagnostic'
            and record['eligible_for_training'] is False
            and record['eligible_for_primary_history'] is False
        )
        assert (
            not env['reference_enabled']
            and exit_state['ExitCode'] == 0
            and not exit_state['Running']
        )
        assert record['initial'] == env['initial'] and record['result'] == env['result']
        assert len(record['trace']) == len(env['trace']) == record['result']['steps'] <= 100
        assert record['result']['max_steps'] == 100 and record['stagnation_limit'] == 4
        streak, last = 0, None
        for index, (step, raw_step) in enumerate(zip(record['trace'], env['trace'])):
            assert all(step[k] == raw_step[k] for k in ('step', 'before', 'action', 'after'))
            assert set(step['before']) == set(step['after']) == {'feedback', 'admissible_commands'}
            prepared = actor.prepare(
                record['initial'], step['before'], record['trace'][:index], tokenize
            )
            res, rec = check(
                path.parent / f'action-{index + 1:03}', 'action-runtime', prepared['choices']
            )
            assert step['receipt'] == rec and step['action'] == rec['content']
            assert (
                prepared['messages'] == res['wire']['messages']
                and prepared['input_token_ids'] == res['input_token_ids']
            )
            assert prepared['context'] == step['context'] == res['metadata']['context']
            assert (
                res['metadata']['task_id'] == record['task_id']
                and res['metadata']['step'] == index + 1
            )
            assert res['metadata']['public_digest'] == digest(step['before'])
            assert raw_step['format_valid'] and raw_step['admissible']
            same = step['before']['feedback'] == step['after']['feedback'] and sorted(
                step['before']['admissible_commands']
            ) == sorted(step['after']['admissible_commands'])
            streak = (streak + 1 if last == step['action'] else 1) if same else 0
            last = step['action']
            if index < len(record['trace']) - 1:
                assert streak < 4
        if record['result']['termination'] == 'stagnation_terminated':
            assert streak == 4 and not record['result']['success']
        assert record['result']['format_valid_rate'] == record['result']['admissible_rate'] == 1
        records.append(record)
    calculated = summarize_records(records)
    assert all(summary[k] == v for k, v in calculated.items())
    assert (
        len(ledgers['action-runtime']) == calculated['task_action_calls']
        and len(ledgers['canary-runtime']) == 4
    )
    q = read(root / 'qualification/receipt.json')
    assert (
        q['passed']
        and q['cases'] == q['passed_cases'] == len(ledgers['qualification-runtime']) == 4
    )
    assert q['profile_digest'] == profile['digest']
    for entry in ledgers['qualification-runtime'].values():
        assert (
            entry['state'] == 'completed'
            and entry['cleanup'] == 'confirmed'
            and entry['lease_released'] == 1
        )
        obs = json.loads(entry['observation'])
        assert (
            obs['release']
            == released[obs['release']['request_id']]
            == json.loads(entry['cleanup_evidence'])
        )
        assert obs['attempt_id'] == entry['attempt_id'] and obs['passed']
        assert obs['wire_payload'] == serving_profile.payload(
            read(root / 'qualification-runtime/requests' / (entry['id'] + '.json'))
        )
        usage.update(
            {k: obs['usage'][k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        )
    for label in ('plain', 'stream'):
        response = read(root / 'bootstrap' / (label + '-response.json'))
        assert (
            response['release'] == released[response['release']['request_id']]
            and response['token_counts_match']
        )
        usage.update(
            {
                k: response['usage'][k]
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
            }
        )
    total = calculated['task_action_calls'] + 10
    assert (
        total
        == summary['total_generation_calls']
        == len(reserved_generations(root))
        == len(released)
        <= 610
    )
    cleanup = read(root / 'cleanup.json')
    assert (
        cleanup['stopped'] and not cleanup.get('remaining') and not cleanup['client_close_errors']
    )
    result = {
        'verified': True,
        **calculated,
        'total_generations': total,
        'all_generation_usage': dict(usage),
        'source_files': len(initial['source_hashes']),
        'training_runs': 0,
        'primary_history_sources': 0,
        'scope': 'raw protocol and seen-task diagnostic; no learning or transfer claim',
    }
    write_once(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-run')
    parser.add_argument('--audit')
    args = parser.parse_args()
    if args.audit:
        result = audit(args.audit, args.output)
    else:
        if not args.cpu_run:
            parser.error('运行必须给出已验证CPU来源')
        result = asyncio.run(run(args.output, args.cpu_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))
