"""先CPU canary，再冻结程序基线和配对actor探针；全部为开发诊断。"""

import argparse
import asyncio
import json
from pathlib import Path
from serving_lab.actor_diagnostic import (
    INSTRUCTIONS,
    paired_contexts,
    score_action,
    closed_loop_gate,
    TOOL_JSON_CONTRACT,
    parse_json_action,
    score_json_action,
    parse_action_envelope,
    score_envelope,
)
from serving_lab.public_tasks import action_context
from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.execution import write_once
from serving_lab.lab import file_hash, digest
from serving_lab.profile import application_history
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.task_protocol import validate_manifest
from serving_lab.transport import qualify_transport
from q2_run import episode, docker
from start_engine import start, stop
from probe_serving import probe
from compare_runtime_baseline import copied_database

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:d3938e3d3b2d4d7a2aa15fac50c9a42a6a65f0603ba025339f7e2c30a1118ae4'
PANEL = REPO / 'fixtures/tasks/q2-rescue-panel.json'
DATA = REPO / 'artifacts/q1-data-20261102'


def validate_cpu_baseline(root):
    root = Path(root)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    if not (root / 'input.json').exists():
        raise ValueError('CPU原始输入/回执缺失')
    initial = read(root / 'input.json')
    summary = read(root / 'summary.json')
    manifest = read(DATA / 'manifest.json')
    validate_manifest(manifest)
    assert initial['manifest_sha256'] == file_hash(DATA / 'manifest.json')
    for name, h in initial['source_hashes'].items():
        assert file_hash(root / 'source' / name) == h
    for name in (
        'src/serving_lab/public_tasks.py',
        'src/serving_lab/task_protocol.py',
        'scripts/q2_environment.py',
    ):
        assert file_hash(REPO / name) == initial['source_hashes'][name], (
            'CPU actor/environment semantics changed',
            name,
        )
    rows = []
    for path in root.glob('**/record.json'):
        r = read(path)
        e = read(path.parent / 'environment.json')
        exit = read(path.parent / 'exit.json')
        assert r['record_hash'] == digest({k: v for k, v in r.items() if k != 'record_hash'})
        assert exit['ExitCode'] == 0 and not exit['Running'] and e['reference_enabled'] is False
        assert (
            r['initial'] == e['initial']
            and r['result'] == e['result']
            and len(r['trace']) == len(e['trace'])
        )
        assert (
            r['origin'] == 'public_policy'
            and r['purpose'] == 'actor_diagnostic'
            and r['eligible_for_training'] is False
        )
        assert r['result']['success'] and r['result']['steps'] == len(r['trace'])
        for a, b in zip(r['trace'], e['trace']):
            assert all(a[k] == b[k] for k in ('step', 'before', 'action', 'after'))
            assert set(a['before']) == set(a['after']) == {'feedback', 'admissible_commands'}
        rows.append((path.parent.parent.name, r))
    expected = {t['task_id'] for t in manifest['tasks'] if t['role'] != 'protected'}
    assert len(rows) == 18 and {r['task_id'] for phase, r in rows if phase == 'formal'} == expected
    support = {t['task_id'] for t in manifest['tasks'] if t['role'] == 'support'}
    assert (
        len([r for phase, r in rows if phase == 'canary']) == 6
        and {r['task_id'] for phase, r in rows if phase == 'canary'} == support
    )
    assert summary['formal_success'] == summary['formal_episodes'] == 12 and summary[
        'cpu_actions'
    ] == sum(r['result']['steps'] for _, r in rows)
    return {
        'raw_verified': True,
        'records': len(rows),
        'steps': summary['cpu_actions'],
        'source_deltas': {
            n: {'recorded': h, 'current': file_hash(REPO / n)}
            for n, h in initial['source_hashes'].items()
            if file_hash(REPO / n) != h
        },
    }


def reserved_generations(root):
    """连同中断后尚无receipt的预约一起计费；不因换目录重置修复额度。"""
    root = Path(root)
    ids = set()
    for dbpath in root.glob('*runtime*/service.sqlite'):
        with copied_database(dbpath) as db:
            ids.update(
                'attempt:' + r[0]
                for r in db.execute('SELECT attempt_id FROM arrivals WHERE attempt_id IS NOT NULL')
            )
    for path in root.glob('**/reservation.json'):
        ids.add('attempt:' + json.loads(path.read_text())['attempt_id'])
    for path in (root / 'bootstrap').glob('*-request.json'):
        ids.add('bootstrap:' + json.loads(path.read_text())['request_id'])
    return ids


def validate_repair_source(source, panel, reserve_max=130):
    source = Path(source).resolve()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    initial = read(source / 'input.json')
    old = read(source / 'summary.json')
    rows = read(source / 'probe-results.json')
    assert initial['panel_sha256'] == file_hash(PANEL) and initial['panel'] == panel
    assert (
        old['total_generation_calls'] == 86 and old['closed_loop_gate'] is False and len(rows) == 72
    )
    expected = {
        (c['id'], condition, seed)
        for c in panel['cases']
        for condition in ('legacy', 'goal_action')
        for seed in panel['seeds']
    }
    assert {(r['state_id'], r['condition'], r['seed']) for r in rows} == expected
    for name, h in initial['source_hashes'].items():
        assert file_hash(source / 'source' / name) == h
    roots = [source]
    for path in source.parent.glob('*/repair-input.json'):
        if Path(read(path)['source']).resolve() == source:
            roots.append(path.parent)
    ids = set()
    for root in roots:
        ids.update(reserved_generations(root))
    assert len(reserved_generations(source)) == 86
    if len(ids) + reserve_max > panel['model_generation_cap']:
        raise ValueError('剩余包预算不足完整预约修复；先复盘，不重置额度')
    return len(ids), read(source / 'profiles.json'), [str(r) for r in roots]


async def close_models(engines, binding_path, out):
    errors = []
    cleanup = {'stopped': True, 'no_worker_started': True}
    for e in engines.values():
        try:
            await e.close()
        except Exception as exc:
            errors.append({'client_close': repr(exc)})
    if binding_path.exists():
        try:
            cleanup = stop(binding_path)
        except Exception as exc:
            errors.append({'worker_stop': repr(exc)})
            cleanup = {'stopped': False}
    write_once(out / 'cleanup.json', {**cleanup, 'client_close_errors': errors})
    if errors or not cleanup.get('stopped') or cleanup.get('remaining'):
        raise RuntimeError('owned cleanup incomplete')


def setup(output, stage):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    panel = json.loads(PANEL.read_text(encoding='utf-8'))
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest(manifest)
    assert (
        file_hash(DATA / 'manifest.json')
        == json.loads((DATA / 'manifest-anchor.json').read_text())['sha256']
    )
    assert json.loads(docker('image', 'inspect', IMAGE))[0]['Id'] == IMAGE
    for case in panel['cases']:
        assert (
            case['source']['role'] == 'support'
            and case['source']['origin'] == 'public_policy_diagnostic'
        )
        assert file_hash(REPO / case['source']['path']) == case['source']['sha256']
    names = [
        'scripts/' + n
        for n in (
            'q2_rescue.py',
            'q2_audit.py',
            'q2_run.py',
            'q2_environment.py',
            'start_engine.py',
            'probe_serving.py',
        )
    ]
    names += [
        'src/serving_lab/' + n + '.py'
        for n in (
            '__init__',
            'actor_diagnostic',
            'public_tasks',
            'task_protocol',
            'research_action',
            'profile',
            'engine',
            'lab',
            'runtime',
            'execution',
            'transport',
            'qualification',
        )
    ]
    hashes = {n: file_hash(REPO / n) for n in names}
    for name in names:
        p = out / 'source' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((REPO / name).read_bytes())
    write_once(
        out / 'input.json',
        {
            'stage': stage,
            'purpose': 'actor_diagnostic',
            'panel_sha256': file_hash(PANEL),
            'panel': panel,
            'manifest_sha256': file_hash(DATA / 'manifest.json'),
            'image': IMAGE,
            'source_hashes': hashes,
            'eligible_for_training': False,
            'eligible_for_primary_history': False,
            'training_runs': 0,
            'protected_executed': 0,
        },
    )
    return out, panel, manifest, hashes


async def cpu(output, dev_remaining=600, formal_remaining=1200):
    out, panel, manifest, hashes = setup(output, 'public_policy')
    tasks = {t['task_id']: t for t in manifest['tasks']}
    records = []
    for i in (1, 6, 3, 5, 4, 2):
        remaining = dev_remaining - sum(r['result']['steps'] for r in records)
        if remaining < 100:
            raise ValueError('本包剩余开发预算不足以预约下一个100步canary')
        task = tasks[f'q1_support_{i:02}']
        r = await episode(
            task,
            DATA,
            out / 'canary' / task['task_id'],
            IMAGE,
            stagnation_limit=4,
            purpose='actor_diagnostic',
        )
        records.append(r)
        if not r['result']['success']:
            write_once(
                out / 'summary.json',
                {
                    'canary_passed': False,
                    'formal_executed': False,
                    'cpu_actions': sum(x['result']['steps'] for x in records),
                    'failed_task': task['task_id'],
                    'model_calls': 0,
                },
            )
            return {'canary_passed': False, 'output': str(out)}
    formal = []
    if formal_remaining < 1200:
        raise ValueError('完整12题需要独立1200步预约上限')
    for task in manifest['tasks']:
        if task['role'] == 'protected':
            continue
        formal.append(
            await episode(
                task,
                DATA,
                out / 'formal' / task['task_id'],
                IMAGE,
                stagnation_limit=4,
                purpose='actor_diagnostic',
            )
        )
    assert all(file_hash(REPO / n) == h for n, h in hashes.items())
    result = {
        'canary_passed': True,
        'formal_executed': True,
        'formal_success': sum(r['result']['success'] for r in formal),
        'formal_episodes': len(formal),
        'canary_actions': sum(r['result']['steps'] for r in records),
        'formal_actions': sum(r['result']['steps'] for r in formal),
        'cpu_actions': sum(r['result']['steps'] for r in records + formal),
        'model_calls': 0,
        'purpose': 'public-policy diagnostic; no teacher/history export',
    }
    write_once(out / 'summary.json', result)
    return result


async def model(output, cpu_run, repair_from=None):
    cpu_run = Path(cpu_run).resolve()
    baseline = json.loads((cpu_run / 'summary.json').read_text())
    if not (
        baseline.get('canary_passed')
        and baseline.get('formal_success') == baseline.get('formal_episodes') == 12
    ):
        raise ValueError('程序基线未通过，停止模型批次')
    cpu_verification = validate_cpu_baseline(cpu_run)
    out, panel, manifest, hashes = setup(output, 'paired_actor')
    engines = {}
    binding_path = out / 'worker/binding.json'
    used = 0
    result = None
    write_once(out / 'cpu-verification.json', cpu_verification)
    prior = 0
    if repair_from:
        source = Path(repair_from).resolve()
        prior, old_profiles, charged_roots = validate_repair_source(source, panel)
        write_once(
            out / 'repair-input.json',
            {
                'source': str(source),
                'source_summary_sha256': file_hash(source / 'summary.json'),
                'prior_generation_calls': prior,
                'charged_roots': charged_roots,
                'condition': 'tool_json',
                'contract': TOOL_JSON_CONTRACT,
                'normalization': 'JSON string decode, then strip surrounding whitespace; no other repair',
                'scope': 'new actor configuration diagnosis; original A/B scores unchanged',
            },
        )
    try:
        start(REPO / 'configs/serving/r3-large.json', out / 'worker')
        probe(binding_path, out / 'bootstrap')
        used += 2
        binding = EngineBinding.load(binding_path)
        application = application_history(
            REPO / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json', binding
        )
        for seed in panel['seeds']:
            e = HttpEngine(
                binding,
                profile=True,
                profile_policy={'name': 'q2_public_non_thinking_v2', 'seed': seed},
            )
            engines[seed] = e
            if repair_from:
                assert e.profile.digest == old_profiles[str(seed)]['digest'], (
                    'repair must retain model/decoding profile'
                )
            with RuntimeStore(out / f'qualification-runtime-{seed}') as store:
                receipt = await qualify_transport(
                    e,
                    json.loads((REPO / 'fixtures/serving/r2-transport-v2.json').read_text()),
                    out / f'qualification-{seed}',
                    store=store,
                    application_result=application,
                )
                used += receipt['cases']
            if not receipt['passed']:
                raise RuntimeError('seed profile未通过预资格')
        e = engines[panel['seeds'][0]]
        tokenize = lambda ms: e.tokenizer.apply_chat_template(
            ms, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        prepared = {
            c['id']: paired_contexts(c, tokenize, panel['max_input_tokens']) for c in panel['cases']
        }
        if repair_from:
            prepared = {}
            for c in panel['cases']:
                messages, ids, meta = action_context(
                    c['initial']['feedback'],
                    c['view'],
                    c['prefix'],
                    None,
                    tokenize,
                    3000,
                    instruction=TOOL_JSON_CONTRACT,
                )
                prepared[c['id']] = {
                    'tool_json': {'messages': messages, 'input_token_ids': ids, 'context': meta}
                }
        write_once(out / 'prepared-contexts.json', prepared)
        write_once(
            out / 'profiles.json',
            {
                str(s): {'digest': e.profile.digest, 'document': e.profile.document}
                for s, e in engines.items()
            },
        )
        rows = []
        with RuntimeStore(out / 'probe-runtime') as store:
            for index, case in enumerate(panel['cases']):
                for si, seed in enumerate(panel['seeds']):
                    order = (
                        ['tool_json']
                        if repair_from
                        else (
                            ['legacy', 'goal_action']
                            if (index + si) % 2 == 0
                            else ['goal_action', 'legacy']
                        )
                    )
                    for condition in order:
                        if used + prior >= panel['model_generation_cap']:
                            raise RuntimeError('model budget exhausted')
                        e = engines[seed]
                        context = prepared[case['id']][condition]
                        metadata = {
                            'purpose': 'actor_diagnostic',
                            'state_id': case['id'],
                            'source_view_hash': digest(case['view']),
                            'condition': condition,
                            'seed': seed,
                            'profile_digest': e.profile.digest,
                            'context': context['context'],
                        }
                        used += 1
                        receipt = await generate_action(
                            e,
                            store,
                            out / 'probes' / case['id'] / str(seed) / condition,
                            context['messages'],
                            context['input_token_ids'],
                            metadata,
                        )
                        row = {
                            **metadata,
                            'score': (score_json_action if repair_from else score_action)(
                                receipt['content'], case
                            ),
                            'receipt': receipt,
                        }
                        rows.append(row)
                print(
                    json.dumps(
                        {
                            'state': case['id'],
                            'key_actions': [
                                r['score']['acceptable_key_action']
                                for r in rows
                                if r['state_id'] == case['id']
                            ],
                        }
                    ),
                    flush=True,
                )
        write_once(out / 'probe-results.json', rows)
        gate = closed_loop_gate(
            [{**r, 'condition': 'goal_action'} for r in rows] if repair_from else rows
        )
        write_once(
            out / 'closed-loop-gate.json',
            {
                'passed': gate,
                'rule': 'candidate admissible >=33/36 and at least one acceptable key action for both s01 and s03',
                'calls_so_far': used,
                'prior_calls': prior,
            },
        )
        closed = []
        if gate:
            e = engines[20261103]
            tasks = {t['task_id']: t for t in manifest['tasks']}
            with RuntimeStore(out / 'closed-runtime') as store:
                for i, task_id in enumerate(('q1_support_01', 'q1_support_03')):
                    order = (
                        ['tool_json']
                        if repair_from
                        else (['legacy', 'goal_action'] if i == 0 else ['goal_action', 'legacy'])
                    )
                    for condition in order:
                        if used + prior + 40 > panel['model_generation_cap']:
                            raise RuntimeError(
                                'remaining budget cannot reserve complete short episode'
                            )
                        r = await episode(
                            tasks[task_id],
                            DATA,
                            out / 'closed' / task_id / condition,
                            IMAGE,
                            e,
                            store,
                            condition=condition,
                            max_steps=40,
                            stagnation_limit=4,
                            instruction=TOOL_JSON_CONTRACT
                            if repair_from
                            else INSTRUCTIONS[condition],
                            purpose='actor_diagnostic',
                            action_parser=parse_json_action if repair_from else None,
                        )
                        used += r['result']['steps']
                        closed.append(r)
        assert all(file_hash(REPO / n) == h for n, h in hashes.items())
        result = {
            'purpose': 'actor_diagnostic',
            'probe_calls': len(rows),
            'qualification_calls': 12,
            'bootstrap_calls': 2,
            'closed_loop_gate': gate,
            'closed_loop_calls': sum(r['result']['steps'] for r in closed),
            'total_generation_calls': used,
            'conditions': {
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
                for condition in (['tool_json'] if repair_from else INSTRUCTIONS)
            },
            'closed': [
                {'task_id': r['task_id'], 'condition': r['condition'], 'result': r['result']}
                for r in closed
            ],
            'training_runs': 0,
            'primary_history_sources': 0,
            'scope': 'seen support-state diagnostic; not memory/weight learning or task generalization',
        }
        result['prior_generation_calls'] = prior
        result['package_generation_calls'] = prior + used
    finally:
        await close_models(engines, binding_path, out)
    if result:
        write_once(out / 'summary.json', result)
        return result


async def adapter_closed(output, cpu_run, adapter_from):
    cpu_verification = validate_cpu_baseline(Path(cpu_run).resolve())
    source = Path(adapter_from).resolve()
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    source_input = read(source / 'input.json')
    source_rows = read(source / 'probe-results.json')
    out, panel, manifest, hashes = setup(output, 'adapter_closed')
    from q2_audit import audit_rescue

    verification = audit_rescue(source, out / 'adapter-source-verification.json')
    assert (
        verification['verified']
        and verification['probe_calls'] == 36
        and verification['total_generations'] == 50
        and verification['closed_loop_gate'] is False
    )
    assert source_input['panel'] == panel and source_input['panel_sha256'] == file_hash(PANEL)
    assert len(source_rows) == 36 and {
        (r['state_id'], r['seed'], r['condition']) for r in source_rows
    } == {(c['id'], s, 'tool_json') for c in panel['cases'] for s in panel['seeds']}
    original = Path(read(source / 'repair-input.json')['source'])
    prior, old_profiles, charged = validate_repair_source(original, panel, reserve_max=86)
    cases = {c['id']: c for c in panel['cases']}
    replay = [
        {
            'state_id': r['state_id'],
            'seed': r['seed'],
            'condition': 'goal_action',
            'source_attempt_id': r['receipt']['attempt_id'],
            'score': score_envelope(r['receipt']['content'], cases[r['state_id']]),
        }
        for r in source_rows
    ]
    gate = closed_loop_gate(replay)
    write_once(
        out / 'adapter-input.json',
        {
            'source': str(source),
            'source_probe_results_sha256': file_hash(source / 'probe-results.json'),
            'scope': 'offline conservative decoder replay; same C prompt/profile; no new static model samples',
            'instruction': TOOL_JSON_CONTRACT,
            'discarded_model_feedback': True,
            'normalization': 'JSON scalar or unique action/command envelope; strip action whitespace; reject ambiguity/extra keys',
        },
    )
    write_once(
        out / 'repair-input.json',
        {
            'source': str(original),
            'prior_generation_calls': prior,
            'charged_roots': charged,
            'condition': 'tool_envelope',
        },
    )
    write_once(out / 'cpu-verification.json', cpu_verification)
    write_once(out / 'adapter-replay.json', replay)
    write_once(
        out / 'closed-loop-gate.json',
        {
            'passed': gate,
            'rule': 'adapted replay admissible >=33/36 and key01/key03 each hit; original C gate stays false',
        },
    )
    if not gate:
        write_once(
            out / 'summary.json',
            {
                'purpose': 'actor_diagnostic',
                'total_generation_calls': 0,
                'closed_loop_gate': False,
                'prior_generation_calls': prior,
                'package_generation_calls': prior,
            },
        )
        return {'adapter_gate': False}
    engines = {}
    binding_path = out / 'worker/binding.json'
    used = 0
    result = None
    try:
        start(REPO / 'configs/serving/r3-large.json', out / 'worker')
        probe(binding_path, out / 'bootstrap')
        used += 2
        binding = EngineBinding.load(binding_path)
        e = HttpEngine(
            binding,
            profile=True,
            profile_policy={'name': 'q2_public_non_thinking_v2', 'seed': 20261103},
        )
        engines[20261103] = e
        assert (
            e.profile.digest
            == old_profiles['20261103']['digest']
            == read(source / 'profiles.json')['20261103']['digest']
        )
        write_once(
            out / 'profiles.json',
            {'20261103': {'digest': e.profile.digest, 'document': e.profile.document}},
        )
        application = application_history(
            REPO / 'artifacts/r2-large-quality-20261103-a/qualification/receipt.json', binding
        )
        with RuntimeStore(out / 'qualification-runtime-20261103') as store:
            receipt = await qualify_transport(
                e,
                read(REPO / 'fixtures/serving/r2-transport-v2.json'),
                out / 'qualification-20261103',
                store=store,
                application_result=application,
            )
            used += receipt['cases']
        if not receipt['passed']:
            raise RuntimeError('adapter profile prequalification failed')
        tasks = {t['task_id']: t for t in manifest['tasks']}
        closed = []
        with RuntimeStore(out / 'closed-runtime') as store:
            for task_id in ('q1_support_01', 'q1_support_03'):
                if prior + used + 40 > panel['model_generation_cap']:
                    raise RuntimeError('adapter budget exhausted')
                r = await episode(
                    tasks[task_id],
                    DATA,
                    out / 'closed' / task_id / 'tool_envelope',
                    IMAGE,
                    e,
                    store,
                    condition='tool_envelope',
                    max_steps=40,
                    stagnation_limit=4,
                    instruction=TOOL_JSON_CONTRACT,
                    purpose='actor_diagnostic',
                    action_parser=parse_action_envelope,
                )
                used += r['result']['steps']
                closed.append(r)
        assert all(file_hash(REPO / n) == h for n, h in hashes.items())
        result = {
            'purpose': 'actor_diagnostic',
            'probe_calls': 0,
            'replayed_source_probe_calls': 36,
            'qualification_calls': 4,
            'bootstrap_calls': 2,
            'closed_loop_gate': True,
            'closed_loop_calls': sum(r['result']['steps'] for r in closed),
            'total_generation_calls': used,
            'prior_generation_calls': prior,
            'package_generation_calls': prior + used,
            'replay_conditions': {
                'admissible': sum(r['score']['admissible'] for r in replay),
                'key_action_hits': sum(r['score']['acceptable_key_action'] is True for r in replay),
                'total': 36,
                'key_action_total': 18,
            },
            'closed': [
                {'task_id': r['task_id'], 'condition': r['condition'], 'result': r['result']}
                for r in closed
            ],
            'training_runs': 0,
            'primary_history_sources': 0,
            'scope': 'same C prompt, conservative input adapter; seen support development only',
        }
    finally:
        await close_models(engines, binding_path, out)
    if result:
        write_once(out / 'summary.json', result)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['cpu', 'model'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cpu-run')
    p.add_argument('--repair-from')
    p.add_argument('--adapter-from')
    p.add_argument('--dev-remaining', type=int, default=600)
    p.add_argument('--formal-remaining', type=int, default=1200)
    a = p.parse_args()
    task = (
        cpu(a.output, a.dev_remaining, a.formal_remaining)
        if a.stage == 'cpu'
        else (
            adapter_closed(a.output, a.cpu_run, a.adapter_from)
            if a.adapter_from
            else model(a.output, a.cpu_run, a.repair_from)
        )
    )
    result = asyncio.run(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
