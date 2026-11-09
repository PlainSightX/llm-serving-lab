"""冷启动固定三条件的独立dev闭环；开发结果不导出下一轮训练。"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from serving_lab.coldstart_actor import ColdStartActor
from serving_lab.engine import EngineBinding, HttpEngine
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash
from serving_lab.research_action import generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.task_protocol import FAMILIES, canonical
from serving_lab.teacher_data import validate_teacher_manifest, validate_base_binding
from serving_lab.transport import qualify_transport
from probe_serving import probe
from q2_rescue import close_models, reserved_generations
from q2_run import episode
from q2_standard import CANARIES, POLICY
from q2_teacher import IMAGE, read, verify_collection
from start_engine import start

REPO = Path(__file__).resolve().parents[1]


def local_path(value):
    """同一仓库在Windows训练、WSL评价；不把字符串别名当不同源。"""
    if sys.platform == 'linux' and len(value) > 2 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + value[2:].replace('\\', '/')
    result = Path(value).resolve()
    result.relative_to(REPO)
    return result


def select_examples(tasks, records, token_length):
    candidates = []
    for task in tasks:
        if task['role'] != 'train':
            continue
        r = records[task['task_id']]
        if r['task_id'] != task['task_id'] or r['record_hash'] != digest(
            {k: v for k, v in r.items() if k != 'record_hash'}
        ):
            raise ValueError('示范原始记录身份不符')
        if r['result']['success']:
            material = {
                'initial_observation': r['initial']['feedback'],
                'trajectory': [
                    {'action': s['action'], 'feedback': s['after']['feedback']} for s in r['trace']
                ],
            }
            size = token_length(canonical(material))
            candidates.append((size, task['task_id'], task['family'], material, r['record_hash']))
    chosen = []
    for c in sorted(candidates):
        if not chosen or c[2] != chosen[0][2]:
            chosen.append(c)
        if len(chosen) == 2:
            break
    text = canonical([c[3] for c in chosen])
    if len(chosen) != 2 or token_length(text) > 1280:
        raise ValueError('两条固定完整示范不能在事前预算内共存')
    return text, [{'task_id': c[1], 'family': c[2], 'record_hash': c[4]} for c in chosen]


def validate_examples(control, tasks, records, token_length):
    text, sources = select_examples(tasks, records, token_length)
    if (
        text != control['examples_text']
        or digest(text) != control['examples_digest']
        or sources != control['sources']
    ):
        raise ValueError('示范必须从真实训练原件重建，不能只验证自洽摘要')
    return text, sources


def build_control(dataset, expected_control=None):
    dataset = local_path(str(dataset))
    source = read(dataset / 'manifest.json')
    data, collection = local_path(source['data']), local_path(source['collection'])
    verification = verify_collection(data, collection)
    if (
        not verification['data_gate_passed']
        or verification != source['source_verification']
        or file_hash(data / 'manifest.json') != source['data_manifest_sha256']
        or file_hash(dataset / 'train.jsonl') != source['data_sha256']
    ):
        raise ValueError('教师/导出资格未通过')
    binding = read(dataset / 'base-binding.json')
    validate_base_binding(binding, binding['base'], file_hash(dataset / 'manifest.json'))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        local_path(source['model']), local_files_only=True, trust_remote_code=False
    )
    tasks = read(data / 'manifest.json')['tasks']
    records = {t['task_id']: read(collection / t['task_id'] / 'record.json') for t in tasks}
    size = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    text, sources = (
        select_examples(tasks, records, size)
        if expected_control is None
        else validate_examples(expected_control, tasks, records, size)
    )
    result = {
        'name': 'q2-coldstart-dev-controls-v1',
        'dataset': str(Path(dataset).resolve()),
        'data_sha256': source['data_sha256'],
        'manifest_sha256': source['data_manifest_sha256'],
        'examples_text': text,
        'examples_digest': digest(text),
        'sources': sources,
        'base_artifact_digest': binding['base_artifact_digest'],
        'base_binding_sha256': file_hash(dataset / 'base-binding.json'),
        'policy': POLICY,
        'max_steps': 100,
        'max_input_tokens': 3500,
        'stagnation_limit': 4,
        'conditions': ['base', 'fixed_demo', 'trained'],
        'model_generation_cap': 7220,
        'qualification': 'C0 requires >=18/24 success and >=2/4 for every family; working task-qualification gate, not population guarantee',
        'comparison_limit': 'fixed_demo uses two trajectories; trained uses all qualified train data; not equal-information parametric vs memory research',
        'selection': 'two shortest complete successful train trajectories with distinct families, shared by all dev cases; chosen before model calls',
    }
    return result


def validate_controls(control):
    expected = build_control(control['dataset'], control)
    expected['dataset'] = control['dataset']  # 路径已按同一仓库的实际位置解析。
    if expected != control:
        raise ValueError('固定对照合同与原始训练来源/基座不一致')


def controls(dataset, output):
    result = build_control(dataset)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / 'controls.json', result)
    return result


def summarize(records, tasks, conditions):
    expected = {
        (k, condition) for k, t in tasks.items() if t['role'] == 'dev' for condition in conditions
    }
    if (
        len(records) != len(expected)
        or {(r['task_id'], r['condition']) for r in records} != expected
    ):
        raise ValueError('完整24题及固定条件缺失或重复，不能报告成功子集')
    result = {}
    for condition in conditions:
        selected = [r for r in records if r['condition'] == condition]
        counts = Counter(tasks[r['task_id']]['family'] for r in selected if r['result']['success'])
        successes = sum(r['result']['success'] for r in selected)
        result[condition] = {
            'episodes': 24,
            'successes': successes,
            'family_successes': {f: counts[f] for f in FAMILIES},
            'action_calls': sum(r['result']['steps'] for r in selected),
            'C0_task_gate': successes >= 18 and all(counts[f] >= 2 for f in FAMILIES),
            'tasks': [{'task_id': r['task_id'], 'result': r['result']} for r in selected],
        }
    return result


async def run(data, controls_path, output, training=None):
    data, out = Path(data).resolve(), Path(output).resolve()
    control = read(controls_path)
    manifest = read(data / 'manifest.json')
    tasks = validate_teacher_manifest(manifest)
    validate_controls(control)
    assert (
        file_hash(data / 'manifest.json') == control['manifest_sha256']
        and digest(control['examples_text']) == control['examples_digest']
    )
    config = read(REPO / 'configs/serving/r3-large.json')
    conditions = ['base', 'fixed_demo']
    if training:
        training = Path(training).resolve()
        trained = read(training / 'summary.json')
        assert (
            trained['data_sha256'] == control['data_sha256']
            and trained['quality_status'] == 'not_evaluated'
        )
        assert (
            trained['base_artifact_digest'] == control['base_artifact_digest']
            and trained['base_binding_sha256'] == control['base_binding_sha256']
        )
        config.update(
            version_id='qwen3-17b-coldstart-candidate',
            model_path=str((training / 'merged').relative_to(REPO)),
            model_revision=trained['adapter_sha256'],
        )
        conditions = ['trained']
    out.mkdir(parents=True, exist_ok=False)
    names = [
        'scripts/' + n
        for n in (
            'q2_coldstart_eval.py',
            'q2_run.py',
            'q2_environment.py',
            'start_engine.py',
            'probe_serving.py',
        )
    ]
    names += ['src/serving_lab/' + p.name for p in (REPO / 'src/serving_lab').glob('*.py')]
    hashes = {n: file_hash(REPO / n) for n in names}
    for name in names:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    write_once(out / 'engine-config.json', config)
    write_once(
        out / 'input.json',
        {
            'data': str(data),
            'controls': control,
            'conditions': conditions,
            'source_hashes': hashes,
            'training': str(training) if training else None,
            'manifest_sha256': file_hash(data / 'manifest.json'),
            'generation_cap': len(conditions) * 2400 + 10,
            'source_contract_digest': manifest['source_contract_digest'],
        },
    )
    binding_path = out / 'worker/binding.json'
    engines, records = {}, []
    try:
        start(out / 'engine-config.json', out / 'worker')
        probe(binding_path, out / 'bootstrap')
        binding = EngineBinding.load(binding_path)
        if training:
            assert binding.artifact_digest == digest(trained['merged_identity']['files'])
        else:
            assert binding.artifact_digest == control['base_artifact_digest']
        engine = HttpEngine(binding, profile=True, profile_policy=POLICY)
        engines['actor'] = engine
        write_once(
            out / 'profile.json',
            {'document': engine.profile.document, 'digest': engine.profile.digest},
        )
        with RuntimeStore(out / 'qualification-runtime') as store:
            q = await qualify_transport(
                engine,
                read(REPO / 'fixtures/serving/r2-transport-v2.json'),
                out / 'qualification',
                store=store,
                application_result={
                    'scope': 'q2_coldstart_dev_v1',
                    'status': 'not_yet_evaluated',
                    'old_application_result': 'not inherited',
                },
            )
        if not q['passed']:
            raise ValueError('传输预资格失败')
        token = lambda ms: engine.tokenizer.apply_chat_template(
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
                rec = await generate_action(
                    engine,
                    store,
                    out / 'canaries' / str(i),
                    messages,
                    token(messages),
                    {'purpose': 'structured_interface_canary', 'case': i},
                    choices=choices,
                )
                ColdStartActor().action(rec, choices)
        with RuntimeStore(out / 'action-runtime') as store:
            for index, task in enumerate(t for t in tasks.values() if t['role'] == 'dev'):
                order = conditions if index % 2 == 0 else list(reversed(conditions))
                for condition in order:
                    actor = ColdStartActor(
                        control['examples_text'] if condition == 'fixed_demo' else None
                    )
                    records.append(
                        await episode(
                            task,
                            data,
                            out / 'dev' / task['task_id'] / condition,
                            IMAGE,
                            engine,
                            store,
                            condition=condition,
                            max_steps=100,
                            stagnation_limit=4,
                            purpose='actor_diagnostic',
                            actor=actor,
                        )
                    )
        result = {
            'conditions': summarize(records, tasks, conditions),
            'task_calls': sum(len(r['trace']) for r in records),
            'bootstrap_calls': 2,
            'qualification_calls': 4,
            'canary_calls': 4,
            'purpose': 'coldstart_development_evaluation',
            'eligible_for_training': False,
            'primary_history_sources': 0,
        }
        result['total_generations'] = result['task_calls'] + 10
        assert (
            result['total_generations']
            == len(reserved_generations(out))
            <= len(conditions) * 2400 + 10
        )
        assert all(file_hash(REPO / n) == h for n, h in hashes.items())
    except BaseException as exc:
        write_once(
            out / 'failure.json',
            {
                'error': repr(exc),
                'completed_records': len(records),
                'reserved_generations': len(reserved_generations(out)),
            },
        )
        raise
    finally:
        await close_models(engines, binding_path, out)
    write_once(out / 'summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['controls', 'run'], required=True)
    p.add_argument('--dataset')
    p.add_argument('--data')
    p.add_argument('--controls')
    p.add_argument('--training')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = (
        controls(a.dataset, a.output)
        if a.stage == 'controls'
        else asyncio.run(run(a.data, a.controls, a.output, a.training))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
