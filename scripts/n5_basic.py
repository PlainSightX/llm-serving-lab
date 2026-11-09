"""共享4B基础适配后的完整24项开发门，未过时不进入流程研究。"""

import argparse
import asyncio
import json
from pathlib import Path

from serving_lab.continual_procedure import validate_actor_binding
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.research_action import check_readiness, generate_action
from serving_lab.runtime import RuntimeStore
from serving_lab.teacher_data import validate_teacher_manifest
from serving_lab.worker_protocol import RemoteEngine
from n5_qualify import episode, reserve

ROOT = Path(__file__).resolve().parents[1]
ROLE = 'Qwen3-4B-shared-basic'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


async def main(args):
    from transformers import AutoTokenizer

    owner = Path(args.owner).resolve()
    out = Path(args.output).resolve()
    plan = read(owner / 'plan.json')
    config = read(args.engine)
    bindings = read(owner / 'actor-bindings.json')
    binding = bindings['actors'][ROLE]
    if (
        file_hash(owner / 'plan.json') != read(owner / 'plan-anchor.json')['sha256']
        or file_hash(owner / 'actor-bindings.json')
        != read(owner / 'actor-bindings-anchor.json')['sha256']
        or bindings['plan_sha256'] != file_hash(owner / 'plan.json')
    ):
        raise ValueError('基础评价协议或共同模型绑定变化')
    validate_actor_binding(ROLE, config['manifest'], binding)
    if file_hash(Path(args.model) / 'tokenizer_config.json') != binding['tokenizer_config_sha256']:
        raise ValueError('共同分词器不同')
    training = read(args.training_evidence)
    if (
        not training['passed']
        or training['artifact_digest'] != config['manifest']['artifact_digest']
        or training['parameter_digest'] != config['manifest']['parameter_digest']
        or file_hash(args.training_evidence) != bindings['training_evidence_sha256']
    ):
        raise ValueError('没有实际共享基础训练证明')
    data = ROOT / 'artifacts/q2-teacher-data-20261103-b'
    tasks = validate_teacher_manifest(read(data / 'manifest.json'))
    if file_hash(data / 'manifest.json') != plan['basic_gate']['data_manifest_sha256']:
        raise ValueError('基础24项数据不同')
    selected = [t for t in tasks.values() if t['role'] == 'dev']
    if len(selected) != 24:
        raise ValueError('基础面板不完整')
    cases = [
        {
            'id': 'n5-basic-' + t['task_id'],
            'source_task_id': t['task_id'],
            'condition': 'basic',
            'role': 'dev',
            'family': t['family'],
        }
        for t in selected
    ]
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in list((ROOT / 'src/serving_lab').glob('*.py')) + list(
        (ROOT / 'scripts').glob('*.py')
    ):
        relative = path.relative_to(ROOT)
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources[relative.as_posix()] = file_hash(path)
    write_once(
        out / 'input.json',
        {
            'worker_manifest': config['manifest'],
            'actor_role': ROLE,
            'cases': cases,
            'data': str(data),
            'data_manifest_sha256': file_hash(data / 'manifest.json'),
            'source_hashes': sources,
            'plan_sha256': file_hash(owner / 'plan.json'),
            'actor_bindings_sha256': file_hash(owner / 'actor-bindings.json'),
            'training_evidence_sha256': file_hash(args.training_evidence),
            'scope': 'basic actor readiness only, not workflow/continual-learning results',
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    engine = RemoteEngine(
        config['endpoint'],
        Path(config['token_file']).read_text().strip(),
        config['manifest'],
        tls=config['tls'],
    )
    store = RuntimeStore(out / 'runtime')
    records = []
    try:
        ready, checks = await check_readiness(engine, 2)
        write_once(out / 'readiness.json', {'ready': ready, 'checks': checks})
        if not ready:
            raise ValueError('共同worker未就绪')
        for i, command in enumerate(['look', 'inventory'], 1):
            messages = [{'role': 'user', 'content': 'Return the single permitted command.'}]
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            reserve(
                owner,
                str(out),
                'choice-canary-' + str(i),
                0,
                plan['qualification']['generation_cap'],
            )
            receipt = await generate_action(
                engine,
                store,
                out / 'canaries' / str(i),
                messages,
                ids,
                {'purpose': 'n5_interface_canary', 'case': 'canary-' + str(i), 'step': 0},
                choices=[command],
                readiness_attempts=2,
            )
            if receipt['content'] != command or receipt['finish_reason'] != 'stop':
                raise ValueError('choice接口不符')
        for case, task in zip(cases, selected):
            row = await episode(
                engine,
                store,
                tokenizer,
                owner,
                case,
                task,
                data,
                None,
                out / 'episodes' / case['id'],
                plan,
                basic=True,
            )
            records.append(row)
            print(
                json.dumps(
                    {
                        'case': case['id'],
                        'success': row['result']['success'],
                        'actions': len(row['trace']),
                    }
                ),
                flush=True,
            )
        passed = sum(row['result']['success'] for row in records)
        actions = sum(len(row['trace']) for row in records)
        result = {
            'completed': True,
            'qualified': passed >= plan['basic_gate']['minimum_success'],
            'passed_cases': passed,
            'cases': 24,
            'actual_generations': actions + 2,
            'study_actions': actions,
            'interface_canaries': 2,
            'future_episodes': 0,
            'artifact_digest': config['manifest']['artifact_digest'],
            'parameter_digest': config['manifest']['parameter_digest'],
            'scope': 'shared basic actor development readiness, no workflow or continual-learning claim',
        }
        write_once(out / 'summary.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write_once(
            out / 'failure.json',
            {'error': repr(error), 'completed_episodes': len(records), 'implicit_retry': False},
        )
        raise
    finally:
        await engine.close()
        store.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['owner', 'output', 'model', 'engine', 'training-evidence']:
        p.add_argument('--' + name, required=True)
    asyncio.run(main(p.parse_args()))
