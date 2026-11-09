"""先冻结信息/资格协议，再从既有合格train来源构建开发资格输入。"""

import argparse
import json
from pathlib import Path
import re

from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.task_protocol import digest
from serving_lab.workflow import validate_workflow_manifest
from serving_lab.workflow_memory import memory_entry
from serving_lab.workflow_study import retrieve_procedural

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reconstruct_memories(data, collection, tokenizer, plan):
    from q2_workflow import audit as audit_teacher

    data, collection = Path(data).resolve(), Path(collection).resolve()
    if (
        data != ROOT / 'artifacts/q2-workflow-data-20261105-a'
        or collection != ROOT / 'artifacts/q2-workflow-collection-20261105-a'
    ):
        raise ValueError('开发来源不是既有已核工作流集合')
    audit = audit_teacher(data, collection)
    if not audit['data_gate_passed']:
        raise ValueError('原始教师来源没有资格')
    manifest = read(data / 'manifest.json')
    tasks = validate_workflow_manifest(manifest)
    entries = [
        memory_entry(
            read(collection / 'workflow_teacher' / t['task_id'] / 'record.json'),
            t,
            manifest['source_contract_digest'],
            file_hash(data / 'manifest.json'),
        )
        for t in tasks.values()
        if t['role'] == 'train'
    ]
    memories = {}
    for task in tasks.values():
        if task['role'] != 'dev':
            continue
        initial = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')['initial']
        goal = re.search(r'Your task is to:\s*([^\n]+)', initial['feedback'])[1].strip()
        retrieval = retrieve_procedural(
            entries,
            goal,
            task['workflow'],
            cutoff=task['phase'] * 42 + 1,
            tokenize=lambda s: tokenizer.encode(s, add_special_tokens=False),
            representation='combined',
            token_budget=plan['qualification']['memory_tokens'],
        )
        if not retrieval['selected']:
            raise ValueError('强M缺少预算内完整同版本训练轨迹')
        memories[task['task_id']] = {
            **retrieval,
            'stage': task['phase'],
            'workflow': task['workflow'],
            'source_role': 'train',
            'entry_digests': [s['entry_digest'] for s in retrieval['selected']],
            'initial_digest': digest(initial),
        }
    return memories, audit


def freeze(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        'schema': 1,
        'name': 'n5-announced-procedure-v1',
        'created_on': '2026-11-06',
        'purpose': 'supervised procedural knowledge consolidation and revision adaptation; not autonomous human-like memory',
        'stages': ['preparation-v1', 'inspection-v1', 'preparation-v2'],
        'arms': {
            'F': 'frozen shared actor',
            'M': 'same actor plus current-version rulebook and one complete verified train trajectory',
            'P': 'ordinary sequential LoRA plus uniform replay, revision labels preserved',
            'PM': 'exact P checkpoint plus exact M snapshot',
            'G': 'conditional step/dependency-aware replay candidate; no activation before interference gate',
        },
        'information': {
            'announcement': 'same published rule/revision/timestamp and verified train trajectories for M/P/PM/G',
            'query': 'goal,procedure name,revision,thermal type,current public observation; no complete rule',
            'explicit_control': 'separate full-rule development panel, never pooled with announcement-query result',
            'future': 'new underlying sources; old future identities remain excluded; never used for training/retrieval/tuning',
        },
        'training_seeds': [20261106, 20261107, 20261108],
        'training': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'epochs': 3,
            'micro_batch': 1,
            'accumulation': 8,
            'learning_rate': 2e-4,
            'weight_decay': 0.0,
            'warmup_ratio': 0.05,
            'scheduler': 'cosine',
            'precision': 'bfloat16',
            'max_length': 3564,
            'gradient_checkpointing': True,
            'gradient_clip_norm': 1.0,
            'target_modules': [
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            'loss': 'sum supervised token loss divided by batch supervised-token count',
            'replay': 'new stage and prior allowed examples approximately1:1 supervised tokens; log exact counts',
            'selection': 'fixed final epoch, no future-dependent choice; PM never trains separately',
        },
        'qualification': {
            'panel': 'existing six development bases x three procedures,18 paired cases per condition',
            'conditions': ['explicit', 'memory'],
            'min_control_success': 12,
            'min_memory_success': 12,
            'max_memory_gap': 3,
            'min_each_phase_success': 3,
            'min_each_thermal_success': 1,
            'backbone_order': ['C0-Qwen3-1.7B', 'Qwen3-4B-fixed-original'],
            'upgrade': 'if C0 fails, qualify fixed4B jointly for all arms; no candidate-only upgrade',
            'generation_cap': 8000,
            'max_steps': 100,
            'stagnation_limit': 4,
            'input_tokens': 3500,
            'memory_tokens': 2048,
            'selection_policy': 'first common actor passing both conditions; otherwise keep research gate failed and diagnose',
        },
        'future_selection': {
            'base_count': 12,
            'families': 2,
            'per_family_per_scene': 2,
            'scenes': 3,
            'seed': 20261106,
            'source': 'same fixed ALFWorld public train archives; distinct source keys and scenes from prior coldstart/workflow sets',
            'partition': 'metadata-only deterministic sha256 ordering; exclude all old protected/source identities',
            'opening': 'after protocol, source audit, common actor and all stage checkpoints are frozen',
        },
        'matrix': 'each stage x trained seed: current skills, still-valid earlier skills, obsolete-order misuse, new future instances and base skill retention; F/M deterministic references reported once',
        'g_activation': {
            'required': 'repeatable obsolete-rule interference in P development across >=2 training seeds, and simple version-drop loses still-valid skill relative to P',
            'controls': [
                'uniform replay P',
                'simple whole-version-drop',
                'same-information current-rulebook M',
            ],
            'mechanism': 'remove invalidated supervision and its dependent state/prefix closure, keep only proven-valid prefix or separately valid episodes',
            'fairness': 'same published invalidation metadata, training token/update budget; explicit padding/cycling receipts',
            'absence': 'G remains unexecuted and full five-arm R incomplete; do not invent a method to fill the table',
        },
        'cost': [
            'GPU wall time',
            'input tokens',
            'supervised tokens',
            'actual model generations',
            'environment actions',
            'memory construction calls',
            'failed attempts',
        ],
        'unit_of_analysis': 'base environment, paired procedures are not independent samples; seed and instance variability separate',
        'references': [
            'https://arxiv.org/html/2508.06433v4',
            'https://github.com/zjunlp/MemP',
            'https://arxiv.org/html/2402.18865v1',
            'https://github.com/which47/LLMCL',
            'https://arxiv.org/html/2608.15165v1',
        ],
    }
    write_once(out / 'plan.json', plan)
    write_once(
        out / 'plan-anchor.json', {'sha256': file_hash(out / 'plan.json'), 'future_accessed': False}
    )
    return plan


def prepare(out, model):
    from transformers import AutoTokenizer
    from q2_workflow import audit as audit_teacher

    out = Path(out)
    plan = read(out / 'plan.json')
    if file_hash(out / 'plan.json') != read(out / 'plan-anchor.json')['sha256']:
        raise ValueError('协议已变')
    data = ROOT / 'artifacts/q2-workflow-data-20261105-a'
    collection = ROOT / 'artifacts/q2-workflow-collection-20261105-a'
    audit = audit_teacher(data, collection)
    if not audit['data_gate_passed']:
        raise ValueError('旧公开教师来源未通过重算')
    manifest = read(data / 'manifest.json')
    tasks = validate_workflow_manifest(manifest)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    entries = [
        memory_entry(
            read(collection / 'workflow_teacher' / task['task_id'] / 'record.json'),
            task,
            manifest['source_contract_digest'],
            file_hash(data / 'manifest.json'),
        )
        for task in tasks.values()
        if task['role'] == 'train'
    ]
    directory = out / ('qualification-' + Path(model).name)
    directory.mkdir(exist_ok=False)
    memories = {}
    cases = []
    for task in tasks.values():
        if task['role'] != 'dev':
            continue
        record = read(collection / 'workflow_teacher' / task['task_id'] / 'record.json')
        goal = re.search(r'Your task is to:\s*([^\n]+)', record['initial']['feedback'])[1].strip()
        retrieval = retrieve_procedural(
            entries,
            goal,
            task['workflow'],
            cutoff=task['phase'] * 42 + 1,
            tokenize=lambda s: tokenizer.encode(s, add_special_tokens=False),
            representation='combined',
            token_budget=plan['qualification']['memory_tokens'],
        )
        if not retrieval['selected']:
            raise ValueError('强M缺少预算内完整同版本训练轨迹')
        selected = {e['source_task_id']: e for e in entries}
        if any(
            selected[s['source_task_id']]['phase'] > task['phase'] for s in retrieval['selected']
        ):
            raise ValueError('记忆越过阶段')
        memories[task['task_id']] = {
            **retrieval,
            'stage': task['phase'],
            'workflow': task['workflow'],
            'source_role': 'train',
            'entry_digests': [s['entry_digest'] for s in retrieval['selected']],
            'initial_digest': digest(record['initial']),
        }
        for condition in plan['qualification']['conditions']:
            cases.append(
                {
                    'id': 'n5-' + task['task_id'] + '-' + condition,
                    'source_task_id': task['task_id'],
                    'condition': condition,
                    'phase': task['phase'],
                    'role': 'dev',
                    'base_id': task['base_id'],
                    'thermal': task['workflow']['thermal'],
                }
            )
    write_once(directory / 'memories.json', memories)
    write_once(
        directory / 'input.json',
        {
            'plan_sha256': file_hash(out / 'plan.json'),
            'data': str(data),
            'data_manifest_sha256': file_hash(data / 'manifest.json'),
            'collection': str(collection),
            'source_audit': audit,
            'cases': cases,
            'memories_sha256': file_hash(directory / 'memories.json'),
            'tokenizer_config_sha256': file_hash(Path(model) / 'tokenizer_config.json'),
            'future_accessed': False,
            'source': {
                'scripts/n5_prepare.py': file_hash(__file__),
                'src/serving_lab/continual_procedure.py': file_hash(
                    ROOT / 'src/serving_lab/continual_procedure.py'
                ),
            },
        },
    )
    return {
        'output': str(directory),
        'cases': len(cases),
        'train_entries': len(entries),
        'future_accessed': False,
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'prepare'])
    p.add_argument('--output', required=True)
    p.add_argument('--model')
    a = p.parse_args()
    result = freeze(a.output) if a.stage == 'freeze' else prepare(a.output, a.model)
    print(json.dumps(result, ensure_ascii=False))
