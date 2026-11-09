"""公开程序教师的新来源合同；不接纳旧oracle或开发诊断。"""

import hashlib
from collections import Counter

from .standard_actor import StandardActor
from .task_protocol import FAMILIES, digest, public_view

# 已核的原1.7B完整工件身份；相同tokenizer不能证明相同权重。
FIXED_BASE_DIGEST = '2d5f13caf0959c19912d4ed0ac556e6c6949e2fe855028e97233c5b35fa7f013'
FIXED_BASE_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def validate_base_binding(binding, actual_base, dataset_manifest_sha256):
    if (
        digest(actual_base['files']) != FIXED_BASE_DIGEST
        or binding['base_artifact_digest'] != FIXED_BASE_DIGEST
        or binding['base']['files'] != actual_base['files']
        or binding['base_revision'] != FIXED_BASE_REVISION
        or binding['dataset_manifest_sha256'] != dataset_manifest_sha256
    ):
        raise ValueError('训练/对照必须绑定固定1.7B完整权重工件，tokenizer相同不足以通过')


def select_candidates(candidates, seed):
    """只按源元数据选题；先按场景分组，任何执行结果都不参与选择。"""
    partitions = {}
    for row in candidates:
        if row['scene'] in partitions and partitions[row['scene']] != row['scene_partition']:
            raise ValueError('同一场景不能分属训练与开发')
        partitions[row['scene']] = row['scene_partition']
    rank = lambda r: hashlib.sha256(f'{seed}:{r["key"]}'.encode()).hexdigest()
    selected, dev_params = [], set()
    for role, count in (('dev', 4), ('train', 8)):
        for family in FAMILIES:
            chosen, scenes, parameters = [], set(), set()
            for row in sorted(candidates, key=rank):
                if row['family'] != family or row['scene_partition'] != role:
                    continue
                if row['scene'] in scenes or row['parameter_digest'] in parameters:
                    continue
                if role == 'train' and row['parameter_digest'] in dev_params:
                    continue
                chosen.append({**row, 'role': role})
                scenes.add(row['scene'])
                parameters.add(row['parameter_digest'])
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'{family}/{role}独立场景和参数组合不足，不降低隔离标准')
            selected.extend(chosen)
            if role == 'dev':
                dev_params.update(parameters)
    return sorted(
        selected, key=lambda r: (r['role'] != 'train', FAMILIES.index(r['family']), rank(r))
    )


def validate_teacher_manifest(manifest):
    if manifest.get('name') != 'q2-public-teacher-coldstart-v1' or manifest.get('max_steps') != 100:
        raise ValueError('不是冻结的公开教师数据合同')
    contract = manifest['source_contract']
    if (
        contract.get('origin') != 'teacher_public_program'
        or contract.get('old_records_allowed') is not False
        or digest(contract) != manifest.get('source_contract_digest')
    ):
        raise ValueError('教师来源合同不符')
    tasks = manifest['tasks']
    ids = {t['task_id']: t for t in tasks}
    if (
        len(tasks) != len(ids)
        or len(tasks) != 72
        or [t['sequence'] for t in tasks] != list(range(1, 73))
    ):
        raise ValueError('必须冻结72个独立任务及顺序')
    counts = Counter((t['role'], t['family']) for t in tasks)
    if counts != Counter(
        {(role, f): n for role, n in [('train', 8), ('dev', 4)] for f in FAMILIES}
    ):
        raise ValueError('训练/开发任务族覆盖不完整')
    for key in ('scene', 'parameter_digest', 'source_key'):
        if {t[key] for t in tasks if t['role'] == 'train'} & {
            t[key] for t in tasks if t['role'] == 'dev'
        }:
            raise ValueError('训练开发身份重叠：' + key)
    if len({t['source_key'] for t in tasks}) != 72:
        raise ValueError('重复源实例')
    for t in tasks:
        if (
            t['source_split'] != 'train'
            or not t['source_key'].startswith('train/')
            or t['scene'] in manifest['excluded_old_scenes']
        ):
            raise ValueError('旧开发或保护来源不得进入新集')
        files = {f'games/{t["task_id"]}/game.tw-pddl', f'games/{t["task_id"]}/traj_data.json'}
        if set(t['files']) != files or t['game_path'] != f'games/{t["task_id"]}/game.tw-pddl':
            raise ValueError('文件与任务身份不符')
        if any(not isinstance(h, str) or len(h) != 64 for h in t['files'].values()):
            raise ValueError('源文件摘要缺失')
    return ids


def action_samples(record, tokenize, contract_digest):
    """监督本步动作，反馈只能用于事后筛选，不能泄漏进本步输入。"""
    if (
        record.get('purpose') != 'teacher_coldstart_source'
        or record.get('origin') != 'public_policy'
        or record.get('role') != 'train'
        or record.get('source_contract_digest') != contract_digest
        or record.get('eligible_for_training') is not True
        or record.get('result', {}).get('success') is not True
        or record.get('record_hash')
        != digest({k: v for k, v in record.items() if k != 'record_hash'})
    ):
        raise ValueError('只接纳新合同下成功训练源，旧诊断/开发/特权记录不可训练')
    samples = []
    for index, step in enumerate(record['trace']):
        before, after = public_view(step['before']), public_view(step['after'])
        choices = set(before['admissible_commands']) | {'look', 'inventory'}
        feedback = after['feedback'].strip()
        if (
            step['action'] not in choices
            or feedback == 'Nothing happens.'
            or feedback.startswith('Action rejected:')
        ):
            continue
        prepared = StandardActor().prepare(
            record['initial'], before, record['trace'][:index], tokenize
        )
        samples.append(
            {
                'messages': prepared['messages']
                + [{'role': 'assistant', 'content': step['action']}],
                'task_id': record['task_id'],
                'step': index + 1,
                'source_record_hash': record['record_hash'],
                'source_contract_digest': contract_digest,
                'context': prepared['context'],
            }
        )
    return samples
