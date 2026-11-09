"""从已核验公开教师轨迹构建流程记忆，按时间、当前版本和预算检索。"""

from collections import Counter
import math
import re

from .memory import tokens
from .task_protocol import canonical, digest
from .workflow import assess_workflow, workflow_spec


def memory_entry(record, task, source_contract_digest, data_manifest_sha256):
    """调用方先审计整份CPU采集，本函数继续约束单条来源身份。"""
    if (
        record.get('record_hash')
        != digest({key: value for key, value in record.items() if key != 'record_hash'})
        or record.get('source_contract_digest') != source_contract_digest
        or record.get('purpose') != 'workflow_teacher_source'
        or record.get('origin') != 'public_policy'
        or record.get('condition') != 'workflow_teacher'
        or record.get('role') != 'train'
        or task.get('role') != 'train'
        or record.get('eligible_for_training') is not True
    ):
        raise ValueError('流程记忆仅接纳本合同的成功训练来源')
    for key in ('task_id', 'role', 'sequence', 'phase', 'workflow'):
        if record[key] != task[key]:
            raise ValueError('记忆记录与任务清单身份不符')
    assessment = assess_workflow(
        record['trace'], task['workflow'], record['result']['base_success']
    )
    if not assessment['success'] or any(
        record['result'].get(key) != value for key, value in assessment.items()
    ):
        raise ValueError('记忆来源没有通过独立流程重算')
    goal = re.search(r'Your task is to:\s*([^\n]+)', record['initial']['feedback'])
    if not goal:
        raise ValueError('记忆来源缺公开目标')
    previous = record['initial']['feedback']
    for row in record['trace']:
        if row['before']['feedback'] != previous:
            raise ValueError('公开反馈序列的因果前缀不连续，不能紧凑编码')
        previous = row['after']['feedback']
    # 保留完整动作与feedback序列，相邻重复反馈只存一次；旧admissible_commands不进入记忆。
    public = {
        'goal': goal[1].strip(),
        'workflow': task['workflow'],
        'initial_observation': record['initial']['feedback'],
        'trajectory': [
            {'action': row['action'], 'feedback': row['after']['feedback']}
            for row in record['trace']
        ],
    }
    entry = {
        'source_task_id': task['task_id'],
        'record_hash': record['record_hash'],
        'source_contract_digest': source_contract_digest,
        'data_manifest_sha256': data_manifest_sha256,
        'sequence': task['sequence'],
        'phase': task['phase'],
        'workflow': task['workflow'],
        'public': public,
    }
    entry['entry_digest'] = digest(entry)
    return entry


def retrieve(entries, goal, workflow, *, cutoff, tokenize, token_budget=1024, top_k=3):
    """元数据留在回执，模型只收到不同实例的公开经验；经验按整条打包。"""
    variant = workflow['name'] + '-v' + str(workflow['revision'])
    if workflow != workflow_spec(variant, workflow['thermal']):
        raise ValueError('当前公开流程不是冻结版本')
    if (
        type(cutoff) is not int
        or cutoff < 1
        or type(token_budget) is not int
        or token_budget < 1
        or type(top_k) is not int
        or top_k < 1
    ):
        raise ValueError('记忆时间截断或预算无效')
    if len(tokenize(canonical([]))) > token_budget:
        raise ValueError('预算连空记忆的序列化开销也无法容纳')
    eligible, seen = [], set()
    for entry in entries:
        if (
            entry.get('entry_digest')
            != digest({key: value for key, value in entry.items() if key != 'entry_digest'})
            or type(entry.get('sequence')) is not int
            or entry['sequence'] < 1
            or entry['source_task_id'] in seen
        ):
            raise ValueError('记忆身份重复或原件摘要变化')
        seen.add(entry['source_task_id'])
        if entry['sequence'] < cutoff and entry['workflow'] == workflow:
            eligible.append(entry)
    query = set(tokens(goal))
    documents = [Counter(tokens(entry['public']['goal'])) for entry in eligible]
    frequency = Counter(word for document in documents for word in document)
    mean_length = sum(sum(document.values()) for document in documents) / max(1, len(documents))
    ranked = []
    for entry, document in zip(eligible, documents):
        score = 0.0
        for word in sorted(query & document.keys()):
            tf = document[word]
            idf = math.log(1 + (len(documents) - frequency[word] + 0.5) / (frequency[word] + 0.5))
            score += (
                idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(document.values()) / mean_length))
            )
        ranked.append((score, entry))
    ranked.sort(key=lambda item: (-item[0], -item[1]['sequence'], item[1]['source_task_id']))
    prompt, selected, skipped = [], [], 0
    for score, entry in ranked:
        candidate = prompt + [entry['public']]
        if len(tokenize(canonical(candidate))) > token_budget:
            skipped += 1
            continue
        prompt = candidate
        selected.append(
            {
                'source_task_id': entry['source_task_id'],
                'record_hash': entry['record_hash'],
                'entry_digest': entry['entry_digest'],
                'score': score,
            }
        )
        if len(selected) == top_k:
            break
    actual_tokens = len(tokenize(canonical(prompt)))
    if prompt and actual_tokens > token_budget:
        raise ValueError('实际记忆token超过冻结预算')
    return {
        'policy': 'workflow_bm25_whole_trajectory_v1',
        'cutoff_sequence': cutoff,
        'workflow_digest': digest(workflow),
        'eligible_count': len(eligible),
        'budget_skipped': skipped,
        'token_budget': token_budget,
        'prompt_tokens': actual_tokens,
        'selected': selected,
        'prompt': prompt,
    }
