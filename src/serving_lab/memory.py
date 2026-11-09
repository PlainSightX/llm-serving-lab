"""可审计的初始经历检索基线：只读过去已验证训练经历，不访问隐藏面板标签。"""

from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from .lab import LabError, canonical, check_output, digest

SOURCE_BYTES = Path(__file__).read_bytes()
POLICIES = {'none', 'episodic_v1'}
TOP_K = 3
MAX_CHARS = 2600


def public_summary(task):
    """从公开文本抽取输入概要；不读取evaluation、family或因素标签。"""
    try:
        case = json.loads(task['messages'][-1]['content'])
        return {
            'request': case['request'],
            'rule': case['rule'],
            'runs': [
                {'id': r['id'], 'unit': r['unit'], 'times': [p['time'] for p in r['rows']]}
                for r in case['runs']
            ],
            'expected_count': len(case['expected_times']),
            'observation_times': [p['time'] for p in case['observations']],
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise LabError('经历基线只支持已声明的业务公开输入') from exc


def tokens(value):
    return re.findall(r'[a-z_]+|[\u4e00-\u9fff]|\d+', canonical(value).lower())


def build_context(lab, task_id, policy):
    if policy not in POLICIES:
        raise LabError('未知记忆策略')
    task = lab.task(task_id)
    result = {
        'policy': policy,
        'cutoff_sequence': task['sequence'] - 1,
        'top_k': TOP_K,
        'max_chars': MAX_CHARS,
        'entries': [],
    }
    if policy == 'none':
        return result
    query = set(tokens(public_summary(task)))
    candidates = []
    rows = lab.db.execute(
        '''SELECT a.*,t.seq,t.split,f.body feedback,e.receipt_hash
      FROM attempts a JOIN tasks t ON t.id=a.task_id JOIN feedback f ON f.attempt_id=a.id
      JOIN executions e ON e.attempt_id=a.id
      WHERE t.split='train' AND t.seq<? AND a.actor='local_hf' AND e.state='completed'
      ORDER BY t.seq,a.created,a.id''',
        (task['sequence'],),
    ).fetchall()
    seen = set()
    for row in rows:
        past = lab.task(row['task_id'])
        feedback = json.loads(row['feedback'])
        output = json.loads(row['output'])
        if (
            past['rule_version'] != task['rule_version']
            or not feedback['passed']
            or feedback['available_sequence'] >= task['sequence']
            or row['task_id'] in seen
        ):
            continue
        checked = check_output(past, output)
        if {k: v for k, v in feedback.items() if k != 'available_sequence'} != checked:
            raise LabError('经历反馈与来源不符')
        seen.add(row['task_id'])
        example = {'input_summary': public_summary(past), 'verified_plan': output}
        candidates.append(
            {
                'attempt_id': row['id'],
                'task_id': row['task_id'],
                'task_hash': digest(past),
                'output_hash': digest(output),
                'feedback_hash': digest(feedback),
                'receipt_hash': row['receipt_hash'],
                'sequence': past['sequence'],
                'example': example,
            }
        )
    if not candidates:
        return result
    documents = [collections.Counter(tokens(c['example']['input_summary'])) for c in candidates]
    mean_length = sum(sum(d.values()) for d in documents) / len(documents)
    frequency = collections.Counter(word for d in documents for word in d)
    ranked = []
    for candidate, document in zip(candidates, documents):
        score = 0.0
        length = sum(document.values())
        for word in sorted(query & document.keys()):
            tf = document[word]
            idf = math.log(1 + (len(documents) - frequency[word] + 0.5) / (frequency[word] + 0.5))
            score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / mean_length))
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: (-item[0], -item[1]['sequence'], item[1]['attempt_id']))
    used = 0
    cache = {}
    from .execution import verify_execution

    for score, entry in ranked:
        size = len(canonical(entry['example']))
        if used + size > MAX_CHARS:
            continue
        evidence = verify_execution(lab, entry['attempt_id'], _cache=cache)
        if not all(evidence[k] for k in ('complete', 'output_complete', 'json_valid')):
            raise LabError('经历缺少完整真实执行')
        result['entries'].append({**entry, 'retrieval_score': score})
        used += size
        if len(result['entries']) == TOP_K:
            break
    return result


def input_messages(request):
    messages = request['task']['messages']
    entries = request.get('context', {}).get('entries', [])
    if not entries:
        return messages
    history = {
        'role': 'user',
        'content': '以下为过去已验证案例的公开输入概要与原始计划。ID仅属于旧案例；本次必须使用当前请求ID和当前规则。\n'
        + canonical([e['example'] for e in entries]),
    }
    return [messages[0], history, *messages[1:]]


def verify_context(lab, request, cache):
    context = request['context']
    task = lab.task(request['task']['task_id'])
    if (
        context['policy'] not in POLICIES
        or context['cutoff_sequence'] != task['sequence'] - 1
        or context['top_k'] != TOP_K
        or context['max_chars'] != MAX_CHARS
        or len(context['entries']) > TOP_K
        or (context['policy'] == 'none' and context['entries'])
    ):
        raise LabError('记忆合同不一致')
    seen = set()
    size = 0
    for entry in context['entries']:
        if entry['attempt_id'] in seen:
            raise LabError('重复经历')
        seen.add(entry['attempt_id'])
        row = lab.db.execute(
            '''SELECT a.*,f.body feedback,e.receipt_hash FROM attempts a
          JOIN feedback f ON f.attempt_id=a.id JOIN executions e ON e.attempt_id=a.id WHERE a.id=?''',
            (entry['attempt_id'],),
        ).fetchone()
        if not row:
            raise LabError('经历来源不存在')
        past = lab.task(row['task_id'])
        feedback = json.loads(row['feedback'])
        output = json.loads(row['output'])
        if (
            past['split'] != 'train'
            or past['sequence'] >= task['sequence']
            or feedback['available_sequence'] >= task['sequence']
            or not feedback['passed']
            or past['rule_version'] != task['rule_version']
            or row['arm'] != request['arm']
            or row['actor'] != 'local_hf'
            or entry['task_id'] != row['task_id']
            or entry['sequence'] != past['sequence']
            or entry['task_hash'] != digest(past)
            or entry['output_hash'] != digest(output)
            or entry['feedback_hash'] != digest(feedback)
            or entry['receipt_hash'] != row['receipt_hash']
            or entry['example'] != {'input_summary': public_summary(past), 'verified_plan': output}
        ):
            raise LabError('记忆来源/时间/切分或字节不一致')
        if {k: v for k, v in feedback.items() if k != 'available_sequence'} != check_output(
            past, output
        ):
            raise LabError('记忆反馈已变化')
        from .execution import verify_execution

        evidence = verify_execution(lab, row['id'], _cache=cache)
        if not all(evidence[k] for k in ('complete', 'output_complete', 'json_valid')):
            raise LabError('记忆执行未完成')
        size += len(canonical(entry['example']))
    if size > MAX_CHARS:
        raise LabError('记忆超出固定预算')
