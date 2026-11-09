"""N1公开流程研究：确定性脚本投影、固定状态诊断和阶段一监督。"""

import random

from .task_protocol import canonical, digest, public_view
from .workflow import WorkflowActor, assess_workflow
from .workflow_memory import memory_entry, retrieve


def procedure_script(entry):
    """将已验证的公开规则写成操作脚本；不声称模型自主抽象。"""
    operations = entry['workflow']['operations']
    actions = {
        'clean': 'Visit a sinkbasin while holding the target; execute the available clean command.',
        'heat': 'Visit a microwave while holding the target; open it if required; execute the available heat command.',
        'cool': 'Visit a fridge while holding the target; open it if required; execute the available cool command.',
        'inventory': 'While holding the target, execute inventory and check the public feedback names it.',
    }
    return {
        'builder': 'deterministic_public_rule_projection_v1',
        'procedure': entry['workflow']['name'],
        'revision': entry['workflow']['revision'],
        'operations': operations,
        'steps': ['Find and take the object required by the current goal.']
        + [actions[op] for op in operations]
        + [
            'Only after every operation succeeds in order, place that same object at the current goal destination.'
        ],
        'reuse': 'Use current objects and executable commands. Never copy instance numbers from another episode.',
    }


def retrieve_procedural(
    entries, goal, workflow, *, cutoff, tokenize, representation, token_budget=1024
):
    if representation not in {'script', 'combined'}:
        raise ValueError('只允许固定脚本或完整轨迹加脚本')
    if type(token_budget) is not int or token_budget < len(tokenize(canonical([]))):
        raise ValueError('记忆预算不足')
    # 共用已验证的版本/时间/BM25排序；表示转换后重新核真实token预算。
    ranked = retrieve(
        entries,
        goal,
        workflow,
        cutoff=cutoff,
        tokenize=tokenize,
        token_budget=1_000_000,
        top_k=max(1, len(entries)),
    )
    by_id = {row['source_task_id']: row for row in entries}
    result = {
        **ranked,
        'policy': 'workflow_bm25_whole_' + representation + '_v1',
        'representation': representation,
        'prompt': [],
        'selected': [],
        'budget_skipped': 0,
        'token_budget': token_budget,
        'prompt_tokens': len(tokenize(canonical([]))),
    }
    for receipt in ranked['selected']:
        entry = by_id[receipt['source_task_id']]
        item = procedure_script(entry)
        if representation == 'combined':
            item = {
                **item,
                'example_goal': entry['public']['goal'],
                'initial_observation': entry['public']['initial_observation'],
                'trajectory': entry['public']['trajectory'],
            }
        size = len(tokenize(canonical([item])))
        if size > token_budget:
            result['budget_skipped'] += 1
            continue
        result.update(prompt=[item], selected=[receipt], prompt_tokens=size)
        break
    return result


def decision_state(record):
    """按首次额外操作冻结公开前缀；真值和未来反馈不进入public对象。"""
    if (
        record.get('role') != 'dev'
        or record.get('condition') != 'workflow_teacher'
        or record.get('record_hash')
        != digest({k: v for k, v in record.items() if k != 'record_hash'})
    ):
        raise ValueError('仅允许完整且未改写的教师开发诊断源')
    assessment = assess_workflow(
        record['trace'], record['workflow'], record['result']['base_success']
    )
    if not assessment['success'] or any(
        record['result'].get(k) != v for k, v in assessment.items()
    ):
        raise ValueError('诊断源流程未通过独立判定')
    extra = next(
        op for op in record['workflow']['operations'] if op != record['workflow']['thermal']
    )
    for index, row in enumerate(record['trace']):
        if row['action'].split()[0] != extra:
            continue
        prefix = [
            {
                'action': r['action'],
                'before': public_view(r['before']),
                'after': public_view(r['after']),
            }
            for r in record['trace'][:index]
        ]
        public = {
            'initial': public_view(record['initial']),
            'view': public_view(row['before']),
            'trace': prefix,
        }
        return {
            'task_id': record['task_id'],
            'source_record_hash': record['record_hash'],
            'step': index + 1,
            'public': public,
            'public_digest': digest(public),
            'expected_action': row['action'],
        }
    raise ValueError('没有额外操作决策点')


def workflow_samples(record, task, contract_digest, data_digest, tokenize, max_input=3500):
    """只编译第一阶段的成功训练经历，目标动作的反馈不能进入输入。"""
    if task.get('phase') != 1 or record.get('phase') != 1:
        raise ValueError('C1只允许阶段一，禁止未来阶段混入')
    memory_entry(record, task, contract_digest, data_digest)
    rows = []
    actor = WorkflowActor(task['workflow'])
    for index, row in enumerate(record['trace']):
        before, after = public_view(row['before']), public_view(row['after'])
        if (
            row['action'] not in set(before['admissible_commands']) | {'look', 'inventory'}
            or after['feedback'].strip() == 'Nothing happens.'
            or after['feedback'].startswith('Action rejected:')
        ):
            continue
        prepared = actor.prepare(
            record['initial'], before, record['trace'][:index], tokenize, max_input=max_input
        )
        rows.append(
            {
                'messages': prepared['messages']
                + [{'role': 'assistant', 'content': row['action']}],
                'task_id': task['task_id'],
                'step': index + 1,
                'cohort': 'workflow_stage1',
                'source_record_hash': record['record_hash'],
                'source_contract_digest': contract_digest,
                'context': prepared['context'],
            }
        )
    return rows


def replay_indices(supervised_counts, target_tokens, seed):
    """固定顺序循环取旧样本；按监督token接近1:1，保留重复次数。"""
    if (
        not supervised_counts
        or any(type(n) is not int or n <= 0 for n in supervised_counts)
        or target_tokens <= 0
    ):
        raise ValueError('监督token计数必须为正整数')
    order = list(range(len(supervised_counts)))
    random.Random(seed).shuffle(order)
    result, count = [], 0
    while count < target_tokens:
        index = order[len(result) % len(order)]
        result.append(index)
        count += supervised_counts[index]
    return result
