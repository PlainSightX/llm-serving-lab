"""公开命令选择actor：记录观察，不替模型制定计划或判断任务完成。"""

import re

from .task_protocol import canonical, digest, public_view


INSTRUCTION = (
    'Complete the exact household goal. Select one command from CURRENT_COMMANDS. '
    'Keep searching for the goal object when it is not visible; use past observations to avoid searching the same places without a reason. '
    'Objects unrelated to the goal do not need to be picked up. Previously seen options may be stale; only CURRENT_COMMANDS are executable. '
    'Tool effects: go to visits; take picks up; move places a held object; open opens a container; '
    'clean with a sinkbasin cleans a held object; heat with a microwave heats it; cool with a fridge cools it; '
    'use a lamp examines the held object by its light; examine describes an object. '
    'Return only the chosen command. Do not generate observations, feedback or explanations.'
)


def public_working_state(trace):
    """只采公开反馈及曾见命令；没有held/已完成等未经证实的隐状态。"""
    visited, seen, interactions = {}, {}, {}
    for i, row in enumerate(trace, 1):
        before, after = public_view(row['before']), public_view(row['after'])
        for step, observation in ((i - 1, before), (i, after)):
            for command in observation['admissible_commands']:
                if command.startswith('take '):
                    seen[command] = {'step': step, 'command': command}
        action, feedback = row['action'], after['feedback']
        if not isinstance(action, str):
            continue
        if action.startswith('go to ') and feedback.startswith('You arrive at ' + action[6:] + '.'):
            visited[action[6:]] = {'place': action[6:], 'step': i}
        if not action.startswith('go to ') and action not in {'look', 'inventory'}:
            # 反馈逐字保留，包括Nothing happens；不能把选择过动作写成成功完成。
            interactions[action] = {'step': i, 'action': action, 'feedback': feedback}
    return {
        'visited_places': sorted(visited.values(), key=lambda x: x['step']),
        'previously_seen_take_options': sorted(
            seen.values(), key=lambda x: (x['step'], x['command'])
        ),
        'interaction_evidence': sorted(interactions.values(), key=lambda x: x['step']),
    }


class StandardActor:
    def prepare(self, initial, view, trace, tokenize, max_input=3500):
        initial, view = public_view(initial), public_view(view)
        goal = re.search(r'Your task is to:\s*([^\n]+)', initial['feedback'])
        if not goal:
            raise ValueError('初始公开观察缺少目标')
        goal = goal[1].strip()
        choices = sorted(set(view['admissible_commands']) | {'look', 'inventory'})
        state = public_working_state(trace)
        state_digest = digest(state)
        original_counts = {key: len(rows) for key, rows in state.items()}
        history = [
            {'step': i, 'action': row['action'], 'feedback': public_view(row['after'])['feedback']}
            for i, row in enumerate(trace, 1)
        ][-8:]
        while True:
            document = {
                'GOAL': goal,
                'INITIAL_OBSERVATION': initial['feedback'],
                'PAST_PUBLIC_EVIDENCE_MAY_BE_STALE': state,
                'RECENT_ACTION_FEEDBACK': history,
                'CURRENT_FEEDBACK': view['feedback'],
                'CURRENT_COMMANDS': choices,
            }
            messages = [
                {'role': 'system', 'content': INSTRUCTION},
                {'role': 'user', 'content': canonical(document)},
            ]
            ids = tokenize(messages)
            if len(ids) <= max_input:
                break
            if len(history) > 2:
                history.pop(0)
            else:
                # 完整删除最老的状态条目；不截断事实字符串，也不移除目标/当前选择集。
                available = [(rows[0]['step'], key) for key, rows in state.items() if rows]
                if available:
                    state[min(available)[1]].pop(0)
                elif history:
                    history.pop(0)
                else:
                    raise ValueError('目标与当前公开观察无法容纳，禁止派发')
        context = {
            'actor': 'public_choice_working_state_v1',
            'initial_public_digest': digest(initial),
            'current_public_digest': digest(view),
            'choices_digest': digest(choices),
            'full_working_state_digest': state_digest,
            'dropped_state_entries': {
                key: original_counts[key] - len(rows) for key, rows in state.items()
            },
            'dropped_history_steps': len(trace) - len(history),
            'input_tokens': len(ids),
            'max_input_tokens': max_input,
        }
        return {
            'messages': messages,
            'input_token_ids': ids,
            'context': context,
            'choices': choices,
        }

    def action(self, receipt, choices):
        content = receipt['content']
        if receipt['finish_reason'] != 'stop' or content not in choices:
            raise ValueError('约束输出未完整匹配当前动作集合；停止，不猜测或重试')
        return content
