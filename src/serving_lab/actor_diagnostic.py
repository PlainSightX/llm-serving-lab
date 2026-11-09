"""只改变system提示的配对动作诊断；标签和来源路径不进入模型。"""

import json
from .public_tasks import action_context
from .task_protocol import TOOL_CONTRACT, canonical, digest

GOAL_ACTION_CONTRACT = (
    'Your objective is to complete the household task stated in the initial observation. '
    'Choose the next executable action that advances that goal, using the current public observation and prior actions and feedback. '
    'Use object names and numbers exactly as shown. Prefer an interaction that advances the task when the needed information is already available. '
    'Return exactly one command from the current available commands, or look or inventory when more observation is needed. '
    'Return only the command: no explanation, quotation marks, code fences, or multiple actions.'
)
INSTRUCTIONS = {'legacy': TOOL_CONTRACT, 'goal_action': GOAL_ACTION_CONTRACT}

TOOL_JSON_CONTRACT = (
    'Complete the household goal stated in the initial observation. Select your next action using the current available commands and prior feedback. '
    'Tool effects: go to visits a location; take picks up an object; move places a held object and does not clean, heat, or cool it; '
    'clean with a sinkbasin cleans a held object; heat with a microwave heats it; cool with a fridge cools it; '
    'use a lamp turns the light on to examine the held object; examine only describes an object; open opens a container. '
    'Use an action when it is available; do not invent extra prerequisites. Match the object numbers shown. '
    'Return one JSON string whose decoded value is exactly one available command. No other text, no list, no object, no explanation.'
)


def parse_json_action(content):
    try:
        action = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(action, str):
        return None
    action = action.strip()
    return (
        action
        if action and len(action) <= 256 and '\n' not in action and '\r' not in action
        else None
    )


def score_json_action(content, case):
    action = parse_json_action(content)
    score = score_action(action or '', case)
    score['json_contract_valid'] = action is not None
    return score


def parse_action_envelope(content):
    """仅解码唯一动作；模型生成的feedback不具有环境事实地位。"""

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out

    try:
        value = json.loads(content, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if isinstance(value, dict):
        keys = set(value) & {'action', 'command'}
        if (
            len(keys) != 1
            or set(value) - keys - {'feedback'}
            or ('feedback' in value and not isinstance(value['feedback'], str))
        ):
            return None
        value = value[next(iter(keys))]
    if not isinstance(value, str):
        return None
    return parse_json_action(json.dumps(value))


def score_envelope(content, case):
    action = parse_action_envelope(content)
    score = score_action(action or '', case)
    score['envelope_valid'] = action is not None
    return score


def paired_contexts(case, tokenize, max_input=3000):
    """共同裁剪完整的旧动作/反馈对；两条件仅system提示不同。"""
    drop = 0
    prefix = case['prefix']
    while True:
        result = {}
        for condition, instruction in INSTRUCTIONS.items():
            messages, ids, meta = action_context(
                case['initial']['feedback'],
                case['view'],
                prefix[drop:],
                None,
                tokenize,
                max_input,
                instruction=instruction,
            )
            result[condition] = {'messages': messages, 'input_token_ids': ids, 'context': meta}
        further = max(r['context']['dropped_steps'] for r in result.values())
        if not further:
            break
        drop += further
    assert result['legacy']['messages'][1:] == result['goal_action']['messages'][1:]
    for r in result.values():
        r['context'].update(
            common_dropped_steps=drop,
            prefix_origin='public_policy_diagnostic',
            source_view_hash=digest(case['view']),
        )
    return result


def score_action(content, case):
    action = content.strip()
    valid = bool(action) and len(action) <= 256 and '\n' not in action and '\r' not in action
    legal = valid and (
        action in case['view']['admissible_commands'] or action in {'look', 'inventory'}
    )
    accepted = (
        None
        if case['acceptable_actions'] is None
        else legal and action in case['acceptable_actions']
    )
    return {
        'action': action,
        'format_valid': valid,
        'admissible': legal,
        'acceptable_key_action': accepted,
        'exploration_action': bool(legal and action.startswith('go to '))
        if case['kind'] == 'initial'
        else None,
    }


def closed_loop_gate(rows):
    candidates = [r for r in rows if r['condition'] == 'goal_action']
    if len(candidates) != 36:
        return False
    return sum(r['score']['admissible'] for r in candidates) >= 33 and all(
        any(
            r['state_id'] == sid and r['score']['acceptable_key_action'] is True for r in candidates
        )
        for sid in ('s01-key_action', 's03-key_action')
    )
