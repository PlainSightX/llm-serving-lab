"""防止隐藏信息、坏动作重试和不合格历史进入研究。"""

import pytest
from serving_lab import public_tasks as p


class Environment:
    def __init__(self):
        self.calls = []

    def reset(self):
        return {
            'feedback': 'Your task is to: put a mug in/on shelf.',
            'admissible_commands': ['look'],
            'facts': ['secret'],
            'won': False,
        }

    def step(self, action):
        self.calls.append(action)
        return {'feedback': 'done', 'admissible_commands': [], 'won': True}, 1, True

    def close(self):
        pass


def test_private_fields_never_leave_and_bad_action_consumes_budget():
    env = Environment()
    ep = p.PublicEpisode(env, 2)
    assert set(ep.reset()) == {'feedback', 'admissible_commands'}
    ep.step('SECRET')
    ep.step('look\nlook')
    assert env.calls == [] and ep.result()['steps'] == 2 and not ep.result()['success']
    with pytest.raises(ValueError):
        ep.step('look')


def test_goal_policy_uses_only_public_observation():
    policy = p.PublicPolicy()
    assert (
        policy.choose(
            {
                'feedback': 'Your task is to: put a mug in/on shelf.',
                'admissible_commands': ['go to shelf 1', 'go to table 1'],
            }
        )
        == 'go to shelf 1'
    )
    assert (
        policy.choose(
            {
                'feedback': 'On shelf 1, you see a mug 2.',
                'admissible_commands': ['take mug 2 from shelf 1', 'go to table 1'],
            }
        )
        == 'take mug 2 from shelf 1'
    )
    assert (
        policy.choose(
            {
                'feedback': 'You pick up the mug 2 from shelf 1.',
                'admissible_commands': ['put mug 2 in/on shelf 1'],
            }
        )
        == 'put mug 2 in/on shelf 1'
    )


@pytest.mark.parametrize(
    'goal,obj,target,process',
    [
        ('put some candle on toilet.', 'candle', 'toilet', None),
        ('examine the pillow with the desklamp.', 'pillow', None, None),
        ('clean some potato and put it in diningtable.', 'potato', 'diningtable', 'clean'),
        ('heat some mug and put it in shelf.', 'mug', 'shelf', 'heat'),
        ('put a cool mug in coffeemachine.', 'mug', 'coffeemachine', 'cool'),
        ('find two soapbottle and put them in cabinet.', 'soapbottle', 'cabinet', None),
    ],
)
def test_standard_public_goal_grammar(goal, obj, target, process):
    p0 = p.PublicPolicy()
    p0.choose({'feedback': 'Your task is to: ' + goal, 'admissible_commands': ['look']})
    assert (p0.object, p0.target, p0.process) == (obj, target, process)


def test_policy_uses_textworld_move_command_after_pickup():
    actor = p.PublicPolicy()
    actor.choose(
        {
            'feedback': 'Your task is to: put some candle on toilet.',
            'admissible_commands': ['take candle 1 from countertop 1'],
        }
    )
    assert (
        actor.choose(
            {
                'feedback': 'You pick up the candle 1 from the countertop 1.',
                'admissible_commands': ['move candle 1 to toilet 1'],
            }
        )
        == 'move candle 1 to toilet 1'
    )


def test_context_retains_current_view_and_drops_only_old_pairs():
    # 字符计数替代昂贵外部tokenizer，只验证真实裁剪和无法容纳时拒绝的分支。
    count = lambda messages: list(range(sum(len(m['content']) for m in messages)))
    view = {'feedback': 'current', 'admissible_commands': ['look']}
    messages, ids, meta = p.action_context(
        'goal', view, [{'action': 'old', 'feedback': 'x' * 2000}], None, count, 1500
    )
    assert meta['dropped_steps'] == 1 and len(ids) <= 1500 and 'current' in messages[-1]['content']
    with pytest.raises(ValueError):
        p.action_context('goal', view, [], None, count, 1)


def test_history_requires_all_real_action_receipts_and_terminal_success():
    record = {
        'origin': 'model_public',
        'task_id': 's',
        'role': 'support',
        'sequence': 1,
        'initial': {'feedback': 'g', 'admissible_commands': ['look']},
        'result': {'success': True},
        'trace': [
            {'action': 'look', 'after': {'feedback': 'ok'}, 'receipt': {'protocol_valid': False}}
        ],
    }
    with pytest.raises(ValueError):
        p.verified_history(record, cutoff=7)
    record['origin'] = 'privileged_reference'
    with pytest.raises(ValueError):
        p.verified_history(record, cutoff=7)


def test_research_profile_does_not_silently_use_greedy():
    from serving_lab.profile import ServingProfile

    profile = ServingProfile(
        {
            'max_tokens': 64,
            'context_tokens': 4096,
            'decoding': {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 20,
                'min_p': 0.0,
                'seed': 20261103,
            },
        },
        'm',
    )
    wire = profile.payload({'model': 'm', 'messages': [], 'max_tokens': 64, 'stream': False})
    assert (wire['temperature'], wire['top_p'], wire['top_k'], wire['seed']) == (
        0.7,
        0.8,
        20,
        20261103,
    )
    with pytest.raises(ValueError):
        profile.payload({'model': 'm', 'messages': [], 'temperature': 0})
