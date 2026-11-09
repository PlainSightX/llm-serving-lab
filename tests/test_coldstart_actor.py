"""固定示范是训练来源的显式输入；验证不允许用开发回合补示范。"""

import importlib
import importlib.util
import json
import pytest


def test_fixed_demonstrations_are_in_actual_context_and_share_full_command_set():
    assert importlib.util.find_spec('serving_lab.coldstart_actor') is not None
    cls = importlib.import_module('serving_lab.coldstart_actor').ColdStartActor
    initial = {
        'feedback': 'Your task is to: put a mug on table.',
        'admissible_commands': ['go to table 1'],
    }
    tokenizer = lambda messages: list(range(sum(len(m['content']) for m in messages)))
    actor = cls('FIXED_TRAIN_EXAMPLE')
    out = actor.prepare(initial, initial, [], tokenizer)
    assert 'FIXED_TRAIN_EXAMPLE' in json.dumps(out['messages'])
    assert out['choices'] == ['go to table 1', 'inventory', 'look']
    assert out['context']['input_tokens'] == len(out['input_token_ids'])
    with pytest.raises(ValueError):
        cls('x' * 4000).prepare(initial, initial, [], tokenizer)


def test_rehashed_demo_text_cannot_substitute_dev_content_under_train_ids():
    import sys
    from pathlib import Path
    from serving_lab.task_protocol import digest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import q2_coldstart_eval as q

    assert hasattr(q, 'validate_examples')
    tasks = [
        {'task_id': 't1', 'role': 'train', 'family': 'f1'},
        {'task_id': 't2', 'role': 'train', 'family': 'f2'},
        {'task_id': 'd1', 'role': 'dev', 'family': 'f3'},
    ]
    records = {
        t['task_id']: {
            'task_id': t['task_id'],
            'initial': {'feedback': 'PUBLIC ' + t['task_id']},
            'trace': [{'action': 'look', 'after': {'feedback': 'REAL'}}],
            'result': {'success': True},
        }
        for t in tasks
    }
    for r in records.values():
        r['record_hash'] = digest(r)
    text, sources = q.select_examples(tasks, records, len)
    legitimate = {'examples_text': text, 'sources': sources, 'examples_digest': digest(text)}
    q.validate_examples(legitimate, tasks, records, len)
    contaminated = {
        **legitimate,
        'examples_text': 'DEV_ANSWER',
        'examples_digest': digest('DEV_ANSWER'),
    }
    with pytest.raises(ValueError):
        q.validate_examples(contaminated, tasks, records, len)
