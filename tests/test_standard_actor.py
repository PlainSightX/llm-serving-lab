"""反例覆盖候选约束、公开状态隔离与真正消费的预算边界。"""

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serving_lab.profile import make_profile


def profile(tmp_path, policy=None):
    model = tmp_path / 'model'
    model.mkdir(exist_ok=True)
    (model / 'config.json').write_text('{}')
    (tmp_path / 'launch.json').write_text(
        json.dumps(
            {
                'command': ['--generation-config', 'vllm'],
                'versions': {'vllm': '0.11.2'},
                'config': {'dtype': 'bfloat16', 'max_model_len': 4096},
            }
        )
    )
    binding = SimpleNamespace(
        manifest_path=tmp_path / 'binding.json',
        model_path=model,
        artifact_digest='artifact',
        served_model='m',
    )
    tokenizer = SimpleNamespace(get_chat_template=lambda: 'template', eos_token_id=1)
    return make_profile(binding, tokenizer, policy)


def actor_module():
    assert importlib.util.find_spec('serving_lab.standard_actor') is not None
    return importlib.import_module('serving_lab.standard_actor')


def view(feedback, commands=None, **hidden):
    return {'feedback': feedback, 'admissible_commands': commands or ['look'], **hidden}


def test_choice_is_explicit_new_profile_and_wire_cannot_override_decoding(tmp_path):
    p = profile(tmp_path, {'name': 'q2_public_choice_v1', 'seed': 20261103})
    body = {
        'messages': [],
        'max_tokens': 64,
        'stream': False,
        'action_choices': ['inventory', 'look'],
    }
    wire = p.payload(body)
    assert wire['structured_outputs'] == {'choice': ['inventory', 'look']}
    assert 'action_choices' not in wire and wire['seed'] == 20261103
    assert body['action_choices'] == ['inventory', 'look']
    for addition in ({'temperature': 0}, {'seed': 99}, {'structured_outputs': {'regex': '.*'}}):
        with pytest.raises(ValueError):
            p.payload({**body, **addition})
    with pytest.raises(ValueError):
        profile(tmp_path).payload(body)
    for choices in ([], ['look', 'look'], ['look\ninventory'], [None], ['x' * 257]):
        with pytest.raises(ValueError):
            p.payload({**body, 'action_choices': choices})


def test_public_state_keeps_real_evidence_and_does_not_turn_failure_into_success():
    a = actor_module()
    trace = [
        {
            'step': 1,
            'before': view('start'),
            'action': 'go to cabinet 1',
            'after': view('Nothing happens.', ['go to cabinet 1']),
            'receipt': {'content': '{"feedback":"SECRET success"}'},
        },
        {
            'step': 2,
            'before': view('before'),
            'action': 'go to countertop 1',
            'after': view(
                'You arrive at countertop 1. You see a candle 1.',
                ['take candle 1 from countertop 1', 'take soapbar 1 from countertop 1'],
            ),
        },
        {
            'step': 3,
            'before': view('before'),
            'action': 'take candle 1 from countertop 1',
            'after': view('Nothing happens.', facts='SECRET facts'),
        },
    ]
    state = a.public_working_state(trace)
    assert state['visited_places'] == [{'place': 'countertop 1', 'step': 2}]
    assert {x['command'] for x in state['previously_seen_take_options']} == {
        'take candle 1 from countertop 1',
        'take soapbar 1 from countertop 1',
    }
    assert state['interaction_evidence'][-1] == {
        'step': 3,
        'action': 'take candle 1 from countertop 1',
        'feedback': 'Nothing happens.',
    }
    assert 'SECRET' not in json.dumps(state) and 'held' not in state


def test_context_preserves_goal_and_all_commands_under_truncation_without_labels():
    a = actor_module()
    initial = view('Room.\nYour task is to: put some candle on toilet.', facts='SECRET facts')
    commands = ['move soapbar 1 to toilet 1', 'go to cabinet 1', 'look']
    current = view('You see soapbar 1.', commands, expert_plan='SECRET answer')
    trace = [
        {
            'step': i,
            'before': view('x'),
            'action': 'examine soapbar 1',
            'after': view('Seen ' + 'x' * 400),
            'receipt': {'content': 'SECRET generated feedback'},
        }
        for i in range(1, 20)
    ]
    token = lambda messages: list(range(sum(len(m['content']) for m in messages)))
    result = a.StandardActor().prepare(initial, current, trace, token, max_input=2500)
    assert len(result['input_token_ids']) <= 2500
    assert result['choices'] == [
        'go to cabinet 1',
        'inventory',
        'look',
        'move soapbar 1 to toilet 1',
    ]
    text = json.dumps(result['messages'])
    assert 'put some candle on toilet.' in text and 'SECRET' not in text
    assert result['context']['dropped_history_steps'] > 0
    assert 'move soapbar 1 to toilet 1' in text
    with pytest.raises(ValueError):
        a.StandardActor().prepare(initial, view('x' * 4000), [], token, max_input=2500)


def test_strict_actor_never_repairs_or_executes_truncated_or_unlisted_output():
    a = actor_module().StandardActor()
    assert a.action({'content': 'look', 'finish_reason': 'stop'}, ['look', 'inventory']) == 'look'
    for content, finish in [
        (' look', 'stop'),
        ('"look"', 'stop'),
        ('look', 'length'),
        ('go to kitchen 1', 'stop'),
    ]:
        with pytest.raises(ValueError):
            a.action({'content': content, 'finish_reason': finish}, ['look', 'inventory'])


async def test_real_episode_consumer_sends_constraints_and_uses_only_environment_feedback(
    tmp_path, monkeypatch
):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import q2_run
    from serving_lab.runtime import RuntimeStore

    class Response:
        status_code = 200

        def __init__(self, value):
            self.value = value

        async def aread(self):
            return json.dumps(self.value).encode()

        async def aclose(self):
            pass

    class Engine:
        version_id = 'm'
        identity = {'startup_generation': 'test'}

        def __init__(self):
            self.profile = profile(tmp_path, {'name': 'q2_public_choice_v1', 'seed': 20261103})
            self.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])

        def wire_payload(self, body):
            return self.profile.payload(body)

        async def ready(self):
            return True

        async def engine_token_ids(self, messages):
            return [1, 2]

        async def open(self, attempt, body):
            wire = self.wire_payload(body)
            # 外部替身只接受约定候选，缺失约束不能假装生成成功。
            if wire.get('structured_outputs') != {'choice': ['inventory', 'look']}:
                raise ValueError('missing action constraint at actual engine boundary')
            return Response(
                {
                    'id': 'chatcmpl-' + attempt,
                    'model': 'm',
                    'choices': [{'message': {'content': 'look'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
                }
            )

        async def wait_release(self, attempt, timeout):
            return {'status': 'FINISHED_STOPPED'}

    class Bridge:
        def __init__(self, task, data, out, image, max_steps, **kwargs):
            out.mkdir()
            self.out = out

        def read(self):
            return {'public': view('Your task is to: find a candle.'), 'terminal': False}

        async def step(self, action):
            self.value = {
                'public': view('REAL environment feedback'),
                'terminal': True,
                'result': {'success': False, 'steps': 1},
            }
            return self.value

        def close(self):
            (self.out / 'closed').write_text('closed')

    monkeypatch.setattr(q2_run, 'Bridge', Bridge)
    with RuntimeStore(tmp_path / 'runtime') as store:
        result = await q2_run.episode(
            {'task_id': 's', 'role': 'support', 'sequence': 1},
            tmp_path,
            tmp_path / 'episode',
            'image',
            Engine(),
            store,
            purpose='actor_diagnostic',
            actor=actor_module().StandardActor(),
        )
        rows = store.db.execute('SELECT state,cleanup,lease_released FROM arrivals').fetchall()
        assert [tuple(r) for r in rows] == [('completed', 'confirmed', 1)]
    reservation = json.loads((tmp_path / 'episode/action-001/reservation.json').read_text())
    assert reservation['wire']['structured_outputs'] == {'choice': ['inventory', 'look']}
    assert result['trace'][0]['after']['feedback'] == 'REAL environment feedback'
    assert result['eligible_for_training'] is False and (tmp_path / 'episode/closed').exists()


def test_standard_summary_cannot_drop_failed_family_or_duplicate_a_success():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    assert importlib.util.find_spec('q2_standard') is not None
    from q2_standard import summarize_records

    records = [
        {
            'task_id': f'q1_support_{i:02}',
            'initial': view('Your task is to: put a candle on toilet.'),
            'trace': [{'step': 1, 'before': view('a'), 'action': 'look', 'after': view('a')}],
            'result': {
                'success': i == 1,
                'steps': 1,
                'termination': 'step_budget',
                'admissible_rate': 1.0,
            },
        }
        for i in range(1, 7)
    ]
    result = summarize_records(records)
    assert result['episodes'] == 6 and result['successes'] == 1
    assert result['all_families_ready'] is False and result['task_action_calls'] == 6
    with pytest.raises(ValueError):
        summarize_records(records[:1])
    with pytest.raises(ValueError):
        summarize_records(records[:5] + records[:1])


def test_audit_rejects_consistent_but_wrong_sampling_and_wrong_environment_task(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import q2_standard as q

    assert hasattr(q, 'validate_profile_binding') and hasattr(q, 'validate_episode_binding')
    p = profile(tmp_path, {'name': 'q2_public_choice_v1', 'seed': 20261103})
    initial = {
        'policy': q.POLICY,
        'image': q.IMAGE,
        'source_hashes': {
            'src/serving_lab/' + k: v for k, v in p.document['implementation'].items()
        },
    }
    binding = {'artifact_digest': 'artifact'}
    q.validate_profile_binding(initial, {'document': p.document, 'digest': p.digest}, binding)
    from serving_lab.profile import ServingProfile

    for key, value in [('seed', 1), ('temperature', 0.0), ('top_p', 1.0), ('top_k', -1)]:
        document = p.document
        document['decoding'][key] = value
        changed = ServingProfile(document, 'm')
        with pytest.raises(ValueError):
            q.validate_profile_binding(
                initial, {'document': document, 'digest': changed.digest}, binding
            )
    task = {'task_id': 'q1_support_01', 'role': 'support', 'sequence': 1}
    env = {'task_id': 'q1_support_01'}
    launch = {'image': q.IMAGE, 'command': ['create', q.IMAGE, '/runner.py']}
    q.validate_episode_binding(task, env, launch, task, 'q1_support_01', q.IMAGE)
    for wrong_env, wrong_launch in [
        ({'task_id': 'q1_support_02'}, launch),
        (env, {'image': 'wrong', 'command': ['create', 'wrong', '/runner.py']}),
    ]:
        with pytest.raises(ValueError):
            q.validate_episode_binding(
                task, wrong_env, wrong_launch, task, 'q1_support_01', q.IMAGE
            )
