"""新诊断修复可见停滞和公开策略状态；不改写旧运行语义。"""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import asyncio
import threading
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from serving_lab.public_tasks import PublicEpisode, PublicPolicy, verified_history


def test_second_object_location_survives_first_placement():
    p = PublicPolicy()
    assert (
        p.choose(
            {
                'feedback': 'Your task is to: find two soapbottle and put them in cabinet.',
                'admissible_commands': ['go to countertop 1'],
            }
        )
        == 'go to countertop 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You arrive at countertop 1. You see soapbottle 1 and soapbottle 2.',
                'admissible_commands': [
                    'take soapbottle 1 from countertop 1',
                    'take soapbottle 2 from countertop 1',
                ],
            }
        )
        == 'take soapbottle 1 from countertop 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You pick up the soapbottle 1 from the countertop 1.',
                'admissible_commands': ['move soapbottle 1 to cabinet 1'],
            }
        )
        == 'move soapbottle 1 to cabinet 1'
    )
    assert (
        p.choose(
            {
                'feedback': 'You move the soapbottle 1 to the cabinet 1.',
                'admissible_commands': ['go to countertop 1', 'go to cabinet 2'],
            }
        )
        == 'go to countertop 1'
    )


def test_held_object_opens_current_destination_before_moving_elsewhere():
    p = PublicPolicy()
    p.choose(
        {
            'feedback': 'Your task is to: put a candle in cabinet.',
            'admissible_commands': ['take candle 1 from countertop 1'],
        }
    )
    assert (
        p.choose(
            {
                'feedback': 'You pick up the candle 1 from the countertop 1.',
                'admissible_commands': ['open cabinet 1', 'go to cabinet 2'],
            }
        )
        == 'open cabinet 1'
    )


def test_destination_name_is_not_a_substring_and_unknown_feedback_is_not_success():
    p = PublicPolicy()
    p.choose(
        {
            'feedback': 'Your task is to: put some candle on toilet.',
            'admissible_commands': ['take candle 1 from countertop 1'],
        }
    )
    assert (
        p.choose(
            {
                'feedback': 'You pick up the candle 1 from the countertop 1.',
                'admissible_commands': ['go to toiletpaperhanger 1', 'go to kitchen 1'],
            }
        )
        == 'go to kitchen 1'
    )
    p = PublicPolicy()
    p.choose(
        {
            'feedback': 'Your task is to: clean some potato and put it in diningtable.',
            'admissible_commands': ['take potato 1 from countertop 1'],
        }
    )
    commands = ['clean potato 1 with sinkbasin 1', 'move potato 1 to diningtable 1']
    assert (
        p.choose(
            {
                'feedback': 'You pick up the potato 1 from the countertop 1.',
                'admissible_commands': commands,
            }
        )
        == commands[0]
    )
    assert (
        p.choose({'feedback': 'Unrecognized response.', 'admissible_commands': commands})
        == commands[0]
    )


class StillEnvironment:
    def reset(self):
        return {
            'feedback': 'same public view',
            'admissible_commands': ['look', 'inventory'],
            'won': False,
        }

    def step(self, action):
        return self.reset(), 0, False


def test_stagnation_is_new_opt_in_protocol_and_does_not_invent_steps():
    ep = PublicEpisode(StillEnvironment(), 100, stagnation_limit=4)
    ep.reset()
    for _ in range(4):
        ep.step('look')
    assert (
        ep.result()['termination'] == 'stagnation_terminated'
        and ep.result()['steps'] == 4
        and not ep.result()['success']
    )
    with pytest.raises(ValueError):
        ep.step('look')
    old = PublicEpisode(StillEnvironment(), 100)
    old.reset()
    for _ in range(4):
        old.step('look')
    assert not old.terminal
    varied = PublicEpisode(StillEnvironment(), 100, stagnation_limit=4)
    varied.reset()
    for action in ['look', 'inventory'] * 3:
        varied.step(action)
    assert not varied.terminal


def test_development_success_does_not_become_training_or_primary_history():
    r = {
        'purpose': 'actor_diagnostic',
        'eligible_for_primary_history': False,
        'origin': 'model_public',
        'role': 'support',
        'sequence': 1,
        'initial': {'feedback': 'g'},
        'result': {'success': True},
        'trace': [
            {
                'before': {'feedback': 'before'},
                'action': 'look',
                'after': {'feedback': 'after'},
                'receipt': {'protocol_valid': True, 'release': {'status': 'ok'}},
            }
        ],
    }
    with pytest.raises(ValueError):
        verified_history(r, cutoff=7)


def test_three_seeds_have_separate_immutable_profiles(tmp_path):
    from serving_lab.profile import make_profile

    model = tmp_path / 'model'
    model.mkdir()
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
    profiles = [
        make_profile(binding, tokenizer, {'name': 'q2_public_non_thinking_v2', 'seed': s})
        for s in (20261103, 20261104, 20261105)
    ]
    assert len({p.digest for p in profiles}) == 3
    assert [
        p.payload({'messages': [], 'max_tokens': 4, 'stream': False})['seed'] for p in profiles
    ] == [20261103, 20261104, 20261105]
    with pytest.raises(ValueError):
        profiles[0].payload({'messages': [], 'seed': 99})


def test_paired_prompts_share_visible_prefix_and_do_not_receive_labels():
    from serving_lab.actor_diagnostic import paired_contexts

    case = {
        'initial': {'feedback': 'Your task is to: put a candle on toilet.'},
        'view': {
            'feedback': 'holding candle',
            'admissible_commands': ['move candle 1 to toilet 1'],
        },
        'prefix': [{'action': 'look', 'after': {'feedback': 'x' * 100}} for _ in range(20)],
        'acceptable_actions': ['move candle 1 to toilet 1'],
        'source': {'path': 'secret-controller-only'},
    }
    tokenize = lambda ms: list(range(sum(len(m['content']) for m in ms)))
    contexts = paired_contexts(case, tokenize, 1200)
    assert contexts['legacy']['messages'][1:] == contexts['goal_action']['messages'][1:]
    assert (
        contexts['legacy']['context']['common_dropped_steps']
        == contexts['goal_action']['context']['common_dropped_steps']
        > 0
    )
    for item in contexts.values():
        assert len(item['input_token_ids']) <= 1200
        assert 'acceptable_actions' not in json.dumps(
            item['messages']
        ) and 'secret-controller-only' not in json.dumps(item['messages'])


def test_cpu_gate_does_not_trust_success_summary_alone(tmp_path):
    import q2_rescue

    assert hasattr(q2_rescue, 'validate_cpu_baseline')
    (tmp_path / 'summary.json').write_text(
        json.dumps({'canary_passed': True, 'formal_success': 12, 'formal_episodes': 12})
    )
    with pytest.raises(ValueError):
        q2_rescue.validate_cpu_baseline(tmp_path)


async def test_cancelled_bridge_construction_still_closes_its_owner(tmp_path, monkeypatch):
    import q2_run

    entered = threading.Event()
    allow = threading.Event()
    closed = threading.Event()

    class SlowBridge:
        def __init__(self, task, data, out, image, max_steps):
            out.mkdir()
            entered.set()
            allow.wait(2)

        def read(self):
            raise AssertionError('cancelled construction must not start episode')

        def close(self):
            closed.set()

    monkeypatch.setattr(q2_run, 'Bridge', SlowBridge)
    task = asyncio.create_task(
        q2_run.episode({'task_id': 'x'}, tmp_path, tmp_path / 'out', 'image')
    )
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.005)
    task.cancel()
    allow.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


def test_json_contract_is_explicit_and_does_not_repair_arbitrary_text():
    from serving_lab import actor_diagnostic as a

    assert hasattr(a, 'parse_json_action')
    assert a.parse_json_action('"look"') == 'look'
    assert a.parse_json_action('look') is None
    assert a.parse_json_action('["look"]') is None
    assert a.parse_json_action('"look" extra') is None
    assert a.parse_json_action('"look\\ninventory"') is None


def test_nonzero_environment_exit_cannot_be_accepted_even_after_cleanup(tmp_path, monkeypatch):
    import q2_run

    calls = []
    info = {
        'Name': '/own',
        'Config': {'Labels': {'serving_lab.owner': 'q2', 'serving_lab.run_id': 'r'}},
        'State': {'Running': False, 'ExitCode': 9},
    }

    def docker(*args):
        calls.append(args)
        return json.dumps([info]) if args[0] == 'inspect' else ''

    monkeypatch.setattr(q2_run, 'docker', docker)
    b = q2_run.Bridge.__new__(q2_run.Bridge)
    b.cid = 'id'
    b.name = 'own'
    b.run_id = 'r'
    b.process = None
    b.out = tmp_path
    b.log = (tmp_path / 'log').open('wb')
    with pytest.raises(RuntimeError):
        b.close()
    assert (
        ('rm', 'id') in calls
        and b.log.closed
        and json.loads((tmp_path / 'exit.json').read_text())['ExitCode'] == 9
    )


async def test_failed_client_close_does_not_skip_worker_stop(tmp_path, monkeypatch):
    import q2_rescue

    binding = tmp_path / 'binding.json'
    binding.write_text('{}')
    stops = []

    class BadClient:
        async def close(self):
            raise OSError('client close failed')

    monkeypatch.setattr(
        q2_rescue, 'stop', lambda path: stops.append(path) or {'stopped': True, 'remaining': []}
    )
    with pytest.raises(RuntimeError):
        await q2_rescue.close_models({1: BadClient()}, binding, tmp_path)
    assert (
        stops == [binding]
        and json.loads((tmp_path / 'cleanup.json').read_text())['client_close_errors']
    )


def test_envelope_never_uses_generated_feedback_or_guesses_between_actions():
    from serving_lab import actor_diagnostic as a

    assert hasattr(a, 'parse_action_envelope')
    assert a.parse_action_envelope('{"action":"look","feedback":"You have won!"}') == 'look'
    assert a.parse_action_envelope('{"command":" look "}') == 'look'
    for bad in (
        '{"action":"look","command":"inventory"}',
        '{"action":"look","action":"inventory"}',
        '{"action":"look","feedback":true}',
        '{"action":"look","other":"x"}',
        'look',
        '{"action":"look\\ninventory"}',
    ):
        assert a.parse_action_envelope(bad) is None
