"""独立审查中的启动泄漏、同代际回执替换和模板边界反例。"""

import sys
import threading
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import q2_run
import q2_audit


@pytest.mark.asyncio
async def test_initial_read_timeout_always_closes_created_bridge(tmp_path, monkeypatch):
    closed = threading.Event()

    class WaitingBridge:
        def __init__(self, task, data, out, image, max_steps):
            self.value = None
            out.mkdir()

        def read(self):
            closed.wait(2)
            return None

        def close(self):
            closed.set()

    monkeypatch.setattr(q2_run, 'Bridge', WaitingBridge)
    monkeypatch.setattr(q2_run, 'BRIDGE_TIMEOUT', 0.01, raising=False)
    with pytest.raises(TimeoutError):
        await q2_run.episode({'task_id': 't'}, tmp_path, tmp_path / 'out', 'test')
    assert closed.is_set()


def test_audit_rejects_other_request_from_same_worker_generation():
    assert hasattr(q2_audit, 'check_action_identity')
    reservation = {
        'arrival_id': 'a',
        'attempt_id': 'b',
        'worker': {'startup_generation': 'g', 'worker_instance_id': 'w'},
    }
    receipt = {
        'arrival_id': 'a',
        'attempt_id': 'b',
        'release': {
            'request_id': 'chatcmpl-other',
            'startup_generation': 'g',
            'worker_instance_id': 'w',
        },
    }
    with pytest.raises(AssertionError):
        q2_audit.check_action_identity(
            reservation, receipt, {'id': 'chatcmpl-b'}, {'id': 'a', 'attempt_id': 'b'}
        )


def test_history_pair_is_checked_inside_rendered_chat():
    from serving_lab import public_tasks

    assert hasattr(public_tasks, 'matched_histories')

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)

        def decode(self, ids, skip_special_tokens=False):
            return ''.join(ids)

        def apply_chat_template(self, messages, **kw):
            # 相同孤立长度，模板边界仍可能产生不同实际长度。
            text = ''.join(m['content'] for m in messages)
            return list(text) + ([0] if 'abc' in text else [])

    view = {'feedback': 'now', 'admissible_commands': ['look']}
    with pytest.raises(ValueError):
        public_tasks.matched_histories('abc', 'xyz', Tokenizer(), 'initial', view, 3)


def test_cleanup_removes_owned_container_even_if_exit_receipt_write_fails(tmp_path, monkeypatch):
    import json

    calls = []
    info = {
        'Name': '/own',
        'Config': {'Labels': {'serving_lab.owner': 'q2', 'serving_lab.run_id': 'r'}},
        'State': {'Running': False, 'ExitCode': 0},
    }

    def docker(*args):
        calls.append(args)
        return json.dumps([info]) if args[0] == 'inspect' else ''

    monkeypatch.setattr(q2_run, 'docker', docker)
    monkeypatch.setattr(
        q2_run, 'write_once', lambda *a: (_ for _ in ()).throw(OSError('controlled disk error'))
    )
    b = q2_run.Bridge.__new__(q2_run.Bridge)
    b.cid = 'id'
    b.name = 'own'
    b.run_id = 'r'
    b.process = None
    b.out = tmp_path
    b.log = (tmp_path / 'log').open('wb')
    with pytest.raises(OSError):
        b.close()
    assert ('rm', 'id') in calls and b.log.closed


def test_launch_receipt_failure_also_removes_created_container(tmp_path, monkeypatch):
    import json

    calls = []
    labels = {}

    def docker(*args):
        calls.append(args)
        if args[0] == 'create':
            labels['name'] = args[args.index('--name') + 1]
            labels['run'] = next(
                a.split('=', 1)[1] for a in args if a.startswith('serving_lab.run_id=')
            )
            return 'id'
        if args[0] == 'inspect':
            return json.dumps(
                [
                    {
                        'Name': '/' + labels['name'],
                        'Config': {
                            'Labels': {
                                'serving_lab.owner': 'q2',
                                'serving_lab.run_id': labels['run'],
                            }
                        },
                        'State': {'Running': False},
                    }
                ]
            )
        return ''

    monkeypatch.setattr(q2_run, 'docker', docker)
    monkeypatch.setattr(q2_run, 'windows', str)
    monkeypatch.setattr(
        q2_run, 'write_once', lambda *a: (_ for _ in ()).throw(OSError('controlled disk error'))
    )
    with pytest.raises(OSError):
        q2_run.Bridge({'task_id': 't'}, tmp_path, tmp_path / 'out', 'test', 100)
    assert ('rm', 'id') in calls
