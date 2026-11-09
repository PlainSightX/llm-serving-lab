import json
import socket
import sys

import pytest


def launch_record(path, data):
    from serving_lab.lab import digest

    files = {'test-only': 'hash'}
    data['artifact_digest'] = digest(files)
    launch = {
        'worker_instance_id': data['worker_instance_id'],
        'startup_generation': data['startup_generation'],
        'config': {'version_id': data['version_id'], 'host': '127.0.0.1', 'port': 19080},
        'base_identity': {'path': data['model_path'], 'files': files},
        'command': [
            'python',
            '--model',
            data['model_path'],
            '--served-model-name',
            data['served_model'],
            '--host',
            '127.0.0.1',
            '--port',
            '19080',
        ],
    }
    (path / 'launch.json').write_text(json.dumps(launch), encoding='utf-8')


def test_process_identity_rejects_pid_reuse(tmp_path):
    from serving_lab.engine import process_identity

    p = tmp_path / '17'
    p.mkdir()
    boot = tmp_path / 'sys/kernel/random'
    boot.mkdir(parents=True)
    (boot / 'boot_id').write_text('boot-a')
    (p / 'stat').write_text(
        '17 (a name with spaces) S ' + ' '.join(['0'] * 18 + ['123'] + ['0'] * 4)
    )
    identity = process_identity(17, proc_root=tmp_path)
    assert identity == {'pid': 17, 'start_ticks': '123', 'boot_id': 'boot-a'}
    (p / 'stat').write_text(
        '17 (a name with spaces) S ' + ' '.join(['0'] * 18 + ['124'] + ['0'] * 4)
    )
    assert process_identity(17, proc_root=tmp_path) != identity


def test_release_probe_runs_original_once_and_records_request_kv_release(tmp_path):
    from serving_lab.engine_probe import install_release_probe

    class KV:
        def __init__(self):
            self.blocks = {'chatcmpl-test': ([1, 2],)}

        def get_block_ids(self, key):
            return self.blocks.get(key, ([],))

    class Scheduler:
        def __init__(self):
            self.requests = {'chatcmpl-test': True}
            self.kv_cache_manager = KV()
            self.calls = 0

        def _free_blocks(self, request):
            self.calls += 1
            self.kv_cache_manager.blocks.pop(request.request_id)
            self.requests.pop(request.request_id)

    class Request:
        request_id = 'chatcmpl-test'
        status = 'FINISHED_ABORTED'

    path = tmp_path / 'events.jsonl'
    install_release_probe(Scheduler, path, 'instance', 'generation')
    install_release_probe(Scheduler, path, 'instance', 'generation')
    scheduler = Scheduler()
    scheduler._free_blocks(Request())
    records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    released = [r for r in records if r['event'] == 'request_released']
    assert scheduler.calls == 1 and len(released) == 1
    assert released[0]['before_blocks'] == [[1, 2]] and released[0]['after_blocks'] == [[]]
    assert released[0]['request_absent'] and released[0]['status'] == 'FINISHED_ABORTED'
    assert released[0]['worker_instance_id'] == 'instance'


def test_release_journal_does_not_accept_old_generation_or_natural_stop(tmp_path):
    from serving_lab.engine import ReleaseJournal

    path = tmp_path / 'events.jsonl'
    events = [
        {
            'event': 'request_released',
            'request_id': 'chatcmpl-a',
            'worker_instance_id': 'old',
            'startup_generation': 'g',
            'request_absent': True,
            'after_blocks': [[]],
            'status': 'FINISHED_ABORTED',
        },
        {
            'event': 'request_released',
            'request_id': 'chatcmpl-b',
            'worker_instance_id': 'new',
            'startup_generation': 'g',
            'request_absent': True,
            'after_blocks': [[]],
            'status': 'FINISHED_STOPPED',
        },
    ]
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    reader = ReleaseJournal(path, 'new', 'g')
    assert reader.get('a') is None
    assert reader.get('b')['status'] == 'FINISHED_STOPPED'
    assert reader.get('b')['status'] != 'FINISHED_ABORTED'


def test_binding_refuses_non_loopback_and_manifest_change(tmp_path):
    from serving_lab.engine import EngineBinding

    data = {
        'version_id': 'base',
        'endpoint': 'http://example.com:19080',
        'served_model': 'base',
        'artifact_digest': 'hash',
        'worker_instance_id': 'i',
        'startup_generation': 'g',
        'process': {'pid': 17, 'start_ticks': '1', 'boot_id': 'b'},
        'model_path': str(tmp_path),
        'events_path': str(tmp_path / 'events.jsonl'),
    }
    p = tmp_path / 'binding.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        EngineBinding.load(p)
    data['endpoint'] = 'http://127.0.0.1:19080'
    p.write_text(json.dumps(data), encoding='utf-8')
    launch_record(tmp_path, data)
    p.write_text(json.dumps(data), encoding='utf-8')
    binding = EngineBinding.load(p)
    assert binding.version_id == 'base'
    p.write_text('{}', encoding='utf-8')
    assert not binding.manifest_unchanged()


def test_probe_installation_does_not_identify_a_live_scheduler(tmp_path):
    from serving_lab.engine import ReleaseJournal

    path = tmp_path / 'events.jsonl'
    common = {'worker_instance_id': 'new', 'startup_generation': 'g'}
    rows = [
        {**common, 'event': 'probe_installed', 'pid': 10},
        {
            **common,
            'event': 'request_released',
            'pid': 20,
            'request_id': 'chatcmpl-r',
            'status': 'FINISHED_STOPPED',
            'request_absent': True,
            'after_blocks': [[]],
        },
    ]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    reader = ReleaseJournal(path, 'new', 'g')
    reader.refresh()
    assert reader.installed_pids == {10} and reader.scheduler_pids == {20}


def test_pre_start_binding_drift_cannot_reuse_old_launch(tmp_path):
    from serving_lab.engine import EngineBinding

    data = {
        'version_id': 'base',
        'endpoint': 'http://127.0.0.1:19080',
        'served_model': 'base',
        'artifact_digest': 'hash',
        'worker_instance_id': 'i',
        'startup_generation': 'g',
        'process': {'pid': 17, 'start_ticks': '1', 'boot_id': 'b'},
        'model_path': str(tmp_path),
        'events_path': str(tmp_path / 'events.jsonl'),
    }
    launch_record(tmp_path, data)
    p = tmp_path / 'binding.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    EngineBinding.load(p)
    data['endpoint'] = 'http://127.0.0.1:19099'
    p.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        EngineBinding.load(p)


def test_release_index_is_bounded_but_recovery_can_find_old_event(tmp_path):
    from serving_lab.engine import ReleaseJournal

    path = tmp_path / 'events.jsonl'
    common = {
        'event': 'request_released',
        'pid': 20,
        'worker_instance_id': 'i',
        'startup_generation': 'g',
        'status': 'FINISHED_ABORTED',
        'request_absent': True,
        'after_blocks': [[]],
    }
    path.write_text(
        ''.join(
            json.dumps({**common, 'request_id': 'chatcmpl-' + str(i)}) + '\n' for i in range(8)
        ),
        encoding='utf-8',
    )
    journal = ReleaseJournal(path, 'i', 'g', max_entries=3)
    journal.refresh()
    assert len(journal.released) == 3 and journal.get('0') is None
    assert journal.find_historical('0')['request_id'] == 'chatcmpl-0'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux TIME_WAIT语义需在实际服务环境验证')
def test_port_preflight_allows_time_wait_but_rejects_live_listener():
    from serving_lab.engine import check_port_available

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.listen()
    with pytest.raises(OSError):
        check_port_available('127.0.0.1', port)
    client = socket.create_connection(('127.0.0.1', port))
    accepted, _ = listener.accept()
    accepted.shutdown(socket.SHUT_WR)
    client.recv(1)
    client.close()
    accepted.close()
    listener.close()
    with socket.socket() as plain:
        with pytest.raises(OSError):
            plain.bind(('127.0.0.1', port))
    check_port_available('127.0.0.1', port)
