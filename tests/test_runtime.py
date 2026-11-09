import pytest


def test_arrivals_dedup_terminal_and_restart(tmp_path):
    from serving_lab.runtime import RuntimeStore

    with RuntimeStore(tmp_path) as store:
        store.record_arrival('a', 'key')
        store.record_arrival('b', 'key')
        assert store.reserve_attempt('a', 'attempt-a', 'v1', 'hash')
        assert not store.reserve_attempt('b', 'attempt-b', 'v1', 'hash')
        assert store.finish('b', 'rejected', {'http_status': 409})
        assert not store.finish('b', 'completed', {})
        store.dispatched('a', {'worker_instance_id': 'worker', 'startup_generation': 'generation'})
    with RuntimeStore(tmp_path) as store:
        assert store.row('a')['state'] == 'unknown'
        assert store.unresolved()[0]['attempt_id'] == 'attempt-a'
        store.record_arrival('c', 'key')
        assert not store.reserve_attempt('c', 'attempt-c', 'v1', 'hash')
        assert len(store.rows()) == 3
        store.cleanup('a', 'confirmed', {'status': 'FINISHED_ABORTED'})
        assert store.unresolved() == []


def test_single_owner_and_rejected_arrival_does_not_consume_key(tmp_path):
    from serving_lab.runtime import RuntimeStore

    with RuntimeStore(tmp_path) as store:
        with pytest.raises(ValueError):
            RuntimeStore(tmp_path)
        store.record_arrival('bad', 'key')
        store.finish('bad', 'rejected', {'http_status': 422})
        store.record_arrival('good', 'key')
        assert store.reserve_attempt('good', 'run', 'v', 'h')
