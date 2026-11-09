"""真实SQLite发布合同；引擎替身只替代昂贵的模型服务。"""

from types import SimpleNamespace

import pytest

from test_online import FakeEngine


def engines():
    result = {}
    for name in ('small', 'large'):
        engine = FakeEngine()
        engine.version_id = name
        engine.identity = {'worker_instance_id': name + '-worker', 'startup_generation': 'g1'}
        engine.binding = SimpleNamespace(
            artifact_digest=name + '-bundle',
            manifest_sha256=name + '-binding',
            manifest_path='test-binding',
            model_path='test-model',
        )
        result[name] = engine
    return result


def controller(store, workers=None, **kwargs):
    from serving_lab.releases import ReleaseController

    workers = workers or engines()
    return ReleaseController(
        store,
        workers,
        {name: {'digest': name + '-parameters'} for name in workers},
        'panel-v1',
        **kwargs,
    )


def mark_ready(pool, version):
    pool.qualified(
        version,
        {
            'passed': True,
            'version_id': version,
            'panel_digest': 'panel-v1',
            'worker': pool.engines[version].identity,
            'artifact_digest': version + '-bundle',
        },
    )


def test_failed_quality_duplicate_parameters_and_stale_cas_keep_active(tmp_path):
    from serving_lab.runtime import RuntimeStore
    from serving_lab.releases import ReleaseController

    with RuntimeStore(tmp_path) as store:
        pool = controller(store)
        mark_ready(pool, 'small')
        pool.change('promote', 'small', 0)
        with pytest.raises(ValueError):
            pool.change('promote', 'large', 1)
        pool.qualified(
            'large',
            {
                'passed': False,
                'version_id': 'large',
                'panel_digest': 'panel-v1',
                'worker': pool.engines['large'].identity,
                'artifact_digest': 'large-bundle',
            },
        )
        assert pool.snapshot()['route']['active'] == 'small'
        with pytest.raises(ValueError):
            pool.change('promote', 'small', 0)
    with RuntimeStore(tmp_path / 'aliases') as store:
        with pytest.raises(ValueError):
            ReleaseController(
                store,
                engines(),
                {'small': {'digest': 'same'}, 'large': {'digest': 'same'}},
                'panel',
            )


def test_route_switch_preserves_lease_and_blocks_retire_until_released(tmp_path):
    from serving_lab.runtime import RuntimeStore

    with RuntimeStore(tmp_path) as store:
        pool = controller(store)
        mark_ready(pool, 'small')
        mark_ready(pool, 'large')
        pool.change('promote', 'small', 0)
        store.record_arrival('old', None)
        lease = pool.pin('active', 'old')
        pool.change('canary', 'large', 1, percentage=50)
        pool.change('promote', 'large', 2)
        assert lease['version_id'] == 'small' and lease['route_generation'] == 1
        store.record_arrival('late', None)
        with pytest.raises(ValueError):
            pool.pin('small', 'late')
        with pytest.raises(ValueError):
            pool.retire('small')
        store.finish('old', 'cancelled', {})
        store.release_lease('old')
        pool.retire('small')
        with pytest.raises(ValueError):
            pool.change('rollback', 'small', 3)


def test_restart_preserves_route_and_rebinding_requires_new_qualification(tmp_path):
    from serving_lab.runtime import RuntimeStore

    workers = engines()
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers)
        mark_ready(pool, 'small')
        mark_ready(pool, 'large')
        pool.change('promote', 'small', 0)
        pool.change('promote', 'large', 1)
        store.record_arrival('orphan', 'key')
        pool.pin('active', 'orphan')
        assert store.reserve_attempt('orphan', 'attempt', 'large', 'hash')
        store.dispatched('orphan', workers['large'].identity)
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers)
        assert pool.snapshot()['route']['generation'] == 2
        assert store.row('orphan')['state'] == 'unknown' and pool.pending('large') == 1
        store.record_arrival('duplicate', 'key')
        assert not store.reserve_attempt('duplicate', 'other', 'small', 'hash')
        workers['small'].identity = {**workers['small'].identity, 'startup_generation': 'g2'}
        with pytest.raises(ValueError):
            controller(store, workers)
        pool = controller(store, workers, allow_rebind={'small'})
        with pytest.raises(ValueError):
            pool.change('rollback', 'small', 2)
        mark_ready(pool, 'small')
        pool.change('rollback', 'small', 2)
        assert pool.snapshot()['route']['active'] == 'small'


def test_dedup_and_pin_are_shared_across_versions(tmp_path):
    from serving_lab.runtime import RuntimeStore

    with RuntimeStore(tmp_path) as store:
        pool = controller(store)
        mark_ready(pool, 'small')
        mark_ready(pool, 'large')
        pool.change('promote', 'small', 0)
        store.record_arrival('a', 'same')
        pool.pin('active', 'a')
        assert store.reserve_attempt('a', 'a-attempt', 'small', 'a-body')
        pool.change('promote', 'large', 1)
        store.record_arrival('b', 'same')
        lease = pool.pin('active', 'b')
        assert lease['version_id'] == 'large'
        assert not store.reserve_attempt('b', 'b-attempt', 'large', 'b-body')


def test_rebind_preserves_orphan_until_original_worker_release_is_confirmed(tmp_path):
    from serving_lab.runtime import RuntimeStore

    workers = engines()
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers)
        mark_ready(pool, 'small')
        pool.change('promote', 'small', 0)
        store.record_arrival('orphan', None)
        pool.pin('active', 'orphan')
        store.reserve_attempt('orphan', 'attempt', 'small', 'hash')
        store.dispatched('orphan', workers['small'].identity)
    workers['small'].identity = {**workers['small'].identity, 'startup_generation': 'g2'}
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers, allow_rebind={'small'})
        assert pool.pending('small') == 1
        with pytest.raises(ValueError):
            mark_ready(pool, 'small')
        store.cleanup('orphan', 'confirmed', {'status': 'OLD_WORKER_TERMINATED'})
        store.release_lease('orphan')
        mark_ready(pool, 'small')


def test_active_rebind_requires_new_route_commit_after_qualification(tmp_path):
    from serving_lab.runtime import RuntimeStore

    workers = engines()
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers)
        mark_ready(pool, 'small')
        pool.change('promote', 'small', 0)
    workers['small'].identity = {**workers['small'].identity, 'startup_generation': 'g2'}
    with RuntimeStore(tmp_path) as store:
        pool = controller(store, workers, allow_rebind={'small'})
        mark_ready(pool, 'small')
        store.record_arrival('before', None)
        with pytest.raises(ValueError):
            pool.pin('active', 'before')
        pool.change('promote', 'small', 1)
        assert pool.route()['generation'] == 2 and pool.row('small')['state'] == 'active'
