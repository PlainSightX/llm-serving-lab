"""版本可有多个副本；参数版本、实例代际和未清租约不能相互冒充。"""

from types import SimpleNamespace
import pytest
from serving_lab.profile import ServingProfile, TRANSPORT_SCOPE
from serving_lab.runtime import RuntimeStore
from test_online import FakeEngine


def workers():
    result = {}
    for rid, version in [('a1', 'a'), ('a2', 'a'), ('b1', 'b')]:
        e = FakeEngine()
        e.version_id = version
        e.identity = {'worker_instance_id': rid, 'startup_generation': 'g1'}
        e.binding = SimpleNamespace(
            artifact_digest=version + '-artifact', manifest_sha256=rid + '-binding'
        )
        e.profile = ServingProfile(
            {'name': 'test', 'max_tokens': 128, 'artifact_digest': version + '-artifact'}, version
        )
        result[rid] = e
    return result


def make_pool(store, ws=None, **kwargs):
    from serving_lab.replicas import ReplicaController

    ws = ws or workers()
    return ReplicaController(
        store,
        ws,
        {v: {'digest': v + '-parameters'} for v in ('a', 'b')},
        'panel',
        {k: {'max_inflight': 1, 'max_queue': 1} for k in ws},
        {v: {'status': 'historical-failed'} for v in ('a', 'b')},
        **kwargs,
    )


def qualify(pool, rid):
    e = pool.engines[rid]
    pool.qualified(
        rid,
        {
            'passed': True,
            'qualification_scope': TRANSPORT_SCOPE,
            'profile_digest': e.profile.digest,
            'version_id': e.version_id,
            'worker': e.identity,
            'artifact_digest': e.binding.artifact_digest,
            'panel_digest': 'panel',
            'application_history': pool.application_results[e.version_id],
        },
    )
    pool.observed_ready[rid] = True


def test_same_version_replicas_pin_independently_and_drain_does_not_move_lease(tmp_path):
    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store)
        for rid in pool.engines:
            qualify(pool, rid)
        pool.change('promote', 'a', 0)
        for aid in ('one', 'two'):
            store.record_arrival(aid, None)
        a = pool.pin('active', 'one')
        b = pool.pin('active', 'two')
        assert a['version_id'] == b['version_id'] == 'a' and {a['replica_id'], b['replica_id']} == {
            'a1',
            'a2',
        }
        pool.drain_replica(a['replica_id'])
        with pytest.raises(ValueError):
            pool.retire_replica(a['replica_id'])
        pool.change('promote', 'b', 1)
        assert a['engine'] is pool.engines[a['replica_id']] and a['route_generation'] == 1
        store.finish('one', 'cancelled', {})
        store.release_lease('one')
        pool.retire_replica(a['replica_id'])
        assert pool.pending('a') == 1


def test_replica_capacity_qualification_and_profile_cannot_be_bypassed(tmp_path):
    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store)
        qualify(pool, 'a1')
        pool.change('promote', 'a', 0)
        for aid in ('one', 'two', 'three'):
            store.record_arrival(aid, None)
        pool.pin('a', 'one')
        pool.pin('a', 'two')
        with pytest.raises(ValueError):
            pool.pin('a', 'three')
        qualify(pool, 'a2')
        assert pool.pin('a', 'three')['replica_id'] == 'a2'
    bad = workers()
    bad['a2'].profile = ServingProfile({'max_tokens': 4}, 'a')
    with RuntimeStore(tmp_path / 'bad') as store:
        with pytest.raises(ValueError):
            make_pool(store, bad)


def test_unreleased_generation_blocks_rebind_and_other_replica_stays_available(tmp_path):
    ws = workers()
    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store, ws)
        qualify(pool, 'a1')
        qualify(pool, 'a2')
        pool.change('promote', 'a', 0)
        store.record_arrival('orphan', 'key')
        old = pool.pin('a', 'orphan')
        store.reserve_attempt('orphan', 'attempt', 'a', 'hash')
        store.dispatched('orphan', old['engine'].identity)
        pool.isolate(old['engine'])
        store.record_arrival('next', None)
        assert pool.pin('a', 'next')['replica_id'] != old['replica_id']
        store.finish('next', 'cancelled', {})
        store.release_lease('next')
    ws[old['replica_id']].identity = {**ws[old['replica_id']].identity, 'startup_generation': 'g2'}
    with RuntimeStore(tmp_path) as store:
        assert store.row('orphan')['state'] == 'unknown'
        with pytest.raises(ValueError):
            make_pool(store, ws, allow_rebind={old['replica_id']})
        store.cleanup('orphan', 'confirmed', {'status': 'test-supervisor-confirmed-old-death'})
        store.release_lease('orphan')
        pool = make_pool(store, ws, allow_rebind={old['replica_id']})
        assert pool.replica(old['replica_id'])['state'] == 'loading'
        assert len(pool.history(old['replica_id'])) == 2


def test_aliasing_parameters_still_cannot_fake_two_versions(tmp_path):
    from serving_lab.replicas import ReplicaController

    with RuntimeStore(tmp_path) as store:
        with pytest.raises(ValueError):
            ReplicaController(
                store,
                workers(),
                {'a': {'digest': 'same'}, 'b': {'digest': 'same'}},
                'panel',
                {},
                {},
            )


def test_version_retirement_waits_for_all_replica_leases_and_reload_needs_qualification(tmp_path):
    ws = workers()
    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store, ws)
        for rid in ws:
            qualify(pool, rid)
        pool.change('promote', 'a', 0)
        store.record_arrival('old', None)
        pool.pin('a', 'old')
        pool.change('promote', 'b', 1)
        with pytest.raises(ValueError):
            pool.retire('a')
        store.finish('old', 'completed', {})
        store.release_lease('old')
        pool.retire('a')
        assert pool.replica('a1')['state'] == pool.replica('a2')['state'] == 'retired'
        ws['a1'].identity = {**ws['a1'].identity, 'startup_generation': 'g2'}
        pool = make_pool(store, ws, allow_rebind={'a1'})
        with pytest.raises(ValueError):
            pool.change('rollback', 'a', 2)
        qualify(pool, 'a1')
        pool.change('rollback', 'a', 2)
        assert pool.route()['active'] == 'a' and pool.replica('a2')['state'] == 'retired'


def test_reconciled_quarantine_requires_explicit_reload_and_requalification(tmp_path):
    with RuntimeStore(tmp_path) as store:
        pool = make_pool(store)
        qualify(pool, 'a1')
        pool.isolate(pool.engines['a1'])
        assert hasattr(pool, 'reload_replica')
        pool.reload_replica('a1')
        assert pool.replica('a1')['state'] == 'loading'
        assert pool.eligible('a') == []
        qualify(pool, 'a1')
        assert pool.eligible('a') == ['a1']
