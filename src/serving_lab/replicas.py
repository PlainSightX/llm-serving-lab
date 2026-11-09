"""新运行合同：发布选版本，副本池选实例，持久lease固定实际代际。"""

import asyncio
import hashlib
import json
from .lab import canonical, digest, identifier
from .profile import TRANSPORT_SCOPE
from .releases import ReleaseError

REPLICA_SCOPE = 'replica_pool_v1'


def replica_contract(
    engines, parameters, panel_digest, capacities, applications, application_policy=None
):
    versions = {e.version_id for e in engines.values()}
    if (
        not engines
        or set(parameters) != versions
        or set(applications) != versions
        or set(capacities) != set(engines)
    ):
        raise ValueError('副本配置不完整')
    if len({p['digest'] for p in parameters.values()}) != len(parameters):
        raise ValueError('相同参数不能冒充不同版本')
    if (
        application_policy is not None
        and application_policy.bind(engines, parameters) != applications
    ):
        raise ValueError('应用资格摘要不符')
    identities = set()
    docs = {}
    for rid, e in engines.items():
        identifier(rid)
        identifier(e.version_id)
        if e.version_id == 'active' or not e.profile:
            raise ValueError('副本必须有显式版本/profile')
        if digest(e.identity) in identities:
            raise ValueError('同一worker不能重复登记为副本')
        identities.add(digest(e.identity))
        cap = capacities[rid]
        if (
            set(cap) != {'max_inflight', 'max_queue'}
            or type(cap['max_inflight']) is not int
            or cap['max_inflight'] < 1
            or type(cap['max_queue']) is not int
            or cap['max_queue'] < 0
        ):
            raise ValueError('无效副本容量')
        doc = {
            'parameter_digest': parameters[e.version_id]['digest'],
            'artifact_digest': e.binding.artifact_digest,
            'profile_digest': e.profile.digest,
            'application_history': applications[e.version_id],
        }
        if e.version_id in docs and docs[e.version_id] != doc:
            raise ValueError('同版本副本的参数/profile不同')
        docs[e.version_id] = doc
    result = {
        'panel_digest': panel_digest,
        'versions': docs,
        'replicas': {
            rid: {'version': e.version_id, **capacities[rid]} for rid, e in engines.items()
        },
    }
    if application_policy is not None:
        result['application_policy'] = application_policy.document
    return result


class ReplicaController:
    replicated = True
    scope = REPLICA_SCOPE

    def __init__(
        self,
        store,
        engines,
        parameters,
        panel_digest,
        capacities,
        application_results,
        *,
        allow_rebind=(),
        application_policy=None,
    ):
        self.store = store
        self.db = store.db
        self.engines = engines
        self.panel_digest = panel_digest
        self.application_results = application_results
        self.application_policy = application_policy
        if application_policy is not None:
            from .application_release import POOL_SCOPE

            self.scope = POOL_SCOPE
        contract = replica_contract(
            engines, parameters, panel_digest, capacities, application_results, application_policy
        )
        self.contract = contract
        self.observed_ready = {r: False for r in engines}
        self.slots = {r: asyncio.Semaphore(capacities[r]['max_inflight']) for r in engines}
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS release_contract(id INTEGER PRIMARY KEY,scope TEXT,document TEXT);
          CREATE TABLE IF NOT EXISTS pool_versions(version TEXT PRIMARY KEY,state TEXT,identity TEXT,ever_active INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS pool_replicas(id TEXT PRIMARY KEY,version TEXT,state TEXT,identity TEXT,qualification TEXT);
          CREATE TABLE IF NOT EXISTS pool_history(replica TEXT,generation TEXT,identity TEXT,PRIMARY KEY(replica,generation));
          CREATE TABLE IF NOT EXISTS pool_route(id INTEGER PRIMARY KEY,generation INTEGER,active TEXT,canary TEXT,percentage INTEGER);
          INSERT OR IGNORE INTO pool_route VALUES(1,0,NULL,NULL,0);
        ''')
        row = self.db.execute('SELECT * FROM release_contract WHERE id=1').fetchone()
        if row and (row['scope'] != self.scope or row['document'] != canonical(contract)):
            raise ValueError('新runtime配置不同，禁止原地升级')
        # 先完整检查所有代际变化，不能在中途失败后留下半个rebind。
        for rid, e in engines.items():
            old = self.db.execute('SELECT * FROM pool_replicas WHERE id=?', (rid,)).fetchone()
            if old and old['identity'] != canonical(e.identity):
                if rid not in allow_rebind or self.pending_replica(rid):
                    raise ValueError('换代需显式rebind且旧lease已清理')
                if old['version'] != e.version_id:
                    raise ValueError('副本不能原地改版本')
                if self.db.execute(
                    'SELECT 1 FROM pool_history WHERE replica=? AND generation=?',
                    (rid, e.identity['startup_generation']),
                ).fetchone():
                    raise ValueError('代际不能复用或回退')
        with self.store.atomic(durable=True):
            self.db.execute(
                'INSERT OR IGNORE INTO release_contract VALUES(1,?,?)',
                (self.scope, canonical(contract)),
            )
            for v, doc in contract['versions'].items():
                self.db.execute(
                    "INSERT OR IGNORE INTO pool_versions VALUES(?,'registered',?,0)",
                    (v, canonical(doc)),
                )
            for rid, e in engines.items():
                old = self.db.execute('SELECT * FROM pool_replicas WHERE id=?', (rid,)).fetchone()
                if not old or old['identity'] != canonical(e.identity):
                    self.db.execute(
                        "INSERT OR REPLACE INTO pool_replicas VALUES(?,?,'loading',?,NULL)",
                        (rid, e.version_id, canonical(e.identity)),
                    )
                    self.db.execute(
                        'INSERT INTO pool_history VALUES(?,?,?)',
                        (rid, e.identity['startup_generation'], canonical(e.identity)),
                    )
                    self.store.event(
                        None,
                        'replica_bound',
                        {
                            'replica_id': rid,
                            'worker': e.identity,
                            'previous': dict(old) if old else None,
                        },
                    )

    def row(self, version):
        row = self.db.execute('SELECT * FROM pool_versions WHERE version=?', (version,)).fetchone()
        if not row:
            raise ReleaseError('unknown_model', 404)
        result = dict(row)
        result['identity'] = json.loads(result['identity'])
        return result

    def replica(self, rid):
        row = self.db.execute('SELECT * FROM pool_replicas WHERE id=?', (rid,)).fetchone()
        if not row:
            raise ReleaseError('unknown_replica', 404)
        result = dict(row)
        result['identity'] = json.loads(result['identity'])
        result['qualification'] = (
            json.loads(result['qualification']) if result['qualification'] else None
        )
        return result

    def history(self, rid):
        return [
            dict(r) for r in self.db.execute('SELECT * FROM pool_history WHERE replica=?', (rid,))
        ]

    def route(self):
        return dict(self.db.execute('SELECT * FROM pool_route WHERE id=1').fetchone())

    def pending(self, version):
        return self.db.execute(
            'SELECT COUNT(*) FROM arrivals WHERE version=? AND lease_released=0', (version,)
        ).fetchone()[0]

    def pending_replica(self, rid):
        return self.db.execute(
            'SELECT COUNT(*) FROM arrivals WHERE lease_released=0 AND worker IN (SELECT identity FROM pool_history WHERE replica=?)',
            (rid,),
        ).fetchone()[0]

    def snapshot(self):
        return {
            'qualification_scope': self.scope,
            'route': self.route(),
            'versions': [
                {**self.row(v), 'pending': self.pending(v)} for v in self.contract['versions']
            ],
            'replicas': [
                {
                    **self.replica(r),
                    'pending': self.pending_replica(r),
                    'observed_ready': self.observed_ready[r],
                    **self.contract['replicas'][r],
                }
                for r in self.engines
            ],
        }

    async def refresh(self):
        async def check(r, e):
            # 后台探测运行时只读最近结果：单个挂起副本的探测不会拖住每一个请求。
            nowait = getattr(e, 'ready_nowait', None)
            try:
                state = nowait() if nowait else None
                self.observed_ready[r] = bool(await e.ready()) if state is None else bool(state)
            except Exception:
                self.observed_ready[r] = False

        await asyncio.gather(*(check(r, e) for r, e in self.engines.items()))

    def eligible(self, version):
        return [
            r
            for r, e in self.engines.items()
            if e.version_id == version
            and self.replica(r)['state'] == 'ready'
            and self.observed_ready[r]
            and self.replica(r)['identity'] == e.identity
            and e.profile.digest == self.row(version)['identity']['profile_digest']
        ]

    def recovery_engine(self, row):
        return next(
            (e for e in self.engines.values() if canonical(e.identity) == row['worker']), None
        )

    def isolate(self, engine):
        rid = next(r for r, e in self.engines.items() if e is engine)
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE pool_replicas SET state='quarantined' WHERE id=?", (rid,))
            self.store.event(
                None, 'replica_quarantined', {'replica_id': rid, 'worker': engine.identity}
            )
        self.observed_ready[rid] = False

    def qualified(self, rid, receipt):
        row = self.replica(rid)
        e = self.engines[rid]
        v = self.row(e.version_id)
        if row['state'] != 'loading' or self.pending_replica(rid):
            raise ReleaseError('replica_not_qualifiable')
        scope = TRANSPORT_SCOPE
        if self.application_policy is not None:
            from .application_release import WORKER_SCOPE

            scope = WORKER_SCOPE
            if receipt.get('passed') is True and not (
                receipt.get('transport_passed') is True
                and receipt.get('interface_passed') is True
                and receipt.get('transport_cases') == 4
                and receipt.get('interface_cases') == 4
            ):
                raise ReleaseError('replica_qualification_mismatch')
        expected = {
            'qualification_scope': scope,
            'version_id': e.version_id,
            'worker': e.identity,
            'artifact_digest': e.binding.artifact_digest,
            'profile_digest': e.profile.digest,
            'panel_digest': self.panel_digest,
            'application_history': self.application_results[e.version_id],
        }
        if any(receipt.get(k) != value for k, value in expected.items()):
            raise ReleaseError('replica_qualification_mismatch')
        with self.store.atomic(durable=True):
            self.db.execute(
                'UPDATE pool_replicas SET state=?,qualification=? WHERE id=?',
                ('ready' if receipt.get('passed') is True else 'failed', canonical(receipt), rid),
            )
            if receipt.get('passed') is True and v['state'] in {'registered', 'retired'}:
                self.db.execute(
                    "UPDATE pool_versions SET state='ready' WHERE version=?", (e.version_id,)
                )
            self.store.event(None, 'replica_qualified', {'replica_id': rid, 'receipt': receipt})

    def pin(self, requested, arrival, *, durable=True):
        route = self.route()
        version = requested
        if requested == 'active':
            bucket = int.from_bytes(hashlib.sha256(arrival.encode()).digest()[:8], 'big') % 100
            version = (
                route['canary']
                if route['canary'] and bucket < route['percentage']
                else route['active']
            )
        if not version:
            raise ReleaseError('no_active_version', 503)
        row = self.row(version)
        if self.application_policy is not None:
            self.application_policy.require(self.contract['versions'][version])
            if row['state'] not in {'active', 'canary'}:
                raise ReleaseError('application_version_not_published', 503)
        if row['state'] not in {'ready', 'active', 'canary'}:
            raise ReleaseError('version_not_accepting', 503)
        if requested == 'active' and row['state'] not in {'active', 'canary'}:
            raise ReleaseError('route_requires_republication', 503)
        ready = self.eligible(version)
        candidates = [
            r
            for r in ready
            if self.pending_replica(r)
            < sum(self.contract['replicas'][r][k] for k in ('max_inflight', 'max_queue'))
        ]
        if not candidates:
            raise ReleaseError(
                'replica_capacity_full' if ready else 'no_ready_replica', 429 if ready else 503
            )
        rid = min(
            candidates,
            key=lambda r: (
                self.pending_replica(r) / self.contract['replicas'][r]['max_inflight'],
                r,
            ),
        )
        e = self.engines[rid]
        lease = {
            'version_id': version,
            'replica_id': rid,
            'worker_generation': e.identity['startup_generation'],
            'route_generation': route['generation'],
            'requested_model': requested,
            'qualification_scope': self.scope,
            **row['identity'],
        }
        self.store.bind_arrival(arrival, version, e.identity, lease, durable=durable)
        return {**lease, 'engine': e, 'slot_semaphore': self.slots[rid]}

    def change(self, operation, target, expected_generation, *, percentage=0):
        route = self.route()
        row = self.row(target)
        if type(expected_generation) is not int or expected_generation != route['generation']:
            raise ReleaseError('stale_route_generation')
        if operation == 'withdraw' and self.application_policy is not None:
            if route['active'] is None or target != route['active']:
                raise ReleaseError('withdraw_requires_active_target')
            pending = {v: self.pending(v) for v in {route['active'], route['canary']} - {None}}
            bound = [
                r['id']
                for r in self.db.execute('SELECT id,version FROM arrivals WHERE lease_released=0')
                if r['version'] in pending
            ]
            updated = {
                **route,
                'active': None,
                'canary': None,
                'percentage': 0,
                'generation': route['generation'] + 1,
            }
            with self.store.atomic(durable=True):
                self.db.execute(
                    'UPDATE pool_route SET generation=?,active=NULL,canary=NULL,percentage=0 WHERE id=1',
                    (updated['generation'],),
                )
                for version in {route['active'], route['canary']} - {None}:
                    self.db.execute(
                        "UPDATE pool_versions SET state='draining' WHERE version=?", (version,)
                    )
                self.store.event(
                    None,
                    'pool_route_changed',
                    {
                        'operation': operation,
                        'previous': route,
                        'current': updated,
                        'bound_requests_at_withdraw': pending,
                        'bound_arrivals': bound,
                    },
                )
            return {**updated, 'bound_requests_at_withdraw': pending, 'bound_arrivals': bound}
        if self.application_policy is not None:
            self.application_policy.require(self.contract['versions'][target])
            if operation == 'rollback' and (not route['active'] or route['active'] == target):
                raise ReleaseError('rollback_requires_different_active_version')
        if row['state'] not in {'ready', 'active', 'canary', 'draining'} or not self.eligible(
            target
        ):
            raise ReleaseError('target_not_qualified', 503)
        if operation == 'canary':
            if (
                not route['active']
                or target == route['active']
                or type(percentage) is not int
                or not 1 <= percentage <= 99
            ):
                raise ReleaseError('invalid_canary')
            updated = {**route, 'canary': target, 'percentage': percentage}
        elif operation in {'promote', 'rollback'}:
            if operation == 'rollback' and not row['ever_active']:
                raise ReleaseError('not_a_previous_active_version')
            updated = {**route, 'active': target, 'canary': None, 'percentage': 0}
        else:
            raise ReleaseError('unknown_release_operation', 422)
        updated['generation'] += 1
        with self.store.atomic(durable=True):
            self.db.execute(
                'UPDATE pool_route SET generation=?,active=?,canary=?,percentage=? WHERE id=1',
                (
                    updated['generation'],
                    updated['active'],
                    updated['canary'],
                    updated['percentage'],
                ),
            )
            for v in {route['active'], route['canary']} - {
                None,
                updated['active'],
                updated['canary'],
            }:
                self.db.execute("UPDATE pool_versions SET state='draining' WHERE version=?", (v,))
            self.db.execute(
                "UPDATE pool_versions SET state='active',ever_active=1 WHERE version=?",
                (updated['active'],),
            )
            if updated['canary']:
                self.db.execute(
                    "UPDATE pool_versions SET state='canary' WHERE version=?", (updated['canary'],)
                )
            self.store.event(
                None,
                'pool_route_changed',
                {'operation': operation, 'previous': route, 'current': updated},
            )
        return updated

    def drain_replica(self, rid):
        if self.replica(rid)['state'] not in {'ready', 'draining', 'quarantined'}:
            raise ReleaseError('replica_not_drainable')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE pool_replicas SET state='draining' WHERE id=?", (rid,))
            self.store.event(None, 'replica_draining', {'replica_id': rid})

    def reload_replica(self, rid):
        row = self.replica(rid)
        e = self.engines[rid]
        if row['state'] not in {
            'quarantined',
            'failed',
            'retired',
            'draining',
        } or self.pending_replica(rid):
            raise ReleaseError('replica_not_reloadable')
        if (
            row['identity'] != e.identity
            or e.profile.digest != self.row(e.version_id)['identity']['profile_digest']
        ):
            raise ReleaseError('replica_identity_changed')
        with self.store.atomic(durable=True):
            self.db.execute(
                "UPDATE pool_replicas SET state='loading',qualification=NULL WHERE id=?", (rid,)
            )
            self.store.event(None, 'replica_reload', {'replica_id': rid})
        self.observed_ready[rid] = False

    def retire_replica(self, rid):
        if self.replica(rid)['state'] != 'draining' or self.pending_replica(rid):
            raise ReleaseError('replica_not_drained')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE pool_replicas SET state='retired' WHERE id=?", (rid,))
            self.store.event(None, 'replica_retired', {'replica_id': rid})
        return {'replica_id': rid, 'state': 'retired', 'worker_stop_observed': False}

    def drain(self, version):
        route = self.route()
        row = self.row(version)
        if version in {route['active'], route['canary']} or row['state'] not in {
            'ready',
            'draining',
        }:
            raise ReleaseError('version_still_routed_or_not_drainable')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE pool_versions SET state='draining' WHERE version=?", (version,))
            self.store.event(None, 'version_draining', {'version': version})

    def retire(self, version):
        if self.row(version)['state'] != 'draining' or self.pending(version):
            raise ReleaseError('version_not_drained')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE pool_versions SET state='retired' WHERE version=?", (version,))
            self.db.execute("UPDATE pool_replicas SET state='retired' WHERE version=?", (version,))
            self.store.event(None, 'version_retired', {'version': version})
        return {'version_id': version, 'state': 'retired', 'worker_stop_observed': False}
