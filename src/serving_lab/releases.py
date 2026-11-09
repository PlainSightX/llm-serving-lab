"""单控制者发布状态；路由提交和请求绑定之间不跨异步等待。"""

from __future__ import annotations

import hashlib
import json

from .lab import canonical, identifier
from .profile import TRANSPORT_SCOPE, LEGACY_SCOPE


class ReleaseError(ValueError):
    def __init__(self, code, status=409):
        self.code = code
        self.status = status
        super().__init__(code)


class ReleaseController:
    def __init__(
        self,
        store,
        engines,
        parameters,
        panel_digest,
        *,
        allow_rebind=(),
        scope=LEGACY_SCOPE,
        application_results=None,
    ):
        self.store = store
        self.db = store.db
        self.engines = engines
        self.panel_digest = panel_digest
        self.scope = scope
        self.application_results = application_results or {}
        if scope not in {LEGACY_SCOPE, TRANSPORT_SCOPE}:
            raise ValueError('未知资格用途')
        if scope == TRANSPORT_SCOPE and (
            set(self.application_results) != set(engines)
            or any(not getattr(e, 'profile', None) for e in engines.values())
        ):
            raise ValueError('新合同缺少profile或原应用结果')
        if not engines or set(engines) != set(parameters):
            raise ValueError('版本与参数身份不完整')
        if len({p['digest'] for p in parameters.values()}) != len(parameters):
            raise ValueError('相同参数不能换名注册为不同版本')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS releases(version TEXT PRIMARY KEY,state TEXT,identity TEXT,quality TEXT,ever_active INTEGER DEFAULT 0);
          CREATE TABLE IF NOT EXISTS route(id INTEGER PRIMARY KEY CHECK(id=1),generation INTEGER,active TEXT,canary TEXT,percentage INTEGER);
          INSERT OR IGNORE INTO route VALUES(1,0,NULL,NULL,0);
        ''')
        self.store.durable()
        if scope == TRANSPORT_SCOPE:
            document = {
                'panel_digest': panel_digest,
                'profiles': {v: e.profile.digest for v, e in engines.items()},
                'application_results': self.application_results,
            }
            self.db.execute(
                'CREATE TABLE IF NOT EXISTS release_contract(id INTEGER PRIMARY KEY,scope TEXT,document TEXT)'
            )
            self.db.execute(
                'INSERT OR IGNORE INTO release_contract VALUES(1,?,?)', (scope, canonical(document))
            )
            self.store.durable()
            stored = self.db.execute(
                'SELECT scope,document FROM release_contract WHERE id=1'
            ).fetchone()
            if stored['scope'] != scope or stored['document'] != canonical(document):
                raise ValueError('合同或profile身份变化')
        existing = {r['version'] for r in self.db.execute('SELECT version FROM releases')}
        if existing - set(engines):
            raise ValueError('恢复不能遗忘已登记的实例与原请求')
        with self.store.atomic(durable=True):
            for version, engine in engines.items():
                identifier(version)
                if version == 'active' or engine.version_id != version:
                    raise ValueError('版本标识冲突')
                identity = {
                    'parameter_digest': parameters[version]['digest'],
                    'artifact_digest': engine.binding.artifact_digest,
                    'binding_sha256': engine.binding.manifest_sha256,
                    'worker': engine.identity,
                    'panel_digest': panel_digest,
                }
                if scope == TRANSPORT_SCOPE:
                    identity.update(
                        profile_digest=engine.profile.digest,
                        qualification_scope=scope,
                        application_history=self.application_results[version],
                    )
                old = self.db.execute(
                    'SELECT * FROM releases WHERE version=?', (version,)
                ).fetchone()
                if old:
                    previous = json.loads(old['identity'])
                    if any(
                        previous[k] != identity[k]
                        for k in ('parameter_digest', 'artifact_digest', 'panel_digest')
                    ):
                        raise ValueError('已登记版本的参数、工件或质量门发生改变')
                    if previous != identity:
                        if version not in allow_rebind:
                            raise ValueError('新worker代际需要显式rebind')
                        self.db.execute(
                            "UPDATE releases SET state='loading',identity=?,quality=NULL WHERE version=?",
                            (canonical(identity), version),
                        )
                        self.store.event(
                            None,
                            'release_rebind',
                            {'version': version, 'previous': previous, 'current': identity},
                        )
                else:
                    self.db.execute(
                        "INSERT INTO releases VALUES(?,'registered',?,NULL,0)",
                        (version, canonical(identity)),
                    )
                    for state in ('registered', 'validated', 'loading'):
                        self.db.execute(
                            'UPDATE releases SET state=? WHERE version=?', (state, version)
                        )
                        self.store.event(
                            None,
                            'release_state',
                            {'version': version, 'state': state, 'identity': identity},
                        )

    def row(self, version):
        row = self.db.execute('SELECT * FROM releases WHERE version=?', (version,)).fetchone()
        if not row:
            raise ReleaseError('unknown_model', 404)
        result = dict(row)
        result['identity'] = json.loads(result['identity'])
        result['quality'] = json.loads(result['quality']) if result['quality'] else None
        return result

    def route(self):
        return dict(self.db.execute('SELECT * FROM route WHERE id=1').fetchone())

    def snapshot(self):
        return {
            'qualification_scope': self.scope,
            'route': self.route(),
            'versions': [{**self.row(v), 'pending': self.pending(v)} for v in self.engines],
        }

    def qualified(self, version, receipt):
        row = self.row(version)
        identity = row['identity']
        if self.scope == TRANSPORT_SCOPE and (
            receipt.get('qualification_scope') != self.scope
            or receipt.get('profile_digest') != identity['profile_digest']
            or receipt.get('application_history') != identity['application_history']
        ):
            raise ReleaseError('qualification_scope_or_profile_mismatch')
        if row['state'] != 'loading':
            raise ReleaseError('qualification_not_allowed')
        if self.pending(version) and receipt.get('passed') is True:
            raise ReleaseError('orphan_leases_not_reconciled')
        if (
            receipt.get('version_id') != version
            or receipt.get('panel_digest') != self.panel_digest
            or receipt.get('worker') != identity['worker']
            or receipt.get('artifact_digest') != identity['artifact_digest']
        ):
            raise ReleaseError('qualification_identity_mismatch')
        state = 'ready' if receipt.get('passed') is True else 'failed'
        with self.store.atomic(durable=True):
            self.db.execute(
                'UPDATE releases SET state=?,quality=? WHERE version=?',
                (state, canonical(receipt), version),
            )
            self.store.event(
                None, 'qualification', {'version': version, 'state': state, 'receipt': receipt}
            )

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
        if version is None:
            raise ReleaseError('no_active_version', 503)
        row = self.row(version)
        if requested == 'active' and row['state'] not in {'active', 'canary'}:
            raise ReleaseError('route_requires_republication', 503)
        if row['state'] not in {'ready', 'canary', 'active'}:
            raise ReleaseError('version_not_accepting', 503)
        engine = self.engines[version]
        if (
            self.scope == TRANSPORT_SCOPE
            and engine.profile.digest != row['identity']['profile_digest']
        ):
            raise ReleaseError('profile_changed', 503)
        lease = {
            'version_id': version,
            'route_generation': route['generation'],
            'requested_model': requested,
            'parameter_digest': row['identity']['parameter_digest'],
            'artifact_digest': row['identity']['artifact_digest'],
        }
        if self.scope == TRANSPORT_SCOPE:
            lease.update(
                profile_digest=engine.profile.digest,
                qualification_scope=self.scope,
                application_history=row['identity']['application_history'],
            )
        self.store.bind_arrival(arrival, version, engine.identity, lease, durable=durable)
        return {**lease, 'engine': engine}

    def change(self, operation, target, expected_generation, *, percentage=0):
        route = self.route()
        row = self.row(target)
        if (
            self.scope == TRANSPORT_SCOPE
            and self.engines[target].profile.digest != row['identity']['profile_digest']
        ):
            raise ReleaseError('profile_changed')
        if type(expected_generation) is not int or expected_generation != route['generation']:
            raise ReleaseError('stale_route_generation')
        if (
            row['state'] not in {'ready', 'canary', 'active', 'draining'}
            or not row['quality']
            or row['quality'].get('passed') is not True
        ):
            raise ReleaseError('target_not_qualified')
        if operation == 'canary':
            if (
                not route['active']
                or route['active'] == target
                or type(percentage) is not int
                or not 1 <= percentage <= 99
            ):
                raise ReleaseError('invalid_canary')
            if self.row(route['active'])['state'] != 'active':
                raise ReleaseError('active_requires_republication')
            updated = {**route, 'canary': target, 'percentage': percentage}
        elif operation in {'promote', 'rollback'}:
            if operation == 'rollback' and not row['ever_active']:
                raise ReleaseError('not_a_previous_active_version')
            if target == route['active'] and route['canary'] is None and row['state'] == 'active':
                raise ReleaseError('route_unchanged')
            updated = {**route, 'active': target, 'canary': None, 'percentage': 0}
        else:
            raise ReleaseError('unknown_release_operation', 422)
        updated['generation'] += 1
        with self.store.atomic(durable=True):
            self.db.execute(
                'UPDATE route SET generation=?,active=?,canary=?,percentage=? WHERE id=1',
                (
                    updated['generation'],
                    updated['active'],
                    updated['canary'],
                    updated['percentage'],
                ),
            )
            for version in {route['active'], route['canary']} - {
                None,
                updated['active'],
                updated['canary'],
            }:
                self.db.execute(
                    "UPDATE releases SET state='draining' WHERE version=? AND state IN ('active','canary','ready')",
                    (version,),
                )
            self.db.execute(
                "UPDATE releases SET state='active',ever_active=1 WHERE version=?",
                (updated['active'],),
            )
            if updated['canary']:
                self.db.execute(
                    "UPDATE releases SET state='canary' WHERE version=?", (updated['canary'],)
                )
            self.store.event(
                None,
                'route_changed',
                {
                    'operation': operation,
                    'previous': route,
                    'current': updated,
                    'qualification_scope': self.scope,
                    'application_history': self.application_results,
                    'profiles': {
                        v: self.row(v)['identity'].get('profile_digest') for v in self.engines
                    },
                },
            )
        return updated

    def pending(self, version):
        return self.db.execute(
            'SELECT COUNT(*) FROM arrivals WHERE version=? AND lease_released=0', (version,)
        ).fetchone()[0]

    def drain(self, version):
        route = self.route()
        row = self.row(version)
        if version in {route['active'], route['canary']} or row['state'] not in {
            'ready',
            'draining',
        }:
            raise ReleaseError('version_still_routed_or_not_drainable')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE releases SET state='draining' WHERE version=?", (version,))
            self.store.event(
                None, 'drain_requested', {'version': version, 'pending': self.pending(version)}
            )

    def retire(self, version):
        if self.row(version)['state'] != 'draining' or self.pending(version):
            raise ReleaseError('version_not_drained')
        with self.store.atomic(durable=True):
            self.db.execute("UPDATE releases SET state='retired' WHERE version=?", (version,))
            self.store.event(None, 'retired', {'version': version, 'worker_stop_observed': False})
        return {'state': 'retired', 'worker_stop_observed': False, 'version_id': version}
