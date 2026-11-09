"""应用资格不能被传输资格、换名或自洽但错误的证明代替。"""

import importlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from serving_lab.lab import digest, file_hash
from serving_lab.profile import make_profile
from serving_lab.runtime import RuntimeStore
from serving_lab.standard_actor import StandardActor
from serving_lab.task_protocol import FAMILIES
from test_online import FakeEngine


def transport_panel():
    return json.loads(
        (Path(__file__).resolve().parents[1] / 'fixtures/serving/r2-transport-v2.json').read_text(
            encoding='utf-8'
        )
    )


def module():
    assert importlib.util.find_spec('serving_lab.application_release') is not None
    return importlib.import_module('serving_lab.application_release')


def setup(tmp_path):
    m = module()
    engines = {}
    certs = {}
    parameters = {}
    for version, passed in [('base', False), ('c0', True)]:
        folder = tmp_path / version
        folder.mkdir()
        model = folder / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (folder / 'launch.json').write_text(
            json.dumps(
                {
                    'command': ['--generation-config', 'vllm'],
                    'versions': {'vllm': '0.11.2'},
                    'config': {'dtype': 'bfloat16', 'max_model_len': 4096},
                }
            )
        )
        artifact = digest([version, 'artifact'])
        parameter = digest([version, 'parameter'])
        binding = SimpleNamespace(
            manifest_path=folder / 'binding.json',
            artifact_digest=artifact,
            model_path=model,
            served_model=version,
        )
        e = FakeEngine()
        e.version_id = version
        e.identity = {'worker_instance_id': version, 'startup_generation': 'g1'}
        e.binding = binding
        e.profile = make_profile(
            binding,
            SimpleNamespace(get_chat_template=lambda: 'template', eos_token_id=1),
            {'name': 'q2_public_choice_v1', 'seed': 20261103},
        )
        engines[version] = e
        parameters[version] = {'digest': parameter}
        certs[artifact] = {
            'artifact_digest': artifact,
            'parameter_digest': parameter,
            'profile_digest': e.profile.digest,
            'profile': e.profile.document,
            'case_results': [
                {
                    'task_id': f'cs_dev_{i // 4 + 1:02}_{i % 4 + 1:02}',
                    'family': FAMILIES[i // 4],
                    'success': passed and i != 23,
                }
                for i in range(24)
            ],
            'source': {'run': 'software_fixture', 'audit_sha256': digest(version)},
        }
    bundle = {
        'schema': 1,
        'name': m.BUNDLE_NAME,
        'policy': m.policy_document(),
        'certificates': certs,
        'interface_panel': {
            'name': 'public_command_interface_v1',
            'cases': [
                {
                    'id': str(i),
                    'messages': [{'role': 'user', 'content': 'Select one available command.'}],
                    'choices': choices,
                }
                for i, choices in enumerate(
                    [
                        ['look', 'inventory'],
                        ['look'],
                        ['take apple 1 from table 1', 'take apple 2 from table 1'],
                        ['go to cabinet 1', 'open cabinet 1'],
                    ]
                )
            ],
        },
        # 测试替身模拟导入器格式，测试通过不构成真实模型证据。
        'source_verification': {
            'producer': 'serving_lab.coldstart_release_import.v1',
            'evidence_level': 'real_local_model_evaluation',
        },
    }
    bundle['bundle_digest'] = digest(bundle)
    policy = m.ApplicationPolicy(bundle)
    applications = policy.bind(engines, parameters)
    return m, engines, parameters, policy, applications, bundle


def ready(pool, rid):
    e = pool.engines[rid]
    # 这里仅是控制器软件替身，真实worker资格由独立运行核验。
    receipt = {
        'qualification_scope': module().WORKER_SCOPE,
        'passed': True,
        'version_id': e.version_id,
        'worker': e.identity,
        'artifact_digest': e.binding.artifact_digest,
        'profile_digest': e.profile.digest,
        'panel_digest': pool.panel_digest,
        'application_history': pool.application_results[e.version_id],
        'transport_passed': True,
        'interface_passed': True,
        'transport_cases': 4,
        'interface_cases': 4,
    }
    pool.qualified(rid, receipt)
    pool.observed_ready[rid] = True


def test_bad_quality_cannot_pin_publish_or_rollback_even_when_transport_ready(tmp_path):
    m, engines, parameters, policy, apps, _ = setup(tmp_path)
    from serving_lab.replicas import ReplicaController

    with RuntimeStore(tmp_path / 'runtime') as store:
        pool = ReplicaController(
            store,
            engines,
            parameters,
            'panel',
            {k: {'max_inflight': 1, 'max_queue': 2} for k in engines},
            apps,
            application_policy=policy,
        )
        assert pool.scope == m.POOL_SCOPE
        for rid in engines:
            ready(pool, rid)
        for op in ['promote', 'canary', 'rollback']:
            with pytest.raises(ValueError, match='application_quality_rejected'):
                pool.change(op, 'base', 0)
        store.record_arrival('bad', None)
        with pytest.raises(ValueError, match='application_quality_rejected'):
            pool.pin('base', 'bad')
        store.record_arrival('before_publish', None)
        with pytest.raises(ValueError, match='application_version_not_published'):
            pool.pin('c0', 'before_publish')
        assert pool.change('promote', 'c0', 0)['active'] == 'c0'
        store.record_arrival('ok', None)
        lease = pool.pin('active', 'ok')
        assert lease['artifact_digest'] == engines['c0'].binding.artifact_digest
        assert lease['application_history']['successes'] == 23


def test_withdraw_works_without_ready_worker_keeps_existing_binding_and_restore_is_explicit(
    tmp_path,
):
    m, engines, parameters, policy, apps, _ = setup(tmp_path)
    from serving_lab.replicas import ReplicaController

    with RuntimeStore(tmp_path / 'runtime') as store:
        pool = ReplicaController(
            store,
            engines,
            parameters,
            'panel',
            {k: {'max_inflight': 1, 'max_queue': 2} for k in engines},
            apps,
            application_policy=policy,
        )
        ready(pool, 'c0')
        pool.change('promote', 'c0', 0)
        store.record_arrival('inflight', None)
        lease = pool.pin('active', 'inflight')
        pool.observed_ready['c0'] = False
        assert pool.change('withdraw', 'c0', 1)['active'] is None
        assert (
            pool.pending('c0') == 1
            and lease['engine'] is engines['c0']
            and lease['route_generation'] == 1
        )
        store.record_arrival('new', None)
        with pytest.raises(ValueError, match='no_active_version'):
            pool.pin('active', 'new')
        pool.observed_ready['c0'] = True
        with pytest.raises(ValueError, match='rollback_requires_different_active_version'):
            pool.change('rollback', 'c0', 2)
        assert pool.change('promote', 'c0', 2)['active'] == 'c0'


def test_certificate_is_bound_to_file_hash_actual_parameters_and_profile(tmp_path):
    m, engines, parameters, policy, apps, bundle = setup(tmp_path)
    p = tmp_path / 'bundle.json'
    p.write_text(json.dumps(bundle))
    anchor = file_hash(p)
    assert m.load_policy(p, anchor).document == policy.document
    changed = json.loads(p.read_text())
    changed['certificates'][engines['base'].binding.artifact_digest]['case_results'][0][
        'success'
    ] = True
    changed['bundle_digest'] = digest({k: v for k, v in changed.items() if k != 'bundle_digest'})
    p.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        m.load_policy(p, anchor)
    wrong_source = json.loads(json.dumps(bundle))
    wrong_source['source_verification']['evidence_level'] = 'control_plane'
    wrong_source['bundle_digest'] = digest(
        {k: v for k, v in wrong_source.items() if k != 'bundle_digest'}
    )
    with pytest.raises(ValueError):
        m.ApplicationPolicy(wrong_source)
    wrong = {**parameters, 'c0': {'digest': digest('different parameters')}}
    with pytest.raises(ValueError):
        policy.bind(engines, wrong)
    engines['c0'].profile = engines['base'].profile
    with pytest.raises(ValueError):
        policy.bind(engines, parameters)


def test_mixed_local_cloud_import_requires_explicit_producer_and_linked_audit(tmp_path):
    m, engines, _, _, _, bundle = setup(tmp_path)
    artifact = engines['c0'].binding.artifact_digest
    bundle['certificates'][artifact]['source']['raw_audit_sha256'] = 'd' * 64
    bundle['source_verification'] = {
        'producer': 'serving_lab.application_release_import.v2',
        'evidence_level': 'real_local_and_cloud_model_evaluation',
        'parent_bundle_sha256': 'a' * 64,
        'added_audits': {artifact: 'd' * 64},
    }

    def seal():
        bundle['bundle_digest'] = digest({k: v for k, v in bundle.items() if k != 'bundle_digest'})

    seal()
    m.ApplicationPolicy(bundle)
    bundle['source_verification']['added_audits'][artifact] = 'e' * 64
    seal()
    with pytest.raises(ValueError):
        m.ApplicationPolicy(bundle)
    bundle['source_verification']['added_audits'][artifact] = 'd' * 64
    bundle['source_verification']['producer'] = 'serving_lab.coldstart_release_import.v1'
    seal()
    with pytest.raises(ValueError):
        m.ApplicationPolicy(bundle)


@pytest.mark.parametrize(
    'family_successes,passed',
    [([3, 3, 3, 3, 3, 2], False), ([4, 4, 4, 3, 2, 1], False), ([3, 3, 3, 3, 3, 3], True)],
)
def test_exact_global_and_each_family_quality_thresholds(tmp_path, family_successes, passed):
    m, engines, parameters, _, _, bundle = setup(tmp_path)
    cert = bundle['certificates'][engines['c0'].binding.artifact_digest]
    for i, row in enumerate(cert['case_results']):
        row['success'] = i % 4 < family_successes[i // 4]
    bundle['bundle_digest'] = digest({k: v for k, v in bundle.items() if k != 'bundle_digest'})
    assert m.ApplicationPolicy(bundle).bind(engines, parameters)['c0']['passed'] is passed


def test_application_request_keeps_constraints_and_rejects_wrong_actor_or_legacy_bypass(tmp_path):
    m, engines, parameters, policy, apps, _ = setup(tmp_path)
    from serving_lab.online import Service, ServiceLimits, Rejected

    view = {
        'feedback': 'Your task is to: put a mug on table.',
        'admissible_commands': ['go to table 1'],
    }
    prepared = StandardActor().prepare(view, view, [], lambda ms: [1, 2])
    body = {
        'model': 'c0',
        'messages': prepared['messages'],
        'action_choices': prepared['choices'],
        'max_tokens': 64,
        'stream': False,
    }
    service = Service(
        engines['c0'],
        tmp_path / 'request-runtime',
        ServiceLimits(max_input_tokens=3500, max_tokens=64),
    )
    try:
        service.pool = SimpleNamespace(application_policy=policy)
        assert service.validate(body)['action_choices'] == prepared['choices']
        with pytest.raises(Rejected):
            service.validate({k: v for k, v in body.items() if k != 'action_choices'})
        wrong = {**body, 'messages': [{'role': 'user', 'content': 'unqualified arbitrary prompt'}]}
        with pytest.raises(Rejected):
            service.validate(wrong)
        service.pool = None
        with pytest.raises(Rejected):
            service.validate(body)
    finally:
        service.store.close()


async def test_actual_http_consumer_preserves_choices_and_withdraw_does_not_wait_for_worker(
    tmp_path,
):
    import asyncio
    import httpx
    from serving_lab.replica_api import create_replica_app
    from serving_lab.online import ServiceLimits

    m, engines, parameters, policy, apps, _ = setup(tmp_path)
    seen = []

    class Response:
        status_code = 200

        def __init__(self, e, attempt, body):
            self.e, self.attempt, self.body = e, attempt, body

        async def aread(self):
            await self.e.release.wait()
            self.e.released[self.attempt] = {'status': 'FINISHED_STOPPED'}
            return json.dumps(
                {
                    'id': 'chatcmpl-' + self.attempt,
                    'model': self.e.version_id,
                    'choices': [{'message': {'content': 'look'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
                }
            ).encode()

        async def aclose(self):
            self.e.released.setdefault(self.attempt, {'status': 'FINISHED_ABORTED'})

    for e in engines.values():

        async def token_ids(ms):
            return [1, 2]

        e.token_ids = token_ids
        e.wire_payload = e.profile.payload

        async def opening(attempt, body, e=e):
            wire = e.profile.payload(body)
            if wire.get('structured_outputs') != {'choice': ['inventory', 'look']}:
                raise ValueError('constraint lost before engine')
            seen.append(wire)
            e.started.set()
            return Response(e, attempt, body)

        e.open = opening
    panel = transport_panel()
    app = create_replica_app(
        engines,
        tmp_path / 'http-runtime',
        parameters,
        panel,
        'secret-long-enough',
        {k: {'max_inflight': 1, 'max_queue': 2} for k in engines},
        apps,
        limits=ServiceLimits(max_input_tokens=3500, max_tokens=64),
        application_policy=policy,
    )
    v = {'feedback': 'Your task is to: find a mug.', 'admissible_commands': ['look']}
    prepared = StandardActor().prepare(v, v, [], lambda ms: [1, 2])
    body = {
        'model': 'active',
        'messages': prepared['messages'],
        'action_choices': prepared['choices'],
        'max_tokens': 64,
        'stream': False,
    }
    async with app.router.lifespan_context(app):
        pool = app.state.service.pool
        for rid in engines:
            ready(pool, rid)
        pool.change('promote', 'c0', 0)
        engines['c0'].release.clear()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            running = asyncio.create_task(client.post('/v1/chat/completions', json=body))
            await engines['c0'].started.wait()

            async def unavailable():
                await asyncio.Future()

            original = engines['c0'].ready
            engines['c0'].ready = unavailable
            response = await asyncio.wait_for(
                client.post(
                    '/admin/route',
                    headers={'Authorization': 'Bearer secret-long-enough'},
                    json={'operation': 'withdraw', 'target': 'c0', 'expected_generation': 1},
                ),
                0.5,
            )
            assert (
                response.status_code == 200
                and response.json()['bound_requests_at_withdraw']['c0'] == 1
            )
            engines['c0'].ready = original
            engines['c0'].release.set()
            old = await running
            assert old.status_code == 200 and old.headers['x-model-version'] == 'c0'
            assert old.headers['x-application-evidence'] == apps['c0']['evidence_digest']
            assert (await client.post('/v1/chat/completions', json=body)).status_code == 503
        assert len(seen) == 1 and pool.pending('c0') == 0


def test_new_application_scope_rejects_old_runtime_before_recovering_its_rows(tmp_path):
    import sqlite3
    from serving_lab.replica_api import create_replica_app
    from serving_lab.replicas import ReplicaController
    from serving_lab.online import ServiceLimits

    m, engines, parameters, policy, apps, _ = setup(tmp_path)
    runtime = tmp_path / 'old-runtime'
    caps = {k: {'max_inflight': 1, 'max_queue': 1} for k in engines}
    with RuntimeStore(runtime) as store:
        ReplicaController(store, engines, parameters, 'old-panel', caps, apps)
        store.record_arrival('pending', None)
    with sqlite3.connect(runtime / 'service.sqlite') as db:
        before = db.execute('SELECT * FROM arrivals').fetchall()
    with pytest.raises(ValueError):
        create_replica_app(
            engines,
            runtime,
            parameters,
            transport_panel(),
            'secret-long-enough',
            caps,
            apps,
            limits=ServiceLimits(max_input_tokens=3500, max_tokens=64),
            application_policy=policy,
        )
    with sqlite3.connect(runtime / 'service.sqlite') as db:
        assert db.execute('SELECT * FROM arrivals').fetchall() == before


@pytest.mark.parametrize('bad_choice', [False, True])
async def test_worker_needs_real_choice_probes_in_addition_to_transport(tmp_path, bad_choice):
    import httpx

    m, engines, _, policy, apps, _ = setup(tmp_path)
    e = engines['c0']
    emitted = []

    async def tokens(messages):
        return [1, 2]

    e.token_ids = e.engine_token_ids = tokens
    e.wire_payload = e.profile.payload

    async def opening(attempt, body):
        wire = e.wire_payload(body)
        emitted.append(wire)
        choices = body.get('action_choices')
        content = ('wrong-command' if bad_choice else choices[0]) if choices else 'OK'
        finish = 'length' if body['max_tokens'] == 1 else 'stop'
        e.released[attempt] = {
            'status': 'FINISHED_LENGTH_CAPPED' if finish == 'length' else 'FINISHED_STOPPED',
            'request_id': 'chatcmpl-' + attempt,
            'worker_instance_id': 'c0',
            'startup_generation': 'g1',
        }
        data = {
            'id': 'chatcmpl-' + attempt,
            'model': 'c0',
            'choices': [
                {
                    'message': {'content': content},
                    'delta': {'content': content},
                    'finish_reason': finish,
                }
            ],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
        }
        if body['stream']:
            return httpx.Response(200, text='data: ' + json.dumps(data) + '\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json=data)

    e.open = opening
    panel = transport_panel()
    with RuntimeStore(tmp_path / 'qualification-runtime') as store:
        result = await m.qualify_application_worker(
            e, panel, tmp_path / 'qualification', store, policy, apps['c0']
        )
        assert result['transport_passed'] and result['passed'] is (not bad_choice)
        assert result['interface_cases'] == (1 if bad_choice else 4)
        assert len(emitted) == (5 if bad_choice else 8)
        assert all('structured_outputs' in wire for wire in emitted[4:])
        assert all(r['lease_released'] == 1 for r in store.rows())


async def test_oversized_qualification_panel_is_rejected_before_any_call(tmp_path):
    from serving_lab.profile import TRANSPORT_SCOPE

    m, engines, _, policy, apps, _ = setup(tmp_path)
    e = engines['c0']

    async def tokens(messages):
        return [1]

    e.token_ids = e.engine_token_ids = tokens
    e.wire_payload = e.profile.payload
    panel = {
        'qualification_scope': TRANSPORT_SCOPE,
        'deadline_seconds': 1,
        'required_passes': 100,
        'cases': [
            {
                'id': str(i),
                'messages': [{'role': 'user', 'content': 'x'}],
                'max_tokens': 64,
                'stream': False,
                'expected_finish': 'stop',
            }
            for i in range(100)
        ],
    }
    with RuntimeStore(tmp_path / 'panel-runtime') as store:
        with pytest.raises(ValueError):
            await m.qualify_application_worker(
                e, panel, tmp_path / 'must-not-create', store, policy, apps['c0']
            )
        assert not e.calls and not store.rows() and not (tmp_path / 'must-not-create').exists()


def test_application_factory_rejects_wrong_probe_count_before_runtime_creation(tmp_path):
    from serving_lab.replica_api import create_replica_app
    from serving_lab.profile import TRANSPORT_SCOPE
    from serving_lab.online import ServiceLimits

    _, engines, parameters, policy, apps, _ = setup(tmp_path)
    app = None
    try:
        with pytest.raises(ValueError):
            app = create_replica_app(
                engines,
                tmp_path / 'must-not-create',
                parameters,
                {'qualification_scope': TRANSPORT_SCOPE, 'cases': [], 'required_passes': 0},
                'secret-long-enough',
                {k: {'max_inflight': 1, 'max_queue': 2} for k in engines},
                apps,
                limits=ServiceLimits(max_input_tokens=3500, max_tokens=64),
                application_policy=policy,
            )
    finally:
        if app:
            app.state.service.store.close()
