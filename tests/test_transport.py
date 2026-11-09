"""预资格验证协议与执行，不能隐含旧应用题通过或整体发布完成。"""

import json
from types import SimpleNamespace
import httpx
import pytest
from serving_lab.profile import ServingProfile, TRANSPORT_SCOPE


@pytest.mark.parametrize('fault', [None, 'model', 'tokens', 'usage', 'finish', 'done', 'release'])
async def test_transport_requires_full_protocol_and_release(tmp_path, fault):
    from serving_lab.transport import qualify_transport
    from serving_lab.runtime import RuntimeStore

    class Engine:
        version_id = 'test'
        identity = {'worker_instance_id': 'worker', 'startup_generation': 'g'}
        binding = SimpleNamespace(artifact_digest='artifact')
        profile = ServingProfile(
            {'name': 'test', 'max_tokens': 128, 'context_tokens': 4096}, 'test'
        )

        async def ready(self):
            return True

        async def token_ids(self, messages):
            return [1, 2]

        async def engine_token_ids(self, messages):
            return [2, 1] if fault == 'tokens' else [1, 2]

        def wire_payload(self, body):
            return self.profile.payload(body)

        async def open(self, attempt, body):
            data = {
                'model': 'other' if fault == 'model' else 'test',
                'choices': [
                    {
                        'message': {'content': 'ANY'},
                        'finish_reason': 'length' if fault == 'finish' else 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': 3 if fault == 'usage' else 2,
                    'completion_tokens': 1,
                    'total_tokens': 3,
                },
            }
            if body['stream']:
                data['choices'][0]['delta'] = {'content': 'ANY'}
                return httpx.Response(200, text='data: ' + json.dumps(data) + '\n\n')
            return httpx.Response(200, json=data)

        async def wait_release(self, attempt, timeout):
            return None if fault == 'release' else {'status': 'FINISHED_STOPPED'}

    panel = {
        'qualification_scope': TRANSPORT_SCOPE,
        'deadline_seconds': 1,
        'required_passes': 1,
        'cases': [
            {
                'id': 'plain',
                'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 8,
                'stream': fault == 'done',
                'expected_finish': 'stop',
            }
        ],
    }
    application = {'status': 'failed', 'passed_cases': 3, 'cases': 8}
    with RuntimeStore(tmp_path / 'runtime') as store:
        result = await qualify_transport(
            Engine(), panel, tmp_path / 'qualification', store=store, application_result=application
        )
        assert result['passed'] is (fault is None)
        assert (
            result['application_history'] == application
            and result['system_acceptance'] == 'not_established_by_prequalification'
        )


def test_profile_and_scope_cannot_reuse_old_receipt_or_runtime(tmp_path):
    from serving_lab.releases import ReleaseController
    from serving_lab.runtime import RuntimeStore
    from serving_lab.profile import check_runtime_contract
    from test_releases import engines

    workers = engines()
    application = {v: {'status': 'failed', 'passed_cases': 3, 'cases': 8} for v in workers}
    for v, e in workers.items():
        e.profile = ServingProfile({'name': 'test', 'max_tokens': 128}, v)
    with RuntimeStore(tmp_path) as store:
        pool = ReleaseController(
            store,
            workers,
            {v: {'digest': v} for v in workers},
            'panel',
            scope=TRANSPORT_SCOPE,
            application_results=application,
        )
        receipt = {
            'passed': True,
            'version_id': 'small',
            'panel_digest': 'panel',
            'artifact_digest': 'small-bundle',
            'worker': workers['small'].identity,
        }
        with pytest.raises(ValueError):
            pool.qualified('small', receipt)
        receipt.update(
            qualification_scope=TRANSPORT_SCOPE,
            profile_digest=workers['small'].profile.digest,
            application_history=application['small'],
        )
        pool.qualified('small', receipt)
        pool.change('promote', 'small', 0)
        workers['small'].profile = ServingProfile({'name': 'changed', 'max_tokens': 128}, 'small')
        store.record_arrival('new', None)
        with pytest.raises(ValueError):
            pool.pin('active', 'new')
    before = (tmp_path / 'service.sqlite').read_bytes()
    with pytest.raises(ValueError):
        check_runtime_contract(tmp_path, TRANSPORT_SCOPE, {'profiles': 'changed'})
    assert (tmp_path / 'service.sqlite').read_bytes() == before
