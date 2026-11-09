"""服务内容门与释放门独立；失败不得被成功HTTP掩盖。"""

import json
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.parametrize('fault', [None, 'content', 'finish', 'release'])
async def test_quality_requires_content_finish_and_resource_release(tmp_path, fault):
    from serving_lab.qualification import qualify_engine
    from serving_lab.runtime import RuntimeStore

    class Engine:
        version_id = 'test'
        identity = {'worker_instance_id': 'worker', 'startup_generation': 'g1'}
        binding = SimpleNamespace(artifact_digest='artifact')

        async def ready(self):
            return True

        async def count_tokens(self, messages):
            return 5

        async def open(self, attempt, body):
            return httpx.Response(
                200,
                json={
                    'choices': [
                        {
                            'message': {'content': 'WRONG' if fault == 'content' else 'OK'},
                            'finish_reason': 'length' if fault == 'finish' else 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 5, 'completion_tokens': 1},
                },
            )

        async def wait_release(self, attempt, timeout):
            return None if fault == 'release' else {'status': 'FINISHED_STOPPED'}

    panel = {
        'name': 'test',
        'max_tokens': 8,
        'deadline_seconds': 1,
        'required_passes': 1,
        'cases': [
            {
                'id': 'literal',
                'messages': [{'role': 'user', 'content': 'OK'}],
                'expected_text': 'OK',
            }
        ],
    }
    with RuntimeStore(tmp_path / 'runtime') as store:
        result = await qualify_engine(Engine(), panel, tmp_path / 'qualification', store=store)
        assert result['passed'] is (fault is None)
        row = store.rows()[0]
        assert row['dispatched'] == 1 and row['lease_released'] == int(fault != 'release')
        assert (
            json.loads((tmp_path / 'qualification/receipt.json').read_text())['passed']
            == result['passed']
        )


async def test_qualification_timeout_keeps_counting_lease_until_task_finishes(tmp_path):
    import asyncio
    from serving_lab.qualification import qualify_engine
    from serving_lab.runtime import RuntimeStore

    gate = asyncio.Event()
    cleanups = set()

    class Engine:
        version_id = 'test'
        identity = {'worker_instance_id': 'worker', 'startup_generation': 'g1'}
        binding = SimpleNamespace(artifact_digest='artifact')

        async def ready(self):
            return True

        async def count_tokens(self, messages):
            await gate.wait()
            return 1

    panel = {
        'name': 'test',
        'max_tokens': 8,
        'deadline_seconds': 0.01,
        'required_passes': 1,
        'cases': [
            {
                'id': 'literal',
                'messages': [{'role': 'user', 'content': 'OK'}],
                'expected_text': 'OK',
            }
        ],
    }
    with RuntimeStore(tmp_path / 'runtime') as store:
        receipt = await qualify_engine(
            Engine(), panel, tmp_path / 'qualification', store=store, cleanups=cleanups
        )
        assert not receipt['passed'] and store.rows()[0]['lease_released'] == 0
        gate.set()
        await asyncio.gather(*cleanups)
        assert store.rows()[0]['lease_released'] == 1
