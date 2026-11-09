"""预约后异常不能以同身份补发；真实结果与原始wire绑定。"""

import pytest
from serving_lab.runtime import RuntimeStore
from serving_lab.research_action import generate_action


class BrokenEngine:
    version_id = 'v'
    identity = {'generation': 'test'}

    class Profile:
        digest = 'profile'

    profile = Profile()

    def wire_payload(self, body):
        return body

    async def ready(self):
        return True

    async def engine_token_ids(self, messages):
        return [1, 2]

    async def open(self, attempt, body):
        raise ConnectionError('controlled external failure')

    async def wait_release(self, attempt, timeout):
        return None


@pytest.mark.asyncio
async def test_reserved_action_cannot_be_resent_after_connection_failure(tmp_path):
    out = tmp_path / 'step'
    with RuntimeStore(tmp_path / 'runtime') as store:
        with pytest.raises(RuntimeError):
            await generate_action(
                BrokenEngine(), store, out, [{'role': 'user', 'content': 'hi'}], [1, 2], {'step': 1}
            )
        assert (out / 'reservation.json').exists() and (out / 'receipt.json').exists()
        with pytest.raises(FileExistsError):
            await generate_action(BrokenEngine(), store, out, [], [], {})
        rows = store.db.execute('SELECT state FROM arrivals').fetchall()
        assert len(rows) == 1 and rows[0][0] == 'failed'


@pytest.mark.asyncio
async def test_bounded_readiness_recheck_still_dispatches_generation_once(tmp_path):
    import httpx

    class Engine(BrokenEngine):
        def __init__(self):
            self.checks = 0
            self.calls = 0

        async def ready(self):
            self.checks += 1
            return self.checks == 2

        async def open(self, attempt, body):
            self.calls += 1
            return httpx.Response(
                200,
                json={
                    'id': 'chatcmpl-' + attempt,
                    'model': 'v',
                    'choices': [{'message': {'content': 'look'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
                },
            )

        async def wait_release(self, attempt, timeout):
            return {'status': 'FINISHED_STOPPED', 'request_id': 'chatcmpl-' + attempt}

    engine = Engine()
    with RuntimeStore(tmp_path / 'runtime') as store:
        receipt = await generate_action(
            engine,
            store,
            tmp_path / 'step',
            [{'role': 'user', 'content': 'x'}],
            [1, 2],
            {},
            readiness_attempts=2,
        )
        assert engine.calls == 1 and engine.checks == 2
        assert [check['ready'] for check in receipt['readiness_checks']] == [False, True]
        assert store.db.execute('SELECT COUNT(*) FROM arrivals').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_readiness_recheck_never_retries_a_dispatched_failure(tmp_path):
    class Engine(BrokenEngine):
        calls = 0

        async def open(self, attempt, body):
            self.calls += 1
            raise ConnectionError('controlled dispatched failure')

    engine = Engine()
    with RuntimeStore(tmp_path / 'runtime') as store:
        with pytest.raises(RuntimeError):
            await generate_action(
                engine, store, tmp_path / 'step', [], [1, 2], {}, readiness_attempts=2
            )
        assert engine.calls == 1
        assert store.db.execute('SELECT lease_released FROM arrivals').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_generation_mismatch_stops_readiness_recheck():
    from serving_lab.research_action import check_readiness

    class Engine:
        calls = 0

        async def readiness(self):
            self.calls += 1
            return {'ready': False, 'same_manifest': False, 'http_status': 200}

    engine = Engine()
    ready, checks = await check_readiness(engine, 2)
    assert not ready and engine.calls == len(checks) == 1


@pytest.mark.asyncio
async def test_wrong_response_id_stops_before_action_but_keeps_real_release(tmp_path):
    import httpx, json

    class Engine(BrokenEngine):
        calls = 0

        async def open(self, attempt, body):
            self.calls += 1
            return httpx.Response(
                200,
                json={
                    'id': 'chatcmpl-other',
                    'model': 'v',
                    'choices': [{'message': {'content': 'look'}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3},
                },
            )

        async def wait_release(self, attempt, timeout):
            return {'status': 'FINISHED_STOPPED', 'request_id': 'chatcmpl-' + attempt}

    engine = Engine()
    with RuntimeStore(tmp_path / 'runtime') as store:
        with pytest.raises(RuntimeError):
            await generate_action(engine, store, tmp_path / 'call', [], [1, 2], {})
        receipt = json.loads((tmp_path / 'call/receipt.json').read_text())
        assert not receipt['protocol_valid'] and receipt['release'] and engine.calls == 1
        assert store.db.execute('SELECT state,lease_released FROM arrivals').fetchone()[:] == (
            'failed',
            1,
        )
