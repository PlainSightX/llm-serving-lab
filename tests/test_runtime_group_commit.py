"""账本组提交的语义边界：保存点只回滚自身、写前确认、提交次数与提交失败后关闭准入。"""

import asyncio
import json
import sqlite3

import httpx
import pytest

from serving_lab.runtime import RuntimeStore


def committed_rows(root):
    """独立只读连接只能看到已经 COMMIT 的数据，用它判断是否真正持久化。"""
    db = sqlite3.connect((root / 'service.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        return {r['id']: dict(r) for r in db.execute('SELECT * FROM arrivals')}
    finally:
        db.close()


def test_failed_operation_rolls_back_only_itself(tmp_path):
    with RuntimeStore(tmp_path) as store:
        store.record_arrival('a', 'same', durable=False)
        store.record_arrival('b', 'same', durable=False)
        assert store.reserve_attempt('a', 'attempt-a', 'v', 'h', durable=False)
        # 同一幂等键的第二次预约触发唯一约束：只回滚这次预约，批内其他请求的记录不受影响。
        assert not store.reserve_attempt('b', 'attempt-b', 'v', 'h', durable=False)
        assert committed_rows(tmp_path) == {}
        assert store.durable()
        rows = committed_rows(tmp_path)
        assert rows['a']['state'] == 'queued' and rows['b']['state'] == 'received'
        assert not store.durable()


def test_dispatch_record_commits_everything_executed_before_it(tmp_path):
    with RuntimeStore(tmp_path) as store:
        store.record_arrival('a', None, durable=False)
        store.record_arrival('other', None, durable=False)
        assert store.reserve_attempt('a', 'attempt', 'v', 'h', durable=False)
        before = store.commit_count
        store.dispatched('a', {'worker_instance_id': 'w'})
        assert store.commit_count == before + 1
        rows = committed_rows(tmp_path)
        assert rows['a']['state'] == 'running' and rows['a']['dispatched'] == 1
        # 组提交顺带让同批并发请求的到达也落盘。
        assert rows['other']['state'] == 'received'


def test_only_effect_free_tail_is_lost_on_controller_crash(tmp_path):
    """边界：最后一次提交之后、尚无任何对外动作的记录可能随崩溃丢失；已派发的一定在账本里。"""
    store = RuntimeStore(tmp_path)
    store.record_arrival('kept', None, durable=False)
    assert store.reserve_attempt('kept', 'attempt', 'v', 'h', durable=False)
    store.dispatched('kept', {'worker_instance_id': 'w'})
    store.record_arrival('tail', None, durable=False)
    # 模拟控制者崩溃：不提交直接关闭连接，未提交事务由SQLite回滚。
    store.db.close()
    store.lock.close()
    with RuntimeStore(tmp_path) as recovered:
        rows = {r['id']: r for r in recovered.rows()}
    assert set(rows) == {'kept'}
    assert rows['kept']['state'] == 'unknown' and rows['kept']['dispatched'] == 1


def test_open_lease_counts_use_partial_indexes(tmp_path):
    with RuntimeStore(tmp_path) as store:
        store.db.execute('CREATE TABLE pool_history(replica TEXT,generation TEXT,identity TEXT)')

        def plan(sql, params):
            return ' '.join(r[3] for r in store.db.execute('EXPLAIN QUERY PLAN ' + sql, params))

        assert 'arrivals_open_version' in plan(
            'SELECT COUNT(*) FROM arrivals WHERE version=? AND lease_released=0', ('v',)
        )
        assert 'arrivals_open_worker' in plan(
            'SELECT COUNT(*) FROM arrivals WHERE lease_released=0 AND worker IN '
            '(SELECT identity FROM pool_history WHERE replica=?)',
            ('r',),
        )


class FailingCommit:
    """只让 COMMIT 失败的连接代理，用于模拟磁盘错误。"""

    def __init__(self, db):
        self._db = db

    def execute(self, sql, *args):
        if sql == 'COMMIT':
            raise sqlite3.OperationalError('disk I/O error')
        return self._db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._db, name)


def test_commit_failure_refuses_further_writes(tmp_path):
    store = RuntimeStore(tmp_path)
    real = store.db
    store.db = FailingCommit(real)
    store.record_arrival('a', None, durable=False)
    with pytest.raises(sqlite3.OperationalError):
        store.durable()
    assert store.failed is not None
    with pytest.raises(RuntimeError):
        store.record_arrival('b', None, durable=False)
    real.close()
    store.lock.close()


def test_transaction_lost_inside_an_operation_closes_writes(tmp_path):
    store = RuntimeStore(tmp_path)
    store.record_arrival('a', None, durable=False)
    with pytest.raises(ValueError):
        with store.atomic():
            # 模拟SQLite因磁盘错误整体回滚了事务，保存点随之消失。
            store.db.execute('ROLLBACK')
            raise ValueError('statement failed')
    assert store.failed is not None and store.depth == 0
    with pytest.raises(RuntimeError):
        store.record_arrival('b', None, durable=False)
    store.close()


class FakeEngine:
    """CPU合同替身：记录派发瞬间账本里已提交的状态。"""

    version_id = 'base'
    identity = {'worker_instance_id': 'test', 'startup_generation': 'g1'}

    def __init__(self, root):
        self.root = root
        self.seen_at_dispatch = None

    async def ready(self):
        return True

    async def count_tokens(self, messages):
        return 1

    async def open(self, attempt, body):
        self.seen_at_dispatch = committed_rows(self.root)
        return FakeStream(self, attempt)

    async def wait_release(self, attempt, timeout):
        return {'status': 'FINISHED_STOPPED'}

    async def close(self):
        pass


class FakeStream:
    status_code = 200

    def __init__(self, engine, attempt):
        self.engine = engine
        self.attempt = attempt

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield 'data: [DONE]'

    async def aclose(self):
        pass


def request(**kwargs):
    return {
        'model': 'base',
        'messages': [{'role': 'user', 'content': 'x'}],
        'max_tokens': 8,
        'stream': True,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_stream_request_is_durable_before_dispatch_and_uses_three_commits(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    engine = FakeEngine(tmp_path)
    app = create_app(engine, tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        store = app.state.service.store
        before = store.commit_count
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            response = await client.post('/v1/chat/completions', json=request())
            assert response.status_code == 200 and '[DONE]' in response.text
        # 原实现每个请求5次提交（到达、预约、派发、终态、清理）；组提交后为派发前、最后一帧之前、清理结束各一次。
        assert store.commit_count - before == 3
        (seen,) = engine.seen_at_dispatch.values()
        assert seen['state'] == 'running' and seen['dispatched'] == 1
        row = committed_rows(tmp_path)[seen['id']]
        assert row['state'] == 'completed' and row['cleanup'] == 'confirmed'


@pytest.mark.asyncio
async def test_rejection_is_durable_before_the_response_and_failure_closes_admission(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    app = create_app(FakeEngine(tmp_path), tmp_path, ServiceLimits())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            response = await client.post('/v1/chat/completions', json=request(model='other'))
            assert response.status_code == 404
            arrival = response.headers['x-arrival-id']
            row = committed_rows(tmp_path)[arrival]
            assert row['state'] == 'rejected'
            assert json.loads(row['observation'])['code'] == 'unknown_model'
            app.state.service.store.failed = RuntimeError('simulated commit failure')
            assert (await client.get('/health/ready')).status_code == 503
            app.state.service.store.failed = None


@pytest.mark.asyncio
async def test_concurrent_requests_complete_with_at_most_three_commits_each(tmp_path):
    from serving_lab.online import create_app, ServiceLimits

    app = create_app(FakeEngine(tmp_path), tmp_path, ServiceLimits(max_inflight=8))
    async with app.router.lifespan_context(app):
        store = app.state.service.store
        before = store.commit_count
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test'
        ) as client:
            responses = await asyncio.gather(
                *(client.post('/v1/chat/completions', json=request()) for _ in range(8))
            )
        assert all(r.status_code == 200 for r in responses)
        rows = committed_rows(tmp_path)
        assert len(rows) == 8 and all(r['state'] == 'completed' for r in rows.values())
        assert store.commit_count - before <= 24
