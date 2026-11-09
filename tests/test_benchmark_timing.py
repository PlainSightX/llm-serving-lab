"""诊断代理保持事务提交、异常回滚和原返回值，不全局替换正式RuntimeStore。"""

import importlib.util
from pathlib import Path
import threading
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/instrument_gateway.py'
    spec = importlib.util.spec_from_file_location('scoped_timing', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_timed_store_preserves_commit_rollback_and_thread_identity(tmp_path):
    timing = module()
    collector = timing.Collector()
    Store = timing.measured_store(collector)
    with Store(tmp_path) as store:
        collector.start()
        store.record_arrival('a', 'key')
        assert store.reserve_attempt('a', 'attempt', 'v', 'hash') is True
        store.record_arrival('b', 'key')
        assert store.reserve_attempt('b', 'duplicate', 'v', 'hash') is False
        assert store.row('b')['state'] == 'received'
        with pytest.raises(RuntimeError):
            with store.atomic():
                store.db.execute("UPDATE arrivals SET state='bad' WHERE id='a'")
                raise RuntimeError('force rollback')
        assert store.row('a')['state'] == 'queued'
        observed = collector.stop()
        assert observed['records'] and all(
            r['thread'] == threading.get_ident() for r in observed['records']
        )
        assert any(
            r['operation'] == 'sqlite_transaction_exit' and r['outcome'] == 'rollback'
            for r in observed['records']
        )
