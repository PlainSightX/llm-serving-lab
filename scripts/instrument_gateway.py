"""诊断专用主线程同步调用计时；不使用跨线程cProfile，不改变事务结果。"""

import functools
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from serving_lab.runtime import RuntimeStore


class Collector:
    def __init__(self):
        self.enabled = False
        self.rows = []
        self.owner_thread = threading.get_ident()
        self.owner_native = threading.get_native_id()
        self.local = threading.local()
        self.started = None
        self.ended = None

    def start(self):
        self.rows = []
        self.started = time.monotonic()
        self.enabled = True

    def stop(self):
        self.ended = time.monotonic()
        self.enabled = False
        return {
            'pid': os.getpid(),
            'owner_thread': self.owner_thread,
            'owner_native': self.owner_native,
            'started': self.started,
            'ended': self.ended,
            'records': self.rows,
            'scope': 'synchronous method wall time; nested method and SQL intervals must not be added together',
        }

    def add(self, operation, started, ended, caller, outcome='returned'):
        if not self.enabled:
            return
        self.rows.append(
            {
                'operation': operation,
                'start': started,
                'end': ended,
                'seconds': ended - started,
                'thread': threading.get_ident(),
                'native_thread': threading.get_native_id(),
                'arrival_id': getattr(self.local, 'arrival', None),
                'caller_file': caller.f_code.co_filename,
                'caller_function': caller.f_code.co_name,
                'caller_line': caller.f_lineno,
                'outcome': outcome,
            }
        )


class TimedConnection:
    def __init__(self, connection, collector):
        self.connection = connection
        self.collector = collector

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, kind, value, tb):
        if not self.collector.enabled:
            return self.connection.__exit__(kind, value, tb)
        caller = sys._getframe(1)
        pending = self.connection.in_transaction
        started = time.monotonic()
        try:
            return self.connection.__exit__(kind, value, tb)
        finally:
            self.collector.add(
                'sqlite_transaction_exit',
                started,
                time.monotonic(),
                caller,
                'rollback' if kind else ('commit' if pending else 'noop'),
            )

    def execute(self, sql, *args, **kwargs):
        if not self.collector.enabled:
            return self.connection.execute(sql, *args, **kwargs)
        started = time.monotonic()
        caller = sys._getframe(1)
        # 组提交改用显式 COMMIT / ROLLBACK [TO]；它们按事务结束计时，与 with conn 的退出同类。
        words = sql.split(None, 1) if isinstance(sql, str) else []
        keyword = words[0].upper() if words else ''
        try:
            return self.connection.execute(sql, *args, **kwargs)
        finally:
            if keyword in {'COMMIT', 'ROLLBACK'}:
                self.collector.add(
                    'sqlite_transaction_exit',
                    started,
                    time.monotonic(),
                    caller,
                    'commit' if keyword == 'COMMIT' else 'rollback',
                )
            else:
                self.collector.add('sqlite_execute', started, time.monotonic(), caller)


def measured_store(collector):
    class DiagnosticStore(RuntimeStore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.db = TimedConnection(self.db, collector)

    for name in (
        'record_arrival',
        'bind_arrival',
        'reserve_attempt',
        'save_payload',
        'save_wire',
        'dispatched',
        'finish',
        'cleanup',
        'release_lease',
    ):
        original = getattr(RuntimeStore, name)

        def wrap(function, operation):
            @functools.wraps(function)
            def measured(self, *args, **kwargs):
                if not collector.enabled:
                    return function(self, *args, **kwargs)
                previous = getattr(collector.local, 'arrival', None)
                collector.local.arrival = args[0] if args else kwargs.get('arrival_id')
                started = time.monotonic()
                caller = sys._getframe(1)
                try:
                    return function(self, *args, **kwargs)
                finally:
                    collector.add(operation, started, time.monotonic(), caller)
                    collector.local.arrival = previous

            return measured

        setattr(DiagnosticStore, name, wrap(original, name))
    return DiagnosticStore


def main():
    from serving_lab import online
    from serve_releases import main as serve

    path = Path(os.environ['SERVING_LAB_TIMING_PATH'])
    collector = Collector()
    online.RuntimeStore = measured_store(collector)

    def start(*args):
        collector.start()
        path.with_suffix('.start.json').write_text(
            json.dumps(
                {
                    'pid': os.getpid(),
                    'thread': collector.owner_thread,
                    'enabled_at': collector.started,
                }
            )
        )

    def stop(*args):
        result = collector.stop()
        path.write_text(json.dumps(result))
        path.with_suffix('.stop.json').write_text(
            json.dumps({'pid': os.getpid(), 'disabled_at': result['ended']})
        )

    signal.signal(signal.SIGUSR1, start)
    signal.signal(signal.SIGUSR2, stop)
    serve()


if __name__ == '__main__':
    main()
