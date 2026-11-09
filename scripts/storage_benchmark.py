"""相同初态和操作序列的CPU持久化诊断；不伪装成模型请求或端到端加速。"""

import time
from pathlib import Path
from serving_lab.runtime import RuntimeStore
from serving_lab.benchmark import distribution
from serving_lab.execution import write_once


def storage_diagnosis(windows_root, linux_root, count=32):
    results = {}
    for label, root in [('mounted', Path(windows_root)), ('native', Path(linux_root))]:
        if root.exists():
            raise ValueError('诊断目录存在，禁止覆盖')
        timings = {}
        rows = []
        with RuntimeStore(root) as store:
            actual = {
                'journal_mode': store.db.execute('PRAGMA journal_mode').fetchone()[0],
                'synchronous': store.db.execute('PRAGMA synchronous').fetchone()[0],
            }
            assert actual == {'journal_mode': 'wal', 'synchronous': 2}
            for i in range(count):
                arrival = f'cpu-storage-{i:03}'
                body = {
                    'model': 'cpu_fixture_not_a_model',
                    'messages': [{'role': 'user', 'content': 'storage diagnostic'}],
                }

                def measured(name, operation):
                    started = time.perf_counter()
                    operation()
                    elapsed = time.perf_counter() - started
                    timings.setdefault(name, []).append(elapsed)
                    rows.append({'index': i, 'operation': name, 'seconds': elapsed})

                worker = {'worker_instance_id': 'cpu_fixture_not_a_worker'}
                measured('arrival', lambda: store.record_arrival(arrival, arrival))
                measured(
                    'bind',
                    lambda: store.bind_arrival(
                        arrival, 'cpu_fixture', worker, {'scope': 'CPU storage fixture only'}
                    ),
                )
                measured(
                    'reserve',
                    lambda: store.reserve_attempt(arrival, arrival, 'cpu_fixture', 'fixture'),
                )
                measured('payload', lambda: store.save_payload(arrival, body))
                measured('wire', lambda: store.save_wire(arrival, body, 'fixture', [1, 2]))
                measured('dispatch_record_only', lambda: store.dispatched(arrival, worker))
                measured(
                    'terminal_record_only',
                    lambda: store.finish(
                        arrival, 'completed', {'scope': 'CPU fixture, no model executed'}
                    ),
                )
                measured(
                    'cleanup_record_only',
                    lambda: store.cleanup(arrival, 'confirmed', {'scope': 'CPU fixture only'}),
                )
                measured('lease_release', lambda: store.release_lease(arrival))
            assert len(store.rows()) == count and all(
                r['lease_released'] == 1 for r in store.rows()
            )
        result = {
            'scope': 'CPU identical transaction/file sequence; no model or end-to-end performance claim',
            'rows': rows,
            'operations': {name: distribution(values) for name, values in timings.items()},
            'count': count,
            'root': str(root),
            'journal_mode': 'WAL',
            'synchronous': 'FULL',
            'observed_pragmas': actual,
        }
        write_once(root / 'diagnostic.json', result)
        results[label] = result
    return results
