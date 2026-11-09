"""在途租约计数的查询代价随历史请求数的变化：有无部分索引对照（纯CPU，不需要GPU或服务）。

副本池每次分配请求都要统计各副本、各版本未释放的租约；没有索引时这是对arrivals整表的扫描，
代价随网关运行时间（历史请求数）线性增长。本脚本用与RuntimeStore相同的表结构构造数据，
分别在“无索引”和“部分索引”下计时两条实际查询。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

SCHEMA = '''
CREATE TABLE arrivals(id TEXT PRIMARY KEY,client_key TEXT,received REAL,state TEXT,attempt_id TEXT,
  version TEXT,input_hash TEXT,worker TEXT,dispatched INTEGER DEFAULT 0,cleanup TEXT DEFAULT 'none',
  observation TEXT DEFAULT '{}',cleanup_evidence TEXT DEFAULT '{}',lease_released INTEGER DEFAULT 1,
  route TEXT DEFAULT '{}');
CREATE TABLE pool_history(replica TEXT,generation TEXT,identity TEXT,PRIMARY KEY(replica,generation));
'''
INDEXES = [
    'CREATE INDEX arrivals_open_version ON arrivals(version) WHERE lease_released=0',
    'CREATE INDEX arrivals_open_worker ON arrivals(worker) WHERE lease_released=0',
]
QUERIES = {
    'pending(version)': (
        'SELECT COUNT(*) FROM arrivals WHERE version=? AND lease_released=0',
        ('v1',),
    ),
    'pending_replica(replica)': (
        'SELECT COUNT(*) FROM arrivals WHERE lease_released=0 AND worker IN '
        '(SELECT identity FROM pool_history WHERE replica=?)',
        ('r1',),
    ),
}


def build(path, history, open_leases):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    workers = ['{"replica":"r1","generation":"g1"}', '{"replica":"r2","generation":"g1"}']
    db.executemany(
        'INSERT INTO pool_history VALUES(?,?,?)',
        [('r1', 'g1', workers[0]), ('r2', 'g1', workers[1])],
    )
    observation = json.dumps({'phase_ms': {'x': 1.0}, 'content': 'y' * 200})
    rows = (
        (
            f'a{i}',
            None,
            float(i),
            'completed',
            f't{i}',
            f'v{i % 2 + 1}',
            'h',
            workers[i % 2],
            1,
            'confirmed',
            observation,
            '{}',
            0 if i >= history else 1,
            '{}',
        )
        for i in range(history + open_leases)
    )
    db.executemany('INSERT INTO arrivals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    db.commit()
    return db


def timed(db, sql, params, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        db.execute(sql, params).fetchone()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(samples)


def main(sizes, open_leases, repeats):
    results = []
    with tempfile.TemporaryDirectory() as folder:
        for size in sizes:
            db = build(Path(folder) / f'n{size}.sqlite', size, open_leases)
            row = {'history_rows': size, 'open_leases': open_leases}
            for name, (sql, params) in QUERIES.items():
                row[f'{name} no-index µs'] = timed(db, sql, params, repeats)
            for statement in INDEXES:
                db.execute(statement)
            for name, (sql, params) in QUERIES.items():
                row[f'{name} partial-index µs'] = timed(db, sql, params, repeats)
                plan = ' '.join(r[3] for r in db.execute('EXPLAIN QUERY PLAN ' + sql, params))
                row[f'{name} plan'] = plan
            db.close()
            results.append(row)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[1_000, 10_000, 100_000])
    parser.add_argument('--open-leases', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--output')
    args = parser.parse_args()
    data = main(args.sizes, args.open_leases, args.repeats)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(text)
