"""在线到达与尝试的短事务账本，独立于离线Lab与研究首尝语义。

持久化采用“组提交”：每个账本操作是当前批量事务里的一个保存点，失败只回滚自身；
COMMIT（WAL + synchronous=FULL，含fsync）只在持久化边界执行，一次覆盖此前所有已执行操作。
在线网关的边界是：派发到引擎之前（写前确认）、向客户端返回之前、清理结束时。
默认参数 durable=True 保持“每个操作各自提交”的原有语义，供资格、研究与运维脚本使用。
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from pathlib import Path

from .lab import canonical

TERMINAL = {'completed', 'failed', 'rejected', 'timed_out', 'cancelled', 'unknown'}


class RuntimeStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'controller.lock').open('a+b')
        self.lock.seek(0, 2)
        if not self.lock.tell():
            self.lock.write(b'0')
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise ValueError('在线目录已有控制者') from exc
        # isolation_level=None：由本类显式控制BEGIN/SAVEPOINT/COMMIT，避免驱动隐式提交或整体回滚。
        self.db = sqlite3.connect(self.root / 'service.sqlite', isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.depth = 0
        self.commit_count = 0
        self.failed = None
        # 批量事务里是否有尚未提交的改动；用于发现事务被 SQLite 整体回滚。
        self.pending = False
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS arrivals(id TEXT PRIMARY KEY,client_key TEXT,received REAL,state TEXT,
            attempt_id TEXT,version TEXT,input_hash TEXT,worker TEXT,dispatched INTEGER DEFAULT 0,
            cleanup TEXT DEFAULT 'none',observation TEXT DEFAULT '{}',cleanup_evidence TEXT DEFAULT '{}');
          CREATE TABLE IF NOT EXISTS client_keys(key TEXT PRIMARY KEY,arrival_id TEXT);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,arrival_id TEXT,kind TEXT,body TEXT,created REAL);
        ''')
        columns = {row['name'] for row in self.db.execute('PRAGMA table_info(arrivals)')}
        if 'lease_released' not in columns:
            self.db.execute('ALTER TABLE arrivals ADD COLUMN lease_released INTEGER DEFAULT 1')
        if 'route' not in columns:
            self.db.execute("ALTER TABLE arrivals ADD COLUMN route TEXT DEFAULT '{}'")
        # 未释放租约的部分索引：版本/副本的在途计数不随历史请求数线性扫描整表。
        self.db.execute(
            'CREATE INDEX IF NOT EXISTS arrivals_open_version ON arrivals(version) WHERE lease_released=0'
        )
        self.db.execute(
            'CREATE INDEX IF NOT EXISTS arrivals_open_worker ON arrivals(worker) WHERE lease_released=0'
        )
        self.recover()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            if self.failed is None:
                self.durable()
        finally:
            self.db.close()
            self.lock.close()

    @contextlib.contextmanager
    def atomic(self, *, durable=False):
        """一个账本操作的原子边界：当前批量事务中的保存点，异常只回滚本操作。

        durable=True 且处于最外层时，退出前执行一次组提交；嵌套调用把提交交给最外层。
        保存点深度在任何异常下都会恢复；保存点语句本身失败说明事务已不可用，立即关闭写入。
        """
        self._check_batch()
        if not self.db.in_transaction:
            self.db.execute('BEGIN')
        name = f'op{self.depth}'
        self.db.execute(f'SAVEPOINT {name}')
        self.depth += 1
        before = self.db.total_changes
        try:
            yield
        except BaseException:
            try:
                self.db.execute(f'ROLLBACK TO {name}')
                self.db.execute(f'RELEASE {name}')
            except sqlite3.Error as lost:
                # SQLite已整体回滚事务（如磁盘错误）：批内其他请求的未提交记录也已丢失，关闭写入。
                self.failed = lost
            finally:
                self.depth -= 1
            if (
                not self.depth
                and not self.pending
                and self.failed is None
                and self.db.in_transaction
            ):
                # 批内没有任何待提交改动：结束这个空事务，不让失败的操作一直占着写锁。
                with contextlib.suppress(sqlite3.Error):
                    self.db.execute('ROLLBACK')
            raise
        try:
            self.db.execute(f'RELEASE {name}')
        except sqlite3.Error as lost:
            self.failed = lost
            raise
        finally:
            self.depth -= 1
        if self.db.total_changes != before:
            self.pending = True
        if durable and not self.depth:
            self.durable()

    def _check_batch(self):
        """拒绝在失败或丢失的批量事务上继续写：有未提交改动却已不在事务中，说明批次被整体回滚。"""
        if self.failed is not None:
            raise RuntimeError('账本提交曾失败，拒绝继续写入') from self.failed
        if self.pending and not self.db.in_transaction:
            self.failed = sqlite3.OperationalError('未提交的批量事务已被整体回滚')
            raise RuntimeError('账本批量事务丢失，拒绝继续写入') from self.failed

    def durable(self):
        """组提交：一次COMMIT让此前已执行的全部操作持久化；无待提交内容时不做事。"""
        if self.depth:
            raise RuntimeError('不能在账本操作内部提交')
        self._check_batch()
        if not self.db.in_transaction:
            return False
        try:
            self.db.execute('COMMIT')
        except sqlite3.Error as exc:
            # 提交失败时批内其他请求的记录也未落盘：关闭写入，由上层停止准入。
            self.failed = exc
            with contextlib.suppress(sqlite3.Error):
                self.db.execute('ROLLBACK')
            raise
        self.pending = False
        self.commit_count += 1
        return True

    def event(self, arrival, kind, body):
        self.db.execute(
            'INSERT INTO events(arrival_id,kind,body,created) VALUES(?,?,?,?)',
            (arrival, kind, canonical(body), time.time()),
        )

    def recover(self):
        with self.atomic(durable=True):
            rows = self.db.execute(
                "SELECT id,state FROM arrivals WHERE state NOT IN ('completed','failed','rejected','timed_out','cancelled','unknown')"
            ).fetchall()
            for row in rows:
                self.db.execute("UPDATE arrivals SET state='unknown' WHERE id=?", (row['id'],))
                self.event(row['id'], 'controller_interrupted', {'previous': row['state']})
            # 控制者已退出，尚未派发的计数/排队不再占worker；已派发者必须单独核实。
            self.db.execute(
                'UPDATE arrivals SET lease_released=1 WHERE dispatched=0 OR cleanup=\'confirmed\''
            )

    def bind_arrival(self, arrival_id, version, worker, route, *, durable=True):
        with self.atomic(durable=durable):
            cur = self.db.execute(
                "UPDATE arrivals SET version=?,worker=?,route=?,lease_released=0 WHERE id=? AND state='received' AND version IS NULL",
                (version, canonical(worker), canonical(route), arrival_id),
            )
            if cur.rowcount != 1:
                raise ValueError('请求已绑定或不在准入状态')
            self.event(arrival_id, 'binding', {'version_id': version, 'worker': worker, **route})

    def release_lease(self, arrival_id, *, durable=True):
        with self.atomic(durable=durable):
            row = self.row(arrival_id)
            if row['dispatched'] and row['cleanup'] != 'confirmed':
                raise ValueError('worker释放未确认，禁止解除租约')
            self.db.execute('UPDATE arrivals SET lease_released=1 WHERE id=?', (arrival_id,))

    def record_arrival(self, arrival_id, client_key, *, durable=True):
        with self.atomic(durable=durable):
            self.db.execute(
                'INSERT INTO arrivals(id,client_key,received,state) VALUES(?,?,?,?)',
                (arrival_id, client_key, time.time(), 'received'),
            )
            self.event(arrival_id, 'arrival', {})

    def reserve_attempt(self, arrival_id, attempt_id, version_id, payload_hash, *, durable=True):
        row = self.row(arrival_id)
        if row['state'] != 'received':
            raise ValueError('到达不处于可预约状态')
        try:
            with self.atomic(durable=durable):
                if row['client_key'] is not None:
                    self.db.execute(
                        'INSERT INTO client_keys VALUES(?,?)', (row['client_key'], arrival_id)
                    )
                self.db.execute(
                    "UPDATE arrivals SET attempt_id=?,version=?,input_hash=?,state='queued' WHERE id=?",
                    (attempt_id, version_id, payload_hash, arrival_id),
                )
                self.event(
                    arrival_id, 'reserved', {'attempt_id': attempt_id, 'version_id': version_id}
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def dispatched(self, arrival_id, worker):
        """写前确认：派发记录及此前全部操作在发给引擎之前必须已经持久化。"""
        with self.atomic(durable=True):
            cur = self.db.execute(
                "UPDATE arrivals SET state='running',dispatched=1,cleanup='pending',worker=? WHERE id=? AND state='queued'",
                (canonical(worker), arrival_id),
            )
            if cur.rowcount != 1:
                raise ValueError('尝试未预约或已终止')
            self.event(arrival_id, 'dispatched', worker)

    def finish(self, arrival_id, state, observation, *, durable=True):
        if state not in TERMINAL:
            raise ValueError('无效终态')
        with self.atomic(durable=durable):
            row = self.row(arrival_id)
            if row['state'] in TERMINAL:
                return False
            self.db.execute(
                'UPDATE arrivals SET state=?,observation=? WHERE id=?',
                (state, canonical(observation), arrival_id),
            )
            self.event(arrival_id, 'terminal', {'state': state, **observation})
        return True

    def cleanup(self, arrival_id, status, evidence, *, durable=True):
        if status not in {'confirmed', 'unconfirmed'}:
            raise ValueError('无效清理状态')
        with self.atomic(durable=durable):
            self.db.execute(
                'UPDATE arrivals SET cleanup=?,cleanup_evidence=? WHERE id=?',
                (status, canonical(evidence), arrival_id),
            )
            self.event(arrival_id, 'cleanup', {'status': status, 'evidence': evidence})

    def save_payload(self, arrival_id, body):
        folder = self.root / 'requests'
        folder.mkdir(exist_ok=True)
        with (folder / (arrival_id + '.json')).open('x', encoding='utf-8') as stream:
            stream.write(canonical(body) + '\n')

    def save_wire(self, arrival_id, payload, profile_digest, token_ids):
        folder = self.root / 'wire-requests'
        folder.mkdir(exist_ok=True)
        with (folder / (arrival_id + '.json')).open('x', encoding='utf-8') as stream:
            stream.write(
                canonical(
                    {
                        'payload': payload,
                        'profile_digest': profile_digest,
                        'input_token_ids': token_ids,
                    }
                )
                + '\n'
            )

    def row(self, arrival_id):
        row = self.db.execute('SELECT * FROM arrivals WHERE id=?', (arrival_id,)).fetchone()
        if not row:
            raise ValueError('到达不存在')
        return dict(row)

    def rows(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM arrivals ORDER BY received,id')]

    def unresolved(self):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM arrivals WHERE dispatched=1 AND cleanup!='confirmed'"
            )
        ]
