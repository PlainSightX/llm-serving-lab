"""原生持久化不能只迁DB、覆盖运行或复制未停下的快照。"""

import json
import pytest


def test_native_runtime_rejects_mounted_storage_before_creation(tmp_path):
    from serving_lab import measurement_runtime as m

    with pytest.raises(ValueError):
        m.prepare_runtime(tmp_path, '/mnt/d/not-native')


def test_finished_snapshot_preserves_all_runtime_files_and_owner(tmp_path, monkeypatch):
    from serving_lab import measurement_runtime as m

    monkeypatch.setattr(m, 'filesystem', lambda path: {'fstype': 'ext4', 'target': '/'})
    out = tmp_path / 'evidence'
    out.mkdir()
    base = tmp_path / 'native'
    root = m.prepare_runtime(out, base)
    assert root != out
    runtime = root / 'control-runtime'
    runtime.mkdir()
    (runtime / 'wire-requests').mkdir()
    (runtime / 'wire-requests/a.json').write_text('wire')
    m.snapshot_runtimes(out, root, ['control'])
    assert (out / 'control-runtime/wire-requests/a.json').read_text() == 'wire'
    assert json.loads((out / 'runtime-placement.json').read_text())['native_root'] == str(root)
    with pytest.raises(ValueError):
        m.prepare_runtime(out, base)


def test_arrival_export_closes_sqlite_even_when_output_exists(tmp_path, monkeypatch):
    import sqlite3
    from serving_lab import measurement_runtime as m

    assert hasattr(m, 'export_arrivals')
    db = sqlite3.connect(tmp_path / 'service.sqlite')
    db.execute('CREATE TABLE arrivals(id TEXT)')
    db.execute("INSERT INTO arrivals VALUES('a')")
    db.commit()
    db.close()
    closed = []
    connect = sqlite3.connect

    class Tracked(sqlite3.Connection):
        def close(self):
            closed.append(True)
            return super().close()

    monkeypatch.setattr(m.sqlite3, 'connect', lambda *a, **kw: connect(*a, **kw, factory=Tracked))
    m.export_arrivals(tmp_path / 'service.sqlite', tmp_path / 'arrivals.json')
    assert closed == [True] and json.loads((tmp_path / 'arrivals.json').read_text()) == [
        {'id': 'a'}
    ]
    with pytest.raises(FileExistsError):
        m.export_arrivals(tmp_path / 'service.sqlite', tmp_path / 'arrivals.json')
    assert closed == [True, True]
