"""候选关闭必须来自已核验原始计时，不能相信另一份可漂移摘要。"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_runtime_baseline as c


def test_stale_timing_summary_cannot_change_decision(tmp_path):
    assert hasattr(c, 'verified_timing')
    report = {
        'scoped_timing_verified': True,
        'scoped_timing': {
            'directory': str(tmp_path),
            'method_union_seconds': 2.0,
            'trace_seconds': 10.0,
        },
    }
    summary = {
        'trace_seconds': 10.0,
        'operation_seconds': {'record_arrival': 0.01, 'sqlite_execute': 0.1},
    }
    with pytest.raises(AssertionError):
        c.verified_timing(report, summary, tmp_path)
    summary['operation_seconds']['record_arrival'] = 2.0
    assert c.verified_timing(report, summary, tmp_path) == (2.0, 10.0)


def test_diagnosis_without_native_provenance_cannot_close_candidate(tmp_path):
    import json
    import sqlite3

    assert hasattr(c, 'verify_native_diagnosis')
    root = str(tmp_path / 'native')
    runtime = str(Path(root) / 'control-runtime')
    values = {
        'runtime-placement.json': {'filesystem': {'fstype': 'drvfs'}, 'native_root': root},
        'runtime-settings.json': {'journal_mode': 'wal', 'synchronous': 2, 'runtime': runtime},
        'gateway.json': {'runtime': runtime},
        'runtime-snapshot.json': {
            'runtimes': {
                'control': {'source': runtime, 'snapshot': str(tmp_path / 'copy'), 'files': {}}
            }
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        c.verify_native_diagnosis(tmp_path)
    values['runtime-placement.json']['filesystem']['fstype'] = 'ext4'
    (tmp_path / 'runtime-placement.json').write_text(json.dumps(values['runtime-placement.json']))
    for folder in (Path(runtime), tmp_path / 'copy'):
        folder.mkdir(parents=True)
        sqlite3.connect(folder / 'service.sqlite').close()
    c.verify_native_diagnosis(tmp_path)


def test_checkpoint_changes_bytes_but_not_snapshot_logical_identity(tmp_path):
    import sqlite3, shutil
    from serving_lab.lab import file_hash

    native = tmp_path / 'native'
    copy = tmp_path / 'copy'
    native.mkdir()
    copy.mkdir()
    db = sqlite3.connect(native / 'service.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE records(id INTEGER PRIMARY KEY,body TEXT)')
    db.execute("INSERT INTO records VALUES(1,'committed')")
    db.commit()
    files = {}
    for p in native.iterdir():
        shutil.copyfile(p, copy / p.name)
        files[p.name] = file_hash(p)
    db.close()
    assert file_hash(native / 'service.sqlite') != files['service.sqlite']
    snapshot = {'source': str(native), 'snapshot': str(copy), 'files': files}
    c.verify_snapshot(snapshot)
    assert {p.name: file_hash(p) for p in copy.iterdir()} == files
    db = sqlite3.connect(native / 'service.sqlite')
    db.execute("INSERT INTO records VALUES(2,'changed')")
    db.commit()
    db.close()
    with pytest.raises(AssertionError):
        c.verify_snapshot(snapshot)
