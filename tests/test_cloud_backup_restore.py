"""恢复包不能越过新建暂存根，也不能把损坏内容当恢复成功。"""

import hashlib
import json
import runpy
import pytest


def fixture_backup(tmp_path, remote='/opt/project/data'):
    root = tmp_path / 'backup'
    (root / 'nodes/test').mkdir(parents=True)
    (root / 'blobs').mkdir()
    payload = b'original'
    sha = hashlib.sha256(payload).hexdigest()
    (root / 'blobs' / sha).write_bytes(payload)
    manifest = {
        'role': 'test',
        'instance_id': 'instance-test',
        'capture_complete': True,
        'files': [
            {
                'type': 'file',
                'path': remote,
                'sha256': sha,
                'bytes': len(payload),
                'mode': 0o644,
                'uid': 0,
                'gid': 0,
                'mtime_ns': 1,
            }
        ],
    }
    path = root / 'nodes/test/files.json'
    path.write_text(json.dumps(manifest))
    anchor = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'instance_id': 'instance-test',
        'regular_files': 1,
        'logical_bytes': len(payload),
        'entries': 1,
    }
    (root / 'restore-index.json').write_text(
        json.dumps(
            {
                'refs': {sha: {'kind': 'blob', 'path': 'blobs/' + sha}},
                'roles': ['test'],
                'manifests': {'test': anchor},
            }
        )
    )
    return root, sha


def test_restore_verifies_actual_bytes_before_success(tmp_path):
    restore = runpy.run_path('scripts/restore_cloud_backup.py')['restore']
    root, sha = fixture_backup(tmp_path)
    result = restore(root, 'test', tmp_path / 'good', prefixes=[], symlinks=True, owners=False)
    assert result['files'] == 1 and (tmp_path / 'good/opt/project/data').read_bytes() == b'original'
    (root / 'blobs' / sha).write_bytes(b'corrupt!')
    with pytest.raises(ValueError):
        restore(root, 'test', tmp_path / 'bad', prefixes=[], symlinks=True, owners=False)
    assert not (tmp_path / 'bad/restore-verification.json').exists()


def test_restore_rejects_parent_escape_and_occupied_target(tmp_path):
    restore = runpy.run_path('scripts/restore_cloud_backup.py')['restore']
    root, _ = fixture_backup(tmp_path, '/opt/../outside')
    with pytest.raises(ValueError):
        restore(root, 'test', tmp_path / 'new', prefixes=[], symlinks=True, owners=False)
    target = tmp_path / 'existing'
    target.mkdir()
    with pytest.raises(FileExistsError):
        restore(root, 'test', target, prefixes=[], symlinks=True, owners=False)


def test_omitted_manifest_entry_cannot_be_reported_as_complete(tmp_path):
    restore = runpy.run_path('scripts/restore_cloud_backup.py')['restore']
    root, _ = fixture_backup(tmp_path)
    p = root / 'nodes/test/files.json'
    data = json.loads(p.read_text())
    second = {**data['files'][0], 'path': '/opt/project/second'}
    data['files'].append(second)
    p.write_text(json.dumps(data))
    index = json.loads((root / 'restore-index.json').read_text())
    index['manifests']['test'].update(
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        regular_files=2,
        entries=2,
        logical_bytes=16,
    )
    (root / 'restore-index.json').write_text(json.dumps(index))
    data['files'].pop()
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        restore(root, 'test', tmp_path / 'target', prefixes=[], symlinks=True, owners=False)


def test_report_path_is_reserved_even_when_a_backup_file_uses_it(tmp_path):
    restore = runpy.run_path('scripts/restore_cloud_backup.py')['restore']
    root, _ = fixture_backup(tmp_path, '/restore-verification.json')
    with pytest.raises(ValueError):
        restore(root, 'test', tmp_path / 'target', prefixes=[], symlinks=True, owners=False)
