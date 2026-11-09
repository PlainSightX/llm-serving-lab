"""错误blob和越界映射不能变成可交付的恢复工件。"""

import hashlib
import io
import json
import runpy
import tarfile
import pytest


def archive(path, name, payload, expected):
    manifest = {
        'files': {name: {'sha256': hashlib.sha256(expected).hexdigest(), 'bytes': len(payload)}},
        'distinct_blobs': 1,
    }
    with tarfile.open(path, 'w:gz') as tar:
        for member, data in [
            ('tensor-archive-map.json', json.dumps(manifest).encode()),
            ('blobs/' + hashlib.sha256(expected).hexdigest(), payload),
        ]:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_archive_restores_only_hash_verified_owned_paths(tmp_path):
    restore = runpy.run_path('scripts/n4_restore_archive.py')['restore']
    path = tmp_path / 'good.tgz'
    archive(path, 'export-a/model/data', b'weights', b'weights')
    assert restore(path, tmp_path / 'good', 'export-a/')['all_blob_bytes_verified']
    bad = tmp_path / 'bad.tgz'
    archive(bad, 'export-a/model/data', b'damaged', b'weights')
    with pytest.raises(ValueError):
        restore(bad, tmp_path / 'bad', 'export-a/')
    assert not (tmp_path / 'bad/verification.json').exists()
    escape = tmp_path / 'escape.tgz'
    archive(escape, '../escape', b'weights', b'weights')
    with pytest.raises(ValueError):
        restore(escape, tmp_path / 'escape-dir', 'export-a/')
