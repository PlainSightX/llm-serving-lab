"""OCI index、运行manifest与config属于不同身份，归档必须连到实际层。"""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def archive(path, corrupt=False):
    files = {}

    def blob(data):
        encoded = json.dumps(data).encode() if isinstance(data, dict) else data
        sha = hashlib.sha256(encoded).hexdigest()
        files['blobs/sha256/' + sha] = encoded
        return {'digest': 'sha256:' + sha, 'size': len(encoded)}

    config = blob({'architecture': 'amd64', 'os': 'linux'})
    layer = blob(b'test-only-runtime-layer')
    manifest = blob({'schemaVersion': 2, 'config': config, 'layers': [layer]})
    index = blob(
        {
            'schemaVersion': 2,
            'manifests': [{**manifest, 'platform': {'architecture': 'amd64', 'os': 'linux'}}],
        }
    )
    files['index.json'] = json.dumps({'schemaVersion': 2, 'manifests': [index]}).encode()
    if corrupt:
        files['blobs/sha256/' + layer['digest'].split(':')[1]] = b'changed-layer'
    with tarfile.open(path, 'w:gz') as stream:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    return index['digest'], config['digest']


def test_archive_resolves_index_identity_and_rejects_changed_layer(tmp_path):
    from export_cluster_images import verify_archive

    path = tmp_path / 'correct.tar.gz'
    identity, config = archive(path)
    assert identity != config
    assert verify_archive(path, identity)['configuration_digests'] == [config]
    bad = tmp_path / 'bad.tar.gz'
    changed_id, _ = archive(bad, corrupt=True)
    with pytest.raises(ValueError, match='摘要'):
        verify_archive(bad, changed_id)
    with pytest.raises(ValueError, match='指定'):
        verify_archive(path, 'sha256:' + '0' * 64)
