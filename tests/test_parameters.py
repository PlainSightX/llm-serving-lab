"""参数身份不受分片、目录说明或JSON头部布局影响。"""

import json
import struct

import pytest


def tensor_file(path, tensors):
    header = {}
    payload = b''
    for name, values in tensors.items():
        raw = struct.pack('<' + 'f' * len(values), *values)
        header[name] = {
            'dtype': 'F32',
            'shape': [len(values)],
            'data_offsets': [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def test_same_parameters_in_different_shards_have_same_identity(tmp_path):
    from serving_lab.parameters import parameter_identity

    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    tensor_file(a / 'model.safetensors', {'w': [1, 2], 'bias': [3]})
    tensor_file(b / 'part-b.safetensors', {'bias': [3]})
    tensor_file(b / 'part-a.safetensors', {'w': [1, 2]})
    (b / 'README.md').write_text('different metadata')
    assert parameter_identity(a)['digest'] == parameter_identity(b)['digest']
    tensor_file(b / 'part-a.safetensors', {'w': [1, 4]})
    assert parameter_identity(a)['digest'] != parameter_identity(b)['digest']


def test_duplicate_or_truncated_tensor_is_rejected(tmp_path):
    from serving_lab.parameters import parameter_identity

    tensor_file(tmp_path / 'a.safetensors', {'w': [1]})
    tensor_file(tmp_path / 'b.safetensors', {'w': [2]})
    with pytest.raises(ValueError):
        parameter_identity(tmp_path)
    tensor_file(tmp_path / 'b.safetensors', {'z': [2]})
    path = tmp_path / 'a.safetensors'
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        parameter_identity(tmp_path)


def test_registration_rejects_disk_drift_from_loaded_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import serving_lab.parameters as module
    from serving_lab.lab import digest

    tensor_file(tmp_path / 'model.safetensors', {'w': [1]})
    monkeypatch.setattr(
        module, 'inspect_base', lambda path: {'files': {'model.safetensors': 'new-disk-bytes'}}
    )
    binding = SimpleNamespace(
        model_path=tmp_path, artifact_digest=digest({'model.safetensors': 'loaded-old-bytes'})
    )
    with pytest.raises(ValueError):
        module.bound_parameter_identity(binding)
