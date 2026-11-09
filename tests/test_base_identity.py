import pytest
from serving_lab.lab import Lab, LabError, file_hash


def test_base_manifest_detects_weight_or_tokenizer_change(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'identity-test-only')
    (tmp_path / 'tokenizer.json').write_text('{}')
    identity = {'path': str(tmp_path), 'files': {p.name: file_hash(p) for p in tmp_path.iterdir()}}
    Lab.verify_base(identity)
    (tmp_path / 'tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(LabError):
        Lab.verify_base(identity)
