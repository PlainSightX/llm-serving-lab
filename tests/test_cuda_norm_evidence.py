"""无需GPU也能拒绝旧证明配新二进制、或缩减检查冒充完整验证。"""

import json, runpy
from pathlib import Path
import pytest


def test_cuda_full_proof_is_bound_to_binary_source_and_verifier(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'experiments/cuda_norm'
    m = runpy.run_path(str(root / 'evidence.py'))
    sha = m['sha']
    binary = tmp_path / 'op.so'
    binary.write_bytes(b'first')
    build = tmp_path / 'build.json'
    build.write_text(
        json.dumps(
            {
                'library_sha256': sha(binary),
                'source_hashes': {n: sha(root / n) for n in ['op.cpp', 'op.cu']},
            }
        )
    )
    proof = tmp_path / 'proof.json'
    data = {
        'passed': True,
        'mode': 'full',
        'library_sha256': sha(binary),
        'verifier_sha256': sha(root / 'verify.py'),
    }
    proof.write_text(json.dumps(data))
    m['check_full'](binary, build, proof)
    proof.write_text(json.dumps({**data, 'mode': 'sanitizer'}))
    with pytest.raises(ValueError, match='完整初验'):
        m['check_full'](binary, build, proof)
    proof.write_text(json.dumps(data))
    binary.write_bytes(b'second')
    with pytest.raises(ValueError, match='编译证明'):
        m['check_full'](binary, build, proof)
