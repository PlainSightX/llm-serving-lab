"""资格消费前同时核编译、完整验证和实际二进制，允许挂载路径变化。"""

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_build(library, receipt):
    build = json.loads(Path(receipt).read_text(encoding='utf-8'))
    actual = sha(library)
    if actual != build['library_sha256']:
        raise ValueError('当前二进制不匹配编译证明')
    root = Path(__file__).resolve().parent
    if any(sha(root / name) != expected for name, expected in build['source_hashes'].items()):
        raise ValueError('当前内核源码不匹配编译证明')
    return actual


def check_full(library, receipt, verification):
    actual = check_build(library, receipt)
    proof = json.loads(Path(verification).read_text(encoding='utf-8'))
    if (
        proof.get('passed') is not True
        or proof.get('mode') != 'full'
        or proof.get('library_sha256') != actual
        or proof.get('verifier_sha256') != sha(Path(__file__).with_name('verify.py'))
    ):
        raise ValueError('缺少同一二进制和验证器的完整初验')
    return actual
