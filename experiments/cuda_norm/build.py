"""只在固定服务镜像和隔离CUDA工具链中编译，不改默认服务入口。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch.utils.cpp_extension import load


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = [root / 'op.cpp', root / 'op.cu']
    library = load(
        name='lsl_n3_ops',
        sources=[str(s) for s in sources],
        build_directory=str(out),
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '-lineinfo'],
        is_python_module=False,
        verbose=True,
    )
    path = Path(library)
    receipt = {
        'library': str(path),
        'library_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_hashes': {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'nvcc': subprocess.check_output(['nvcc', '--version'], text=True),
        'scope': 'compilation only; correctness and model integration not yet established',
    }
    (out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
