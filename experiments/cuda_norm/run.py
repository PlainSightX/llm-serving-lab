"""一个自有GPU任务内顺序执行门；失败保留日志并停止后续步骤。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evidence import check_full


def child(command, label, output, timeout=900):
    started = time.monotonic()
    log = output / (label + '.log')
    with log.open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    receipt = {
        'command': command,
        'returncode': result.returncode,
        'seconds': time.monotonic() - started,
    }
    (output / (label + '-process.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-3500:], flush=True)
        raise SystemExit(result.returncode)
    print(json.dumps({'step': label, 'passed': True, 'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['initial', 'reverify', 'qualify'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--initial')
    a = p.parse_args()
    source = Path(__file__).resolve().parent
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if a.stage == 'initial':
        child(
            [sys.executable, str(source / 'build.py'), '--output', str(out / 'build')], 'build', out
        )
        build = json.loads((out / 'build/build-receipt.json').read_text())
        child(
            [
                sys.executable,
                str(source / 'verify.py'),
                '--library',
                build['library'],
                '--build-receipt',
                str(out / 'build/build-receipt.json'),
                '--output',
                str(out / 'verify.json'),
            ],
            'verify',
            out,
        )
    elif a.stage == 'reverify':
        initial = Path(a.initial)
        receipt = initial / 'build/build-receipt.json'
        build = json.loads(receipt.read_text(encoding='utf-8'))
        child(
            [
                sys.executable,
                str(source / 'verify.py'),
                '--library',
                build['library'],
                '--build-receipt',
                str(receipt),
                '--output',
                str(out / 'verify.json'),
            ],
            'verify',
            out,
        )
    else:
        initial = Path(a.initial)
        receipt = initial / 'build/build-receipt.json'
        library = json.loads(receipt.read_text(encoding='utf-8'))['library']
        verification = Path(os.environ.get('LSL_N3_VERIFICATION', str(initial / 'verify.json')))
        library_sha = check_full(library, receipt, verification)
        sanitizer = Path(os.environ['CUDA_HOME']) / 'bin/compute-sanitizer'
        for tool in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
            child(
                [
                    str(sanitizer),
                    '--tool',
                    tool,
                    '--error-exitcode',
                    '86',
                    sys.executable,
                    str(source / 'verify.py'),
                    '--library',
                    library,
                    '--build-receipt',
                    str(receipt),
                    '--sanitizer',
                    '--output',
                    str(out / (tool + '.json')),
                ],
                tool,
                out,
            )
        child(
            [
                sys.executable,
                str(source / 'benchmark.py'),
                '--library',
                library,
                '--build-receipt',
                str(receipt),
                '--verification',
                str(verification),
                '--output',
                str(out / 'benchmark.json'),
            ],
            'benchmark',
            out,
        )
    (out / 'completed.json').write_text(
        json.dumps({'completed': True, 'stage': a.stage}), encoding='utf-8'
    )
