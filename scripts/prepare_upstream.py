"""取得并核验固定公开chart/私有Helm工具；只渲染，不连接Kubernetes。"""

import argparse
import hashlib
import json
import subprocess
import tarfile
import urllib.request
from pathlib import Path
import yaml

from serving_lab.paths import lab_root

CHART = 'https://vllm-project.github.io/production-stack/vllm-stack-0.1.13.tgz'
CHART_SHA = 'c0d9ee20163aec0072ee74a7d68576e62a58358dde48bf651661926ff54e5373'


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]

    def download(url, name, expected=None):
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        sha = hashlib.sha256(data).hexdigest()
        if expected and sha != expected:
            raise ValueError('下载摘要不符: ' + name)
        (out / name).write_bytes(data)
        return {'url': url, 'bytes': len(data), 'sha256': sha}

    chart = download(CHART, 'chart.tgz', CHART_SHA)
    checksum = download(
        'https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz.sha256sum', 'helm.sha256sum'
    )
    expected = (out / 'helm.sha256sum').read_text().split()[0]
    helm_archive = download(
        'https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz', 'helm.tgz', expected
    )
    with tarfile.open(out / 'helm.tgz') as archive:
        data = archive.extractfile('linux-amd64/helm').read()
    private = lab_root() / 'bin/helm-v3.19.0'
    if private.exists():
        if private.read_bytes() != data:
            raise ValueError('已有私有工具身份不同')
    else:
        private.write_bytes(data)
        private.chmod(0o755)
    values = repo / 'configs/upstream/router-contract-values.yaml'
    (out / 'values.yaml').write_bytes(values.read_bytes())
    commands = [
        ['version', '--short'],
        ['show', 'chart', str(out / 'chart.tgz')],
        ['lint', str(out / 'chart.tgz'), '-f', str(values)],
        [
            'template',
            'lsl-upstream',
            str(out / 'chart.tgz'),
            '--namespace',
            'lsl-contract',
            '-f',
            str(values),
        ],
    ]
    receipts = []
    for name, args in zip(('version', 'chart', 'lint', 'rendered'), commands):
        result = subprocess.run([str(private), *args], capture_output=True, text=True, timeout=60)
        (out / (name + '.yaml' if name == 'rendered' else name + '.txt')).write_text(
            result.stdout + result.stderr
        )
        receipts.append({'command': args, 'returncode': result.returncode})
        if result.returncode:
            raise RuntimeError('Helm ' + name + ' failed')
    docs = [v for v in yaml.safe_load_all((out / 'rendered.yaml').read_text()) if v]
    deployments = [v for v in docs if v['kind'] == 'Deployment']
    assert len(deployments) == 1
    container = deployments[0]['spec']['template']['spec']['containers'][0]
    args = container['args']
    assert (
        'sha256:5b138d39a2aeb417be496c17a74b6b1fc7201d23f5ea125aabdb0326be62967b'
        in container['image']
    )
    assert args[args.index('--max-instance-failover-reroute-attempts') + 1] == '0'
    assert args[args.index('--routing-logic') + 1] == 'roundrobin'
    assert '--static-backends' in args and 'nvidia.com/gpu' not in json.dumps(docs)
    assert not any(v['kind'] == 'PersistentVolumeClaim' for v in docs)
    record = {
        'chart': chart,
        'helm_checksum': checksum,
        'helm_archive': helm_archive,
        'private_helm': str(private),
        'commands': receipts,
        'resources': [{'kind': v['kind'], 'name': v['metadata']['name']} for v in docs],
        'scope': 'static render validation only; no helm install, image pull or Kubernetes connection',
    }
    (out / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    main(a.output)
