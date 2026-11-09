"""本地取得固定K3s发布文件，并按上游校验和逐字节核验。"""

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from serving_lab.execution import write_once
from serving_lab.lab import file_hash

REPO = Path(__file__).resolve().parents[1]


def fetch(output):
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    lock = json.loads((REPO / 'deployment/images.lock.json').read_text(encoding='utf-8'))
    version = urllib.parse.quote(lock['k3s_version'], safe='')
    root = 'https://github.com/k3s-io/k3s/releases/download/' + version + '/'

    def download(url, name, expected=None):
        target = directory / name
        with urllib.request.urlopen(url, timeout=120) as response, target.open('xb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual = file_hash(target)
        if expected and actual != expected:
            raise ValueError('上游文件摘要不符: ' + name)
        return {'file': name, 'url': url, 'sha256': actual, 'bytes': target.stat().st_size}

    records = [download(root + 'sha256sum-amd64.txt', 'sha256sum-amd64.txt')]
    checksums = {}
    for line in (directory / 'sha256sum-amd64.txt').read_text(encoding='utf-8').splitlines():
        sha, name = line.split(maxsplit=1)
        checksums[name.lstrip('*')] = sha
    for name in ('k3s', 'k3s-airgap-images-amd64.tar.zst'):
        records.append(download(root + name, name, checksums[name]))
    records.append(
        download(
            'https://raw.githubusercontent.com/k3s-io/k3s/' + version + '/install.sh',
            'install-k3s.sh',
        )
    )
    records.append(
        download(
            'https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/'
            + lock['device_plugin_version']
            + '/deployments/static/nvidia-device-plugin.yml',
            'nvidia-device-plugin-original.yaml',
        )
    )
    (directory / 'k3s').chmod(0o755)
    write_once(
        directory / 'assets.json',
        {
            'k3s_version': lock['k3s_version'],
            'files': records,
            'device_plugin_image': lock['device_plugin_image'],
            'scope': 'downloaded local assets; no host installation',
        },
    )
    print(
        json.dumps(
            {'verified_downloads': len(records), 'output': str(directory)}, ensure_ascii=False
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    fetch(parser.parse_args().output)
