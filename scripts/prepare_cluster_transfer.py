"""按节点职责打包交付材料，CA签名私钥留在本地工作目录。"""

import argparse
import io
import json
from pathlib import Path
import secrets
import shutil
import tarfile

from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from prepare_cluster import read_bundle

REPO = Path(__file__).resolve().parents[1]


def prepare(bundle_path, output_name='transfer'):
    root = Path(bundle_path).resolve()
    bundle, _ = read_bundle(root)
    images = json.loads((root / 'images/images.json').read_text(encoding='utf-8'))
    for item in images.values():
        if file_hash(root / 'images' / item['archive']) != item['sha256']:
            raise ValueError('镜像归档摘要变化')
    deployment = root / 'deployment'
    deployment.mkdir(exist_ok=True)
    for name in ('render_node_plan.py', 'cluster_admin.py'):
        shutil.copyfile(REPO / 'deployment' / name, deployment / name)
    shutil.copyfile(REPO / 'scripts/cluster_node_preflight.py', deployment / 'node_preflight.py')
    loadgen = root / 'secrets/loadgen'
    loadgen.mkdir(exist_ok=True)
    for name in ('ca.pem', 'loadgen.pem', 'loadgen.key'):
        shutil.copyfile(root / 'pki' / name, loadgen / name)
        (loadgen / name).chmod(0o600)
    join = root / 'secrets/cluster-join-token'
    if not join.exists():
        join.write_text(secrets.token_hex(32), encoding='utf-8')
        join.chmod(0o600)
    lock = json.loads((REPO / 'deployment/images.lock.json').read_text(encoding='utf-8'))
    plugin = {
        'apiVersion': 'apps/v1',
        'kind': 'DaemonSet',
        'metadata': {'name': 'lsl-nvidia-device-plugin', 'namespace': 'kube-system'},
        'spec': {
            'selector': {'matchLabels': {'app': 'lsl-nvidia-device-plugin'}},
            'template': {
                'metadata': {
                    'labels': {'app': 'lsl-nvidia-device-plugin', 'serving-lab/owner': 'r5'}
                },
                'spec': {
                    'automountServiceAccountToken': False,
                    'runtimeClassName': 'nvidia',
                    'nodeSelector': {'serving-lab/role': 'gpu'},
                    'priorityClassName': 'system-node-critical',
                    'tolerations': [
                        {'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect': 'NoSchedule'}
                    ],
                    'containers': [
                        {
                            'name': 'device-plugin',
                            'image': lock['device_plugin_image'],
                            'imagePullPolicy': 'Never',
                            'env': [{'name': 'FAIL_ON_INIT_ERROR', 'value': 'true'}],
                            'securityContext': {
                                'allowPrivilegeEscalation': False,
                                'capabilities': {'drop': ['ALL']},
                            },
                            'volumeMounts': [
                                {
                                    'name': 'registration',
                                    'mountPath': '/var/lib/kubelet/device-plugins',
                                }
                            ],
                        }
                    ],
                    'volumes': [
                        {
                            'name': 'registration',
                            'hostPath': {
                                'path': '/var/lib/kubelet/device-plugins',
                                'type': 'Directory',
                            },
                        }
                    ],
                },
            },
        },
    }
    (deployment / 'nvidia-device-plugin.json').write_text(
        json.dumps(plugin, indent=2), encoding='utf-8'
    )
    common = [
        'bundle.json',
        'images/images.json',
        'deployment/render_node_plan.py',
        'deployment/node_preflight.py',
    ]
    assets = [
        str(path.relative_to(root)).replace('\\', '/')
        for path in (root / 'assets').iterdir()
        if path.is_file()
    ]
    roles = {
        'control': common
        + assets
        + [
            'manifests.json',
            'deployment/cluster_admin.py',
            'deployment/nvidia-device-plugin.json',
            'images/control.tar.gz',
            'secrets/cluster-join-token',
        ],
        'worker': common
        + assets
        + ['images/worker.tar.gz', 'images/device-plugin.tar.gz', 'secrets/cluster-join-token']
        + ['model/' + name for name in bundle['model_files']],
        'loadgen': common
        + [
            'images/control.tar.gz',
            'workload.json',
            'secrets/loadgen/ca.pem',
            'secrets/loadgen/loadgen.pem',
            'secrets/loadgen/loadgen.key',
        ],
    }
    roles['control'] += [
        'secrets/gateway/' + path.name
        for path in (root / 'secrets/gateway').iterdir()
        if path.is_file()
    ]
    if Path(output_name).name != output_name:
        raise ValueError('交付目录名不能包含路径')
    destination = root / output_name
    destination.mkdir(exist_ok=False)
    receipts = {}
    for role, names in roles.items():
        if any(name.startswith('pki/') or name.endswith('/ca.key') for name in names):
            raise ValueError('签名私钥进入角色包')
        sums = {name: file_hash(root / name) for name in names}
        archive = destination / (role + '.tar')
        with tarfile.open(archive, mode='w', format=tarfile.PAX_FORMAT) as stream:
            for name in names:
                info = stream.gettarinfo(str(root / name), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = 'root'
                info.mode = (
                    0o600 if name.startswith('secrets/') or name == 'manifests.json' else 0o644
                )
                with (root / name).open('rb') as source:
                    stream.addfile(info, source)
            data = ''.join(sha + '  ' + name + '\n' for name, sha in sums.items()).encode()
            info = tarfile.TarInfo('TRANSFER-SHA256SUMS')
            info.mode, info.size = 0o600, len(data)
            stream.addfile(info, io.BytesIO(data))
        archive.chmod(0o600)
        with tarfile.open(archive) as stream:
            if set(stream.getnames()) != set(names) | {'TRANSFER-SHA256SUMS'}:
                raise ValueError('角色包成员不符')
        receipts[role] = {
            'file': archive.name,
            'bytes': archive.stat().st_size,
            'sha256': file_hash(archive),
            'members': sums,
            'CA_signing_key_included': False,
        }
    write_once(destination / 'transfer.json', receipts)
    print(
        json.dumps(
            {
                role: {'bytes': row['bytes'], 'sha256': row['sha256']}
                for role, row in receipts.items()
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output-name', default='transfer')
    args = parser.parse_args()
    prepare(args.bundle, args.output_name)
