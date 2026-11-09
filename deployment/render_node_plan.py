"""生成目标主机配置和逐条接续命令，渲染过程只写指定输出目录。"""

import argparse
import ipaddress
import json
import hashlib
from pathlib import Path
import re
import shlex


def render(
    bundle_path,
    role,
    node_ip,
    server_ip,
    package_dir,
    output,
    loadgen_ip=None,
    network_cidr='10.10.0.0/24',
    pod_cidr='10.72.0.0/16',
    service_cidr='10.73.0.0/16',
):
    root = Path(bundle_path)
    if (root / 'retired.json').exists():
        raise ValueError('部署包已退役')
    bundle = json.loads((root / 'bundle.json').read_text(encoding='utf-8'))
    images = json.loads((root / 'images/images.json').read_text(encoding='utf-8'))
    networks = [ipaddress.ip_network(value) for value in (network_cidr, pod_cidr, service_cidr)]
    if any(network.version != 4 for network in networks) or any(
        left.overlaps(right) for i, left in enumerate(networks) for right in networks[i + 1 :]
    ):
        raise ValueError('主机、Pod和Service IPv4网段必须相互独立')
    for value in (node_ip, server_ip, loadgen_ip):
        if value:
            address = ipaddress.ip_address(value)
            if (
                address.version != 4
                or not address.is_private
                or address.is_loopback
                or address.is_unspecified
                or address.is_link_local
                or address.is_multicast
            ):
                raise ValueError('节点计划需要可互通的私网IP')
            if address not in networks[0]:
                raise ValueError('节点IP不在指定主机网段')
    if role.startswith('gpu') and not server_ip:
        raise ValueError('GPU节点需要控制节点私网IP')
    if role == 'loadgen' and not server_ip:
        raise ValueError('负载机需要控制节点私网IP')
    if role == 'control' and not loadgen_ip:
        raise ValueError('控制节点计划需要负载机私网IP')
    if not re.fullmatch(r'/[A-Za-z0-9._/-]+', package_dir) or '..' in Path(package_dir).parts:
        raise ValueError('目标包路径需要无空格的Linux绝对路径')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    plan = {
        'role': role,
        'node_ip': node_ip,
        'server_ip': server_ip,
        'package_dir': package_dir,
        'mode': 'rendered_only',
        'requires_cloud_environment': True,
    }
    commands = ['set -eu', f'cd {package_dir}', 'sha256sum -c TRANSFER-SHA256SUMS']
    if role == 'loadgen':
        commands += [
            'docker load -i images/control.tar.gz',
            'mkdir -p /var/lib/llm-serving-lab/loadgen',
            'chown 1000:1000 /var/lib/llm-serving-lab/loadgen',
            'chown 1000:1000 secrets/loadgen secrets/loadgen/ca.pem secrets/loadgen/loadgen.pem secrets/loadgen/loadgen.key',
        ]
        if server_ip:
            plan['name_resolution'] = f'{server_ip} gateway.lsl.internal'
        command = (
            f'docker run --rm --network host --add-host gateway.lsl.internal:{server_ip} --user 1000:1000 '
            f'--mount type=bind,source={package_dir}/secrets/loadgen,target=/certificates,readonly '
            f'--mount type=bind,source={package_dir}/workload.json,target=/workload.json,readonly '
            '--mount type=bind,source=/var/lib/llm-serving-lab/loadgen,target=/results '
            f'--entrypoint python {images["control"]["tag"]} /app/scripts/load_cluster.py '
            '--endpoint https://gateway.lsl.internal:30443 --workload /workload.json '
            '--certificates /certificates --requests 12 --rate 1 --budget 12 --output /results/first-window'
        )
        (directory / 'load-command.sh').write_text(command + '\n', encoding='utf-8', newline='\n')
    else:
        commands += ['test -d /run/systemd/system', 'test ! -e /usr/local/bin/k3s']
        if role.startswith('gpu'):
            commands.append('nvidia-container-runtime --version')
        labels = (
            ['serving-lab/role=control']
            if role == 'control'
            else ['serving-lab/role=gpu', 'serving-lab/slot=' + role[-1]]
        )
        config = {
            'node-name': 'lsl-' + role,
            'node-ip': node_ip,
            'node-label': labels,
            'token-file': '/etc/rancher/k3s/join-token',
        }
        if role == 'control':
            config.update(
                {
                    'tls-san': [node_ip],
                    'advertise-address': node_ip,
                    'secrets-encryption': True,
                    'cluster-cidr': pod_cidr,
                    'service-cidr': service_cidr,
                    'disable': ['traefik', 'servicelb', 'metrics-server'],
                    'write-kubeconfig-mode': '0600',
                }
            )
            encoded = (root / 'manifests.json').read_bytes()
            if hashlib.sha256(encoded).hexdigest() != bundle['manifests_sha256']:
                raise ValueError('源部署清单摘要不符')
            manifests = json.loads(encoded)
            for item in manifests['items']:
                if item['kind'] == 'NetworkPolicy' and item['metadata']['name'] == 'gateway':
                    item['spec']['ingress'][0]['from'][0]['ipBlock']['cidr'] = loadgen_ip + '/32'
            (directory / 'bound-manifests.json').write_text(
                json.dumps(manifests, ensure_ascii=False), encoding='utf-8'
            )
            (directory / 'bound-manifests.json').chmod(0o600)
        else:
            config['server'] = f'https://{server_ip}:6443'
        (directory / 'k3s-config.yaml').write_text(
            json.dumps(config, indent=2) + '\n', encoding='utf-8'
        )
        commands += [
            'test ! -e /etc/rancher/k3s/config.yaml',
            'install -m 0755 assets/k3s /usr/local/bin/k3s',
            'install -d -m 0700 /etc/rancher/k3s',
            'install -m 0600 node-plan/k3s-config.yaml /etc/rancher/k3s/config.yaml',
            'install -m 0600 secrets/cluster-join-token /etc/rancher/k3s/join-token',
            'install -d /var/lib/rancher/k3s/agent/images',
            'cp assets/k3s-airgap-images-amd64.tar.zst /var/lib/rancher/k3s/agent/images/',
        ]
        if role == 'control':
            commands += [
                'cp images/control.tar.gz /var/lib/rancher/k3s/agent/images/',
                'install -d -o 1000 -g 1000 /var/lib/llm-serving-lab/gateway',
            ]
        else:
            commands += [
                'install -d /opt/llm-serving-lab/models',
                'test ! -e /opt/llm-serving-lab/models/c0',
                'cp -a model /opt/llm-serving-lab/models/c0',
                'cp images/worker.tar.gz images/device-plugin.tar.gz /var/lib/rancher/k3s/agent/images/',
            ]
        commands += [
            f'INSTALL_K3S_SKIP_DOWNLOAD=true INSTALL_K3S_SYMLINK=skip sh assets/install-k3s.sh {"server" if role == "control" else "agent"}',
            '/usr/local/bin/k3s --version',
        ]
        if role.startswith('gpu'):
            reference = images['device-plugin']['tag']
            registry_path, separator, image_name = reference.split('@', 1)[0].rpartition('/')
            repository = registry_path + separator + image_name.split(':', 1)[0]
            # 摘要导出的OCI归档可能没有名称，需要显式建立CRI可查询的仓库摘要引用。
            commands += [
                '/usr/local/bin/k3s ctr -n k8s.io images import --platform linux/amd64 '
                f'--digests --base-name {shlex.quote(repository)} images/device-plugin.tar.gz',
                f'/usr/local/bin/k3s crictl inspecti {shlex.quote(reference)} > node-plan/device-plugin-image.json',
            ]
    (directory / 'setup.sh').write_text('\n'.join(commands) + '\n', encoding='utf-8', newline='\n')
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--role', choices=['control', 'gpu-a', 'gpu-b', 'loadgen'], required=True)
    parser.add_argument('--node-ip', required=True)
    parser.add_argument('--server-ip')
    parser.add_argument('--loadgen-ip')
    parser.add_argument('--network-cidr', required=True)
    parser.add_argument('--pod-cidr', default='10.72.0.0/16')
    parser.add_argument('--service-cidr', default='10.73.0.0/16')
    parser.add_argument('--package-dir', default='/opt/lsl-package')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.bundle,
                args.role,
                args.node_ip,
                args.server_ip,
                args.package_dir,
                args.output,
                args.loadgen_ip,
                args.network_cidr,
                args.pod_cidr,
                args.service_cidr,
            ),
            indent=2,
        )
    )
