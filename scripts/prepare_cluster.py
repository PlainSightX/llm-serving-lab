"""本地生成模型、证书与Kubernetes材料，所有输出进入新目录。"""

import argparse
import base64
import ipaddress
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess

from serving_lab.application_release import load_policy
from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash, inspect_base

REPO = Path(__file__).resolve().parents[1]


def read_bundle(directory):
    """消费前核验原清单字节，中文配置显式使用UTF-8。"""
    root = Path(directory)
    if (root / 'retired.json').exists():
        raise ValueError('部署包已退役，请使用当前交付入口')
    bundle = json.loads((root / 'bundle.json').read_text(encoding='utf-8'))
    path = root / 'manifests.json'
    if file_hash(path) != bundle['manifests_sha256']:
        raise ValueError('部署清单摘要与原bundle不符')
    documents = json.loads(path.read_text(encoding='utf-8'))['items']
    for item in documents:
        kind = item.get('kind')
        if kind not in {'Secret', 'ConfigMap'}:
            continue
        name = item['metadata']['name']
        if Path(name).name != name:
            raise ValueError('部署对象名称包含路径')
        for leaf, value in item['data'].items():
            if Path(leaf).name != leaf:
                raise ValueError('部署数据名称包含路径')
            if kind == 'Secret':
                material = root / 'secrets' / name / leaf
                expected = base64.b64decode(value, validate=True)
            else:
                material = root / 'config' / leaf
                expected = value.encode('utf-8')
            if material.read_bytes() != expected:
                raise ValueError('部署文件与冻结清单不符: ' + name + '/' + leaf)
    return bundle, documents


def workloads(namespace, control_image, worker_image, client_cidr):
    """worker按节点槽位固定，控制者持久状态和启动身份采集分别挂载。"""
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,50}', namespace):
        raise ValueError('namespace格式不符')
    ipaddress.ip_network(client_cidr)

    def metadata(name):
        return {'name': name, 'namespace': namespace, 'labels': {'serving-lab/owner': 'r5'}}

    documents = [
        {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {'name': namespace, 'labels': {'serving-lab/owner': 'r5'}},
        }
    ]
    for slot in ('a', 'b'):
        name = 'worker-' + slot
        labels = {'app': name, 'serving-lab/component': 'worker'}
        container = {
            'name': 'worker',
            'image': worker_image,
            'imagePullPolicy': 'Never',
            'args': ['--config', '/config/worker.json'],
            'ports': [{'name': 'https', 'containerPort': 8443}],
            'resources': {
                'requests': {'cpu': '2', 'memory': '8Gi'},
                'limits': {'nvidia.com/gpu': 1, 'memory': '24Gi'},
            },
            'securityContext': {
                'allowPrivilegeEscalation': False,
                'capabilities': {'drop': ['ALL']},
            },
            'volumeMounts': [
                {'name': 'config', 'mountPath': '/config', 'readOnly': True},
                {'name': 'secrets', 'mountPath': '/secrets', 'readOnly': True},
                {'name': 'model', 'mountPath': '/models/c0', 'readOnly': True},
                {'name': 'state', 'mountPath': '/state'},
                {'name': 'shm', 'mountPath': '/dev/shm'},
            ],
            'startupProbe': {
                'exec': {'command': ['python3', '/app/scripts/cluster_health.py', 'worker']},
                'periodSeconds': 10,
                'failureThreshold': 90,
                'timeoutSeconds': 5,
            },
            'readinessProbe': {
                'exec': {'command': ['python3', '/app/scripts/cluster_health.py', 'worker']},
                'periodSeconds': 5,
                'timeoutSeconds': 5,
            },
        }
        pod = {
            'automountServiceAccountToken': False,
            'runtimeClassName': 'nvidia',
            'nodeSelector': {'serving-lab/slot': slot},
            'terminationGracePeriodSeconds': 75,
            'affinity': {
                'podAntiAffinity': {
                    'requiredDuringSchedulingIgnoredDuringExecution': [
                        {
                            'labelSelector': {'matchLabels': {'serving-lab/component': 'worker'}},
                            'topologyKey': 'kubernetes.io/hostname',
                        }
                    ]
                }
            },
            'containers': [container],
            'volumes': [
                {'name': 'config', 'configMap': {'name': 'r5-config'}},
                {'name': 'secrets', 'secret': {'secretName': name, 'defaultMode': 256}},
                {
                    'name': 'model',
                    'hostPath': {'path': '/opt/llm-serving-lab/models/c0', 'type': 'Directory'},
                },
                {
                    'name': 'state',
                    'hostPath': {
                        'path': '/var/lib/llm-serving-lab/' + name,
                        'type': 'DirectoryOrCreate',
                    },
                },
                {'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '2Gi'}},
            ],
        }
        documents.append(
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': metadata(name),
                'spec': {
                    'replicas': 1,
                    'strategy': {'type': 'Recreate'},
                    'selector': {'matchLabels': {'app': name}},
                    'template': {'metadata': {'labels': labels}, 'spec': pod},
                },
            }
        )
        documents.append(
            {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': metadata(name),
                # 原监督桥在native退出后仍要提供清理证明，生成准入由身份与ready检查控制。
                'spec': {
                    'selector': {'app': name},
                    'publishNotReadyAddresses': True,
                    'ports': [{'name': 'https', 'port': 8443, 'targetPort': 'https'}],
                },
            }
        )
    mounts = [
        {'name': 'config', 'mountPath': '/config', 'readOnly': True},
        {'name': 'secrets', 'mountPath': '/secrets', 'readOnly': True},
        {'name': 'bootstrap', 'mountPath': '/bootstrap'},
        {'name': 'state', 'mountPath': '/state'},
    ]
    gateway = {
        'automountServiceAccountToken': False,
        'nodeSelector': {'serving-lab/role': 'control'},
        'terminationGracePeriodSeconds': 45,
        'securityContext': {'runAsUser': 1000, 'runAsGroup': 1000, 'fsGroup': 1000},
        'initContainers': [
            {
                'name': 'bind-workers',
                'image': control_image,
                'imagePullPolicy': 'Never',
                'command': ['python', '/app/scripts/collect_worker_manifests.py'],
                'volumeMounts': mounts,
            }
        ],
        'containers': [
            {
                'name': 'gateway',
                'image': control_image,
                'imagePullPolicy': 'Never',
                'args': ['--config', '/bootstrap/gateway.json'],
                'volumeMounts': mounts,
                'resources': {
                    'requests': {'cpu': '1', 'memory': '512Mi'},
                    'limits': {'memory': '2Gi'},
                },
                'ports': [{'name': 'https', 'containerPort': 8443}],
                'securityContext': {
                    'allowPrivilegeEscalation': False,
                    'capabilities': {'drop': ['ALL']},
                },
                'readinessProbe': {
                    'exec': {'command': ['python', '/app/scripts/cluster_health.py', 'gateway']},
                    'periodSeconds': 5,
                    'timeoutSeconds': 5,
                },
            }
        ],
        'volumes': [
            {'name': 'config', 'configMap': {'name': 'r5-config'}},
            {'name': 'secrets', 'secret': {'secretName': 'gateway', 'defaultMode': 288}},
            {'name': 'bootstrap', 'emptyDir': {}},
            {
                'name': 'state',
                'hostPath': {'path': '/var/lib/llm-serving-lab/gateway', 'type': 'Directory'},
            },
        ],
    }
    documents.append(
        {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': metadata('gateway'),
            'spec': {
                'replicas': 1,
                'strategy': {'type': 'Recreate'},
                'selector': {'matchLabels': {'app': 'gateway'}},
                'template': {
                    'metadata': {
                        'labels': {'app': 'gateway', 'serving-lab/component': 'controller'}
                    },
                    'spec': gateway,
                },
            },
        }
    )
    documents.append(
        {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': metadata('gateway'),
            'spec': {
                'type': 'NodePort',
                'externalTrafficPolicy': 'Local',
                'selector': {'app': 'gateway'},
                'ports': [
                    {'name': 'https', 'port': 8443, 'targetPort': 'https', 'nodePort': 30443}
                ],
            },
        }
    )
    # worker只接收控制者，网关只接收指定负载机或本命名空间的验证客户端。
    for name, selector, sources in (
        (
            'workers',
            {'serving-lab/component': 'worker'},
            [{'podSelector': {'matchLabels': {'serving-lab/component': 'controller'}}}],
        ),
        (
            'gateway',
            {'app': 'gateway'},
            [
                {'ipBlock': {'cidr': client_cidr}},
                {'podSelector': {'matchLabels': {'serving-lab/component': 'client'}}},
            ],
        ),
    ):
        documents.append(
            {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'NetworkPolicy',
                'metadata': metadata(name),
                'spec': {
                    'podSelector': {'matchLabels': selector},
                    'policyTypes': ['Ingress'],
                    'ingress': [{'from': sources, 'ports': [{'protocol': 'TCP', 'port': 8443}]}],
                },
            }
        )
    return documents


def make_certificates(directory, namespace):
    """CA私钥留在本地管理目录，节点仅取得自身叶证书和所需访问密钥。"""
    directory.mkdir()

    def openssl(*arguments):
        subprocess.run(['openssl', *arguments], cwd=directory, check=True, capture_output=True)

    openssl(
        'req',
        '-x509',
        '-newkey',
        'ec',
        '-pkeyopt',
        'ec_paramgen_curve:P-256',
        '-nodes',
        '-keyout',
        'ca.key',
        '-out',
        'ca.pem',
        '-days',
        '365',
        '-subj',
        '/CN=llm-serving-lab-r5',
        '-addext',
        'basicConstraints=critical,CA:TRUE',
        '-addext',
        'keyUsage=critical,keyCertSign,cRLSign',
    )
    for name in ('worker-a', 'worker-b', 'gateway', 'controller', 'loadgen'):
        openssl(
            'req',
            '-new',
            '-newkey',
            'ec',
            '-pkeyopt',
            'ec_paramgen_curve:P-256',
            '-nodes',
            '-keyout',
            name + '.key',
            '-out',
            name + '.csr',
            '-subj',
            '/CN=' + name,
        )
        server = name in ('worker-a', 'worker-b', 'gateway')
        extension = 'basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage='
        extension += 'serverAuth,clientAuth\n' if server else 'clientAuth\n'
        if server:
            extension += f'subjectAltName=DNS:{name},DNS:{name}.{namespace}.svc.cluster.local,DNS:localhost,IP:127.0.0.1'
            if name == 'gateway':
                extension += ',DNS:gateway.lsl.internal'
            extension += '\n'
        (directory / (name + '.ext')).write_text(extension, encoding='utf-8')
        openssl(
            'x509',
            '-req',
            '-in',
            name + '.csr',
            '-CA',
            'ca.pem',
            '-CAkey',
            'ca.key',
            '-CAcreateserial',
            '-out',
            name + '.pem',
            '-days',
            '90',
            '-extfile',
            name + '.ext',
        )
    for path in directory.iterdir():
        path.chmod(0o600)


def prepare(output, control_image, worker_image, namespace='lsl-r5', client_cidr='192.0.2.1/32'):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('部署包目录必须为新目录')
    evidence = REPO / 'artifacts/c0-publication-evidence-20261103-a/application-evidence.json'
    receipt = json.loads(evidence.with_name('import-receipt.json').read_text(encoding='utf-8'))
    policy = load_policy(evidence, receipt['sha256'])
    model = REPO / 'artifacts/q2-coldstart-training-20261103-b/merged'
    identity = inspect_base(model)
    artifact = digest(identity['files'])
    certificate = policy.bundle['certificates'][artifact]
    if not policy.summary(certificate)['passed']:
        raise ValueError('C0质量门未通过')
    output.mkdir(parents=True)
    model_copy = output / 'model'
    model_copy.mkdir()
    for name, expected in identity['files'].items():
        destination = model_copy / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model / name, destination)
        if file_hash(destination) != expected:
            raise ValueError('模型复制后摘要不符')
    pki = output / 'pki'
    make_certificates(pki, namespace)
    config = output / 'config'
    config.mkdir()
    shutil.copyfile(evidence, config / 'application-evidence.json')
    shutil.copyfile(REPO / 'fixtures/serving/r2-transport-v2.json', config / 'transport-panel.json')
    worker = {
        'version_id': 'qwen3-17b-coldstart-candidate',
        'model_path': '/models/c0',
        'vllm_version': '0.11.2',
        'host': '127.0.0.1',
        'port': 8000,
        'dtype': 'bfloat16',
        'max_model_len': 4096,
        'max_num_seqs': 8,
        'gpu_memory_utilization': 0.40,
        'kv_cache_memory_bytes': 805306368,
        'enforce_eager': False,
        'enable_prefix_caching': False,
    }
    write_once(config / 'worker.json', worker)
    tokens = {name: secrets.token_hex(24) for name in ('worker-a', 'worker-b', 'admin')}
    secret_objects = []
    for name in ('worker-a', 'worker-b', 'gateway'):
        destination = output / 'secrets' / name
        destination.mkdir(parents=True)
        materials = {'ca.pem': (pki / 'ca.pem').read_bytes()}
        if name == 'gateway':
            for role in ('gateway', 'controller'):
                for leaf in ('pem', 'key'):
                    key = 'gateway-tls.' + leaf if role == 'gateway' else 'client.' + leaf
                    materials[key] = (pki / (role + '.' + leaf)).read_bytes()
            materials.update(
                {
                    role + '-token': tokens[role].encode()
                    for role in ('worker-a', 'worker-b', 'admin')
                }
            )
        else:
            materials.update(
                {
                    'tls.pem': (pki / (name + '.pem')).read_bytes(),
                    'tls.key': (pki / (name + '.key')).read_bytes(),
                    'worker-token': tokens[name].encode(),
                }
            )
        for leaf, value in materials.items():
            (destination / leaf).write_bytes(value)
            (destination / leaf).chmod(0o600)
        secret_objects.append(
            {
                'apiVersion': 'v1',
                'kind': 'Secret',
                'metadata': {'name': name, 'namespace': namespace},
                'type': 'Opaque',
                'data': {key: base64.b64encode(value).decode() for key, value in materials.items()},
            }
        )
    gateway = {
        'replicas': [
            {
                'id': slot,
                'endpoint': f'https://worker-{slot}.{namespace}.svc.cluster.local:8443',
                'token_file': f'/secrets/worker-{slot}-token',
                'tls': {
                    'ca': '/secrets/ca.pem',
                    'cert': '/secrets/client.pem',
                    'key': '/secrets/client.key',
                },
                'capacity': {'max_inflight': 1, 'max_queue': 2},
            }
            for slot in ('a', 'b')
        ],
        'application_release': {
            'bundle': '/config/application-evidence.json',
            'sha256': receipt['sha256'],
        },
        'panel': '/config/transport-panel.json',
        'runtime': '/state/controller',
        'host': '0.0.0.0',
        'port': 8443,
        'admin_token_file': '/secrets/admin-token',
        'tls': {
            'ca': '/secrets/ca.pem',
            'cert': '/secrets/gateway-tls.pem',
            'key': '/secrets/gateway-tls.key',
        },
        'limits': {
            'max_inflight': 2,
            'max_queue': 4,
            'max_input_tokens': 3500,
            'max_tokens': 64,
            'context_tokens': 4096,
            'deadline_seconds': 90,
            'cancel_grace_seconds': 10,
        },
        'allow_rebind': [],
    }
    write_once(config / 'gateway-template.json', gateway)
    documents = workloads(namespace, control_image, worker_image, client_cidr)
    settings = {
        'apiVersion': 'v1',
        'kind': 'ConfigMap',
        'metadata': {'name': 'r5-config', 'namespace': namespace},
        'data': {path.name: path.read_text(encoding='utf-8') for path in config.iterdir()},
    }
    write_once(
        output / 'manifests.json',
        {'apiVersion': 'v1', 'kind': 'List', 'items': documents + [settings] + secret_objects},
    )
    # 供普通调用演练使用的公开状态，原监督答案不进入请求。
    sample = json.loads(
        (
            REPO / 'artifacts/c0-publication-real-20261104-c/calls/0005-paired-1/request.json'
        ).read_text(encoding='utf-8')
    )['body']
    write_once(output / 'sample-request.json', sample)
    samples = []
    for index in range(1, 4):
        folder = (
            REPO / f'artifacts/c0-publication-real-20261104-c/calls/{index + 4:04}-paired-{index}'
        )
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        response = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
        headers = response['headers']
        samples.append(
            {
                'id': f'public-state-{index}',
                'body': request['body'],
                'input_token_ids': request['input_token_ids'],
                'version': headers['x-model-version'],
                'profile_digest': headers['x-serving-profile'],
                'application_evidence': headers['x-application-evidence'],
            }
        )
    write_once(
        output / 'workload.json',
        {'samples': samples, 'purpose': 'bounded application transport/load checks'},
    )
    record = {
        'schema': 1,
        'namespace': namespace,
        'model_artifact_digest': artifact,
        'model_files': identity['files'],
        'evidence_sha256': receipt['sha256'],
        'images': {'control': control_image, 'worker': worker_image},
        'client_cidr': client_cidr,
        'requires_cloud_binding': client_cidr == '192.0.2.1/32',
        'secrets_mode': 'private local bundle; CA key stays with operator',
        'source_sha256': file_hash(__file__),
        'manifests_sha256': file_hash(output / 'manifests.json'),
    }
    write_once(output / 'bundle.json', record)
    (output / 'manifests.json').chmod(0o600)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--control-image', required=True)
    parser.add_argument('--worker-image', required=True)
    parser.add_argument('--namespace', default='lsl-r5')
    parser.add_argument('--client-cidr', default='192.0.2.1/32')
    args = parser.parse_args()
    result = prepare(
        args.output, args.control_image, args.worker_image, args.namespace, args.client_cidr
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
