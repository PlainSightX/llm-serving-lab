"""N4固定角色的内部训练作业；不修改现有服务或公开rendezvous端口。"""

import argparse
import json
from pathlib import Path
import re

IMAGE = 'docker.io/library/llm-serving-lab-worker@sha256:24c5476d51446805dfd8614cfb41280b0b29e4743890d24b82c6b5940e4cdb45'


def training(
    run_id,
    mode,
    phase,
    nodes,
    root,
    source,
    *,
    scaling='strong',
    fault=False,
    resume=None,
    profile=False,
):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,44}', run_id):
        raise ValueError('无效运行名')
    if mode not in {'single', 'ddp', 'fsdp2'} or phase not in {
        'correctness',
        'performance',
        'recovery',
    }:
        raise ValueError('未知训练模式')
    world = len(nodes)
    if (
        mode == 'single'
        and world != 1
        or mode != 'single'
        and world != 2
        or len({n['node_name'] for n in nodes}) != world
        or len({n['instance_id'] for n in nodes}) != world
    ):
        raise ValueError('当前包只调度单GPU或两个不同节点')
    items = []
    service = 'n4-' + run_id
    if world > 1:
        items.append(
            {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': service,
                    'namespace': 'lsl-r5',
                    'labels': {'serving-lab/owner': 'n4'},
                },
                'spec': {
                    'type': 'ClusterIP',
                    'selector': {'n4-run': run_id, 'n4-rank': '0'},
                    'ports': [{'name': 'rendezvous', 'port': 29500, 'targetPort': 29500}],
                },
            }
        )
    for rank, node in enumerate(nodes):
        env = {
            'PYTHONPATH': '/workspace/src',
            'PYTHONDONTWRITEBYTECODE': '1',
            'OMP_NUM_THREADS': '4',
            'MKL_NUM_THREADS': '4',
            'TOKENIZERS_PARALLELISM': 'false',
            'N4_INSTANCE_ID': node['instance_id'],
            'N4_EXPECTED_NODE': node['node_name'],
            'NCCL_IB_DISABLE': '1',
            'NCCL_SOCKET_IFNAME': 'eth0',
            'NCCL_DEBUG': 'INFO',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
            'RANK': str(rank),
            'WORLD_SIZE': str(world),
            'LOCAL_RANK': '0',
            'MASTER_ADDR': service,
            'MASTER_PORT': '29500',
        }
        command = [
            'python3',
            '/workspace/scripts/n4_train.py',
            '--mode',
            mode,
            '--phase',
            phase,
            '--topology',
            'single_gpu' if world == 1 else 'two_node',
            '--model',
            '/model',
            '--data',
            '/data',
            '--output',
            '/output/' + run_id,
            '--scaling',
            scaling,
        ]
        if fault:
            command.append('--fault')
        if resume:
            command.extend(['--resume', resume])
        if profile:
            command.append('--profile')
        spec = {
            'restartPolicy': 'Never',
            'runtimeClassName': 'nvidia',
            'nodeSelector': {'kubernetes.io/hostname': node['node_name']},
            'automountServiceAccountToken': False,
            'containers': [
                {
                    'name': 'trainer',
                    'image': IMAGE,
                    'imagePullPolicy': 'Never',
                    'command': command,
                    'env': [{'name': k, 'value': v} for k, v in env.items()]
                    + [
                        {
                            'name': 'N4_NODE_NAME',
                            'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}},
                        }
                    ],
                    'resources': {
                        'requests': {'cpu': '4', 'memory': '16Gi', 'nvidia.com/gpu': '1'},
                        'limits': {'memory': '48Gi', 'nvidia.com/gpu': '1'},
                    },
                    'volumeMounts': [
                        {'name': n, 'mountPath': m, 'readOnly': r}
                        for n, m, r in [
                            ('source', '/workspace', True),
                            ('model', '/model', True),
                            ('data', '/data', True),
                            ('output', '/output', False),
                            ('shm', '/dev/shm', False),
                        ]
                    ],
                }
            ],
            'volumes': [
                {'name': n, 'hostPath': {'path': root + '/' + s, 'type': 'Directory'}}
                for n, s in [
                    ('source', source),
                    ('model', 'model'),
                    ('data', source + '/data-b'),
                    ('output', 'results'),
                ]
            ]
            + [{'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '8Gi'}}],
        }
        items.append(
            {
                'apiVersion': 'batch/v1',
                'kind': 'Job',
                'metadata': {'name': service + f'-r{rank}', 'namespace': 'lsl-r5'},
                'spec': {
                    'backoffLimit': 0,
                    'activeDeadlineSeconds': 1800,
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'n4-training',
                                'serving-lab/owner': 'n4',
                                'n4-run': run_id,
                                'n4-rank': str(rank),
                            }
                        },
                        'spec': spec,
                    },
                },
            }
        )
    return {'apiVersion': 'v1', 'kind': 'List', 'items': items}


def network(control_ip):
    peer = {'podSelector': {'matchLabels': {'serving-lab/owner': 'n4'}}}
    control = {'ipBlock': {'cidr': control_ip + '/32'}}
    return {
        'apiVersion': 'networking.k8s.io/v1',
        'kind': 'NetworkPolicy',
        'metadata': {'name': 'n4-private-training', 'namespace': 'lsl-r5'},
        'spec': {
            'podSelector': {'matchLabels': {'serving-lab/owner': 'n4'}},
            'policyTypes': ['Ingress', 'Egress'],
            'ingress': [
                {
                    'from': [peer, control],
                    'ports': [{'protocol': 'TCP', 'port': 1024, 'endPort': 65535}],
                }
            ],
            'egress': [
                {
                    'to': [peer, control],
                    'ports': [{'protocol': 'TCP', 'port': 1024, 'endPort': 65535}],
                },
                {
                    'to': [
                        {
                            'namespaceSelector': {
                                'matchLabels': {'kubernetes.io/metadata.name': 'kube-system'}
                            }
                        }
                    ],
                    'ports': [{'protocol': 'UDP', 'port': 53}, {'protocol': 'TCP', 'port': 53}],
                },
            ],
        },
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text(encoding='utf-8'))
    result = training(**config)
    with Path(a.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
