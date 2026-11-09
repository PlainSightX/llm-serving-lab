"""在专属本地Docker网络内核验K3s API、节点调度和双向TLS。"""

import argparse
import copy
import json
from pathlib import Path
import secrets
import subprocess
import time
import uuid

from serving_lab.engine import check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from prepare_cluster import read_bundle
from owned_containers import cleanup_owned

REPO = Path(__file__).resolve().parents[1]


def run(bundle_path, assets_path, image_receipt, output):
    bundle_path, assets_path = Path(bundle_path).resolve(), Path(assets_path).resolve()
    bundle, documents = read_bundle(bundle_path)
    receipt = json.loads(Path(image_receipt).read_text(encoding='utf-8'))
    lock = json.loads((REPO / 'deployment/images.lock.json').read_text(encoding='utf-8'))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    prefix = 'lsl-r5-' + owner[:10]
    containers = []
    network = None
    summary = None
    kube = None
    namespace = bundle['namespace']
    context = json.loads(
        subprocess.run(
            ['docker', 'context', 'inspect', 'desktop-linux'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    if not context['Endpoints']['docker']['Host'].startswith(('npipe://', 'unix://')):
        raise ValueError('本地预检只允许本机Docker传输')

    def docker(*arguments, check=True):
        result = subprocess.run(
            ['docker', '--context', 'desktop-linux', *map(str, arguments)],
            capture_output=True,
            text=True,
            encoding='utf-8',
        )
        if check and result.returncode:
            raise RuntimeError('Docker操作失败: ' + result.stderr[-2000:])
        return result

    image = json.loads(docker('image', 'inspect', receipt['tag']).stdout)[0]
    if image['Id'] != receipt['image_id'] or receipt['tag'] != bundle['images']['control']:
        raise ValueError('本地演练镜像不是冻结构建产物')
    check_port_available('127.0.0.1', 16443)
    images = output / 'images'
    images.mkdir()
    # 显式输入已核上游文件，避免节点启动时重新取得不同版本。
    assets = json.loads((assets_path / 'assets.json').read_text(encoding='utf-8'))
    source_archive = assets_path / 'k3s-airgap-images-amd64.tar.zst'
    expected = next(
        item['sha256'] for item in assets['files'] if item['file'] == source_archive.name
    )
    if file_hash(source_archive) != expected:
        raise ValueError('K3s离线镜像摘要变化')
    import shutil

    shutil.copyfile(source_archive, images / source_archive.name)
    docker('image', 'save', '--output', images / 'control.tar', receipt['tag'])
    write_once(
        output / 'input.json',
        {
            'owner': owner,
            'bundle_sha256': file_hash(bundle_path / 'bundle.json'),
            'image': receipt,
            'k3s': lock['local_k3s'],
            'evidence_level': 'local_kubernetes_cpu_transport',
            'fixture_sha256': file_hash(REPO / 'scripts/cluster_cpu_fixture.py'),
            'original_containers': docker('ps', '-q').stdout.splitlines(),
        },
    )
    try:
        network = docker(
            'network',
            'create',
            '--label',
            'serving-lab.owner=r5-local',
            '--label',
            'serving-lab.run_id=' + owner,
            prefix,
        ).stdout.strip()
        token = secrets.token_hex(24)
        for role in ('control', 'a', 'b'):
            arguments = [
                'create',
                '--name',
                prefix + '-' + role,
                '--privileged',
                '--memory',
                '4g',
                '--cpus',
                '2',
                '--label',
                'serving-lab.owner=r5-local',
                '--label',
                'serving-lab.run_id=' + owner,
                '--network',
                network,
                '--network-alias',
                role,
                '--tmpfs',
                '/run',
                '--tmpfs',
                '/var/run',
                '--mount',
                f'type=bind,source={images},target=/var/lib/rancher/k3s/agent/images,readonly',
                '-e',
                'K3S_TOKEN=' + token,
            ]
            if role == 'control':
                arguments += [
                    '-p',
                    '127.0.0.1:16443:6443',
                    lock['local_k3s'],
                    'server',
                    '--node-name',
                    'control',
                    '--node-label',
                    'serving-lab/role=control',
                    '--disable',
                    'traefik',
                    '--disable',
                    'servicelb',
                    '--disable',
                    'metrics-server',
                    '--tls-san',
                    '127.0.0.1',
                    '--cluster-cidr',
                    '10.72.0.0/16',
                    '--service-cidr',
                    '10.73.0.0/16',
                ]
            else:
                arguments += [
                    '-e',
                    'K3S_URL=https://control:6443',
                    lock['local_k3s'],
                    'agent',
                    '--node-name',
                    'gpu-' + role,
                    '--node-label',
                    'serving-lab/slot=' + role,
                ]
            cid = docker(*arguments).stdout.strip()
            containers.append((role, cid))
            write_once(
                output / (role + '-container.json'), {'id': cid, 'owner': owner, 'role': role}
            )
            docker('start', cid)
        kubeconfig = output / 'kubeconfig.json'
        raw_config = output / 'kubeconfig.yaml'
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            copied = docker(
                'cp', containers[0][1] + ':/etc/rancher/k3s/k3s.yaml', raw_config, check=False
            )
            if copied.returncode:
                time.sleep(2)
                continue
            raw_config.chmod(0o600)
            result = subprocess.run(
                [
                    'kubectl',
                    '--kubeconfig',
                    str(raw_config),
                    'config',
                    'view',
                    '--raw',
                    '-o',
                    'json',
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
            )
            try:
                config = json.loads(result.stdout)
                if len(config['clusters']) != 1:
                    raise ValueError('本地kubeconfig必须恰有一个集群')
                config['clusters'][0]['cluster']['server'] = 'https://127.0.0.1:16443'
                kubeconfig.write_text(json.dumps(config), encoding='utf-8')
                kubeconfig.chmod(0o600)
                break
            except (ValueError, KeyError, IndexError, TypeError):
                (output / 'kubeconfig-export-error.txt').write_text(result.stderr, encoding='utf-8')
                time.sleep(2)
        else:
            raise RuntimeError('本地K3s未生成kubeconfig')

        def kube(*arguments, check=True):
            result = subprocess.run(
                [
                    'kubectl',
                    '--kubeconfig',
                    str(kubeconfig),
                    '--request-timeout=20s',
                    *map(str, arguments),
                ],
                capture_output=True,
                text=True,
                encoding='utf-8',
            )
            if check and result.returncode:
                raise RuntimeError('本地Kubernetes操作失败: ' + result.stderr[-2500:])
            return result

        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            result = kube('get', 'nodes', '-o', 'json', check=False)
            if result.returncode == 0:
                nodes = json.loads(result.stdout)['items']
                if len(nodes) == 3 and all(
                    any(
                        c['type'] == 'Ready' and c['status'] == 'True'
                        for c in n['status']['conditions']
                    )
                    for n in nodes
                ):
                    break
            time.sleep(2)
        else:
            raise RuntimeError('本地三个K3s节点未就绪')
        (output / 'nodes.json').write_text(result.stdout, encoding='utf-8')
        namespace_file = output / 'namespace.json'
        write_once(namespace_file, documents[0])
        kube('apply', '-f', namespace_file)
        admitted = kube('apply', '--dry-run=server', '-f', bundle_path / 'manifests.json')
        (output / 'production-admission.txt').write_text(admitted.stdout, encoding='utf-8')
        fixtures = []
        for item in documents:
            item = copy.deepcopy(item)
            if item['kind'] == 'Deployment':
                if item['metadata']['name'] == 'gateway':
                    continue
                pod = item['spec']['template']['spec']
                pod.pop('runtimeClassName')
                pod['securityContext'] = {'runAsUser': 1000, 'runAsGroup': 1000, 'fsGroup': 1000}
                container = pod['containers'][0]
                container.update(
                    image=receipt['tag'],
                    command=['python', '/fixture/cluster_cpu_fixture.py', 'server'],
                    args=[],
                )
                container['resources'] = {
                    'requests': {'cpu': '100m', 'memory': '128Mi'},
                    'limits': {'memory': '512Mi'},
                }
                container.pop('startupProbe')
                container['readinessProbe'] = {'tcpSocket': {'port': 8443}, 'periodSeconds': 2}
                container['volumeMounts'] = [
                    mount for mount in container['volumeMounts'] if mount['name'] == 'secrets'
                ]
                pod['volumes'] = [
                    volume for volume in pod['volumes'] if volume['name'] == 'secrets'
                ]
                pod['volumes'][0]['secret']['defaultMode'] = 288
                container['volumeMounts'].append(
                    {'name': 'fixture', 'mountPath': '/fixture', 'readOnly': True}
                )
                pod['volumes'].append(
                    {'name': 'fixture', 'configMap': {'name': 'cpu-fixture-code'}}
                )
            fixtures.append(item)
        fixtures.append(
            {
                'apiVersion': 'v1',
                'kind': 'ConfigMap',
                'metadata': {'name': 'cpu-fixture-code', 'namespace': namespace},
                'data': {
                    'cluster_cpu_fixture.py': (REPO / 'scripts/cluster_cpu_fixture.py').read_text(
                        encoding='utf-8'
                    )
                },
            }
        )
        fixture_file = output / 'cpu-fixtures.json'
        write_once(fixture_file, {'apiVersion': 'v1', 'kind': 'List', 'items': fixtures})
        fixture_file.chmod(0o600)
        kube('apply', '-f', fixture_file)
        for slot in ('a', 'b'):
            kube(
                '-n', namespace, 'rollout', 'status', 'deployment/worker-' + slot, '--timeout=180s'
            )

        def client_job(number):
            name = 'tls-client-' + str(number)
            job = {
                'apiVersion': 'batch/v1',
                'kind': 'Job',
                'metadata': {'name': name, 'namespace': namespace},
                'spec': {
                    'backoffLimit': 0,
                    'activeDeadlineSeconds': 90,
                    'template': {
                        'metadata': {'labels': {'serving-lab/component': 'controller'}},
                        'spec': {
                            'restartPolicy': 'Never',
                            'automountServiceAccountToken': False,
                            'nodeSelector': {'serving-lab/role': 'control'},
                            'securityContext': {
                                'runAsUser': 1000,
                                'runAsGroup': 1000,
                                'fsGroup': 1000,
                            },
                            'containers': [
                                {
                                    'name': 'client',
                                    'image': receipt['tag'],
                                    'imagePullPolicy': 'Never',
                                    'command': [
                                        'python',
                                        '/fixture/cluster_cpu_fixture.py',
                                        'client',
                                        '--namespace',
                                        namespace,
                                    ],
                                    'volumeMounts': [
                                        {
                                            'name': 'secrets',
                                            'mountPath': '/secrets',
                                            'readOnly': True,
                                        },
                                        {
                                            'name': 'fixture',
                                            'mountPath': '/fixture',
                                            'readOnly': True,
                                        },
                                    ],
                                }
                            ],
                            'volumes': [
                                {
                                    'name': 'secrets',
                                    'secret': {'secretName': 'gateway', 'defaultMode': 288},
                                },
                                {'name': 'fixture', 'configMap': {'name': 'cpu-fixture-code'}},
                            ],
                        },
                    },
                },
            }
            path = output / (name + '.json')
            write_once(path, job)
            kube('apply', '-f', path)
            waited = kube(
                '-n',
                namespace,
                'wait',
                '--for=condition=complete',
                'job/' + name,
                '--timeout=120s',
                check=False,
            )
            logs = kube('-n', namespace, 'logs', 'job/' + name, check=False)
            raw = logs.stdout
            (output / (name + '.log')).write_text(raw, encoding='utf-8')
            (output / (name + '-stderr.log')).write_text(logs.stderr, encoding='utf-8')
            if waited.returncode:
                raise RuntimeError('本地TLS客户端未完成，详见保留的Job日志和Pod状态')
            return json.loads(raw)

        first = client_job(1)
        before = json.loads(
            kube('-n', namespace, 'get', 'pods', '-l', 'app=worker-b', '-o', 'json').stdout
        )['items'][0]
        kube(
            '-n',
            namespace,
            'delete',
            'pod',
            before['metadata']['name'],
            '--wait=true',
            '--timeout=45s',
        )
        kube('-n', namespace, 'rollout', 'status', 'deployment/worker-b', '--timeout=120s')
        second = client_job(2)
        assert first['workers']['a']['instance'] == second['workers']['a']['instance']
        assert first['workers']['b']['instance'] != second['workers']['b']['instance']
        (output / 'final-pods.json').write_text(
            kube('-n', namespace, 'get', 'pods', '-o', 'json').stdout, encoding='utf-8'
        )
        summary = {
            'verified': True,
            'nodes': 3,
            'production_manifest_admission': True,
            'mtls_dns_and_auth': True,
            'recreated_worker_changed_identity': True,
            'evidence_level': 'local_kubernetes_cpu_transport',
            'model_calls': 0,
        }
    except BaseException as error:
        write_once(
            output / 'failure.json', {'error_type': type(error).__name__, 'message': str(error)}
        )
        raise
    finally:
        diagnostic_errors = []
        if kube is not None:
            try:
                for resource in ('pods', 'jobs', 'events'):
                    result = kube('-n', namespace, 'get', resource, '-o', 'json', check=False)
                    (output / (resource + '-at-cleanup.json')).write_text(
                        result.stdout, encoding='utf-8'
                    )
                    if resource == 'pods' and result.returncode == 0:
                        for pod in json.loads(result.stdout)['items']:
                            name = pod['metadata']['name']
                            logs = kube(
                                '-n', namespace, 'logs', name, '--all-containers=true', check=False
                            )
                            (output / (name + '.log')).write_text(
                                logs.stdout + logs.stderr, encoding='utf-8'
                            )
            except Exception as error:
                diagnostic_errors.append(
                    {'phase': 'kubernetes', 'type': type(error).__name__, 'message': str(error)}
                )
        cleanup_owned(
            docker, containers, owner, network, output, diagnostic_errors=diagnostic_errors
        )
    write_once(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--image-receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.bundle, args.assets, args.image_receipt, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
