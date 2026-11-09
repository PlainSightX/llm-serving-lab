"""在本机GPU上消费交付镜像和mTLS配置，保留容器内原始请求与清理回执。"""

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import time
import uuid

import httpx

from serving_lab.engine import check_port_available
from serving_lab.execution import write_once
from serving_lab.lab import file_hash
from serving_lab.tls import client_context
from prepare_cluster import read_bundle
from load_cluster import run as load_window
from owned_containers import cleanup_owned

CAP = 64


async def run(bundle_path, image_directory, output, previous=()):
    bundle_path, image_directory = Path(bundle_path).resolve(), Path(image_directory).resolve()
    bundle, _ = read_bundle(bundle_path)
    previous = [Path(path).resolve() for path in previous]
    if len(previous) != len(set(previous)):
        raise ValueError('失败前轮不能重复计数')
    prior_count = 0
    for path in previous:
        initial = json.loads((path / 'input.json').read_text(encoding='utf-8'))
        failed = json.loads((path / 'failure.json').read_text(encoding='utf-8'))
        cleanup = json.loads((path / 'cleanup.json').read_text(encoding='utf-8'))
        if (
            (path / 'summary.json').exists()
            or initial['generation_cap'] != CAP
            or initial['artifact_digest'] != bundle['model_artifact_digest']
        ):
            raise ValueError('前轮不属于同一失败准备包')
        if any(Path(p).resolve() not in previous for p in initial['prior_runs']):
            raise ValueError('前轮成本来源未完整提供')
        if not cleanup['network_removed'] or any(
            item['state']['Running'] for item in cleanup['containers']
        ):
            raise ValueError('前轮尚未清理')
        count = sum(
            json.loads(line)['count']
            for line in (path / 'reservations.jsonl').read_text().splitlines()
        )
        if count != failed['reservations']:
            raise ValueError('前轮预约计数不符')
        prior_count += count
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    prefix = 'lsl-r5-gpu-' + owner[:8]
    containers = []
    network = None
    reservations = 0
    summary = None

    def reserve(count, purpose):
        nonlocal reservations
        if prior_count + reservations + count > CAP:
            raise RuntimeError('本地GPU准备预算耗尽')
        reservations += count
        with (output / 'reservations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(
                json.dumps({'count': count, 'total': reservations, 'purpose': purpose}) + '\n'
            )

    def docker(*arguments, check=True):
        result = subprocess.run(
            ['docker', '--context', 'desktop-linux', *map(str, arguments)],
            capture_output=True,
            text=True,
            encoding='utf-8',
        )
        if check and result.returncode:
            raise RuntimeError('Docker操作失败: ' + result.stderr[-1500:])
        return result

    context = json.loads(docker('context', 'inspect', 'desktop-linux').stdout)[0]
    if not context['Endpoints']['docker']['Host'].startswith('npipe://'):
        raise ValueError('本地GPU演练仅接受本机Docker Desktop上下文')
    receipts = {}
    for role in ('control', 'worker'):
        receipt = json.loads((image_directory / (role + '-image.json')).read_text(encoding='utf-8'))
        actual = json.loads(docker('image', 'inspect', receipt['tag']).stdout)[0]
        if actual['Id'] != receipt['image_id'] or receipt['tag'] != bundle['images'][role]:
            raise ValueError('交付镜像身份变化')
        receipts[role] = receipt
    check_port_available('127.0.0.1', 19383)
    write_once(
        output / 'input.json',
        {
            'owner': owner,
            'bundle_sha256': file_hash(bundle_path / 'bundle.json'),
            'images': receipts,
            'generation_cap': CAP,
            'artifact_digest': bundle['model_artifact_digest'],
            'prior_runs': [str(path) for path in previous],
            'prior_reservations': prior_count,
            'scope': 'local shared GPU container qualification',
        },
    )
    (output / 'source-probe_cluster_gpu.py').write_bytes(Path(__file__).read_bytes())
    write_once(output / 'orchestrator-source.json', {'sha256': file_hash(__file__)})

    def create(role, arguments):
        cid = docker(
            'create',
            '--name',
            prefix + '-' + role,
            '--label',
            'serving-lab.owner=r5-gpu',
            '--label',
            'serving-lab.run_id=' + owner,
            '--network',
            network,
            *arguments,
        ).stdout.strip()
        containers.append((role, cid))
        write_once(output / (role + '-container.json'), {'id': cid, 'owner': owner})
        docker('start', cid)
        return cid

    def state(cid):
        value = json.loads(docker('inspect', cid).stdout)[0]
        if value['Config']['Labels'].get('serving-lab.run_id') != owner:
            raise ValueError('容器所有权不符')
        return value['State']

    def wait_worker(cid):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if not state(cid)['Running']:
                raise RuntimeError('worker容器启动失败')
            ready = docker(
                'exec', cid, 'python3', '/app/scripts/cluster_health.py', 'worker', check=False
            )
            if ready.returncode == 0:
                return
            time.sleep(3)
        raise RuntimeError('worker就绪超时')

    try:
        network = docker(
            'network', 'create', '--label', 'serving-lab.run_id=' + owner, prefix
        ).stdout.strip()
        for slot in ('a', 'b'):
            role = 'worker-' + slot
            directory = output / role
            directory.mkdir()
            reserve(2, role + ' bootstrap')
            cid = create(
                role,
                [
                    '--gpus',
                    'device=0',
                    '--shm-size',
                    '2g',
                    '--memory',
                    '10g',
                    '--network-alias',
                    f'{role}.{bundle["namespace"]}.svc.cluster.local',
                    '--mount',
                    f'type=bind,source={bundle_path / "model"},target=/models/c0,readonly',
                    '--mount',
                    f'type=bind,source={bundle_path / "config"},target=/config,readonly',
                    '--mount',
                    f'type=bind,source={bundle_path / "secrets" / role},target=/secrets,readonly',
                    '--mount',
                    f'type=bind,source={directory},target=/state',
                    receipts['worker']['tag'],
                    '--config',
                    '/config/worker.json',
                ],
            )
            await asyncio.to_thread(wait_worker, cid)
            print(json.dumps({'ready': role, 'reserved_calls': reservations}), flush=True)
        runtime, bootstrap = output / 'runtime', output / 'bootstrap'
        runtime.mkdir()
        bootstrap.mkdir()
        controller = create(
            'gateway',
            [
                '-p',
                '127.0.0.1:19383:8443',
                '--mount',
                f'type=bind,source={bundle_path / "config"},target=/config,readonly',
                '--mount',
                f'type=bind,source={bundle_path / "secrets/gateway"},target=/secrets,readonly',
                '--mount',
                f'type=bind,source={bootstrap},target=/bootstrap',
                '--mount',
                f'type=bind,source={runtime},target=/state',
                '--entrypoint',
                '/bin/sh',
                receipts['control']['tag'],
                '-c',
                'python /app/scripts/collect_worker_manifests.py && exec python /app/scripts/serve_replicas.py --config /bootstrap/gateway.json',
            ],
        )
        pki = bundle_path / 'pki'
        tls = {
            'ca': str(pki / 'ca.pem'),
            'cert': str(pki / 'controller.pem'),
            'key': str(pki / 'controller.key'),
        }
        token = (bundle_path / 'secrets/gateway/admin-token').read_text(encoding='utf-8').strip()
        auth = {'Authorization': 'Bearer ' + token}
        endpoint = 'https://localhost:19383'
        async with httpx.AsyncClient(
            base_url=endpoint, verify=client_context(tls), trust_env=False, timeout=120
        ) as client:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if not state(controller)['Running']:
                    raise RuntimeError('控制者容器启动失败')
                try:
                    response = await client.get('/admin/releases', headers=auth, timeout=3)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError('控制者未就绪')

            async def manage(path, body=None):
                response = await client.post(path, headers=auth, json=body or {})
                value = response.json()
                with (output / 'management.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(
                        json.dumps(
                            {'path': path, 'status': response.status_code, 'response': value}
                        )
                        + '\n'
                    )
                response.raise_for_status()
                return value

            for slot in ('a', 'b'):
                reserve(8, slot + ' transport and choice qualification')
                qualified = await manage('/admin/replicas/' + slot + '/qualify')
                if not qualified['passed']:
                    raise RuntimeError('容器worker资格失败')
            await manage(
                '/admin/route',
                {
                    'operation': 'promote',
                    'target': 'qwen3-17b-coldstart-candidate',
                    'expected_generation': 0,
                },
            )
            body = json.loads((bundle_path / 'sample-request.json').read_text(encoding='utf-8'))
            reserve(1, 'gateway first replica')
            first = await client.post('/v1/chat/completions', json=body)
            first.raise_for_status()
            first_slot = first.headers['x-replica-id']
            write_once(
                output / 'first-request.json',
                {'response': first.json(), 'headers': dict(first.headers)},
            )
            await manage('/admin/replicas/' + first_slot + '/drain')
            reserve(1, 'gateway remaining replica')
            second = await client.post('/v1/chat/completions', json=body)
            second.raise_for_status()
            if second.headers['x-replica-id'] == first_slot:
                raise RuntimeError('排空副本仍收到新请求')
            write_once(
                output / 'second-request.json',
                {'response': second.json(), 'headers': dict(second.headers)},
            )
            reserve(2, 'existing open-loop client through mTLS')
            load = await load_window(
                endpoint, bundle_path / 'workload.json', pki, output / 'load', 2, 1, 2
            )
            if load['valid'] != 2:
                raise RuntimeError('交付负载工具消费失败')
            # SSE终段到达与服务端释放确认分属不同任务，等待原租约完成收尾。
            checks = []
            deadline = time.monotonic() + 15
            while True:
                final = await client.get('/admin/releases', headers=auth)
                final.raise_for_status()
                pending = {item['id']: item['pending'] for item in final.json()['replicas']}
                checks.append(pending)
                if not any(pending.values()):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('请求租约未排空')
                await asyncio.sleep(0.05)
            write_once(output / 'drain-checks.json', checks)
            write_once(output / 'final-state.json', final.json())
        summary = {
            'verified_execution': True,
            'reservations': reservations,
            'cap': CAP,
            'prior_reservations': prior_count,
            'package_reservations': prior_count + reservations,
            'fresh_worker_qualifications': 2,
            'mTLS_application_requests': 4,
            'replicas_observed': [first_slot, second.headers['x-replica-id']],
            'scope': 'two actual containers sharing the local GPU; cloud node checks pending',
        }
    except BaseException as error:
        write_once(
            output / 'failure.json',
            {
                'error_type': type(error).__name__,
                'message': str(error),
                'reservations': reservations,
            },
        )
        raise
    finally:
        cleanup = cleanup_owned(docker, containers, owner, network, output, stop_timeout=60)
        if any(
            item['state']['Running'] or item['state']['ExitCode'] != 0
            for item in cleanup['containers']
        ):
            raise RuntimeError('容器未正常退出，保留失败')
    write_once(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prior-run', action='append', default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.bundle, args.images, args.output, args.prior_run)),
            ensure_ascii=False,
            indent=2,
        )
    )
