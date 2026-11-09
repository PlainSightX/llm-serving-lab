"""启动并回收本包CPU容器；探针和主回合均按确切ID清理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

COMMON = [
    '--network',
    'none',
    '--read-only',
    '--cap-drop',
    'ALL',
    '--security-opt',
    'no-new-privileges',
    '--pids-limit',
    '128',
    '--memory',
    '2g',
    '--cpus',
    '2',
    '--tmpfs',
    '/tmp:rw,exec,nosuid,nodev,size=256m',
    '--env',
    'ALFWORLD_DATA=/tmp/alfworld',
    '--user',
    '65534:65534',
]


def owned_container(image, program, *, output, label, mounts=(), timeout=900):
    """取得ID后立即进入finally保护，写盘或版本探针失败也不能留下容器。"""
    run_id = uuid.uuid4().hex
    name = 'lsl-q1-' + run_id[:12]
    args = [
        'docker',
        'create',
        '--name',
        name,
        '--label',
        'serving_lab.owner=q1',
        '--label',
        'serving_lab.run_id=' + run_id,
        *COMMON,
        *mounts,
        image,
        *program,
    ]
    container_id = subprocess.check_output(args, text=True, encoding='utf-8', timeout=60).strip()
    state = None
    completed = None
    try:
        launch = {
            'container_id': container_id,
            'command': args,
            'image_id': image,
            'created': time.time(),
        }
        (output / (label + '-launch.json')).write_text(
            json.dumps(launch, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        completed = subprocess.run(
            ['docker', 'start', '--attach', container_id],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        (output / (label + '-stdout.txt')).write_bytes(completed.stdout)
        (output / (label + '-stderr.txt')).write_bytes(completed.stderr)
        state = json.loads(
            subprocess.check_output(
                ['docker', 'inspect', container_id], text=True, encoding='utf-8', timeout=30
            )
        )[0]['State']
        (output / (label + '-exit.json')).write_text(
            json.dumps(state, indent=2) + '\n', encoding='utf-8'
        )
    finally:
        owned = json.loads(
            subprocess.check_output(
                ['docker', 'inspect', container_id], text=True, encoding='utf-8', timeout=30
            )
        )[0]
        labels = owned['Config'].get('Labels') or {}
        if (
            labels.get('serving_lab.owner') != 'q1'
            or labels.get('serving_lab.run_id') != run_id
            or owned['Name'] != '/' + name
        ):
            raise RuntimeError('容器所有权变化，拒绝清理')
        if owned['State']['Running']:
            subprocess.run(
                ['docker', 'stop', '--time', '5', container_id],
                check=True,
                capture_output=True,
                timeout=30,
            )
        subprocess.run(['docker', 'rm', container_id], check=True, capture_output=True, timeout=30)
    if state['ExitCode'] != 0:
        raise RuntimeError(label + '容器异常退出，结果不得晋级')
    return completed.stdout.decode('utf-8')


def run(data, output, config_path):
    repo = Path(__file__).resolve().parents[1]
    data = Path(data).resolve()
    output = Path(output).resolve()
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('运行目录已存在，禁止覆盖旧前检')
    for path in (data, output, repo):
        if ',' in str(path):
            raise ValueError('容器挂载路径不能包含逗号')
    from serving_lab.task_protocol import validate_manifest

    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if manifest['max_steps'] != config['max_steps']:
        raise ValueError('任务预算与配置不符')
    image = config['image_id']
    inspected = json.loads(
        subprocess.check_output(
            ['docker', 'image', 'inspect', image], text=True, encoding='utf-8', timeout=30
        )
    )[0]
    if inspected['Id'] != image:
        raise ValueError('CPU镜像身份不符')
    output.mkdir(parents=True)
    probe = owned_container(
        image,
        [
            '-c',
            'import json,importlib.metadata; print(json.dumps({n:importlib.metadata.version(n) for n in ("alfworld","textworld","numpy","pyyaml")}))',
        ],
        output=output,
        label='version-probe',
        timeout=60,
    )
    versions = json.loads(probe.strip())
    if versions != config['versions']:
        raise ValueError('环境版本不符')
    mounts = [
        '--mount',
        f'type=bind,source={repo / "src"},target=/code,readonly',
        '--mount',
        f'type=bind,source={repo / "scripts/q1_qualify.py"},target=/runner.py,readonly',
        '--mount',
        f'type=bind,source={data},target=/data,readonly',
        '--mount',
        f'type=bind,source={output},target=/output',
    ]
    owned_container(
        image,
        [
            '/runner.py',
            '--data',
            '/data',
            '--output',
            '/output',
            '--reference-method',
            config['reference_method'],
        ],
        output=output,
        label='episodes',
        mounts=mounts,
    )
    from q1_contexts import validated_records

    _, summary, _, _ = validated_records(output)
    receipt = {
        'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'image_id': image,
        'versions': versions,
        'raw_bundle_revalidated': True,
        'new_model_calls': 0,
        'container_cleanup': 'exact IDs removed',
    }
    (output / 'launcher-verification.json').write_text(
        json.dumps(receipt, indent=2) + '\n', encoding='utf-8'
    )
    return {
        'passed': True,
        'output': str(output),
        'tasks': summary['tasks_executed'],
        'new_model_calls': 0,
        'container_removed': True,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/evaluation/q1-sources.json')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output, args.config), ensure_ascii=False, indent=2))
