"""从锁文件导出核心依赖，冻结小型构建上下文并构建本地交付镜像。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from serving_lab.execution import write_once
from serving_lab.lab import digest, file_hash

REPO = Path(__file__).resolve().parents[1]


def run(output, role, wheelhouse=None):
    directory = Path(output).resolve()
    source = directory / 'context'
    if source.exists() and not (directory / 'context.json').exists():
        raise ValueError('构建上下文尚未冻结，请检查原prepare日志并选择新的输出目录')
    context = json.loads(
        subprocess.run(
            ['docker', 'context', 'inspect', 'desktop-linux'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    if not context['Endpoints']['docker']['Host'].startswith(('npipe://', 'unix://')):
        raise ValueError('本地构建只允许本机Docker传输')
    dependencies = None
    if role == 'worker':
        if wheelhouse is None:
            raise ValueError('worker构建需要已核验的离线wheelhouse')
        wheelhouse = Path(wheelhouse).resolve()
        dependencies = json.loads((wheelhouse / 'verified.json').read_text(encoding='utf-8'))
        if not dependencies['verified'] or dependencies['lock_sha256'] != file_hash(
            REPO / 'environments/serving-linux/uv.lock'
        ):
            raise ValueError('离线依赖与Linux锁不符')
        for item in dependencies['files']:
            if file_hash(wheelhouse / item['file']) != item['sha256']:
                raise ValueError('离线wheel摘要变化: ' + item['name'])
    if not source.exists():
        source.mkdir(parents=True)
        commands = [
            ['uv', 'build', '--wheel', '--out-dir', str(source / 'wheels')],
            [
                'uv',
                'export',
                '--locked',
                '--no-dev',
                '--no-emit-project',
                '--output-file',
                str(source / 'core-requirements.txt'),
            ],
            [
                'uv',
                'export',
                '--project',
                'environments/serving-linux',
                '--locked',
                '--default-index',
                'https://pypi.org/simple',
                '--no-dev',
                '--no-emit-project',
                '--no-emit-package',
                'llm-serving-lab',
                '--emit-index-url',
                '--output-file',
                str(source / 'worker-requirements.txt'),
            ],
        ]
        for index, command in enumerate(commands):
            with (directory / f'prepare-{index}.log').open('xb') as log:
                subprocess.run(command, cwd=REPO, check=True, stdout=log, stderr=subprocess.STDOUT)
        shutil.copytree(
            REPO / 'scripts',
            source / 'scripts',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )
        shutil.copytree(REPO / 'fixtures', source / 'fixtures')
        shutil.copytree(REPO / 'deployment', source / 'deployment')
        files = {
            p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob('*') if p.is_file()
        }
        write_once(directory / 'context.json', {'files': files, 'digest': digest(files)})
    record = json.loads((directory / 'context.json').read_text(encoding='utf-8'))
    if any(file_hash(source / name) != sha for name, sha in record['files'].items()):
        raise ValueError('冻结构建上下文变化')
    images = json.loads((source / 'deployment/images.lock.json').read_text(encoding='utf-8'))
    tag = 'llm-serving-lab-' + role + ':r5-' + record['digest'][:12]
    command = [
        'docker',
        '--context',
        'desktop-linux',
        'build',
        '--platform',
        images['platform'],
        '--pull=false',
        '--build-arg',
        'BASE_IMAGE=' + images[role + '_base'],
        '--label',
        'serving-lab.owner=r5',
        '--label',
        'serving-lab.source=' + record['digest'],
        '-t',
        tag,
        '-f',
        str(source / ('deployment/Dockerfile.' + role)),
        str(source),
    ]
    if dependencies:
        command[4:4] = ['--build-context', 'runtime_wheels=' + str(wheelhouse)]
    with (directory / (role + '-build.log')).open('xb') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    inspected = json.loads(
        subprocess.run(
            ['docker', '--context', 'desktop-linux', 'image', 'inspect', tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    result = {
        'tag': tag,
        'image_id': inspected['Id'],
        'size_bytes': inspected['Size'],
        'context_digest': record['digest'],
        'base': images[role + '_base'],
        'platform': images['platform'],
    }
    if dependencies:
        result['wheelhouse_sha256'] = file_hash(wheelhouse / 'verified.json')
    write_once(directory / (role + '-image.json'), result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--role', choices=['control', 'worker'], required=True)
    parser.add_argument('--wheelhouse')
    args = parser.parse_args()
    run(args.output, args.role, args.wheelhouse)
