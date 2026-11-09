"""按实际镜像兼容标签选取锁定wheel，完整核验后交给离线构建。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tomllib
import urllib.parse
import urllib.request

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from serving_lab.execution import write_once
from serving_lab.lab import file_hash


def select(lock, requirements, tags_file):
    tags = json.loads(Path(tags_file).read_text(encoding='utf-8'))['tags']
    if len(tags) < 100 or not tags[0].startswith('cp312-'):
        raise ValueError('缺少目标CPython3.12镜像的实际兼容标签')
    rank = {tag: index for index, tag in enumerate(tags)}
    environment = {
        **default_environment(),
        'os_name': 'posix',
        'sys_platform': 'linux',
        'platform_system': 'Linux',
        'platform_machine': 'x86_64',
        'python_version': '3.12',
        'python_full_version': '3.12.14',
        'implementation_name': 'cpython',
        'platform_python_implementation': 'CPython',
    }
    packages = tomllib.loads(Path(lock).read_text(encoding='utf-8'))['package']
    selected = []
    for line in Path(requirements).read_text(encoding='utf-8').splitlines():
        if not re.match(r'^[A-Za-z0-9_.-]+==', line):
            continue
        requirement = Requirement(line.removesuffix(' \\').strip())
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        package = next(
            item
            for item in packages
            if canonicalize_name(item['name']) == canonicalize_name(requirement.name)
            and requirement.specifier.contains(item['version'])
        )
        candidates = []
        for wheel in package.get('wheels', []):
            name = urllib.parse.unquote(urllib.parse.urlsplit(wheel['url']).path.rsplit('/', 1)[-1])
            if Path(name).name != name:
                raise ValueError('wheel文件名包含路径')
            compatible = [
                rank[str(tag)] for tag in parse_wheel_filename(name)[3] if str(tag) in rank
            ]
            if compatible:
                candidates.append((min(compatible), name, wheel))
        if not candidates:
            raise ValueError('目标镜像没有兼容wheel: ' + requirement.name)
        _, name, wheel = min(candidates, key=lambda item: item[0])
        selected.append(
            {
                'name': package['name'],
                'version': package['version'],
                'file': name,
                'url': wheel['url'],
                'sha256': wheel['hash'].split(':')[1],
                'bytes': wheel.get('size'),
            }
        )
    return selected


def download(directory, item):
    target = Path(directory) / item['file']
    if target.exists():
        if file_hash(target) != item['sha256']:
            raise ValueError('缓存wheel摘要不符: ' + item['name'])
        return {**item, 'actual_bytes': target.stat().st_size}
    partial = target.with_name(target.name + '.partial')
    if partial.exists() and file_hash(partial) == item['sha256']:
        partial.replace(target)
        return {**item, 'actual_bytes': target.stat().st_size}
    failures = 0
    while failures < 3:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'Range': f'bytes={offset}-{offset + 8 * 1024**2 - 1}'}
        request = urllib.request.Request(item['url'], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                ranged = response.status == 206
                if ranged and not response.headers.get('Content-Range', '').startswith(
                    f'bytes {offset}-'
                ):
                    raise ValueError('续传范围与已保存字节不符')
                expected_size = item['bytes']
                if expected_size is None:
                    total = response.headers.get('Content-Range', '').rsplit('/', 1)[-1]
                    if not total.isdigit():
                        total = response.headers.get('Content-Length', '')
                    if total.isdigit():
                        expected_size = int(total)
                with partial.open('ab' if ranged and offset else 'wb') as stream:
                    while chunk := response.read(64 * 1024):
                        stream.write(chunk)
        except OSError as error:
            failures += 1
            print(
                json.dumps(
                    {'retry': item['name'], 'offset': offset, 'error_type': type(error).__name__}
                ),
                flush=True,
            )
            if failures == 3:
                raise
            continue
        size = partial.stat().st_size
        if expected_size is not None and size >= expected_size:
            if file_hash(partial) != item['sha256']:
                raise ValueError('完整wheel摘要不符: ' + item['name'])
            partial.replace(target)
            print(
                json.dumps({'verified': item['name'], 'bytes': target.stat().st_size}), flush=True
            )
            return {**item, 'actual_bytes': target.stat().st_size}
        failures = failures + 1 if size <= offset else 0
        if size // (64 * 1024**2) > offset // (64 * 1024**2):
            print(json.dumps({'progress': item['name'], 'bytes': size}), flush=True)
    raise ValueError('wheel下载未完整: ' + item['name'])


def run(lock, requirements, tags, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = select(lock, requirements, tags)
    plan = {
        'lock_sha256': file_hash(lock),
        'requirements_sha256': file_hash(requirements),
        'target_tags_sha256': file_hash(tags),
        'files': items,
    }
    plan_path = output / 'download-plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('wheel目录已绑定另一组输入')
    else:
        write_once(plan_path, plan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(lambda item: download(output, item), items))
    write_once(
        output / 'verified.json',
        {
            **plan,
            'files': files,
            'verified': True,
            'total_bytes': sum(item['actual_bytes'] for item in files),
        },
    )
    print(
        json.dumps(
            {'verified_wheels': len(files), 'bytes': sum(item['actual_bytes'] for item in files)}
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', required=True)
    parser.add_argument('--requirements', required=True)
    parser.add_argument('--tags', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.lock, args.requirements, args.tags, args.output)
