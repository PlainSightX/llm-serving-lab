"""导出可离线导入的镜像，并核验归档中的镜像配置身份。"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from serving_lab.execution import write_once
from serving_lab.lab import file_hash


def verify_archive(path, expected_id):
    """Docker新存储后端返回OCI index身份，沿amd64描述符核到实际配置和层。"""
    hashes, documents = {}, {}
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            checksum = hashlib.sha256()
            small = bytearray()
            while chunk := source.read(1024 * 1024):
                checksum.update(chunk)
                if member.size <= 2 * 1024**2:
                    small.extend(chunk)
            hashes[member.name] = checksum.hexdigest()
            if small:
                try:
                    documents[member.name] = json.loads(small)
                except (ValueError, UnicodeError):
                    pass
    configurations = []

    def descriptor(value):
        algorithm, expected = value['digest'].split(':', 1)
        name = 'blobs/' + algorithm + '/' + expected
        if algorithm != 'sha256' or hashes.get(name) != expected:
            raise ValueError('OCI描述符内容摘要不符')
        document = documents.get(name, {})
        if 'manifests' in document:
            for child in document['manifests']:
                platform = child.get('platform', {})
                if platform.get('architecture') in (None, 'amd64') and platform.get('os') in (
                    None,
                    'linux',
                ):
                    descriptor(child)
        elif 'config' in document and 'layers' in document:
            descriptor(document['config'])
            configurations.append(document['config']['digest'])
            for layer in document['layers']:
                descriptor(layer)

    index = documents.get('index.json')
    if index:
        root = next((value for value in index['manifests'] if value['digest'] == expected_id), None)
        if root is None:
            raise ValueError('归档不包含指定OCI镜像身份')
        descriptor(root)
        if not configurations:
            raise ValueError('归档缺少linux/amd64运行清单')
    else:
        manifest = documents.get('manifest.json')
        if (
            not manifest
            or len(manifest) != 1
            or 'sha256:' + hashes[manifest[0]['Config']] != expected_id
        ):
            raise ValueError('旧格式镜像配置身份不符')
        configurations.append(expected_id)
    return {
        'image_id': expected_id,
        'configuration_digests': configurations,
        'platform': 'linux/amd64',
    }


def export(tag, expected_id, output):
    output = Path(output)
    inspected = json.loads(
        subprocess.run(
            ['docker', '--context', 'desktop-linux', 'image', 'inspect', tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    if inspected['Id'] != expected_id:
        raise ValueError('待导出镜像身份不符')
    if output.exists():
        identity = verify_archive(output, expected_id)
        return {
            'tag': tag,
            'image_id': expected_id,
            'archive': output.name,
            'sha256': file_hash(output),
            'bytes': output.stat().st_size,
            'archive_identity': identity,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('xb') as log:
        process = subprocess.Popen(
            ['docker', '--context', 'desktop-linux', 'image', 'save', tag],
            stdout=subprocess.PIPE,
            stderr=log,
        )
        try:
            with (
                output.open('xb') as destination,
                gzip.GzipFile(
                    filename='', mode='wb', fileobj=destination, compresslevel=1, mtime=0
                ) as zipped,
            ):
                shutil.copyfileobj(process.stdout, zipped, length=1024 * 1024)
            if process.wait() != 0:
                raise RuntimeError('镜像导出失败')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    identity = verify_archive(output, expected_id)
    return {
        'tag': tag,
        'image_id': expected_id,
        'archive': output.name,
        'sha256': file_hash(output),
        'bytes': output.stat().st_size,
        'archive_identity': identity,
    }


def run(images, output):
    context = json.loads(
        subprocess.run(
            ['docker', 'context', 'inspect', 'desktop-linux'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    if not context['Endpoints']['docker']['Host'].startswith(('npipe://', 'unix://')):
        raise ValueError('本地导出只允许本机Docker传输')
    images, output = Path(images), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for role in ('control', 'worker'):
        receipt = json.loads((images / (role + '-image.json')).read_text(encoding='utf-8'))
        results[role] = export(receipt['tag'], receipt['image_id'], output / (role + '.tar.gz'))
        print(json.dumps({'exported': role, 'bytes': results[role]['bytes']}), flush=True)
    lock = json.loads(
        (Path(__file__).resolve().parents[1] / 'deployment/images.lock.json').read_text(
            encoding='utf-8'
        )
    )
    tag = lock['device_plugin_image']
    inspected = json.loads(
        subprocess.run(
            ['docker', '--context', 'desktop-linux', 'image', 'inspect', tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]
    results['device-plugin'] = export(tag, inspected['Id'], output / 'device-plugin.tar.gz')
    write_once(output / 'images.json', results)
    return {role: {'bytes': row['bytes'], 'sha256': row['sha256']} for role, row in results.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.images, args.output), indent=2))
