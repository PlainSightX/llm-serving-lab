"""流式核验去重参数归档，可按逻辑目录恢复；失败产物不会取得校验回执。"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile


def restore(archive, output, prefix):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if (
        not prefix
        or '\\' in prefix
        or PurePosixPath(prefix).is_absolute()
        or '..' in PurePosixPath(prefix).parts
    ):
        raise ValueError('恢复目录必须是归档内相对前缀')
    received = set()
    written = []
    with tarfile.open(archive, 'r|gz') as tar:
        first = next(iter(tar))
        if first.name != 'tensor-archive-map.json' or not first.isfile():
            raise ValueError('缺少首项归档映射')
        manifest = json.load(tar.extractfile(first))
        by_sha = {}
        for name, value in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('映射路径越界')
            if (
                not re.fullmatch('[0-9a-f]{64}', value['sha256'])
                or type(value['bytes']) is not int
                or value['bytes'] < 0
            ):
                raise ValueError('映射身份无效')
            by_sha.setdefault(value['sha256'], []).append((name, value['bytes']))
        for item in tar:
            if item.name == 'tensor-archive-map.json':
                continue
            sha = item.name.removeprefix('blobs/')
            if (
                not item.isfile()
                or item.name != 'blobs/' + sha
                or sha not in by_sha
                or sha in received
            ):
                raise ValueError('未知或重复blob')
            if any(size != item.size for _, size in by_sha[sha]):
                raise ValueError('blob大小不符')
            selected = [name for name, _ in by_sha[sha] if name.startswith(prefix)]
            target = output / selected[0] if selected else None
            stream = None
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = target.open('xb')
            h = hashlib.sha256()
            try:
                source = tar.extractfile(item)
                for block in iter(lambda: source.read(4194304), b''):
                    h.update(block)
                    if stream:
                        stream.write(block)
            finally:
                if stream:
                    stream.close()
            if h.hexdigest() != sha:
                raise ValueError('blob实际SHA不符: ' + sha)
            if selected:
                written.append(selected[0])
                for name in selected[1:]:
                    other = output / name
                    other.parent.mkdir(parents=True, exist_ok=True)
                    os.link(target, other)
                    written.append(name)
            received.add(sha)
    if received != set(by_sha) or len(received) != manifest['distinct_blobs']:
        raise ValueError('归档未覆盖完整blob集合')
    if not written:
        raise ValueError('请求的逻辑前缀没有匹配文件')
    result = {
        'verified': True,
        'archive': str(Path(archive).resolve()),
        'distinct_blobs': len(received),
        'logical_files': len(manifest['files']),
        'restored_files': written,
        'prefix': prefix,
        'all_blob_bytes_verified': True,
    }
    (output / 'tensor-archive-map.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prefix', default='export-a/')
    a = p.parse_args()
    restore(a.archive, a.output, a.prefix)
