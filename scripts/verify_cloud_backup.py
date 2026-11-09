"""独立复核已封存恢复包的每个字节；只读备份，不接触云主机。"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def verify(root):
    root = Path(root).resolve()
    complete = json.loads((root / 'COMPLETE.json').read_text(encoding='utf-8'))
    seal = root / 'SHA256SUMS.json'
    if sha(seal) != complete['seal_sha256']:
        raise ValueError('备份总清单变化')
    records = json.loads(seal.read_text(encoding='utf-8'))['files']

    def check(item):
        name, value = item
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError('备份路径越界')
        if path.stat().st_size != value['bytes'] or sha(path) != value['sha256']:
            raise ValueError('文件校验失败: ' + name)
        return value['bytes']

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(check, records.items()))
    index = json.loads((root / 'restore-index.json').read_text(encoding='utf-8'))
    for digest, ref in index['refs'].items():
        if ref['kind'] not in {'blob', 'tar-member', 'zip-member'}:
            raise ValueError('未知内容引用类型')
        storage = ref['path'] if ref['kind'] == 'blob' else ref['pack']
        if storage not in records:
            raise ValueError('引用了未封存的内容: ' + storage)
    for role in index['roles']:
        manifest_path = root / 'nodes' / role / 'files.json'
        anchor = index['manifests'][role]
        if sha(manifest_path) != anchor['sha256']:
            raise ValueError('节点原件清单身份不符')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (
            manifest['capture_complete'] is not True
            or manifest['role'] != role
            or manifest['instance_id'] != anchor['instance_id']
        ):
            raise ValueError('实例清单未完成')
        files = [row for row in manifest['files'] if row['type'] == 'file']
        if (
            len(files) != anchor['regular_files']
            or len(manifest['files']) != anchor['entries']
            or sum(row['bytes'] for row in files) != anchor['logical_bytes']
            or any(
                row['sha256'] not in index['refs'] or index['sizes'][row['sha256']] != row['bytes']
                for row in files
            )
        ):
            raise ValueError('实例文件覆盖不完整')
    return {
        'verified': True,
        'sealed_files': len(records),
        'sealed_bytes': sum(sizes),
        'roles': index['roles'],
        'scope': 'sealed local backup integrity and project-file coverage; not a live cloud-cluster startup test',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backup', required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.backup)))
