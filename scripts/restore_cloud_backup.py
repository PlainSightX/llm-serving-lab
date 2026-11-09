"""将已校验的项目恢复包重建到全新暂存目录；不会覆盖系统或自动启动旧集群。"""

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import uuid
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative_path(value):
    p = PurePosixPath(value)
    if (
        not p.is_absolute()
        or len(p.parts) < 2
        or '..' in p.parts
        or '\\' in value
        or '\x00' in value
    ):
        raise ValueError('无效原始绝对路径')
    relative = Path(*p.parts[1:])
    if os.name == 'nt' and any(':' in part for part in relative.parts):
        raise ValueError('此文件名需在Linux暂存恢复')
    return relative


def backup_path(root, value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('内容引用越过备份根')
    result = (root / p).resolve()
    if not result.is_relative_to(root):
        raise ValueError('内容引用越过备份根')
    return result


def restore(backup, role, output, *, prefixes, symlinks, owners):
    backup = Path(backup).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('只允许全新暂存目录，不覆盖已有系统或文件')
    index = read(backup / 'restore-index.json')
    if role not in index['roles']:
        raise ValueError('未知实例角色')
    manifest_path = backup / 'nodes' / role / 'files.json'
    anchor = index.get('manifests', {}).get(role)
    if not anchor or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != anchor['sha256']:
        raise ValueError('节点原始清单未绑定或已改变')
    manifest = read(manifest_path)
    all_files = [r for r in manifest['files'] if r['type'] == 'file']
    if (
        manifest.get('capture_complete') is not True
        or manifest.get('role') != role
        or manifest.get('instance_id') != anchor['instance_id']
        or len(manifest['files']) != anchor['entries']
        or len(all_files) != anchor['regular_files']
        or sum(r['bytes'] for r in all_files) != anchor['logical_bytes']
    ):
        raise ValueError('实例身份、完整清单或总覆盖不符')
    rows = []
    for row in manifest['files']:
        relative = relative_path(row['path'])
        if relative.parts[0] == 'restore-verification.json':
            raise ValueError('恢复路径与独立报告保留位置冲突')
        if not prefixes or any(
            row['path'] == p.rstrip('/') or row['path'].startswith(p.rstrip('/') + '/')
            for p in prefixes
        ):
            rows.append(row)
    if not rows:
        raise ValueError('选定范围为空')
    names = [r['path'] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError('原始路径重复')
    links = {r['path'].rstrip('/') for r in rows if r['type'] == 'symlink'}
    if any(any(name.startswith(link + '/') for link in links) for name in names):
        raise ValueError('不能在符号链接下写入恢复文件')
    if owners and (os.name != 'posix' or os.geteuid() != 0):
        raise ValueError('恢复uid/gid需要Linux root')
    if symlinks and os.name == 'nt' and links:
        raise ValueError('包含符号链接的完整重建请在Linux容器/主机执行')
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if os.name == 'posix':
        os.chmod(output, 0o700)
    elif os.name == 'nt':
        info = subprocess.run(
            ['whoami', '/user', '/fo', 'csv', '/nh'], capture_output=True, text=True, check=True
        )
        sid = next(csv.reader(info.stdout.splitlines()))[-1]
        subprocess.run(
            [
                'icacls',
                str(output),
                '/inheritance:r',
                '/grant:r',
                '*' + sid + ':(OI)(CI)F',
                '*S-1-5-18:(OI)(CI)F',
            ],
            capture_output=True,
            check=True,
        )
    else:
        raise ValueError('当前系统没有实现私有暂存权限')
    by_sha = defaultdict(list)
    directories = []
    link_rows = []
    ephemeral = []
    for row in rows:
        target = output / relative_path(row['path'])
        if row['type'] == 'directory':
            target.mkdir(parents=True, exist_ok=True)
            directories.append((target, row))
        elif row['type'] == 'file':
            by_sha[row['sha256']].append((target, row))
        elif row['type'] == 'symlink':
            link_rows.append((target, row))
        else:
            ephemeral.append(row['path'])
    restored = set()
    bytes_written = 0

    def metadata(target, row):
        if owners:
            os.chown(target, row['uid'], row['gid'], follow_symlinks=False)
        if not target.is_symlink():
            os.chmod(target, row['mode'])
            os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))

    def consume(digest, stream):
        nonlocal bytes_written
        destinations = by_sha[digest]
        first, row = destinations[0]
        first.parent.mkdir(parents=True, exist_ok=True)
        temporary = first.parent / ('.restore-' + uuid.uuid4().hex)
        h = hashlib.sha256()
        length = 0
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as out:
            for block in iter(lambda: stream.read(4194304), b''):
                length += len(block)
                if length > row['bytes']:
                    raise ValueError('内容超过冻结长度')
                h.update(block)
                out.write(block)
        if h.hexdigest() != digest or any(r['bytes'] != length for _, r in destinations):
            raise ValueError('内容摘要或长度校验失败: ' + row['path'])
        temporary.rename(first)
        for target, other in destinations[1:]:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as out, first.open('rb') as source:
                shutil.copyfileobj(source, out, 4194304)
        for target, other in destinations:
            metadata(target, other)
            bytes_written += length
        restored.add(digest)

    grouped = defaultdict(dict)
    for digest in by_sha:
        ref = index['refs'].get(digest)
        if not ref:
            raise ValueError('缺少必要文件内容: ' + digest)
        if ref['kind'] == 'blob':
            with backup_path(backup, ref['path']).open('rb') as stream:
                consume(digest, stream)
        elif ref['kind'] in {'tar-member', 'zip-member'}:
            grouped[ref['kind'], ref['pack']][ref['member']] = digest
        else:
            raise ValueError('未知内容格式')
    for (kind, pack), wanted in grouped.items():
        path = backup_path(backup, pack)
        seen = set()
        if kind == 'zip-member':
            with zipfile.ZipFile(path) as archive:
                for member, digest in wanted.items():
                    with archive.open(member) as stream:
                        consume(digest, stream)
                        seen.add(member)
        else:
            with tarfile.open(path, 'r|*') as archive:
                for member in archive:
                    if member.name in wanted:
                        if not member.isfile() or member.name in seen:
                            raise ValueError('内容成员不是唯一普通文件')
                        with archive.extractfile(member) as stream:
                            consume(wanted[member.name], stream)
                        seen.add(member.name)
                        if seen == set(wanted):
                            break
        if seen != set(wanted):
            raise ValueError('归档缺少所需成员')
    if restored != set(by_sha):
        raise ValueError('恢复内容不完整')
    if symlinks:
        for target, row in link_rows:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(row['target'], target)
            metadata(target, row)
    # 目录mtime和权限最后恢复，避免过程写入改动它们。
    for target, row in sorted(directories, key=lambda item: -len(item[0].parts)):
        metadata(target, row)
    result = {
        'verified': True,
        'role': role,
        'prefixes': prefixes,
        'files': sum(len(v) for v in by_sha.values()),
        'unique_contents': len(restored),
        'logical_bytes': bytes_written,
        'symlinks_materialized': len(link_rows) if symlinks else 0,
        'symlinks_recorded_not_materialized': []
        if symlinks
        else link_rows and [row for _, row in link_rows],
        'ephemeral_not_materialized': ephemeral,
        'owners_restored': owners,
        'scope': 'file-tree restoration into a new staging directory; no old runtime or Kubernetes identity started',
    }
    result.update(
        manifest_sha256=anchor['sha256'],
        restore_index_sha256=hashlib.sha256(
            (backup / 'restore-index.json').read_bytes()
        ).hexdigest(),
        restorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        staging_root_private=stat.S_IMODE(output.stat().st_mode) == 0o700
        if os.name == 'posix'
        else True,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(output / 'restore-verification.json', flags, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as report:
        json.dump(result, report, indent=2)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--backup', required=True)
    p.add_argument('--role', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prefix', action='append', default=[])
    p.add_argument('--no-symlinks', action='store_true')
    p.add_argument('--owners', action='store_true')
    a = p.parse_args()
    print(
        json.dumps(
            restore(
                a.backup,
                a.role,
                a.output,
                prefixes=a.prefix,
                symlinks=not a.no_symlinks,
                owners=a.owners,
            )
        )
    )
