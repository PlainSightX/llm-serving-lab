"""测量专用目录所有权与停机快照；不改变在线事务与确认语义。"""

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from .execution import write_once
from .lab import file_hash


def export_arrivals(database, output):
    """事务context不等于关闭连接；导出失败也必须在停机快照前close。"""
    db = sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db.row_factory = sqlite3.Row
        write_once(output, [dict(r) for r in db.execute('SELECT * FROM arrivals')])
    finally:
        db.close()


def filesystem(path):
    value = json.loads(
        subprocess.check_output(['findmnt', '--json', '--target', str(path)], text=True)
    )['filesystems'][0]
    if value['fstype'] != 'ext4':
        raise ValueError('本次原生基线要求实际ext4文件系统')
    return value


def prepare_runtime(output, base=None):
    output = Path(output).resolve()
    if base is None:
        return output
    if str(base).replace('\\', '/').startswith('/mnt/'):
        raise ValueError('原生基线不能位于Windows挂载目录')
    base = Path(base).resolve()
    parent = base
    while not parent.exists():
        parent = parent.parent
    fs = filesystem(parent)
    base.mkdir(parents=True, exist_ok=True)
    root = base / output.name
    if root.exists():
        raise ValueError('运行身份已存在，不覆盖原生runtime')
    root.mkdir()
    owner = {
        'owner': 'llm-serving-lab R4',
        'evidence_output': str(output),
        'native_root': str(root),
        'filesystem': fs,
        'scope': 'whole RuntimeStore, including sqlite/payload/wire; process-crash contract, not power-loss guarantee',
    }
    write_once(root / 'owner.json', owner)
    write_once(output / 'runtime-placement.json', owner)
    return root


def snapshot_runtimes(output, root, names):
    """调用者须先停止所有gateway；保留原生原件，复制为报告器的证据快照。"""
    output = Path(output).resolve()
    root = Path(root).resolve()
    if output == root:
        return
    owner = json.loads((root / 'owner.json').read_text())
    if owner['evidence_output'] != str(output) or owner['native_root'] != str(root):
        raise ValueError('目录所有权不符')
    manifest = {}
    for name in names:
        source = root / (name + '-runtime')
        target = output / (name + '-runtime')
        if not source.exists():
            continue
        shutil.copytree(source, target)
        files = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
        if files != {
            str(p.relative_to(target)): file_hash(p) for p in target.rglob('*') if p.is_file()
        }:
            raise RuntimeError('停机快照不一致')
        manifest[name] = {'source': str(source), 'snapshot': str(target), 'files': files}
    write_once(
        output / 'runtime-snapshot.json',
        {'stopped_gateway_snapshot': True, 'runtimes': manifest, 'native_original_retained': True},
    )
