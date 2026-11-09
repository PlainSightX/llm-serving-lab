"""为固定0.12.2加载器生成元数据兼容副本；所有权重文件字节保持不变。"""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from serving_lab.lab import digest, inspect_base, file_hash
from serving_lab.execution import write_once


def compatible_config(original):
    value = copy.deepcopy(original)
    q = value['quantization_config']
    if (
        q.get('quant_method') != 'compressed-tensors'
        or q.get('version') != '0.13.0'
        or q.get('format') != 'pack-quantized'
    ):
        raise ValueError('不是本次固定生产格式')
    if not q.get('config_groups'):
        raise ValueError('缺少量化组')
    removed = {}
    for name, group in q['config_groups'].items():
        w = group['weights']
        if (
            w.get('num_bits') != 4
            or w.get('group_size') != 128
            or w.get('symmetric') is not False
            or w.get('type') != 'int'
            or w.get('dynamic') is not False
            or w.get('scale_dtype') is not None
            or w.get('zp_dtype') != 'torch.int8'
        ):
            raise ValueError('非默认量化dtype不可降级元数据')
        removed[name] = {'scale_dtype': w.pop('scale_dtype'), 'zp_dtype': w.pop('zp_dtype')}
    return value, removed


def run(source, destination, expected):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('兼容副本必须位于原目录之外')
    before = inspect_base(source)
    if digest(before['files']) != expected:
        raise ValueError('原量化工件身份变化')
    config, removed = compatible_config(
        json.loads((source / 'config.json').read_text(encoding='utf-8'))
    )
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if not path.is_file():
            raise ValueError('本次工件只接受平面文件')
        if path.name in {'config.json', 'source.json'}:
            continue
        if path.suffix == '.safetensors':
            os.link(path, destination / path.name)
        else:
            shutil.copyfile(path, destination / path.name)
    write_once(destination / 'config.json', config)
    provenance = json.loads((source / 'source.json').read_text(encoding='utf-8'))
    provenance['loader_compatibility'] = {
        'target': 'vLLM0.11.2/compressed-tensors0.12.2',
        'producer_artifact_digest': expected,
        'removed_default_fields': removed,
        'weight_bytes_unchanged': True,
        'producer_version_not_relabelled': True,
    }
    write_once(destination / 'source.json', provenance)
    result = inspect_base(destination)
    for name, sha in before['files'].items():
        if name.endswith('.safetensors') and result['files'][name] != sha:
            raise ValueError('权重发生变化')
    receipt = {
        'artifact_digest': digest(result['files']),
        'files': result['files'],
        'parent_artifact_digest': expected,
        'removed': removed,
        'weight_bytes_unchanged': True,
        'script_sha256': file_hash(__file__),
    }
    write_once(destination.parent / (destination.name + '-receipt.json'), receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('source', 'destination', 'expected'):
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    run(a.source, a.destination, a.expected)
