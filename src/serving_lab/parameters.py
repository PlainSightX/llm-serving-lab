"""按真实张量内容标识参数，不把目录说明或重新分片当成新模型。"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from .lab import base_paths, digest, inspect_base


def bound_parameter_identity(binding):
    """只在注册边界做完整工件检查；张量读前读后必须仍匹配装载清单。"""
    for_stage = inspect_base(binding.model_path)
    if digest(for_stage['files']) != binding.artifact_digest:
        raise ValueError('磁盘工件与已装载worker不符')
    result = parameter_identity(binding.model_path)
    if digest(inspect_base(binding.model_path)['files']) != binding.artifact_digest:
        raise ValueError('注册期间工件变化')
    return result


def parameter_identity(directory):
    files = base_paths(directory)
    weights = [path for name, path in files.items() if name.endswith('.safetensors')]
    if not weights:
        raise ValueError('仅接纳显式safetensors参数工件')
    tensors = {}
    widths = {
        'BOOL': 1,
        'U8': 1,
        'I8': 1,
        'I16': 2,
        'U16': 2,
        'I32': 4,
        'U32': 4,
        'I64': 8,
        'U64': 8,
        'F16': 2,
        'BF16': 2,
        'F32': 4,
        'F64': 8,
    }

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('参数头包含重复键')
            result[key] = value
        return result

    for path in sorted(weights):
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('参数文件截断')
            size = struct.unpack('<Q', prefix)[0]
            if not 2 <= size <= 100 * 1024 * 1024:
                raise ValueError('参数头大小异常')
            try:
                header = json.loads(stream.read(size), object_pairs_hook=pairs)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('参数头无效') from exc
            if not isinstance(header, dict):
                raise ValueError('参数头无效')
            total = path.stat().st_size - 8 - size
            ranges = []
            for name, item in header.items():
                if name == '__metadata__':
                    continue
                if name in tensors:
                    raise ValueError('多个分片包含同名参数')
                dtype = item.get('dtype')
                shape = item.get('shape')
                offsets = item.get('data_offsets')
                if (
                    dtype not in widths
                    or not isinstance(shape, list)
                    or any(type(n) is not int or n < 0 for n in shape)
                ):
                    raise ValueError('不支持或无效的参数类型/形状')
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                ):
                    raise ValueError('参数偏移无效')
                start, end = offsets
                if (
                    not 0 <= start <= end <= total
                    or end - start != math.prod(shape) * widths[dtype]
                ):
                    raise ValueError('参数区间或字节数无效')
                ranges.append((start, end))
                stream.seek(8 + size + start)
                h = hashlib.sha256()
                remaining = end - start
                while remaining:
                    data = stream.read(min(4 * 1024 * 1024, remaining))
                    if not data:
                        raise ValueError('参数内容截断')
                    h.update(data)
                    remaining -= len(data)
                tensors[name] = {'dtype': dtype, 'shape': shape, 'sha256': h.hexdigest()}
            offset = 0
            for start, end in sorted(ranges):
                if start != offset:
                    raise ValueError('参数区间重叠或存在未归属字节')
                offset = end
            if offset != total:
                raise ValueError('参数尾部未归属')
    if not tensors:
        raise ValueError('空参数集合')
    return {
        'schema': 1,
        'digest': digest(tensors),
        'tensor_count': len(tensors),
        'tensors': tensors,
    }
