"""从诊断原件聚合CPU独占区间及GPU内核时间；与无剖析器吞吐分开。"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
from serving_lab.lab import file_hash
from serving_lab.execution import write_once


def union(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def rank(events):
    grouped = defaultdict(lambda: [0.0, 0])
    for name, duration in events:
        grouped[name][0] += duration
        grouped[name][1] += 1
    total = sum(v[0] for v in grouped.values())
    rows = [
        {
            'name': k,
            'total_us': v[0],
            'count': v[1],
            'mean_us': v[0] / v[1],
            'fraction_of_sum': v[0] / total if total else 0,
        }
        for k, v in grouped.items()
    ]
    return {'sum_us': total, 'rows': sorted(rows, key=lambda r: -r['total_us'])}


def torch_trace(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    events = document['traceEvents']
    counts = defaultdict(int)
    for e in events:
        counts[e.get('cat', 'unspecified')] += 1
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and e.get('ph') == 'X']
    by_thread = defaultdict(list)
    for e in cpu:
        by_thread[(e['pid'], e['tid'])].append(e)
    exclusive = []
    crossing = 0
    for values in by_thread.values():
        stack = []
        for event in sorted(values, key=lambda e: (e['ts'], -e['dur'])):
            start = event['ts']
            stop = start + event['dur']
            while stack and stack[-1][0] <= start:
                stack.pop()
            index = len(exclusive)
            exclusive.append([event['name'], event['dur']])
            if stack:
                if stop <= stack[-1][0]:
                    exclusive[stack[-1][1]][1] -= event['dur']
                else:
                    crossing += 1
            stack.append((stop, index))
    if crossing or any(v < -0.01 for _, v in exclusive):
        raise ValueError('CPU区间不是可直接扣减的嵌套结构')
    gpu = rank((e['name'], e['dur']) for e in kernels)
    gpu['active_union_us'] = union((e['ts'], e['ts'] + e['dur']) for e in kernels)
    return {
        'file': str(path),
        'sha256': file_hash(path),
        'categories': dict(counts),
        'device_properties': document.get('deviceProperties'),
        'gpu': gpu,
        'cpu_op_exclusive': rank(exclusive),
        'note': 'CPU self subtracts nested cpu_op only; not full process CPU time. Kernel sums are not wall latency.',
    }


def nsight(path):
    with sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro', uri=True) as db:
        rows = db.execute(
            'SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id'
        ).fetchall()
    gpu = rank((n, (b - a) / 1000) for a, b, n in rows)
    gpu['active_union_us'] = union((a / 1000, b / 1000) for a, b, n in rows)
    gpu['first_to_last_kernel_span_us'] = (
        max(b for a, b, n in rows) - min(a for a, b, n in rows)
    ) / 1000
    return {
        'file': str(path),
        'sha256': file_hash(path),
        'gpu': gpu,
        'kernel_events': len(rows),
        'note': 'Nsight cudaProfilerApi range on separate A10; 16 diagnostic requests, overhead included',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--torch-dir', required=True)
    p.add_argument('--nsight', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = {
        'torch': [torch_trace(f) for f in sorted(Path(a.torch_dir).glob('*.gz'))],
        'nsight': nsight(a.nsight),
    }
    write_once(a.output, result)
    print(
        json.dumps(
            {
                'torch_traces': len(result['torch']),
                'nsight_kernels': result['nsight']['kernel_events'],
                'top_nsight': result['nsight']['gpu']['rows'][:3],
            }
        )
    )
