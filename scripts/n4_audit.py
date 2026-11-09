"""离线核验N4实际rank记录，分别报告训练吞吐、显存和独立剖析。"""

import argparse
import gzip
import json
from pathlib import Path
import statistics

from serving_lab.distributed_study import (
    validate_topology,
    validate_training_steps,
    validate_rank_contracts,
    validate_training_contract,
)
from serving_lab.execution import write_once
from serving_lab.lab import file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def intervals_length(intervals):
    total = 0
    right = None
    for begin, end in sorted(intervals):
        if right is None or begin >= right:
            total += end - begin
            right = end
        elif end > right:
            total += end - right
            right = end
    return total


def trace_metrics(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        trace = json.load(f)
    kernels = [r for r in trace['traceEvents'] if r.get('cat') == 'kernel' and r.get('ph') == 'X']
    communication = [r for r in kernels if 'nccl' in r['name'].lower()]
    regions = [
        r
        for r in trace['traceEvents']
        if r.get('name', '').startswith('n4_') and r.get('ph') == 'X'
    ]
    if not kernels or not regions:
        raise ValueError('独立trace缺少实际GPU或阶段事件')
    span = lambda rows: intervals_length([(r['ts'], r['ts'] + r['dur']) for r in rows]) / 1e6
    result = {
        'kernel_count': len(kernels),
        'communication_kernel_count': len(communication),
        'gpu_active_union_seconds': span(kernels),
        'communication_union_seconds': span(communication),
        'summed_kernel_seconds': sum(r['dur'] for r in kernels) / 1e6,
        'summed_communication_kernel_seconds': sum(r['dur'] for r in communication) / 1e6,
        'cpu_region_seconds': {
            name: sum(r['dur'] for r in regions if r['name'] == name) / 1e6
            for name in sorted({r['name'] for r in regions})
        },
        'trace_sha256': file_hash(path),
        'scope': 'two profiled steps; overlapping spans are not additive wall time',
    }
    result['communication_union_fraction'] = (
        result['communication_union_seconds'] / result['gpu_active_union_seconds']
    )
    return result


def audit_run(root, name):
    paths = sorted(root.glob('gpu-*/results/' + name + '/rank-*/summary.json'))
    if not paths:
        raise ValueError('缺少运行: ' + name)
    summaries = [read(p) for p in paths]
    contract = summaries[0]['contract']
    world = contract['world_size']
    if len(paths) != world or sorted(s['rank'] for s in summaries) != list(range(world)):
        raise ValueError('缺少或重复rank')
    plan = read(root / 'data-b/plan.json')
    records = read(root / 'data-b/samples.json')
    if file_hash(root / 'data-b/plan.json') != contract['plan_sha256']:
        raise ValueError('冻结计划身份不符')
    if file_hash(root / 'data-b/samples.json') != plan['samples_sha256']:
        raise ValueError('样本身份不符')
    traces = []
    all_rows = []
    receipts = []
    starts = []
    for path, summary in zip(paths, summaries):
        folder = path.parent
        initial = read(folder / 'input.json')
        rank = summary['rank']
        effective = validate_training_contract(
            contract, initial['config'], plan, file_hash(root / 'data-b/plan.json')
        )
        if (
            not summary['completed']
            or summary['application_qualified']
            or summary['contract'] != contract
            or initial['contract'] != contract
            or summary['topology'] != summaries[0]['topology']
            or (folder / 'failure.json').exists()
        ):
            raise ValueError('训练摘要、输入或拓扑冲突')
        validate_topology(summary['topology']['members'], summary['topology']['kind'])
        source_manifest = read(
            root
            / (
                'source-c-manifest.json'
                if name.startswith('single-') and 'profile' not in name
                else 'source-d-manifest.json'
            )
        )
        for relative, sha in initial['source_hashes'].items():
            if file_hash(folder / 'source' / relative) != sha or source_manifest[relative] != sha:
                raise ValueError('实际执行源码不符')
        if world > 1:
            signatures = read(folder / 'rank-contracts.json')
            validate_rank_contracts(signatures)
            if (
                len(signatures) != world
                or signatures[rank]['contract'] != contract
                or signatures[rank]['source_hashes'] != initial['source_hashes']
            ):
                raise ValueError('各rank握手证据不同')
        start = read(folder / 'resume.json')['next_step'] if summary['resumed'] else 0
        starts.append(start)
        rows = [
            json.loads(s) for s in (folder / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
        ]
        warmup = effective.get('warmup_steps', 0)
        validate_training_steps(rows, contract, records, rank=rank, start_step=start, warmup=warmup)
        if (
            summary['steps_completed'] != contract['planned_steps']
            or summary['steps_this_run'] != len(rows)
            or summary['measured_steps'] != sum(r['measured'] for r in rows)
        ):
            raise ValueError('摘要步数未匹配原始记录')
        all_rows.append(rows)
        receipts.append(
            {
                'rank': rank,
                'summary_sha256': file_hash(path),
                'steps_sha256': file_hash(folder / 'steps.jsonl'),
                'input_sha256': file_hash(folder / 'input.json'),
            }
        )
        if contract['profile_enabled']:
            traces.append({'rank': rank, **trace_metrics(folder / 'trace.json.gz')})
    if len(set(starts)) != 1:
        raise ValueError('各rank恢复位置不同')
    for per_rank in zip(*all_rows):
        for key in ['step', 'global_sample_ids', 'global_valid_tokens', 'loss']:
            if any(r[key] != per_rank[0][key] for r in per_rank):
                raise ValueError('各rank全局记录不同')
    measured = [rows for rows in zip(*all_rows) if rows[0]['measured']]
    performance = None
    if measured:
        seconds = [max(r['wall_seconds'] for r in rows) for rows in measured]
        performance = {
            'steps': len(seconds),
            'total_seconds': sum(seconds),
            'median_step_seconds': statistics.median(seconds),
            'padded_tokens_per_second': sum(rows[0]['processed_tokens'] for rows in measured)
            / sum(seconds),
            'valid_prediction_tokens_per_second': sum(
                rows[0]['global_valid_tokens'] for rows in measured
            )
            / sum(seconds),
            'peak_allocated_bytes_by_rank': [
                max(r['peak_allocated_bytes'] for r in rows if r['measured']) for rows in all_rows
            ],
            'peak_reserved_bytes_by_rank': [
                max(r['peak_reserved_bytes'] for r in rows if r['measured']) for rows in all_rows
            ],
            'timing_scope': 'max per-rank synchronized forward/backward/optimizer step; excludes barrier/report/checkpoint; not end-to-end job throughput',
        }
    return {
        'run': name,
        'passed': True,
        'contract': contract,
        'rank_receipts': receipts,
        'steps': contract['planned_steps'],
        'start_step': starts[0],
        'performance': performance,
        'profiles': traces,
        'source_equivalence': 'single source-c and distributed source-d math matched; exact sources retained',
    }


def main(args):
    root = Path(args.root)
    names = args.runs.split(',')
    values = [audit_run(root, name) for name in names]
    write_once(args.output, {'passed': True, 'runs': values, 'auditor_sha256': file_hash(__file__)})
    print(
        json.dumps(
            {
                'passed': True,
                'runs': len(values),
                'performance': {v['run']: v['performance'] for v in values if v['performance']},
            }
        )
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    main(p.parse_args())
