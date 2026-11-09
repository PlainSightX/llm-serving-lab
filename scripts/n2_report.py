"""从已审计原件生成精度—延迟对照；失败候选不会因速度更快变成通过。"""

import argparse
import json
from pathlib import Path
from serving_lab.lab import file_hash
from serving_lab.execution import write_once


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def report(root, output):
    root = Path(root)
    paths = {
        'bf16': root / 'bf16-benchmarks/audit-v3.json',
        'awq': root / 'awq-benchmarks/audit.json',
    }
    audits = {k: read(p) for k, p in paths.items()}
    assert all(
        d['raw_verified'] and d['requests'] == 864 and d['matched_native_releases'] == 864
        for d in audits.values()
    )
    assert audits['bf16']['rates_sha256'] == audits['awq']['rates_sha256']
    assert audits['awq']['quality_gate_passed'] is False and audits['awq']['adoptable'] is False
    profiles = {v: read(root / (v + '-engine.json'))['manifest']['profile'] for v in audits}
    assert {k for k in profiles['bf16'] if profiles['bf16'][k] != profiles['awq'][k]} == {
        'artifact_digest',
        'quantization',
    }
    rows = []
    for shape in ('short', 'prefill', 'decode', 'mixed'):
        for route in ('native', 'gateway', 'upstream'):
            row = {'shape': shape, 'path': route}
            for variant, audit in audits.items():
                cells = [c for c in audit['windows'] if c['shape'] == shape and c['path'] == route]
                assert len(cells) == 3 and all(c['schedule_valid'] for c in cells)
                row[variant] = {
                    'three_window_median_mean': {
                        m: sum(c[m]['median'] for c in cells) / 3 for m in ('ttft', 'tpot', 'e2e')
                    },
                    'individual_window_medians': [
                        {m: c[m]['median'] for m in ('ttft', 'tpot', 'e2e')}
                        for c in sorted(cells, key=lambda c: c['repeat'])
                    ],
                }
            row['descriptive_tpot_ratio'] = (
                row['bf16']['three_window_median_mean']['tpot']
                / row['awq']['three_window_median_mean']['tpot']
            )
            rows.append(row)
    quality = read(root / 'quality-comparison.json')
    gpu = {}
    for variant, name in [
        ('bf16', 'trace-summary-matched.json'),
        ('awq', 'awq-trace-summary-matched.json'),
    ]:
        trace = read(root / name)['nsight']
        groups = {}
        for r in trace['gpu']['rows']:
            n = r['name'].lower()
            group = (
                'attention'
                if 'flash::' in n
                else 'quantized_matrix'
                if 'marlin::' in n
                else 'matrix'
                if 'gemm' in n or 'cutlass::kernel' in n or 'gemv' in n
                else 'named_residual_rms_fusion'
                if 'rsqrt' in n or 'norm' in n
                else 'other_or_unmapped'
            )
            groups[group] = groups.get(group, 0) + r['total_us']
        gpu[variant] = {
            'kernel_sum_us': trace['gpu']['sum_us'],
            'kernel_events': trace['kernel_events'],
            'categories_percent': {k: 100 * v / trace['gpu']['sum_us'] for k, v in groups.items()},
            'scope': 'mutually exclusive name-based groups; numbered Triton kernels remain unmapped',
        }
    result = {
        'audits': {k: {'path': str(p), 'sha256': file_hash(p)} for k, p in paths.items()},
        'formal_requests': 1728,
        'formal_valid': sum(a['fixed_length_valid'] for a in audits.values()),
        'formal_slo_met': sum(a['slo_met'] for a in audits.values()),
        'windows': 72,
        'quality': quality,
        'latency': rows,
        'gpu_diagnosis': gpu,
        'awq_adopted': False,
        'system_optimization_claim': False,
        'interpretation': 'quality-failed AWQ diagnostic; ratios are latency observations at fixed offered load, not throughput/capacity gain',
    }
    write_once(output, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ('latency', 'gpu_diagnosis', 'audits')}
        )
    )
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report(a.root, a.output)
