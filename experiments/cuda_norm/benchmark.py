"""正确性通过后的配对内核计时；不把微基准外推成模型加速。"""

import argparse
import json
from pathlib import Path
import statistics
import torch
from verify import reference
from evidence import check_full, sha


def inputs(shape, dtype):
    torch.manual_seed(20261105)
    return (
        torch.randn(*shape, device='cuda', dtype=dtype),
        torch.randn(*shape, device='cuda', dtype=dtype),
        torch.randn(shape[-1], device='cuda', dtype=dtype),
    )


def measure(function, original):
    x, r, w = [v.clone() for v in original]
    for _ in range(20):
        function(x, r, w, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            function(x, r, w, 1e-6)
    values = []
    for _ in range(5):
        x.copy_(original[0])
        r.copy_(original[1])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 1000)
    return {
        'trial_us': values,
        'median_us': statistics.median(values),
        'min_us': min(values),
        'max_us': max(values),
    }


def main(args):
    library_sha = check_full(args.library, args.build_receipt, args.verification)
    torch.ops.load_library(args.library)
    custom = torch.ops.lsl_n3.fused_add_rms_norm_
    if args.one:
        x, r, w = inputs((args.rows, args.hidden), torch.bfloat16)
        custom(x, r, w, 1e-6)
        torch.cuda.synchronize()
        print(json.dumps({'profiled_shape': [args.rows, args.hidden], 'dtype': 'bfloat16'}))
        return
    from vllm import _custom_ops

    def naive(x, r, w, eps):
        y, z = reference(x, r, w, eps)
        x.copy_(y)
        r.copy_(z)

    methods = {
        'torch_reference': naive,
        'vllm_cuda': _custom_ops.fused_add_rms_norm,
        'custom_cuda': custom,
    }
    shapes = [
        (1, 2560),
        (4, 2560),
        (8, 2560),
        (512, 2560),
        (3072, 2560),
        (3, 2560),
        (127, 2560),
        (2049, 2560),
        (7, 2573),
    ]
    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for index, shape in enumerate(shapes):
            data = inputs(shape, dtype)
            order = list(methods)
            order = order[index % 3 :] + order[: index % 3]
            result = {
                'shape': shape,
                'dtype': str(dtype),
                'heldout': index >= 5,
                'methods': {name: measure(methods[name], data) for name in order},
            }
            result['custom_speed_ratio_over_vllm'] = (
                result['methods']['vllm_cuda']['median_us']
                / result['methods']['custom_cuda']['median_us']
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        'shape': shape,
                        'dtype': str(dtype),
                        'ratio': result['custom_speed_ratio_over_vllm'],
                    }
                ),
                flush=True,
            )
    result = {
        'rows': rows,
        'trials': 5,
        'library_sha256': library_sha,
        'verification_sha256': sha(args.verification),
        'scope': 'five within-process paired diagnostic trials; not independent system replications',
        'timing': 'CUDA graph of100 in-place calls,10 replays per trial; resets outside timing',
        'system_gain_established': False,
    }
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True)
    p.add_argument('--build-receipt', required=True)
    p.add_argument('--verification', required=True)
    p.add_argument('--output')
    p.add_argument('--one', action='store_true')
    p.add_argument('--rows', type=int, default=8)
    p.add_argument('--hidden', type=int, default=2560)
    with torch.inference_mode():
        main(p.parse_args())
