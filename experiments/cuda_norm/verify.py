"""先固定数值与边界反例；只有全部通过才进入计时和模型接入。"""

import argparse
import json
from pathlib import Path
import torch
from evidence import check_build, sha


TOLERANCES = {
    torch.float32: (1e-5, 2e-6),
    torch.float16: (0.002, 0.002),
    torch.bfloat16: (0.016, 0.008),
}


def reference(x, residual, weight, epsilon):
    # 固定vLLM CUDA语义有两次dtype舍入；它与默认Inductor的FP32 residual路径不同。
    summed = (x.float() + residual.float()).to(x.dtype)
    inverse = torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + epsilon)
    output = (summed.float() * inverse).to(x.dtype) * weight
    return output, summed


def check_case(op, shape, dtype, *, strided=False, scale=1.0, seed=20261105):
    torch.manual_seed(seed)
    rows, hidden = shape
    backing = torch.randn(rows, hidden + (3 if strided else 0), device='cuda', dtype=dtype) * scale
    x = backing[:, :hidden]
    residual = torch.randn(rows, hidden, device='cuda', dtype=dtype) * scale
    weight = torch.randn(hidden, device='cuda', dtype=dtype)
    before = backing.clone()
    xp, rp = x.clone(), residual.clone()
    expected, new_residual = reference(xp, rp, weight, 1e-6)
    pointer = (x.data_ptr(), residual.data_ptr())
    op(x, residual, weight, 1e-6)
    torch.cuda.synchronize()
    assert pointer == (x.data_ptr(), residual.data_ptr())
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(x, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual, new_residual, rtol=0, atol=0)
    if strided:
        assert torch.equal(backing[:, hidden:], before[:, hidden:])
    from vllm import _custom_ops

    mature_x, mature_r = xp.clone(), rp.clone()
    if rows:
        _custom_ops.fused_add_rms_norm(mature_x, mature_r, weight, 1e-6)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, mature_x, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual, mature_r, rtol=0, atol=0)
    return {
        'shape': shape,
        'dtype': str(dtype),
        'strided': strided,
        'scale': scale,
        'max_abs_reference': (x.float() - expected.float()).abs().max().item() if x.numel() else 0,
        'max_abs_mature': (x.float() - mature_x.float()).abs().max().item() if x.numel() else 0,
    }


def failures(op):
    x = torch.ones(3, 33, device='cuda')
    r = x.clone()
    w = torch.ones(33, device='cuda')
    cases = [
        ('alias', lambda: op(x, x, w, 1e-6)),
        ('negative_epsilon', lambda: op(x, r, w, -1)),
        ('nan_epsilon', lambda: op(x, r, w, float('nan'))),
        ('weight_shape', lambda: op(x, r, w[:-1], 1e-6)),
        ('underflow_epsilon', lambda: op(x, r, w, 1e-300)),
        ('overflow_epsilon', lambda: op(x, r, w, 1e300)),
        ('residual_shape', lambda: op(x, r[:2], w, 1e-6)),
        ('dtype', lambda: op(x.double(), r.double(), w.double(), 1e-6)),
        ('cpu', lambda: op(x.cpu(), r.cpu(), w.cpu(), 1e-6)),
        ('column_stride', lambda: op(x[:, ::2], r[:, ::2], w[::2], 1e-6)),
        ('active_autograd', lambda: op(x.clone().requires_grad_(), r, w, 1e-6)),
    ]
    rejected = []
    for name, call in cases:
        try:
            call()
        except (RuntimeError, NotImplementedError):
            rejected.append(name)
        else:
            raise AssertionError('应拒绝的输入被接受: ' + name)
    return rejected


def inference_leaf_parameters(op):
    checked = []
    for context in [torch.no_grad, torch.inference_mode]:
        x = torch.randn(8, 2560, device='cuda').requires_grad_()
        r = torch.randn_like(x).requires_grad_()
        w = torch.randn(2560, device='cuda').requires_grad_()
        expected, z = reference(x.detach(), r.detach(), w.detach(), 1e-6)
        with context():
            op(x, r, w, 1e-6)
        torch.cuda.synchronize()
        torch.testing.assert_close(x, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(r, z, rtol=0, atol=0)
        checked.append(context.__name__)
    return checked


def concurrency_and_graph(op):
    torch.manual_seed(20261105)
    streams = [torch.cuda.Stream() for _ in range(2)]
    cases = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn(7, 2573, device='cuda', dtype=torch.bfloat16)
            r = torch.randn_like(x)
            w = torch.randn(2573, device='cuda', dtype=x.dtype)
            expected = reference(x, r, w, 1e-6)
            op(x, r, w, 1e-6)
            cases.append((x, r, expected))
    for stream in streams:
        stream.synchronize()
    for x, r, (y, z) in cases:
        torch.testing.assert_close(x, y, rtol=0.016, atol=0.008)
        torch.testing.assert_close(r, z, rtol=0, atol=0)
    x = torch.randn(8, 2560, device='cuda', dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.randn(2560, device='cuda', dtype=x.dtype)
    before = (x.clone(), r.clone())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, r, w, 1e-6)
    x.copy_(before[0])
    r.copy_(before[1])
    graph.replay()
    torch.cuda.synchronize()
    y, z = reference(*before, w, 1e-6)
    torch.testing.assert_close(x, y, rtol=0.016, atol=0.008)
    torch.testing.assert_close(r, z, rtol=0, atol=0)
    return {'independent_streams': 2, 'graph_replay': True}


def main(args):
    library_sha = check_build(args.library, args.build_receipt)
    torch.ops.load_library(args.library)
    op = torch.ops.lsl_n3.fused_add_rms_norm_
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
        (0, 2560),
        (2, 1),
        (2, 33),
    ]
    if args.sanitizer:
        shapes = [(7, 2573), (8, 2560), (2, 33)]
    results = [
        check_case(op, shape, dtype, strided=strided)
        for dtype in TOLERANCES
        for shape in shapes
        for strided in (False, True)
    ]
    if not args.sanitizer:
        results.extend(
            check_case(op, (8, 2560), dtype, scale=scale)
            for dtype in TOLERANCES
            for scale in (0.0, 1e-4, 1e3)
        )
    result = {
        'passed': True,
        'mode': 'sanitizer' if args.sanitizer else 'full',
        'library_sha256': library_sha,
        'verifier_sha256': sha(__file__),
        'cases': results,
        'rejected': failures(op),
        'execution': concurrency_and_graph(op),
        'inference_leaf_contexts': inference_leaf_parameters(op),
        'torch': torch.__version__,
        'device': torch.cuda.get_device_name(),
        'tolerances': {str(k): v for k, v in TOLERANCES.items()},
    }
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {
                'passed': True,
                'numeric_cases': len(results),
                'rejected': len(result['rejected']),
                'graph_streams': result['execution'],
            }
        )
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True)
    p.add_argument('--build-receipt', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sanitizer', action='store_true')
    main(p.parse_args())
