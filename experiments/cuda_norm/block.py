"""同一真实Qwen权重的归一化块导出ONNX并执行TensorRT；不是完整LLM部署。"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from evidence import check_full, sha
from verify import reference


class NormalizationBlock(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight.float())

    def forward(self, x, residual):
        summed = x + residual
        normalized = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-6)
        return normalized * self.weight, summed


def execute(args):
    import onnx
    import tensorrt as trt
    from safetensors import safe_open
    from serving_lab.lab import inspect_base, digest

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    library_sha = check_full(args.library, args.build_receipt, args.verification)
    torch.ops.load_library(args.library)
    model = Path(args.model)
    base = inspect_base(model)
    assert (
        digest(base['files']) == 'b45fba1ab01819bd9540b73858cd32e109bfeefed07ac36b5ada9248650dbba2'
    )
    name = 'model.layers.0.post_attention_layernorm.weight'
    index = json.loads((model / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    with safe_open(str(model / index['weight_map'][name]), framework='pt', device='cpu') as handle:
        weight = handle.get_tensor(name).float()
    assert tuple(weight.shape) == (2560,)
    module = NormalizationBlock(weight).eval()
    torch.manual_seed(20261105)
    onnx_path = out / 'qwen-rms-block.onnx'
    torch.onnx.export(
        module,
        (torch.randn(8, 2560), torch.randn(8, 2560)),
        str(onnx_path),
        dynamo=False,
        opset_version=17,
        input_names=['x', 'residual'],
        output_names=['normalized', 'residual_out'],
        dynamic_axes={n: {0: 'tokens'} for n in ['x', 'residual', 'normalized', 'residual_out']},
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for n in ['x', 'residual']:
        profile.set_shape(n, (1, 2560), (8, 2560), (4096, 2560))
    config.add_optimization_profile(profile)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT构建失败')
    (out / 'qwen-rms-block.plan').write_bytes(bytes(serialized))
    build_seconds = time.monotonic() - started
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert all(
        engine.get_tensor_dtype(engine.get_tensor_name(i)) == trt.float32
        for i in range(engine.num_io_tensors)
    )
    original_stream = torch.cuda.current_stream()
    gpu_weight = weight.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(original_stream)
    torch.cuda.set_stream(stream)
    rows = []
    for tokens in [1, 8, 127, 2049, 4096]:
        x = torch.randn(tokens, 2560, device='cuda')
        residual = torch.randn_like(x)
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        for n in ['x', 'residual']:
            assert context.set_input_shape(n, (tokens, 2560))
        for n, tensor in [('x', x), ('residual', residual), ('normalized', y), ('residual_out', z)]:
            assert context.set_tensor_address(n, tensor.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        expected, new_residual = reference(x, residual, gpu_weight, 1e-6)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(z, new_residual, rtol=0, atol=0)
        custom_x, custom_r = x.clone(), residual.clone()
        torch.ops.lsl_n3.fused_add_rms_norm_(custom_x, custom_r, gpu_weight, 1e-6)
        stream.synchronize()
        torch.testing.assert_close(y, custom_x, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(z, custom_r, rtol=0, atol=0)
        for _ in range(10):
            assert context.execute_async_v3(stream.cuda_stream)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            assert context.execute_async_v3(stream.cuda_stream)
        end.record()
        end.synchronize()
        rows.append(
            {
                'shape': [tokens, 2560],
                'dtype': 'float32',
                'max_abs_reference': (y - expected).abs().max().item(),
                'max_abs_custom': (y - custom_x).abs().max().item(),
                'cuda_event_mean_us': start.elapsed_time(end) * 10,
                'output_sha256': hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
            }
        )
    rejected = []
    for tokens in [0, 4097]:
        try:
            accepted = context.set_input_shape('x', (tokens, 2560))
        except RuntimeError:
            accepted = False
        if accepted:
            raise AssertionError('TensorRT接受了profile范围外形状')
        rejected.append(tokens)
    result = {
        'passed': True,
        'scope': 'FP32 normalization block with actual Qwen weight; not full LLM or BF16 serving qualification',
        'model_artifact_digest': digest(base['files']),
        'weight_name': name,
        'weight_sha256': hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
        'library_sha256': library_sha,
        'verification_sha256': sha(args.verification),
        'runner_sha256': sha(__file__),
        'onnx_version': onnx.__version__,
        'tensorrt_version': trt.__version__,
        'onnx_sha256': sha(onnx_path),
        'engine_sha256': sha(out / 'qwen-rms-block.plan'),
        'build_seconds': build_seconds,
        'cases': rows,
        'out_of_profile_shapes_rejected': rejected,
        'timing': 'CUDA event interval includes host submission gaps; separate from graph microbenchmark',
    }
    with (out / 'result.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    stream.synchronize()
    torch.cuda.set_stream(original_stream)
    print(
        json.dumps(
            {
                'passed': True,
                'shapes': len(rows),
                'build_seconds': build_seconds,
                'rejected': rejected,
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ['model', 'library', 'build-receipt', 'verification', 'output']:
        p.add_argument('--' + n, required=True)
    with torch.inference_mode():
        execute(p.parse_args())
