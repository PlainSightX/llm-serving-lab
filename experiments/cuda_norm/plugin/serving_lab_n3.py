"""显式启用的固定版本vLLM扩展；默认服务不会自动接入实验算子。"""

import hashlib
import json
import os
from pathlib import Path
import torch
from vllm.model_executor.layers.layernorm import RMSNorm

_registered = False


class ServingLabRMSNorm(RMSNorm):
    def forward_cuda(self, x, residual=None):
        if residual is None or self.variance_size_override is not None or not self.has_weight:
            return super().forward_cuda(x, residual)
        torch.ops.lsl_n3.fused_add_rms_norm_(x, residual, self.weight.data, self.variance_epsilon)
        return x, residual


def register():
    global _registered
    if os.environ.get('LSL_N3_ENABLE') != '1' or _registered:
        return
    import vllm
    from vllm.model_executor.custom_op import CustomOp

    if vllm.__version__ != '0.11.2':
        raise RuntimeError('实验扩展只支持已核vLLM0.11.2')
    library = Path(os.environ['LSL_N3_LIBRARY'])
    sha = hashlib.sha256(library.read_bytes()).hexdigest()
    if sha != os.environ['LSL_N3_LIBRARY_SHA256']:
        raise RuntimeError('算子二进制未匹配验证身份')
    torch.ops.load_library(str(library))

    @torch.library.register_fake('lsl_n3::fused_add_rms_norm_')
    def fake(input, residual, weight, epsilon):
        return None

    CustomOp.register_oot(name='RMSNorm')(ServingLabRMSNorm)
    # OOT查找键是类名；是否启用仍沿用已批准的+rms_norm配置键。
    ServingLabRMSNorm.name = RMSNorm.name
    _registered = True
    output = Path(os.environ['LSL_N3_AUDIT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'plugin-{os.getpid()}.json').open('x', encoding='utf-8') as stream:
        json.dump(
            {
                'pid': os.getpid(),
                'vllm': vllm.__version__,
                'library_sha256': sha,
                'replacement': 'RMSNorm',
                'dispatch_key': ServingLabRMSNorm.name,
                'scope': 'registration proof only; actual execution requires CUDA trace',
            },
            stream,
            indent=2,
        )
