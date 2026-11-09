#include <torch/extension.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <limits>

void lsl_fused_add_norm_cuda(torch::Tensor input, torch::Tensor residual,
                            torch::Tensor weight, double epsilon);

// 只接受推理时的二维行向量；任何拒绝都发生在修改输入之前。
void fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                        torch::Tensor weight, double epsilon) {
  TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(input.device() == residual.device() &&
                  input.device() == weight.device(), "device mismatch");
  TORCH_CHECK(input.dim() == 2 && residual.sizes() == input.sizes() &&
                  weight.dim() == 1 && weight.size(0) == input.size(1),
              "shape mismatch");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(), "dtype mismatch");
  auto dtype = input.scalar_type();
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf ||
                  dtype == at::kBFloat16, "unsupported dtype");
  TORCH_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                  residual.is_contiguous() && weight.is_contiguous(),
              "unsupported stride");
  TORCH_CHECK(input.size(1) > 0 && input.size(1) <= 65536 &&
                  input.size(0) <= std::numeric_limits<int>::max(),
              "shape outside supported range");
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(static_cast<float>(epsilon)) &&
                  static_cast<float>(epsilon) > 0, "epsilon must be positive finite float32");
  // AOT可保留参数的requires_grad标记；实际是否追踪梯度由调用上下文决定。
  const bool tracking = c10::GradMode::is_enabled() && !c10::InferenceMode::is_enabled();
  TORCH_CHECK(!tracking || (!input.requires_grad() && !residual.requires_grad() &&
                  !weight.requires_grad()), "inference only; grad_mode=", tracking);
  TORCH_CHECK(!input.is_alias_of(residual) && !input.is_alias_of(weight) &&
                  !residual.is_alias_of(weight), "shared storage is unsupported");
  at::assert_no_overlap(input, residual);
  at::assert_no_overlap(input, weight);
  at::assert_no_overlap(residual, weight);
  if (input.size(0) == 0) return;
  lsl_fused_add_norm_cuda(input, residual, weight, epsilon);
}

TORCH_LIBRARY(lsl_n3, m) {
  m.def("fused_add_rms_norm_(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float epsilon) -> ()");
}
TORCH_LIBRARY_IMPL(lsl_n3, CUDA, m) {
  m.impl("fused_add_rms_norm_", &fused_add_rms_norm);
}
