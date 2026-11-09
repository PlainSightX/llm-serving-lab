#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

template <typename T> struct alignas(2 * sizeof(T)) Pair { T a, b; };

__device__ __forceinline__ float warp_sum(float value) {
  // 固定256线程，所有warp完整参与；尾部只屏蔽访存，不屏蔽shuffle。
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

template <typename T, bool Paired>
__global__ void lsl_fused_add_rms_norm(T* input, T* residual, const T* weight,
                                     int hidden, int64_t stride, float epsilon) {
  __shared__ float partial[8];
  __shared__ float inverse;
  const int row = blockIdx.x, tid = threadIdx.x;
  T* x = input + row * stride;
  T* r = residual + static_cast<int64_t>(row) * hidden;
  float squared = 0;
  if constexpr (Paired) {
    auto* xp = reinterpret_cast<Pair<T>*>(x);
    auto* rp = reinterpret_cast<Pair<T>*>(r);
    for (int j = tid; j < hidden / 2; j += blockDim.x) {
      Pair<T> a = xp[j], b = rp[j];
      Pair<T> z{T(float(a.a) + float(b.a)), T(float(a.b) + float(b.b))};
      rp[j] = z;
      squared += float(z.a) * float(z.a) + float(z.b) * float(z.b);
    }
  } else {
    for (int j = tid; j < hidden; j += blockDim.x) {
      T z = T(float(x[j]) + float(r[j]));
      r[j] = z;
      squared += float(z) * float(z);
    }
  }
  squared = warp_sum(squared);
  if (tid % 32 == 0) partial[tid / 32] = squared;
  __syncthreads();
  if (tid < 32) {
    float total = warp_sum(tid < 8 ? partial[tid] : 0.f);
    if (tid == 0) inverse = rsqrtf(total / hidden + epsilon);
  }
  __syncthreads();
  // 原地residual已经舍入。归一化结果也先舍入，再乘权重，匹配固定上游CUDA合同。
  if constexpr (Paired) {
    auto* xp = reinterpret_cast<Pair<T>*>(x);
    auto* rp = reinterpret_cast<const Pair<T>*>(r);
    auto* wp = reinterpret_cast<const Pair<T>*>(weight);
    for (int j = tid; j < hidden / 2; j += blockDim.x) {
      Pair<T> z = rp[j], w = wp[j];
      T na = T(float(z.a) * inverse), nb = T(float(z.b) * inverse);
      xp[j] = Pair<T>{T(float(na) * float(w.a)), T(float(nb) * float(w.b))};
    }
  } else {
    for (int j = tid; j < hidden; j += blockDim.x) {
      T normalized = T(float(r[j]) * inverse);
      x[j] = T(float(normalized) * float(weight[j]));
    }
  }
}

template <typename T>
void launch(torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
            double epsilon, cudaStream_t stream) {
  const int rows = input.size(0), hidden = input.size(1);
  const int64_t stride = input.stride(0);
  constexpr auto alignment = alignof(Pair<T>);
  bool paired = hidden % 2 == 0 && stride % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input.data_ptr<T>()) % alignment == 0 &&
      reinterpret_cast<uintptr_t>(residual.data_ptr<T>()) % alignment == 0 &&
      reinterpret_cast<uintptr_t>(weight.data_ptr<T>()) % alignment == 0;
  if (paired)
    lsl_fused_add_rms_norm<T, true><<<rows, 256, 0, stream>>>(
        input.data_ptr<T>(), residual.data_ptr<T>(), weight.data_ptr<T>(),
        hidden, stride, static_cast<float>(epsilon));
  else
    lsl_fused_add_rms_norm<T, false><<<rows, 256, 0, stream>>>(
        input.data_ptr<T>(), residual.data_ptr<T>(), weight.data_ptr<T>(),
        hidden, stride, static_cast<float>(epsilon));
}

void lsl_fused_add_norm_cuda(torch::Tensor input, torch::Tensor residual,
                            torch::Tensor weight, double epsilon) {
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kFloat) launch<float>(input, residual, weight, epsilon, stream);
  else if (input.scalar_type() == at::kHalf) launch<at::Half>(input, residual, weight, epsilon, stream);
  else launch<at::BFloat16>(input, residual, weight, epsilon, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
