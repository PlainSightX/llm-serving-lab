# CUDA residual RMSNorm实验

当前候选未采用，N3整体尚未通过。已完成范围、数值和性能失败见[实际结果](../../docs/results/2026-11-06-n3-cuda-foundation.md)，不得从某个微基准或TensorRT块通过推导完整模型优化成功。

本目录包含一个256线程、paired2/标量尾部的C++/CUDA实现，固定vLLM0.11.2 OOT接入，真实Qwen3-4B观察及同权重的FP32 ONNX/TensorRT归一化块。它保持独立实验边界；默认服务不启用该插件。

## 固定环境与顺序

已核GPU为A10/SM86，服务镜像摘要为24c5476d51446805dfd8614cfb41280b0b29e4743890d24b82c6b5940e4cdb45，Torch2.9.0+cu128、vLLM0.11.2。CUDA12.8.93编译/检查工具与Nsight Compute2025.1.1解包到项目目录；TensorRT10.11.0.33、ONNX1.19.1使用独立前缀。包摘要及实际Job配置保留在 artifacts/n3-20261105-a 中，未替换主机驱动。

在对应Linux GPU环境、项目根目录运行。每个输出目录必须为新目录：

~~~bash
export CUDA_HOME=/path/to/owned/toolchain/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
python experiments/cuda_norm/run.py --stage initial --output /owned/new-initial
python experiments/cuda_norm/run.py --stage qualify --initial /owned/new-initial --output /owned/new-qualification
~~~

initial生成二进制、build receipt和full验证。qualify先核当前二进制/源码/full验证器摘要，再顺序运行四个sanitizer和微基准；任何一步失败即停止。原地输入不接受共享storage。没有autograd实现；活动梯度跟踪会拒绝，而显式推理上下文允许保留requires_grad的叶子参数。

验证器覆盖81个数值场景、11类拒绝、两种推理上下文、双stream及CUDA Graph。sanitizer是缩减形状检查，不能取代full证明。支持的数值容差和实际最大差异写在每次验证JSON中。微基准是同进程、原地链条、图重放的描述性测量；不代表完整模型性能。

## 模型与块级入口

plugin目录可构建纯Python wheel，安装到独立prefix。模型实验仅通过显式VLLM_PLUGINS/LSL_N3_ENABLE以及二进制摘要启用。integration.py同时需要不可变BF16模型、固定N2输入plan和新的输出目录；自定义分支还必须给library、build-receipt、verification。真实GPU kernel出现须从Nsight原件确认，注册日志不足以证明执行。

default与mature保持原引擎；custom只替换带residual的归一化。每种方案4预热+16观察；独立profile另计16请求。当前候选与成熟实现逐token一致性仅1/16，成熟自身重复16/16，因此不进入默认服务，也不能沿用原BF16应用资格。

block.py导出相同归一化公式、实际Qwen权重的FP32块。动态tokens范围1..4096，hidden固定2560；使用显式非默认stream，检查PyTorch、自定义CUDA、TensorRT数值及越界形状拒绝。它不是完整LLM的TensorRT部署。

## 参考与归属

数学背景：[RMSNorm论文](https://arxiv.org/abs/1910.07467)。成熟比较与舍入合同：[固定vLLM CUDA实现](https://github.com/vllm-project/vllm/blob/v0.11.2/csrc/layernorm_kernels.cu)、[RMSNorm层](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/layers/layernorm.py)、[CustomOp扩展接口](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/custom_op.py)。

工具边界：[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、[TensorRT10.11兼容说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.11.0.html)。
