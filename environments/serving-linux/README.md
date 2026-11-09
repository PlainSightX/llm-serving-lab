# R1独立Linux推理环境

用于固定版本服务资格，不是正式性能优化结果。当前组合：Python3.12.14、vLLM0.11.2、PyTorch2.9.0+cu128、Transformers4.57.1。完整依赖由本目录uv.lock固定；根项目只新增插件入口元数据，原Windows trainer环境保持。

本机环境位于WSL Ubuntu-26.04的独占目录：$HOME/.local/share/llm-serving-lab。uv0.12.5由PyPI官方wheel校验SHA后取得；bootstrap-uv.json记录来源。Linux Python、uv缓存和serving-venv均在该目录，未安装到系统Python或其他项目环境。

在Linux中从项目根运行，下列路径为本机已登记实例；换机器先登记新的独占根和资源，不复用Windows .venv：

```bash
cd /path/to/llm-serving-lab
env UV_PROJECT_ENVIRONMENT=$HOME/.local/share/llm-serving-lab/serving-venv UV_PYTHON_INSTALL_DIR=$HOME/.local/share/llm-serving-lab/python UV_CACHE_DIR=$HOME/.local/share/llm-serving-lab/uv-cache $HOME/.local/share/llm-serving-lab/bin/uv sync --project environments/serving-linux --locked --python 3.12
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/start_engine.py --config configs/serving/local-smoke.json --run-dir artifacts/r1-my-engine
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/probe_serving.py --binding artifacts/r1-my-engine/binding.json --output artifacts/r1-my-direct
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python -m serving_lab.online --binding artifacts/r1-my-engine/binding.json --runtime artifacts/r1-my-service --port 19081
```

引擎默认127.0.0.1:19080，网关默认19081。启动前检查端口，避免已运行的其他业务。引擎为受控后台进程；网关命令前台运行。先完成真实直连probe，再允许网关识别实际Scheduler进程。每次引擎启动必须使用新目录；控制者恢复则复用原runtime目录，保留去重与未知尝试。

停止引擎必须在同一Linux环境执行：

```bash
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/start_engine.py --stop artifacts/r1-my-engine/binding.json
```

完整故障验收会杀掉指定绑定的自有引擎、创建替代实例并重启自有网关；不要对其他项目或非本次所有的binding使用：

```bash
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/probe_online.py --binding artifacts/r1-my-engine/binding.json --output artifacts/r1-my-qualification --exercise-worker-exit
```

probe_online.py保存输入、配置、源码与锁快照，结束时核对未漂移；场景使用真实TCP，不把CPU替身测试当GPU证据。它默认会在退出时停止自己启动的网关和替代引擎；故障前的原引擎是否已退出仍需核验。未带--exercise-worker-exit时原引擎留给调用者显式停止。

## 实现与边界

- 真实GPU入口为vLLM，未使用模拟推理回退；Qwen3-0.6B仅验证服务路径，不代表业务能力。
- enforce_eager与0.55 GPU预算是本地资格配置。正式性能基线需重新选择合理配置和留出负载，不从三组配对耗时宣称加速。
- serving_lab_release_probe通过显式插件白名单观测固定vLLM0.11.2的Scheduler._free_blocks。它先调用原方法，再记录请求状态、KV引用解除、实际进程与启动代际；不改变调度或参数。这是诊断支持，不是框架性能贡献。
- 正常完成、请求终态与引擎清理分别记录。取消资格要求取消前仍在执行，随后观察FINISHED_ABORTED且请求KV引用解除；总显存不回落不代表泄漏。
- 正式验收的在线日志在/mnt/d项目artifacts，耗时包含本机WSL、文件系统、记录与探针开销；不是生产容量结论。
- 默认单控制者和单GPU。多版本切换、双节点集群、训练混布、学习效果均不由R1替代。

版本来源：[vLLM0.11.2](https://pypi.org/project/vllm/0.11.2/)、[版本安装文档](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)、[Scheduler释放路径](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/core/sched/scheduler.py)。Linux运行根作为项目环境保留，停止服务不删除环境；替换环境时再归档和清理确切自有路径。
