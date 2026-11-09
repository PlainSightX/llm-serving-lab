# 文档索引

正文在仓库根目录的 [README](../README.md)。这里说明各份记录的用途，以及记录里出现的内部代号。

记录按“当时做了什么、在什么条件下得到什么、哪些没有通过”的方式书写，所以写得很细，也保留了失败和中断。原始运行数据（模型、请求原件、日志）体积约 200 GB，没有放进 Git；每份结果页都写明了测量条件和适用范围。

## 内部代号

| 代号 | 含义 |
|---|---|
| B / S / R | 三条独立验收线：B 工程基石，S 系统深度（性能/引擎），R 持续学习研究。三者互不抵扣 |
| R1 | 单实例真实服务：有界准入、期限、取消后核对 vLLM 释放 KV、控制者崩溃恢复与幂等去重 |
| R2 | 两个真实参数版本的资格、灰度、在途切流、排空、回滚与故障矩阵 |
| R3 | 本机真实负载基线；定位网关事件循环里的同步持久化阻塞 |
| R4 | 把运行账本移到 Linux 原生 ext4 后的基线与判别 |
| R5 | 阿里云三节点 K3s + 两台 A10 的部署、故障恢复与固定负载 SLO（S-A 中断保留，S-B 完整重跑） |
| Q1 / Q2 | 持续学习研究的任务与评价重建（ALFWorld/TextWorld），以及动作接口、actor 与冷启动准备 |
| C0 / C1 | ALFWorld 公开教师冷启动后的 Qwen3-1.7B LoRA 合并模型 C0；一轮更新与旧技能回放后的 C1 |
| N1–N5 | 后续五个工作包：N1 参数更新与云端热回滚；N2 AWQ 质量—性能；N3 CUDA 算子实验；N4 分布式训练与恢复；N5 持续学习准备度 |
| P0–P2 | 早期离线控制面、真实执行与业务协议试点（已作为历史，仅保留回归用途） |

## 结果记录

| 记录 | 内容 |
|---|---|
| [网关热路径优化](results/2026-11-09-gateway-hotpath.md) | 逐请求阶段计时定位额外延迟，组提交与后台健康探测改造，同协议 A/B 对照 |
| [R1 单实例服务](results/2026-11-02-r1-serving.md) | 14 类真实 TCP 场景：过载、流式取消与 KV 释放、控制者崩溃、worker 替换 |
| [R2 发布验收](results/2026-11-03-r2-v2-serving.md) | 两个参数版本的 13 阶段发布与故障矩阵 |
| [R3 负载基线](results/2026-11-03-r3-baseline.md) | 1,152 个正式请求；同步持久化阻塞定位；cProfile 在 Python 3.12 下跨线程误归因的识别 |
| [R4 原生盘基线](results/2026-11-03-r4-native-baseline.md) | 账本迁到 ext4 后的基线与预先设定的判别条件 |
| [N2 量化质量门](results/2026-11-06-n2-quality-performance.md) | Qwen3-4B BF16 与 AWQ 的同条件质量—性能对照，AWQ 因质量下降被拒绝 |
| [N3 CUDA 实验](results/2026-11-06-n3-cuda-foundation.md) | 自写 residual RMSNorm 接入 vLLM 与 TensorRT 块部署；更慢且一致性不达标，未采用 |
| [N4 分布式训练](results/2026-11-06-n4-distributed-training.md) | 两机 DDP/FSDP2 数值一致、NCCL 测试、检查点故障恢复；普通网络下负扩展 |
| [N5 持续学习准备度](results/2026-11-06-n5-readiness.md) | 共享 4B 基础适配后基础任务 23/24，流程遵循仍未达门，研究主实验未启动 |
| [Q1 任务前检](results/2026-11-02-q1-task-preflight.md)、[Q2-P0](results/2026-11-03-q2-p0-readiness.md) | 研究任务环境与 actor 前置条件 |

操作手册：[R2](r2-operations.md)、[R3](r3-operations.md)、[R4](r4-operations.md)、[多副本](replica-operations.md)、[云端集群](cluster-operations.md)、[持续学习实验](q2-operations.md)。系统设计见 [design.md](design.md)。
