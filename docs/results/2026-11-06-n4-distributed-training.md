# N4：真实两节点训练、通信与恢复

单GPU与两机各一卡的本轮责任已实测：Qwen3-0.6B完整参数DDP/FSDP2、官方NCCL测试、独立剖析、rank故障、完整检查点恢复及恢复产物的服务工程资格。** 训练有效不表示分布式更快，也不产生应用质量或持续学习结论。

## 冻结条件与数值

模型实际596,049,920个独立参数。GSM8K固定train来源第128—255条；所有128条输入在运行前冻结，test与N5未来数据未进入训练。合同为 artifacts/n4-20261106-a/data-b/plan.json，SHA bbfbf48bba9bb358225762aed88eba8d50ec703164848c5a057ee0954a479c91。

两台独立A10、各16vCPU，NCCL通过内网Socket；未发现RDMA设备。重启后的硬件身份重新核对，并重测各节点单卡基线。固定PyTorch2.9.0+cu128、NCCL2.27.5、Transformers4.57.1及镜像sha256:24c5476d51446805dfd8614cfb41280b0b29e4743890d24b82c6b5940e4cdb45。

FP32/SGD、8步、全局batch4、microbatch1、序列128：单卡累积4，双rank各累积2，按全局有效预测token加权。DDP和FSDP2相对单卡的最终参数relative L2均为3.938e-9、maxabs均为2.384e-7；逐步loss最大相对差4.124e-7，均通过事前门。末步梯度relative L2为2.423e-5，作为诊断保留，不声称逐bit梯度相同。

单卡正式运行使用source-c；两机使用source-d。差异是跨rank合同/源码/恢复身份握手、DCP完整分片检查及FSDP根模块恢复PyTorch默认reshard策略，单卡数学路径未变。全部实际执行源码及各rank记录已保存，不能以当前工作树替换历史执行身份。

## 性能与代价

FP32主参数和AdamW状态、BF16 autocast；FSDP没有单独改为BF16参数。无梯度检查点。强扩展保持global batch4，预热4步/测量16步；弱扩展保持每GPU batch4，预热4步/测量12步。

| 模式 | 强扩展padded token/s | 弱扩展padded token/s | 正式峰值allocated/GPU |
|---|---:|---:|---:|
| 单GPU A | 2596.73 | 2619.97 | 12.25 GiB |
| 单GPU B | 2787.40 | 2763.87 | 12.25 GiB |
| 两机DDP | 918.87 | 1768.66 | 14.50 GiB |
| 两机FSDP2 | 527.09 | 601.57 | 9.52 GiB |

时间是每步各rank同步forward/backward/optimizer耗时的最大值；不包括步前barrier、汇报loss、检查点和作业启动。padded token包含padding，不能当有效文本量；原始审计同时报告有效预测token/s（强扩展A/B/DDP/FSDP2分别950.00/1019.76/336.16/192.83）。这是一次冻结短窗口比较，没有多轮重复或生产容量外推。

两个分布式模式均负扩展，FSDP2节省参数/优化器显存，但当前小模型、小microbatch及Socket条件下通信代价更高。独立两步trace中，DDP的NCCL kernel时间并集覆盖GPU活跃并集约85.3%—86.5%，FSDP2约98.0%—98.3%；这些kernel可能含等待，不能解释成链路利用率，重叠CPU/GPU时间不能相加。剖析没有混入正式吞吐窗口。

官方nccl-tests v2.17.6固定提交da0b547b1b9c6e3b1d4c15578087874522ae3761，OpenMPI4.1.8。all-reduce/all-gather/reduce-scatter各5种消息大小、5次预热/20次测量，15组全部数据检查零错误。128MiB的out-of-place bus bandwidth分别1.46/1.45/1.21GB/s。日志绑定实际操作、2个rank、Socket、消息元素数与按耗时重算的带宽；这不是NVLink/IB测试。

## 故障与可消费产物

FSDP2/AdamW连续12步作为参考。故障运行保存第4步完整分布式检查点，在第6步后rank1以73退出；rank0的NCCL watchdog报告120027ms超时，其Pod约181秒后以139终结。检测到超时不等于已完成进程退出，这条故障路径没有实现优雅退出。第5、6步未落盘更新明确丢弃。

两个节点的DCP分片通过已有私网Kubernetes连接汇集，实际文件集合、全部SHA和metadata引用均核对后才提交files.json，两节点清单SHA同为ceb3a68bd6c0d059fa4cd05447b8ba1e6e5cd3f922e3ea57b333428f62543cc0。新Pod从第4步恢复模型、AdamW、RNG和步号，重放第5—12步。loss和最终参数与不中断参考完全相同，最终state文件SHA也相同。dropout为0，该实验没有单独验证有随机dropout时的恢复等价；不是弹性变更world-size或无人值守高可用。

导出前数值比较绑定实际最终state、输入、日志和摘要SHA。恢复模型相对基座311个state条目有数值更新，relative L2约9.178e-4；FP32工件保存重载参数和探针logits差异均0。真实worker以BF16执行，完成原4项transport门，2次bootstrap＋4次资格生成均找到准确原生KV释放，4个账本租约全部解除。服务已停止，原监督确认5个自有进程均退出。未赋予C0应用证书，未发布为业务活动模型。

工件artifact digest为502b64b61ca73219435d23262699b80e4aa8ab8ccb2db83f5d4e3af7f7d8415a；parameter digest为1a058207aea9efd05ece8c8148692d414ec6e3c4e5044b19214136322aeabf0d。

## 原件与复核入口

根目录 artifacts/n4-20261106-a：data-b为计划/样本；evidence-a/audit.json为14组训练与trace核验；evidence-b/recovery-audit.json为连续/恢复步序核验；evidence-b中两个数值compare-single-b及compare-reference保留实际state身份；communication-evidence/audit-b.json为15组官方通信重算；serving-audit.json为6次真实生成、释放和清理核验。training-kubernetes-evidence保存31个训练Pod原始状态与带时间日志，包括前检和预期故障。

scripts/n4_prepare.py、n4_train.py、n4_compare.py、n4_export.py、n4_audit.py、n4_qualify.py和n4_serving_audit.py给出实际消费入口；deployment/n4.py生成两机Job和私网rendezvous。新运行必须新名字、固定source快照和独立输出目录。复核原件不需要启动GPU；重新训练/部署须沿CURRENT的当前资源安排。

首个主机编译OpenMPI与worker glibc不兼容，以及MPI启动器shell片段引用错误都保留；后来在固定worker内编译并修复启动器。这些失败发生在正式collective之前，未拼接到B矩阵。独立审查发现并关闭跨rank签名、DCP完整覆盖、缺步/空state误过、collective身份、最终参数绑定及冻结合同重建等缺口。

最终归档tensor-evidence.tar.gz为21,223,619,004字节，22个独立blob对应29个逻辑文件（含各次完整state/梯度、完整恢复checkpoint和导出模型），全部已流式核SHA。restored-export-a/export-a/model为本地可消费模型，独立重算artifact/parameter身份与云端一致。恢复其他原件可运行 `uv run --locked python scripts/n4_restore_archive.py --archive artifacts/n4-20261106-a/tensor-evidence.tar.gz --output artifacts/n4-my-restore --prefix results/committed-checkpoint-a/`，输出目录必须新建。N4无活跃Pod，34个终态Pod历史保留，主机未释放且N5单独登记。最终软件检查与交付Git锚点见CURRENT。
