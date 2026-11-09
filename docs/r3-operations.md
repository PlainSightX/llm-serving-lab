# 推理测量与诊断

## N2：4B质量、固定长度负载与真实GPU时间线

当前原件根目录是 artifacts/n2-20261105-a；进度只归CURRENT及既有工作包。Qwen3-4B版本固定为 1cfa9a7208912126459214e8b04321603b3df60c。官方Qwen ModelScope分发的每个文件已与Hugging Face的LFS SHA256/Git blob身份核对。它与旧1.7B短命令基线分开，不将两者吞吐混为提升。

质量面板来自GSM8K固定提交 3101c7d5072418e28b9008a6636bde82a006892c。train前128条仅用于一次AWQ校准；test前256条使用同一非thinking提示、greedy/seed0、512输出预算及自然停止。错误与截断均计错。BF16最低70%，AWQ最多下降2个百分点。质量输入、原响应、参数与profile、实际原生释放均独立复核；4题train smoke不能代替正式质量证明。

量化生产环境见 environments/quantizer/，llmcompressor0.9.0与compressed-tensors0.13.0在独立前缀运行；服务仍使用原镜像中的vLLM0.11.2、Torch2.9.0+cu128、compressed-tensors0.12.2。n2_quantize.py只挂载校准记录，固定W4A16_ASYM/group128、忽略lm_head、duo_scaling=both、最多512校准token。生产文件保留；n2_quant_compat.py仅接受本次默认scale_dtype=None/zp_dtype=torch.int8，生成加载器兼容副本并逐文件确认权重不变，拒绝任意dtype降级。新版/旧版pack和unpack函数AST相同，vLLM直接读取打包权重及零点；旧库通用CPU解压不作为本包加载路径。

首次评价BF16为233/256，AWQ为223/256，下降3.90625个百分点。AWQ拒绝采用。为了完整呈现质量—性能取舍，允许在质量原件已完整审计后显式指定 --diagnostic-rejected；计划、汇总和审计均携带 diagnostic_rejected_candidate 与失败标记。默认入口继续拒绝失败候选，此诊断不授予应用资格、可用优化或S亮点，不更换评价题或重调训练配方。

性能卡固定256/128、3072/128、512/512输入/输出token与交替混合请求，关闭prefix cache，4GiB固定KV、4096上下文、max_num_seqs=8、正常CUDA graph。ignore_eos=true只用于固定工作量；自然输出的质量成绩单列。并发1/4/8各8请求的试点完成后，正式速率按BF16并发8实测速率的0.6倍冻结为short1.6176、prefill0.6990、decode0.4049、mixed0.3649 req/s。AWQ使用同一份速率文件，不按候选重新选负载。

每种精度36窗口×24请求：四种形状、native/本项目网关/固定production-stack0.1.13三入口、三次重复并轮换入口顺序。参考路由器固定单backend、round-robin、禁用语义缓存和故障重试；两参考入口复用同一worker原生引擎。网关处于明确的测量scope，传输资格与独立质量哨兵不冒充企业应用资格。负载机、控制机、GPU节点分离，正式窗口期间不运行profiler。

TTFT从预定到达算到首个非空内容；TPOT为每请求( E2E-TTFT )/( 输出token-1 )，不称为真实逐token ITL分布。short/decode TTFT≤1s、prefill≤3s，TPOT≤80ms，E2E≤3+.08×输出token；至少99%计划请求满足，迟发P95不得超过到达间隔的10%。所有计划请求留在分母，实际最后完成与计划窗口长度共同约束吞吐分母。短窗口P95只作描述，不作稳定生产尾延迟估计。

运行入口（在项目根目录，均使用新输出目录）：

~~~powershell
uv run --project environments/trainer --locked python scripts/n2_quality.py --stage run --data <frozen-data> --engine-config <engine.json> --tokenizer <tokenizer> --output <new-quality-run> --variant bf16
uv run --project environments/trainer --locked python scripts/n2_quality_audit.py --run <quality-run> --releases <native-release-events.jsonl> --output <new-audit.json>
uv run --project environments/trainer --locked python scripts/n2_benchmark.py --stage prepare --tokenizer <tokenizer> --engine-config <engine.json> --quality <full-audit.json> --output <new-plan>
~~~

AWQ自然停止评价另传 --variant awq --baseline <audited-bf16.json>，冻结并复核基线文件摘要。只有明确研究被拒绝候选时才在性能prepare附加 --diagnostic-rejected。Linux独立发生器通过 n2_benchmark.py --stage pilot/formal 消费plan、固定rates及mTLS客户端证书。容器使用证书所属的UID1000；显式覆盖原控制镜像entrypoint，源代码按运行版本只读挂载，不能改写正在运行的source目录。

性能重算入口为 scripts/n2_benchmark_audit.py，要求 --run/--plan/--rates/--releases/--gateway-db/--tokenizer/--quality-proof/--payloads/--source-manifest/--output。它核对每窗口逐索引的sample序列、同源分词结果、网关规范化body与实际wire、原始SSE身份与时钟、原生代际/释放、完整窗口分母，再重算摘要。SQLite通过backup导出，原始来源目录及源码包仍保留。

PyTorch Profiler在同一服务worker的独立20请求窗口运行（4预热、16剖析）。Nsight Systems2025.4.1在另一台同型号A10、同权重/引擎配置的离线20请求窗口采集cudaProfilerApi范围；该诊断不能替代在线吞吐。n2_profile.py保留实际输入输出与完成标记，n2_trace_summary.py读取两个Torch trace及Nsight SQLite，区分CPU嵌套算子独占时间、GPU内核累计时间与区间并集，不能把这些分母直接相加。

初次Nsight使用离线默认8192 batched-token预算，原件保留；显式4096的B已重新采集并匹配服务配置，正式负载不重跑。BF16的B诊断内核累计时间中，矩阵类86.995%、attention8.621%、已具名残差/RMS融合核1.351%，分类互斥。此证据支持优先分析矩阵/权重访存；单独改归一化不能据此承诺10%整体收益。所有trace、未插桩窗口、失败启动与加载器兼容记录独立保留。

## R3历史操作与证据边界

从独立Linux环境运行，使用现有模型和serving-venv。当前基线开启正常图执行、关闭prefix cache，CPU工具链见[说明](../environments/compiler-linux/README.md)。旧R2应用分数不因本轮性能指标而升级。

```bash
PY=$HOME/.local/share/llm-serving-lab/serving-venv/bin/python
$PY scripts/run_benchmark.py --output artifacts/my-r3-formal
$PY scripts/diagnose_benchmark.py --source artifacts/my-r3-formal --output artifacts/my-r3-diagnosis
```

正式矩阵1152个请求，预热、单版本预资格和启动探针单列。第一条命令完成并停止自有服务后，第二条才重新启动诊断环境，只执行所选32请求及必要资格/预热。诊断记录完整清理排空后才停止。SQLite事务退出达到追踪区间10%时，才补同初态、同WAL/FULL的各32条CPU存储序列；没有模型调用，不能当成端到端优化。

汇总可在项目核心环境执行：

```powershell
uv run --locked python scripts/report_benchmark.py artifacts/my-r3-formal --diagnosis artifacts/my-r3-diagnosis --output artifacts/my-r3-report
```

所有输出目录必须是新的。脚本管理本包19080/19082模型和19083/19084网关，只关闭自己创建的进程。发生中断时先读各worker binding、网关/遥测进程记录及cleanup.json，核对PID、boot_id和start_ticks，不能按名称或端口批量终止。

配置和计时含义在实施包。正式SLO从预定到达计时，发送后口径也保留。所有计划请求进入分母；漏发/发生器迟到标无效窗口，非200或释放未确认的直连窗口停止后续推进。SSE首个role帧不算首token，只有首个非空content才计TTFT。

发布对照有共同1秒无到达空档，只衡量已预热版本的受控暖发布；如果提交未赶上空档或逐请求版本构成不匹配，不用于因果比较。高负载差异包含准入/排队政策，不能全称纯网关开销。

原B轮cProfile归因已隔离，正式数据不改。当前报告支持读取旧B正式窗口及独立C诊断；新的常规测量不再开启cProfile。诊断方法跨度与内部SQL跨度嵌套，不能相加，不是CPU占比；先在内存记录，停止后写盘。

本地编译器、编译缓存和CPU存储原始诊断目录保留为项目自有依赖/证据，不清理其他项目环境。源代码提交不包含模型、全部原始请求或编译器二进制；原始工件路径及恢复锚点在CURRENT与工作包记录中。
