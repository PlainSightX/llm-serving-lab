# R4原生持久化基线

先读CURRENT和冻结计划。本页是复现工具说明，不表示实验/源码优化已经完成。使用独立WSL serving-venv和既有模型/编译依赖，不改Windows .venv、系统Python或其他容器。

```bash
python scripts/run_benchmark.py --output artifacts/r4-new-baseline --runtime-base $HOME/.local/share/llm-serving-lab/r4-runtimes
python scripts/diagnose_benchmark.py --source artifacts/r4-new-baseline --output artifacts/r4-new-timing --runtime-base $HOME/.local/share/llm-serving-lab/r4-runtimes
python scripts/report_benchmark.py artifacts/r4-new-baseline --output artifacts/r4-new-report --diagnosis artifacts/r4-new-timing
python scripts/compare_runtime_baseline.py --current artifacts/r4-new-baseline --report artifacts/r4-new-report --diagnosis artifacts/r4-new-timing --historical artifacts/r3-measurement-20261103-b --output artifacts/r4-new-decision
```

全部路径必须是新身份；工具拒绝覆盖原生runtime。整个RuntimeStore迁至实际ext4文件系统，包括数据库与请求文件；运行输出/模型路径位置不变。配置、原生文件系统和owner回执、真实gateway连接PRAGMA读回、源文件快照均保留。默认为原R3布局，省略runtime-base不会暗中迁移。

正式矩阵24个稳态+6个暖发布窗口，共1152请求；准备24预热+16资格+4bootstrap另计。独立计时诊断另32测量+4预热+8资格+4bootstrap=48请求。完整两包共1244次生成，不包含管理、metrics、编译和可能触发的CPU存储序列。

网关停止后才复制原生runtime到证据目录供报告器消费，逐文件核摘要。交付工具在导出后显式close SQLite；已完成A轮的旧源码导出连接在runner退出时checkpoint，原快照保留DB/WAL/SHM完整包。审计只在临时副本打开SQLite，核原快照字节及与native全部逻辑表行的一致性，不强行要求checkpoint前后DB物理摘要相同。原生原件保留为可追溯运行数据，后续证据归档时再审查退休；复制目录不是运行时第二写源。所有gateway/worker/遥测PID以各自代际和启动清单核实，自有进程退出后检查端口；不可用全局kill/prune。

判别脚本核对原生位置/WAL-FULL、800条账本、768正式gateway wire、384直连配对、192发布配对和540原生指标差分。历史对照仅核有效配置/权重/负载一致并供描述；本轮与R3不是同时、随机化的存储迁移实验，不能把数字差异写成因果速度提升或源码贡献。独立计时用于判断当前残余阻塞，不与正式窗口混算。

故障语义仍是进程故障合同。payload/wire文件没有新增fsync保证，不把WAL/FULL外推为整个目录的断电持久性。若关闭当前writer候选，S维度源码深度仍待后续实测问题完成，不修改原验收范围。
