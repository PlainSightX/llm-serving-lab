# MemP机制与固定源码核查

2026-11-03核查；没有复现论文分数。原文v4为2026-04-15，固定仓库提交为3066a1b280a39c7433ae37d9a745861903a5d3c4（2026-01-30），两者分别解释，不能假定实现完全对应。本页维护MemP机制与代码差异；综合研究参考见[Q2文献依据](q2-p0-readiness.md)。Q2-P0实际成功历史为0条，不能将下表的预定流程读成已获得数据。

| 机制 | 文献责任 | 旧P0设计处置（非已获数据） |
|---|---|---|
| Build | 训练gold轨迹，逐步轨迹/抽象脚本/组合；源码direct一次LLM，round两次LLM | 先取得模型自采、公开信息成功轨迹；脚本抽象仍未做 |
| Retrieve | Query/关键词向量检索和随机对照；源码query/facts走FAISS，ave_fact显式cosine | 使用事前同任务族候选与固定另一来源作为可用性对照；不是向量检索复现 |
| Update | 普通新增、成功筛选、失败修订及丢弃；构建/反思额外计费 | 本包仅成功后收录并冻结共同历史；在线修订/遗忘未做 |

固定源码静态核查发现：入口所引用的ProcedureMem/config.yaml及Alfworld/base_config.yaml不在该树；run_memp_online.py路径拼接少斜杠，默认YOUR_MODEL_NAME且未传args.model；actor设10次retry，Builder另有重试。prompt_generator将部分动作词限定为TravelPlanner，不能原样作为ALFWorld适配。它需要显式本地适配与成本/信息边界检查。本包采用自己的受控运行器，不执行上游重试入口。

P0试点与论文使用gold轨迹、顺序测试更新的实验条件不同；保护身份不执行、不写回历史。失败或缺少成功历史时不得借助专家轨迹补齐。后续正式Q2仍需可运行的近邻机制复现、抽象脚本与组合对照及完整成本。

再次核原文§4.1–4.2：No Memory使用ReAct，backbone是GPT-4o、Claude-3.5-Sonnet、Qwen2.5-72B；Trajectory/Script/Proceduralization依赖训练gold轨迹。不能把其结果与当前1.7B无示例action-only的0/6直接比较。论文依据也不要求所有后训练都先有模型自身成功；新的公开教师来源须按独立合同说明，旧oracle仍不升级。

来源：[论文](https://arxiv.org/html/2508.06433v4)、[固定代码](https://github.com/zjunlp/MemP/tree/3066a1b280a39c7433ae37d9a745861903a5d3c4)、[memory.py](https://github.com/zjunlp/MemP/blob/3066a1b280a39c7433ae37d9a745861903a5d3c4/ProcedureMem/memory.py)、[线上入口](https://github.com/zjunlp/MemP/blob/3066a1b280a39c7433ae37d9a745861903a5d3c4/ProcedureMem/run_memp_online.py)。只核代码与元数据，未引入其训练轨迹。

Qwen固定模型说明建议非思考模式temperature=.7、top_p=.8、top_k=20、min_p=0。seed=20261103、64输出token、3000输入token和4096引擎上下文是本项目冻结设计，不是官方建议的预算。[固定revision模型说明](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)


2026-11-05接续范围。新流程包已取得72条合格公开程序教师训练轨迹，来源、阶段和时间截断可复核。当前实现是完整动作—feedback轨迹投影、任务级BM25、公开流程版本过滤和整条经验预算。旧候选命令集合未进入记忆。Trajectory开发试点的流程成功为0/18，提示澄清子集也未取得流程成功。Script、Proceduralization、论文embedding检索与完整动态Update继续按各自机制验证，当前来源和局部实现不升级为论文成绩复现。最新结果与接续合同归[Q2依据](q2-p0-readiness.md)。
