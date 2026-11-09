# Q1独立CPU文本任务环境

用于任务与评价前检，不提供模型推理或训练。当前锁定Python3.12、ALFWorld0.4.2、TextWorld1.7.0、NumPy1.26.4及PyYAML6.0.3。环境与R1的Linux推理环境独立；现有镜像保留供后续Q2复用。

## 已有环境的使用

从项目根目录执行。每次使用新输出目录，旧任务与运行不会覆盖：

```powershell
uv sync --locked
uv run --locked python scripts/q1_prepare.py --output artifacts/q1-my-data
uv run --locked python scripts/q1_run.py --data artifacts/q1-my-data --output artifacts/q1-my-run
uv run --locked python scripts/q1_contexts.py --episodes artifacts/q1-my-run --output artifacts/q1-my-contexts
```

prepare按配置下载约108.5MB官方文本资产，核验大小与SHA，只提取12个train关卡；6个valid_unseen关卡只保存身份与摘要。不要用alfworld-download替代，它会引入本包不需要的视觉资产。来源版本、摘要、种子、100步预算和镜像身份在[配置](../../configs/evaluation/q1-sources.json)。数据许可与上游使用条款保留，数据不随Git分发。

run先核验本地镜像与包版本，再执行42个CPU回合。参考策略访问隐藏状态，仅作可解性检查。context导出18份诊断输入，始终标记model_trial_ready=false；禁止作为正式同信息学习比较历史或训练数据。具体边界见[评价合同](../../docs/research/q1-task-protocol.md)。

## 重建环境

本机已存在llm-serving-lab-q1-tasks:alfworld-0.4.2，无需每次重建。新机器安装Docker后可构建：

```powershell
docker build -t llm-serving-lab-q1-tasks:alfworld-0.4.2 environments/tasks-linux
docker image inspect llm-serving-lab-q1-tasks:alfworld-0.4.2 --format '{{.Id}}'
```

基础镜像及依赖锁固定，但重新构建不保证镜像ID逐字相同。若身份不同，保存本次构建日志，复制配置为本次运行专用JSON并记录新的image_id，使用q1_run.py的--config参数；不能改写旧运行的身份或把新镜像冒称已验收环境。新环境仍需完整前检。

容器为非root、无网络、只读根、移除capabilities，限制2CPU/2GB/128进程。仅输出目录可写。/tmp为256MB的可执行tmpfs：Fast Downward需要映射临时动态库，noexec会加载失败。每回合一个子进程及临时目录，退出后回收，避免库副本累计填满tmpfs；这不允许写宿主系统。

## 中断和结果复核

正常结束时，启动器保存容器启动命令、退出码和日志，核验原始回合后删除自己创建的确切容器。写盘失败或超时也尝试清理。主机/应用退出可能使finally无法执行，不能保证无遗留。

恢复时先读运行目录的episodes-launch.json，以完整container_id、名称、owner/run_id标签和镜像ID共同核验归属，再检查状态。保存docker logs与inspect后，运行scripts/q1_contexts.py重新核验原始数据。只有退出码0、完整42份子回合回执和全部前检通过才可接受；宿主恢复另记回执，不能伪造原启动器正常返回。清理只针对确认的自有ID，不全局prune，不覆盖旧目录补跑。

本次F轮已按此恢复，原始文件未改。结果及失败轮次见[Q1结果](../../docs/results/2026-11-02-q1-task-preflight.md)。
