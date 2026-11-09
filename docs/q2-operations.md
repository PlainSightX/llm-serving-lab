# Q2开发诊断运行与复核

## N5：公告后查询的开发资格

工作根为 artifacts/n5-20261106-a，plan/anchor固定科学协议，actor-bindings/anchor固定允许的C0与原4B完整参数、分词器和有效profile。scripts/n5_prepare.py先冻结协议再从旧72条合格train轨迹准备M；scripts/n5_qualify.py在18个真实dev身份上分别运行完整规则控制和仅procedure/revision查询的M，共36回合。当前角色、运行目录和会话状态以CURRENT及既有工作包为准。

运行器先重建每条记忆的train来源和完整36项清单，核实际worker参数/profile，完成两次真实choice canary后才派发任务。每个生成都有事前预约、wire/token、原生释放；每个CPU环境有固定image、源文件及退出回执。独立scripts/n5_audit.py从原件重算输入、动作、环境、准确释放和完整资格摘要。共同4B升级必须消费相同协议下C0的完整失败及审计，不允许只提升某一研究臂。

强记忆资格与三阶段/三seed训练是两个阶段。N5-a的C0和原4B均未过；随后N5-b共享基础适配完成447步并通过独立adapter合并审计，基础23/24，但流程显式1/18、M2/18仍未过。两份协议及原结果保留，当前本轮已按停止条件收口；完整研究未完成。全部7337实际生成、132环境回合和本地模型归档的证据见[N5结果](results/2026-11-06-n5-readiness.md)及 artifacts/n5-20261106-b/summary.json。开发结果不回流训练，未来0回合；N1和旧受保护实例保持原身份。

独立重算原件时，n5_audit.py默认不覆盖已有audit.json，可用 `--output` 指向新的派生文件。基础24项需同时给 `--basic --training-evidence artifacts/n5-20261106-b/training-audit.json`；流程36项不能混用基础模式。两种模式都必须提供该次worker完整native release日志及bootstrap目录；实际入口不把摘要passed当成充分证明。

## N1：已验证的参数更新与云端热回滚

原件总入口为`artifacts/n1-20261105-a/summary.json`。云端完成的C1在`cloud-training/training-b`，固定训练数据在`training-data`。本地21步中断保留在`training`，不得当成完整候选。完整训练、原应用、阶段一与发布分开核验；新的输出路径必须不存在。

```powershell
uv run --project environments/trainer --locked python scripts/n1_audit.py --run artifacts/n1-20261105-a/c1-stage1 --releases artifacts/n1-20261105-a/c1-stage1/native-releases.jsonl --output artifacts/n1-stage1-recheck.json
uv run --project environments/trainer --locked python scripts/audit_parameter_publication.py --root artifacts/n1-20261105-a/publication-export --output artifacts/n1-publication-recheck.json
```

模型运行使用`scripts/n1_study.py`的baseline、stage1、application或限定C0的application_control模式；C1调用必须附`--training`并通过675步最终工件/父模型/数据/编码核验。原已完成矩阵不因展示再次生成；需要新实验先冻结相应工作记录。当前本机转发已关闭，旧engine-config中的endpoint不是仍在线的本机入口。

发布资料为`release-evidence`与`publication-export`。C0/C1当前云端同24题均23/24，成功集合一致；第一阶段F/M/P/PM为2/6、4/6、6/6、5/6。Script是确定性公开规则投影，Combined加入训练轨迹，不等于完成MemP原实现复现。全部4301次真实生成（含8bootstrap）及4302次预约与失败分别保留。15个真实C0绑定满足控制者事件顺序`binding < cutover < dispatched < terminal`，热回滚后active=C0、generation=5。worker仍驻留，不声称冷重载。

部署中须保留`serving-lab/component`标签；NetworkPolicy依赖controller/worker角色。registry版本ID只用字母、数字、下划线与短横线，当前发布C1为`qwen3-1p7b-workflow-c1`；评价时的旧带点alias保留在原件。别名调整未改变参数/profile。网关和worker必须按实际当前manifest重新绑定，不能拿旧Pod地址或旧转发当当前事实。

## 流程任务与轨迹记忆准备

当前准备包为`artifacts/q2-workflow-readiness-20261105-a`。数据清单在`q2-workflow-data-20261105-a`，120个CPU回合在`q2-workflow-collection-20261105-a`。模型A/B/C3是完整开发运行，C2是含一次未派发失败的中断运行，C只完成了失败的只读前检。当前事实与方法边界由CURRENT和研究依据页维护。

新来源准备和CPU执行从项目根运行，使用新的输出路径。Windows原生与已有WSL环境均可使用该容器桥，当前Docker须可用。

```bash
uv run --locked python scripts/q2_workflow.py --stage prepare --output artifacts/workflow-data-new
uv run --locked python scripts/q2_workflow.py --stage collect --data artifacts/workflow-data-new --output artifacts/workflow-cpu-new --budget-ledger artifacts/workflow-cpu-budget-new.jsonl
uv run --locked python scripts/q2_workflow.py --stage audit --data artifacts/q2-workflow-data-20261105-a --collection artifacts/q2-workflow-collection-20261105-a --output artifacts/workflow-cpu-recheck.json
```

记忆与模型试点使用锁定trainer环境的tokenizer。准备入口独立审计教师来源，完整计算基座工件与张量身份，并冻结历史、检索、全部开发案例和执行源码。运行入口在建立模型客户端之前核对远端参数身份。每个普通完整试点最多42个开发回合、4200次预约，基础任务门通过后才进行流程比较。

```bash
uv sync --project environments/trainer --locked
uv run --project environments/trainer --locked python scripts/q2_workflow_probe.py --stage prepare --data artifacts/q2-workflow-data-20261105-a --collection artifacts/q2-workflow-collection-20261105-a --model artifacts/q2-coldstart-training-20261103-b/merged --output artifacts/workflow-probe-plan-new
uv run --project environments/trainer --locked python scripts/q2_workflow_probe.py --stage run --preparation artifacts/workflow-probe-plan-new --engine-config artifacts/q2-workflow-readiness-20261105-a/engine-config-ipv4.json --output artifacts/workflow-model-new
```

上述运行命令需先按当前连接记录建立本机19382到目标worker Pod 8443的SSH转发，确认worker清单与engine-config一致。专用转发的状态以本轮资源交接回执为准。此入口以独立研究账本调用已认证worker，记录独立开发用途和每次派发/释放。研究actor的应用质量由对应试验评价。

无需模型调用即可复核已保存原件。审计器加载每轮准备目录中的冻结源码，重建实际消息、记忆与token，并核对本地C0权重、云端清单、环境记录、账本和下载的原生释放事件。输出文件须不存在。

```bash
uv run --project environments/trainer --locked python scripts/q2_workflow_audit.py --run artifacts/q2-workflow-model-20261105-a --releases artifacts/q2-workflow-model-20261105-a/native-release-events.jsonl --output artifacts/workflow-model-a-recheck.json
uv run --project environments/trainer --locked python scripts/q2_workflow_audit.py --run artifacts/q2-workflow-model-20261105-c3 --releases artifacts/q2-workflow-model-20261105-c3/native-release-events.jsonl --output artifacts/workflow-model-c3-recheck.json
uv run --project environments/trainer --locked python scripts/q2_workflow_audit.py --run artifacts/q2-workflow-model-20261105-c2 --releases artifacts/q2-workflow-model-20261105-c2/native-release-events.jsonl --nondispatched-interruption --output artifacts/workflow-model-c2-partial-recheck.json
```

`--nondispatched-interruption`仅接纳已核定的完整前缀及紧随其后的零动作、未派发就绪失败，并显式列出未执行案例。它不把中断升级为完整结果。当前代码的公开流程actor已采用GOAL合并版本，原A/B提示版本在各自源码快照中保留。未来评价实例尚未提取，准备入口会拒绝future与旧protected角色。


## 公开教师冷启动

当前激活的新来源、固定训练配方与24题三条件资格门见[Q2合同](research/q2-p0-readiness.md)。旧诊断与oracle不进入此入口。正常复现使用全部新目录，保留原数据/参数；现有A/B失败和方法修复是历史，不能重复执行来挑更高分。

本次已完成的工件是`q2-teacher-data-20261103-b`、`q2-coldstart-dataset-20261103-a`、`q2-coldstart-controls-20261103-b`及`q2-coldstart-training-20261103-b`。C0合并产物位于训练B的`merged/`，原开发成绩23/24；后续[本地受控发布](replica-operations.md)另包完成，所有自有服务已停止，网关集成回合不加入训练或主历史。日常核查先用已有原件，以下两条仅离线审计，不调用模型（输出文件必须不存在）：

```bash
python scripts/q2_coldstart_audit.py --run artifacts/q2-coldstart-baselines-20261103-a --data artifacts/q2-teacher-data-20261103-b --output artifacts/coldstart-base-recheck.json
python scripts/q2_coldstart_audit.py --run artifacts/q2-coldstart-trained-20261103-a --data artifacts/q2-teacher-data-20261103-b --output artifacts/coldstart-trained-recheck.json
```

整包核验见`artifacts/q2-coldstart-trained-20261103-a/package-verification.json`，训练张量与重载核验见训练B的`training-verification.json`。原训练A保留为16步中断尝试，旧对照配置A没有执行模型；后续恢复不能把它们当作待续训/待推理任务。

新任务选择/CPU教师采集使用既有WSL serving-venv：

```bash
python scripts/q2_teacher.py --stage prepare --output artifacts/teacher-data-new
python scripts/q2_teacher.py --stage collect --data artifacts/teacher-data-new --output artifacts/teacher-cpu-new
```

教师72题完整原件通过数据门后，在Windows项目根目录导出并用锁定trainer运行：

```powershell
uv run --project environments/trainer --locked python scripts/q2_teacher.py --stage export --data artifacts/teacher-data-new --collection artifacts/teacher-cpu-new --model artifacts/models/qwen3-1.7b --output artifacts/coldstart-data-new
uv run --project environments/trainer --locked python scripts/q2_coldstart_eval.py --stage controls --dataset artifacts/coldstart-data-new --output artifacts/coldstart-controls-new
uv run --project environments/trainer --locked python -m serving_lab.coldstart_train --dataset artifacts/coldstart-data-new --model artifacts/models/qwen3-1.7b --output artifacts/coldstart-train-new
```

同一GPU上的训练与推理顺序运行。训练保存/重载通过后，在WSL执行完整对照（第一条包含base与固定示范两条件，第二条是合并后的真实参数候选）：

```bash
python scripts/q2_coldstart_eval.py --stage run --data artifacts/teacher-data-new --controls artifacts/coldstart-controls-new/controls.json --output artifacts/coldstart-base-new
python scripts/q2_coldstart_eval.py --stage run --data artifacts/teacher-data-new --controls artifacts/coldstart-controls-new/controls.json --training artifacts/coldstart-train-new --output artifacts/coldstart-trained-new
python scripts/q2_coldstart_audit.py --run artifacts/coldstart-base-new --data artifacts/teacher-data-new --output artifacts/coldstart-base-new/audit.json
python scripts/q2_coldstart_audit.py --run artifacts/coldstart-trained-new --data artifacts/teacher-data-new --output artifacts/coldstart-trained-new/audit.json
```

导出附`base-binding.json`，训练/对照均核完整固定1.7B身份；模型路径别名与相同tokenizer不能绕过。固定示范在运行和离线审计时均从真实train轨迹重新构造。训练只计算动作/EOS损失，保留完整上下文；最终任务评价使用合并产物，不能用loss或adapter字节变化代替成功率。所有dev记录不得回流训练，候选不自动发布。首次云购买不在本包授权内。

## 前序标准actor入口

`q2_standard.py`固定一个公开命令选择actor和seed20261103，使用既有1.7B。真实输出受当前完整命令集合约束；工作状态只来自公开观察。每个support一次、最多100步，共六类，另2次bootstrap、4次transport、4次约束canary，总上限610。业务失败仍完成其他任务；协议或释放异常停止，保留原始记录且不自动重试。

```bash
python scripts/q2_standard.py --cpu-run artifacts/q2-new-cpu-qualified --output artifacts/q2-standard-new
python scripts/q2_standard.py --audit artifacts/q2-standard-20261103-a --output artifacts/q2-standard-a-new-audit.json
```

以上命令使用既有WSL serving-venv，普通复核使用第二条，**不重新运行第一条以替换失败成绩**。新输出目录必须不存在。当前程序策略已扩展，旧CPU A不能满足新源码的精确资格绑定；第一条中的CPU目录须先用当前源码完成新的CPU资格，历史复核则使用原记录/源码。离线复核消费原始响应、实际候选集、任务/镜像身份、冻结采样配置、公开上下文、容器退出、账本及准确引擎释放；SQLite只在临时副本读取。此入口不创建训练数据或经验历史。结果和新教师来源合同仍归[同一Q2页面](research/q2-p0-readiness.md)。

## 上一轮救援入口

上一轮救援源码入口为q2_rescue.py；设计/条件门/实际失败结果维护在[既有Q2依据](research/q2-p0-readiness.md)，原P0入口在下方保留回溯。已有A/C/D不应自动重复执行；默认先离线核原件。

```bash
python scripts/q2_audit.py --rescue --run artifacts/q2-rescue-model-20261103-a --output artifacts/q2-a-new-audit.json
python scripts/q2_audit.py --rescue --run artifacts/q2-rescue-model-20261103-c --output artifacts/q2-c-new-audit.json
python scripts/q2_audit.py --rescue --run artifacts/q2-rescue-model-20261103-d --output artifacts/q2-d-new-audit.json
```

这些python命令使用已有WSL serving-venv。原件核验只在临时SQLite副本读取；会核source、真实token、准确arrival/attempt/释放、预定标签和A/B共同前缀。D单列离线解析的C旧输出，不能当作新增36次模型调用。容器退出码非零、原件缺失或worker释放不明均不能通过。

若明确开展新身份/新预算下的复现，可分别使用`q2_rescue.py --stage cpu --output <new>`、`--stage model --cpu-run <verified CPU run> --output <new>`；C另加`--repair-from <A>`，D另加`--adapter-from <C>`。它们保留失败、不自动重发，累计原件预约抵扣修复额度；当前包实际190生成、未用满246上限。它们用于重现开发诊断，不是已合格生产actor或训练入口。所有记录标actor_diagnostic且不具训练/主历史资格。

JSON envelope只接收单字符串或唯一action/command字符串，允许额外feedback字符串但丢弃它；重复键、双动作、列表、额外字段、内部多动作或截断JSON均拒绝。后续公开观察始终来自环境，不消费模型声称的执行结果。

## 原Q2-P0入口（历史）

以下旧入口若使用当前源码，会包含后续策略修复；不能自动称为复现旧失败。严格复核应消费各旧运行自己的source快照和原件，不能把新源码结果写回旧身份。

CPU任务镜像复用[Q1环境](../environments/tasks-linux/README.md)，模型使用独立WSL serving-venv。代码在本仓库，WSL通过docker.exe调用现有Windows Docker客户端，不需要启用WSL全局集成。不要把Windows .venv交给Linux。

本包CPU1200动作、support600模型动作预算已消耗；以下是明确新包/新身份时的复现入口，不是自动补跑授权。禁止覆盖旧输出目录。

```bash
python scripts/q2_run.py --mode cpu --cpu-budget 1200 --output artifacts/q2-new-cpu
python scripts/q2_run.py --mode model --output artifacts/q2-new-model
```

模型模式先启动既有1.7B，直连bootstrap两次，新profile预资格四次；随后六个support每题最多100动作，成功历史不足时不进入目标比较。条件满足才三条件完整执行一个目标组；总模型动作硬上限2400。每次调用之前写reservation和数据库预约，不自动重发或恢复首尝。中断时原目录保留，未确认调用不得补写成功；单独复核不产生模型调用。

```bash
python scripts/q2_audit.py --run artifacts/q2-public-model-20261103-a --output artifacts/q2-a-new-audit.json
```

审计必须使用可访问原模型/tokenizer和引擎events路径的Linux环境；它校验600动作身份、raw、wire、token、公开上下文、环境记录及账本。原输出路径不变时重算应与原audit逐字段相同，除输出文件名不计入内容。

正常finally按完整container ID、名称、owner/run_id清理CPU容器，按binding进程代际停止worker。宿主直接崩溃可能无法运行finally；恢复时从本次launch.json和worker/binding.json逐项核验，保存原日志/inspect，仅清理确切自有对象，不全局prune。现有失败原件不能覆盖或标签升级。
