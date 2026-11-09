# R2 v2工程验收操作

单版本预资格只允许本地工程验收流量；完整发布/故障矩阵通过才算R2工程验收。旧v1两个3/8仍失败，不构成业务可用、生产容量或集群结论。实际状态见CURRENT。

## 从独立Linux环境运行

使用既有WSL serving-venv，依赖锁不变。每次使用新输出目录，从项目根按顺序执行：

```bash
PY=$HOME/.local/share/llm-serving-lab/serving-venv/bin/python
$PY scripts/start_engine.py --config configs/serving/r2-small.json --run-dir artifacts/my-v2-small
$PY scripts/probe_serving.py --binding artifacts/my-v2-small/binding.json --output artifacts/my-v2-small-direct
$PY scripts/start_engine.py --config configs/serving/r2-large.json --run-dir artifacts/my-v2-large
$PY scripts/probe_serving.py --binding artifacts/my-v2-large/binding.json --output artifacts/my-v2-large-direct
$PY scripts/probe_releases.py --small artifacts/my-v2-small/binding.json --large artifacts/my-v2-large/binding.json --output artifacts/my-v2-release
$PY scripts/audit_releases.py artifacts/my-v2-release
```

该验收会终止所给binding的worker、装载替代实例，并强制终止自有网关及活动worker。只能用于本次明确自有实例。引擎端口19080/19082，网关19083；端口占用时先核对所有者，不能批量结束进程。失败目录保留，不覆盖补跑或拼接通过项。

完整参数摘要在启动/注册边界读取。Windows挂载目录的大文件核验和多次网关重启会耗时，不在每请求热路径重复计算权重摘要。运行目录保存进程身份、配置和日志，等待有上限。

## 两级门与固定配置

预资格fixture是fixtures/serving/r2-transport-v2.json，4项全过：普通非流式、重复直连、完整流式、1 token预算截断。核对完整输入token ID、usage、实际模型、finish原因与引擎释放；不评分回答是否等于OK。整体矩阵另有13阶段，包含两个版本入口一致性、在途切流、灰度、取消/过载/期限、退休重载回滚、控制者崩溃及活动/候选worker退出。

profile固定模板/模式、引擎、dtype/量化、EOS与停止参数、采样/seed、上下文/输出上限及实现摘要。请求只提供公开messages、模型选择、预算和stream，不能覆盖采样。实际wire payload、profile摘要及输入token IDs保存在runtime/wire-requests。资格、路由与请求都绑定profile，配置改变不能沿用旧资格。

本版明确保留greedy/non-thinking工程回归行为，不宣称是模型能力评测的推荐配置。旧应用结果通过receipt摘要绑定，明确原v1条件下failed、3/8；不推导旧分数失效或新的应用分数。原qualify_release_model.py默认8题入口保留为历史v1诊断，不作为v2工程入口。

## 管理与恢复

完整验收脚本自动生成本地网关配置和随机密钥。维护者也可运行scripts/serve_releases.py --config，配置包含bindings、v2 panel、runtime、port、admin_token_file、limits和两版本原application_receipts。密钥保存在本地受控文件，不能提交Git。只监听127.0.0.1，管理请求需Authorization: Bearer。

GET /admin/releases返回资格用途、版本身份、原应用结果和未释放租约。POST /admin/releases/{version}/qualify执行预资格。POST /admin/route接受operation（canary/promote/rollback）、target、expected_generation；canary另需percentage。坏候选、过期代际或配置不符不能改变active。

响应头含X-Model-Version、X-Route-Generation、X-Worker-Instance、X-Serving-Profile及X-Qualification-Scope。drain和retire受真实租约约束；HTTP终态不等于执行资源释放。退休后才按binding停止实际worker。实例换代须显式allow_rebind、核验旧调用释放、重新资格，再提交新路由；不会自动继承旧active流量。

旧v1 runtime不能交给v2入口原地升级，旧R1入口也拒绝v2 runtime；检查发生在账本recover之前。改profile需要新runtime，不以allow_rebind绕过；同profile控制者重启则复用原runtime保留去重和unknown。

中断后先读binding的boot_id/start_ticks及网关进程记录，核对同一代际，再收集日志与清理。audit_releases.py检查明确的SIGKILL记录、原始回合、wire/响应转发、13项覆盖和cleanup；失败终态本身不能证明实际故障注入。
