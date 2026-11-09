# 多副本接口的运行入口

当前是静态副本配置、单控制者。原`replica_pool_v1`已用同一0.6B参数的两个真实vLLM实例完成本机接入；C0应用发布scope也已完成本地实际验证，所有自有演练服务已停止，当前状态以CURRENT为准。没有自动发现/扩缩容/重试，也没有宣称真实双节点部署。

## C0应用发布入口

`application_replica_pool_v1`使用新runtime，绑定离线导入的应用证据。C0原开发成绩23/24，旧基座0/24，后者即使传输和命令接口探针通过，也不能进入该应用路由。已有传输scope保留原含义，不原地升级。

先从已保存的原始评价导入证据（只读模型/评价，不生成新模型输出；新目录不能存在）：

```bash
python scripts/import_coldstart_release.py --baseline artifacts/q2-coldstart-baselines-20261103-a --trained artifacts/q2-coldstart-trained-20261103-a --data artifacts/q2-teacher-data-20261103-b --output artifacts/my-publication-evidence
python scripts/probe_application_release.py --evidence artifacts/my-publication-evidence/application-evidence.json --output artifacts/my-c0-publication
```

以上使用既有WSL serving-venv。演练从导入回执取得预先固定的文件摘要，不能用证据文件刚算出的摘要自证来源。它启动两个不同参数的本机worker、认证桥和应用网关，执行质量拒绝、C0首次采用、3组直连/入口等输入比对、6个旧公开support闭环、在途撤回、控制者重启及同参数新代际资格恢复，随后停止全部自有资源。680是包含拒绝探测在内的保守生成预约上限；原开发面板不重跑。旧support清单/任务字节及镜像先冻结，传输面板在发起调用前检查恰为4项并使用本次副本。

本包两个1.7B实例各固定768MiB KV缓存，保留BF16、4096上下文和图执行。原A轮第二实例自动缓存预算为负，保留其失败与2次调用；修复重跑通过`--prior-run <失败轮目录>`累计原预约，不能重置680上限。多次失败需逐项传入所有前轮。`kv_cache_memory_bytes`按[vLLM 0.11.2缓存配置](https://docs.vllm.ai/en/v0.11.2/api/vllm/config/cache/)覆盖自动KV分配，不能把0.40 utilization解读成整个进程显存硬上限。启动期间记录子进程出生身份，父进程提前退出后仍清理已观察的自有子进程；监督环境不符时拒绝停止或死亡证明。

业务调用方复用StandardActor构造两消息公开状态与完整候选列表，提交`action_choices`及固定64输出预算。网关检查消息结构、目标和列表一致性，实际约束传至worker；公开状态的真实性与候选完整性由可信调用方负责。任意聊天输入和解码覆盖不在这个已评估的应用入口范围内。每个worker代际需4项传输和4项命令约束接口资格，源模型质量证据与实例就绪分别检查。

`/admin/route`新增`withdraw`，携带当前活动版本及expected_generation。它移除活动/灰度路由并记录当时已绑定的arrival ID，不依赖故障worker响应，也不解除未清lease。已有绑定继续原版本，尚未绑定的请求可被拒绝；恢复需显式promote。当前只有一个同应用合格参数身份，恢复C0或更换其副本不能称为跨参数应用回滚。旧基座不是该应用的合格回退版本。

新配置在原gateway结构上增加`application_release: {bundle, sha256}`，由导入回执提供值；worker增加`--profile-policy q2_public_choice_v1`，默认仍是旧greedy profile。该应用limits固定max_input_tokens=3500、max_tokens=64、context_tokens=4096、max_request_bytes=65536。质量包、参数、profile或actor不符时，控制者在打开/恢复runtime前拒绝。

离线审计不启动模型：

```bash
python scripts/audit_application_release.py --run artifacts/my-c0-publication --output artifacts/my-c0-publication-audit.json
```

审计绑定原始ledger、wire、准确释放、三组独立请求及完整输入、旧support实际文件与镜像；软件替身不计入真实模型证据。本机同GPU结果不代替正式双节点、TLS、节点丢失或目标负载S验收。

### 2026-11-04实际结果

完整C轮位于`artifacts/c0-publication-real-20261104-c`。三代worker各4项传输+4项命令接口资格通过；初次C0发布、三组真实直连/入口配对（含流式）、六个旧公开support、在途撤回、控制者重启、worker故障和同参数替代代际重新资格全部完成。旧基座在promote/canary/rollback及明确版本调用上因应用质量被拒绝；C0停机时基座仍ready，也未被允许回退。

六个support为6/6、121动作，逐题14/4/26/46/18/13步。撤回时3个已绑定arrival均按原路由完成，恢复后旧去重键仍被拒绝。C轮163次真实生成由6次bootstrap、24次worker资格、3次直连和130次应用网关生成组成；173755输入token、1099输出token。网关另按到达计163条：154完成、9拒绝，所有lease已释放。两个163来自不同分母，不能互换。

| 运行 | 真实生成 | 保守预约 | 结果 |
|---|---:|---:|---|
| A | 2 | 2 | 第二worker自动KV预算为负，未进入网关；失败保留 |
| B | 26 | 30 | 双实例/质量门/发布通过，流式可选空usage字段被误判；未进入support，失败保留 |
| C | 163 | 172 | 完整独立演练及原件审计通过 |
| 合计 | 191 | 204 | 同一680预约上限，包含被拒绝调用的保守预约 |

两项运行修复分别为显式768MiB KV，以及只将`prompt_tokens_details: null`与该可选字段缺省视为同义；非空明细、输出、finish及全部token计数仍须相同。A/B原回执未改。C的首次离线审计因流式事件变量覆盖任务数据路径而报错；修复路径变量后重读同一C原件通过，新增模型调用0。

执行时冻结44份源码/配置。完整回归在缓存与清理修复后为Windows207通过/3项Linux专用跳过、Linux210通过；流式比对修复后相关27项在双环境通过，审计修复后4项及实际C离线审计通过。实际清理另在原监督环境核实18个进程出生身份消失、3个worker进程组无存活成员、6个自有容器移除、19080/19082/19180/19182/19183空闲。B退出后另开环境的清理复核因PID namespace变化被拒绝，不能把那次查询写成原监督证明；B的原运行stop回执保留。

直接证据：C执行摘要、原件审计、同监督环境清理核验。审计结果绑定审计脚本SHA，脚本副本与清理核验源码在同一运行目录；实现锚点`3c85acf`及此前`85b7e39`、`d8a298a`可恢复。唯一合格参数仍是C0；上述结果不建立跨参数应用回滚、持续学习或集群结论。

## 一条命令完成真实演练并清理

在WSL项目根目录，使用已有独立serving环境。模型、编译工具和缓存沿用已验证资产，不自动下载模型。选择未使用的新目录：

```bash
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/probe_replicas.py --output artifacts/my-replica-run
```

脚本依次启动两个同参数实例和认证worker桥、逐副本预资格、发布、流中排空、同版本分流、退役及去重；再演示桥恢复、生成中准确杀掉自有engine组、在原监督侧核验旧代际终止、显式rebind及新资格。最大预约80次生成，本轮实际23次。结束后停止所有自有worker/bridge/gateway；清理失败会非零退出，不写成功summary。此命令会注入故障，只用于本包自有实例。

结果位于指定目录的summary、responses、stages、arrivals及cleanup。故障流缺DONE且为failed，不能作为成功响应；拒绝到达仍进入账本。离线复核不启动模型：

```bash
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/audit_replicas.py --run artifacts/my-replica-run --output artifacts/my-replica-audit.json
```

当前audit核固定演练的23次计数；改变演练须使用新身份并相应审阅核验条件，不能套用旧数字。

## 持续运行的组件

1. 在worker所在Linux环境，用既有start_engine.py启动新实例、probe_serving.py完成直连核验。
2. 启动`serve_worker.py --binding <新binding.json> --token-file <worker密钥文件> --port <端口> --manifest-output <新清单.json>`。清单由worker本机核参数、profile与进程身份；密钥仅保存在本地受控文件。
3. 根据演练输出gateway.json的结构配置副本ID、显式endpoint、清单和容量；选择新runtime，运行`serve_replicas.py --config <配置.json>`。
4. 通过Bearer管理密钥，对每个`/admin/replicas/{id}/qualify`完成实际预资格，再向`/admin/route`提交版本与expected_generation。普通调用使用`/v1/chat/completions`，可指定明确版本或active。响应头包含实际版本、副本和worker代际。
5. `/drain`停止给指定副本新派请求；`/retire`只在lease排空后允许，物理进程停止由维护者执行。`/reconcile`只核准确旧请求/原代际；同代际隔离恢复要显式`/reload`并再次`/qualify`，不会自动ready。

演练结束后的旧binding只作证据，不可拿它冒充新进程。副本身份变化时，旧lease必须先解除，新配置显式allow_rebind，再完整资格。同一个失败去重键不会因换代而被补发。不同参数的版本回滚与同参数副本替换是不同操作。

## 网络和故障边界

控制者通过网络协议消费worker回执，不把远端PID当成本机PID。实际生成携带expected manifest；HTTP桥会拒绝错代际。桥到本地vLLM仍使用本机TCP及前后身份检查，本包没有给原生vLLM端口增加原子身份认证。维护者必须停止旧bridge并确认旧engine结束后才复用后端地址；不能据此宣称任意外部替换下“绝不向错误native进程派发”。

可信终止核验限定原worker的相同boot ID、相同PID namespace和准确PID/start_ticks全部消失。监督环境重启、namespace变化或节点不可达时保持fail closed；网络错误和新代际出现都不是旧代际死亡证明。R5还需补完整节点生命周期、TLS及远端故障资格。

服务脚本默认loopback；RemoteEngine只允许本机HTTP或受控HTTPS，当前没有暴露公网监听。静态版本/拓扑合同变化不能原地升级runtime，运行中增删版本/自动发现尚未实现。单GPU多进程、HTTP假worker、配置渲染各自保持证据范围。

## 上游对照入口

固定[production-stack vllm-stack-0.1.13](https://github.com/vllm-project/production-stack/releases/tag/vllm-stack-0.1.13)，commit `e8cb4959ebfa333714ef468a3aefe158a65a09e6`。上游普通router作为独立对照，不放到本项目已pin worker之后再次选副本。源码的发现ID、/v1/models与内存请求计数不代替本项目参数/profile/代际/释放证据。源码与镜像清单核查不等于容器运行兼容。

独立无extras环境的锁在`environments/upstream-router-linux`；本机环境位于`$HOME/.local/share/llm-serving-lab/upstream-router-venv`。不改变本地vLLM0.11.2环境。官方router镜像包含额外模型框架，当前CPU对照直接运行固定源码的核心依赖，不拉取该镜像。

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.local/share/llm-serving-lab/upstream-router-venv \
  UV_CACHE_DIR=$HOME/.local/share/llm-serving-lab/uv-cache \
  $HOME/.local/share/llm-serving-lab/bin/uv sync --locked --no-dev --project environments/upstream-router-linux \
  --python $HOME/.local/share/llm-serving-lab/serving-venv/bin/python
$HOME/.local/share/llm-serving-lab/serving-venv/bin/python scripts/probe_upstream_router.py --output artifacts/my-upstream-router
```

CPU对照用两个明确HTTP夹具，检查roundrobin、坏model、连接失败/部分流且不重试，不生成真实模型内容。渲染使用configs/upstream下两个values；prepare_upstream.py校验固定chart与私有Helm工具，render_upstream_engines.py核双副本配置。只执行lint/template，未执行helm install。

双副本渲染要求每副本1 GPU、跨hostname、预置RWX模型卷与离线加载；engine滚动策略maxSurge0/maxUnavailable1避免只有两GPU时额外pod永远Pending，但它允许容量下降，不等于本项目在途排空保证。真实GPU容器、存储、TLS和Kubernetes仍待目标环境验收。
