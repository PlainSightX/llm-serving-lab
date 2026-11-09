# 双节点交付准备

## 当前备份与重租接续

本次恢复包在 `artifacts/cloud-recovery-20261106-a`；最终完整性与可释放判断以其COMPLETE.json及校验回执为准。恢复范围、Linux暂存命令、凭据边界和重租顺序见云实例恢复说明。

新机器必须重新核实例与主机密钥、网络、GPU和worker身份。

本页维护R5的资源、当前运行、交付入口与复现步骤。当前云S-B的结果与运行交接位于`artifacts/r5-cloud-slo-20261105-b`。固定运行镜像仍由首轮云B的`artifacts/r5-cloud-deployment-20261104-b/deployment-source.json`与`images-ready.json`追溯，当前网关使用云S-B的新状态目录。基础角色包`artifacts/r5-cloud-package-20261104-b/transfer-final`是已冻结输入，当前镜像与运行清单通过云B补丁交付接续。

## 已购置环境

2026-11-04，两台`ecs.gn7i-c16g1.4xlarge`于18:02创建，每台单张NVIDIA A10、16 vCPU、60GiB内存和200GiB ESSD系统盘。两台`ecs.c9i.xlarge`于18:09创建，每台4 vCPU、8GiB内存和60GiB ESSD系统盘，分别分配为控制机和负载机；负载机的磁盘容量由原目标40GiB提高到统一CPU配置60GiB。四台采用Ubuntu 24.04 x86_64、同VPC和同交换机，节点网段为`172.19.96.0/20`。完整实例ID、地址与本机密钥路径保存在采购记录。

四台共享同一个普通安全组，组内互通，公网入方向为SSH 22与ICMP。四台SSH登录和基础OS/CPU/内存/磁盘检查通过，两台GPU报告不同UUID、23028MiB可见显存、580.126.09驱动和Docker 29.8.0。控制机到其余三台的私网ICMP通过。角色级前检、GPU运行时、三节点K3s和设备注册均已验证，当前B模型服务已资格并发布。

两台GPU合计20.6068元/小时，两台CPU合计1.6752元/小时，四台基础费率22.282元/小时，公网出流量0.8元/GB，每台带宽上限100Mbps。24小时基础费用为534.768元，流量另计。

杭州K区的同规格订单曾因库存不足失败。浏览器自动生成密钥的私钥下载未能核实，随后在本机生成RSA密钥并通过公钥导入完成登录；杭州留有两条此次尝试产生的闲置密钥登记。原始SSH输出和控制台截图与采购记录同目录保存。

## 云端首轮结果与当前运行

A轮在原代码上取得下列观察，随后发生清理证据交接失败。原A的180条网关到达为141完成、38拒绝和1失败，失败请求的清理记录保留为未确认。测试Pod已停止，完整原件位于`artifacts/r5-cloud-deployment-20261104-a`。

| 场景 | 已观察结果 |
|---|---|
| 节点与调度 | 三节点Ready，两个GPU节点各有1个可分配GPU，两个模型Pod位于独立GPU主机 |
| 低负载与并发 | 12请求/1 req/s全部协议有效。24请求/8 req/s全部有效，两个副本各12次 |
| 排空 | 持久管理连接下，排空动作观察到2个待收尾绑定，后续流量由b承接，最终待清理数为0 |
| 控制者Pod更换 | 路由与worker身份保持，重复请求409，后续4/4有效 |
| worker Pod换代 | 旧绑定被隔离，b在替换期4/4。新a经显式重绑定与独立资格后恢复双副本 |
| 节点间Pod网络中断 | 阻断控制节点到gpu-a的UDP/8472，b承接4/4。恢复后4/4，身份保持，规则与定时恢复措施已清理 |
| 原代码进程组故障 | 原监督环境观察到进程退出，但控制面保留1个未确认清理，A按失败归档 |

修复包括匿名OCI设备插件显式导入、正常清理使用现有认证重核接口、先终止native再保留原桥20秒、worker Service发布未就绪监督端点，以及清理阶段停止信号归并。生成端继续执行ready和expected-worker校验。20秒窗口与停止、桥关闭的既有上界合计70秒，低于worker Pod的75秒终止宽限。原监督环境不可用时仍保留不确定状态。

B使用新状态根`/var/lib/llm-serving-lab/r5-cloud-b`和固定修复镜像。B共45条网关到达，35完成、1失败、6拒绝、3未知，全部租约解除。原Pod未就绪后，经同一Service新建TCP连接取得了原代际终止证明，期间发送SIGTERM后证明入口仍可用。native故障后的新代际资格及4请求窗口通过。控制者SIGKILL后保留3条unknown，其中2条已派发请求的释放证据确认，恢复窗口2/2有效。原件位于本页开头的云B入口。

两轮分别预约203与52次生成，累计255，上限256。离线复核关联192条原始Scheduler释放事件，B另有1条原监督进程组终止证明。A的1条未确认记录仍按失败保留。B运行代码的实际字节与本次源码相符，Windows238通过/6跳过、Linux244通过，独立只读审查的问题已关闭。

上述为首轮有界集成与故障观察。正式短命令目标负载与匹配原生、上游基线结果见下一节。A的网络中断窗口首内容延迟中位约2.09秒，恢复后约96ms，这四请求样本中的健康检查等待作为后续S诊断线索。

## 正式负载与匹配对照实施

11月5日恢复时，原实例ID及主机密钥保持一致，公网地址、主机启动身份、物理GPU UUID和native代际发生变化。原S-A在16个正式窗口后中断，2240请求原始观测及两侧日志已取回。新S-B使用`artifacts/r5-cloud-slo-20261105-b`和`/var/lib/llm-serving-lab/r5-cloud-s6b/gateway`，保持原输入与完整矩阵，重新冻结实际代际。为计入中断成本与完整重跑，A/B累计预约上限调整为7000，A已预约2274，B上限4726。按连续计费估算主机费用、20GB流量预留及新增2小时计算，本次接续投影约412元，原600元货币上限继续生效。计费投影尚未经云账单结算核对。

当前连接地址归`r5_cloud.slo.current_connections`所指记录。SSH使用新公网地址，并以采购时保存的原地址作为`HostKeyAlias`，继续严格校验原主机密钥。采购记录中的地址保留为历史事实。下一次主机地址或启动身份变化时，先核验实例与密钥，再处理工作负载代际。

当前执行记录为既有工作包的`r5_cloud.slo`，原S-A原件根为`artifacts/r5-cloud-slo-20261104-a`，当前结果归上述S-B。沿用C0参数、推理profile、两台A10上的当前native代际与CUDA12.8镜像。原部署A/B的失败、调用数和状态分别保留。初始A方案调用预约上限5000，重启后采用上述累计上限，全部准备与失败计入。费用仍归原600元总上限。

本包由root顺序实施，子代理只读核对方法与完整变更。先复用已有open-loop计量器和当前native实例，随后接入固定production-stack路由。独立重新启动另一套引擎会增加编译与代际差异，故采用当前Pod内的临时双向TLS字节转发，将原生API用于受控测量。原生与上游共用这一入口，主服务继续通过自身worker桥。临时入口限定私网来源，客户端证书必需，公网安全组设置保持原采购范围。

1. 冻结测量输入与部署。`scripts/benchmark_cluster.py`复用`serving_lab.benchmark.window`，保存事前请求、完整SSE、真实token与全部失败。原生请求通过`ServingProfile.payload`生成，核对与网关最终wire一致。现有计量器只增加逐请求endpoint支持，以同一时刻表在两个native之间交替。对应反例验证默认单endpoint调用保持原行为。临时TLS转发与上游启动包装器保留独立源码摘要。
2. 同一模型下比较native、自研gateway和固定production-stack入口，每入口3次重复，1/8/16 req/s分别32/320/32请求，共3456正式请求。核心目标窗口持续40秒。顺序按重复与速率轮换，窗口间等待全部引擎请求释放。主负载沿用三个公开状态835/896/1168输入token。另从旧六个公开support的实际请求选取六个不同状态，作为本包留出输入，每入口3次重复、8 req/s、96请求，共864请求。总计4320正式请求，准备、失败与启动成本另计入上述累计预约上限。全部输入、顺序与采样参数在首个正式窗口前冻结。留出指本轮性能调参未使用的工程负载，历史support不升级为能力泛化面板。
3. 实施前先核三入口等价wire、实际worker代际、token计数、合法动作及释放证据。正式窗口前后保存各引擎metrics、GPU旁证和请求释放日志。独立复核计划分母、SSE、每条真实请求身份、账本与释放事件，输出重复分布和留出表现，清理本包临时进程及内网测量入口，恢复可接手状态。

目标在测量前固定为8 req/s下每次重复至少99%的计划请求协议完整、输出属于当次公开候选集合、TTFT不超过1秒且E2E不超过2秒。16 req/s用于压力观察。计时从计划到达开始，发送后口径也保留。迟发超过250ms记漏发，p95迟发超过间隔10%时窗口失去发生器资格，失败与拒绝始终保留分母。输出预算64，报告实际输出token。短动作命令的合格不扩大为完整任务成功或长文本生成能力。三次短窗口的描述性尾分位保持其样本范围。

自研gateway以新runtime使用总执行槽16、排队槽32，每副本8/16，与每个native的`max_num_seqs=8`对应。原B的2/4及每副本1/2是首次交付资格配置，其原件保留。新配置重新取得接口资格并显式发布。此项属于部署容量配置，归因时单独说明。上游保持固定commit、关闭重路由和语义缓存，TLS包装只承担证书与传输接入，路由源码保留原字节。高负载差异包含准入与排队策略。只有低负载、相同wire及实际输出长度匹配时才描述入口额外等待。

检查点包括安全入口拒绝无证书访问、上游各HTTP客户端的证书配置、任一worker换代使当前比较停止、输出长度不匹配时分层而非宣称纯开销，以及失败运行保留后以新身份修复。恢复后的时间与流量预算按上述B轮估算口径记录。研究R的任务流、强记忆和持续更新责任继续留在独立下一包。


### 已核验结果

S-B完成36个正式窗口与6个预热窗口，全部计划发送时刻合格。主负载为835–1168输入token，留出为825–2240输入token。实际输出主负载最多15 token、留出最多11 token，以下尾分位为这些有限样本的描述。

| 8 req/s输入组 | 入口 | 达标/计划请求 | 首内容P50 / P95（ms） | 端到端P95（ms） |
|---|---|---:|---:|---:|
| 主负载 | 原生参考 | 960/960 | 56.754 / 74.757 | 198.930 |
| 主负载 | 项目网关 | 960/960 | 85.717 / 104.385 | 227.925 |
| 主负载 | 上游路由 | 958/960 | 57.384 / 75.178 | 199.634 |
| 留出输入 | 原生参考 | 288/288 | 80.754 / 121.738 | 213.926 |
| 留出输入 | 项目网关 | 288/288 | 115.155 / 158.073 | 248.370 |
| 留出输入 | 上游路由 | 288/288 | 81.935 / 122.468 | 214.843 |

上游第二个主负载重复有2次ConnectError，该次318/320达标，三次重复均通过事前99%门槛。失败请求未形成原生释放事件，已保留客户端原件、Pod事件与路由器日志，当前根因尚未确定。1 req/s下每入口96/96，按每次32个相同输入、相同实际输出与长度配对，项目网关额外首内容等待中位为23.217–26.265ms，上游为1.161–1.623ms。16 req/s的每入口96/96通过，三个两秒窗口只支持该压力观察范围。

完整复核覆盖4374条客户端记录、1474条网关完成记录和4392条Scheduler释放事件，各窗请求数、输入token与输出token的引擎计数差分逐一一致。网关遗留租约为0。S-A的2274次预约和中断原件保留，S-B含自动启动4次共预约4394次，累计6668/7000。报告器修复了信任存档达标布尔值及失格窗口仍作配对归因的两处缺口，3个反例先失败后通过，最终Windows244通过/6跳过、Linux250通过。

临时原生TLS参考进程、上游Deployment/Service、3个NodePort与测量NetworkPolicy已按确切身份清理，负载机无本包活动容器。两个旧Failed Pod的元数据已存档后清理。当前三个业务Pod保持Ready，路由generation为1，两个worker身份保持固定，网关容量为总16执行/32排队、每副本8执行/16排队。

离线重算从项目根执行以下命令，输入为已下载原件。

```bash
uv run --locked python scripts/report_cluster_benchmark.py --run artifacts/r5-cloud-slo-20261105-b --output artifacts/r5-cloud-slo-20261105-b/report-recheck.json
uv run --locked python artifacts/r5-cloud-slo-20261105-b/audit_evidence.py --run artifacts/r5-cloud-slo-20261105-b
```

## 租赁规格

采用四台同地域、私网互通的完整Linux主机。两个GPU节点承载完整C0副本，控制机承载K3s控制面与服务入口，独立负载机采集请求表现。以下为本项目的目标配置，GPU和内存余量用于后续并发与负载试验。

| 主机 | 数量 | 目标配置 | 必要条件 |
|---|---:|---|---|
| GPU worker | 2 | 每台1张独享NVIDIA GPU，显存至少24GB，8 vCPU，32GiB内存，200GB SSD | 两台使用相同型号，优先Ampere或Ada架构，驱动580或更新，可使用NVIDIA Container Runtime |
| 控制机 | 1 | 4 vCPU，8GiB内存，60GB SSD | root/sudo、systemd、持久磁盘 |
| 负载机 | 1 | 4 vCPU，8GiB内存，40GB SSD | Docker Engine、独立CPU资源、到控制机的私网连接 |

系统统一使用Ubuntu 24.04 x86_64。租赁单位需要提供主机级root/sudo权限、systemd以及安装容器运行时的权限。GPU主机的只读前检入口为`deployment/node_preflight.py`，它分别报告硬件与软件就绪条件。CUDA版本兼容与GPU容器运行时的依据见官方资料。[CUDA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[K3s NVIDIA运行时配置](https://docs.k3s.io/advanced#nvidia-container-runtime)

需要收集四台主机的管理连接方式、私网IPv4地址、主机网段、GPU型号与UUID、磁盘规格、计费方式。网络放行按来源限定，22/TCP供管理入口，6443/TCP供GPU节点连接控制面，8472/UDP供节点间Flannel VXLAN，10250/TCP供节点内部管理，30443/TCP供负载机访问网关。Pod和Service网段由渲染器核验与主机网段相互独立，默认分别为10.72.0.0/16和10.73.0.0/16。[K3s网络要求](https://docs.k3s.io/installation/requirements#networking)

当前实例的计费和释放责任见上方已购置环境记录。租赁后的第一个检查点是主机条件、私网互通与GPU设备注册，随后才进入模型部署和负载窗口。

## 固定运行材料

运行镜像采用Python 3.12.14、vLLM 0.11.2、PyTorch 2.9.0+cu128、Transformers 4.57.1。官方vLLM 0.11.2镜像的实际版本为`0.11.2+cu129`，本包因此使用现有Linux依赖锁构建CUDA 12.8运行镜像。171个wheel共5,255,926,551字节已按锁定URL及SHA256校验，构建使用离线wheel目录。镜像内BF16矩阵计算和C0实际生成均已运行。

| 对象 | 固定身份 |
|---|---|
| 基础包控制镜像 | `llm-serving-lab-control:r5-21a63bfeed1d`，image/index digest `be2eabcaa094e9023e70d8dcf8787420a31232c220afba35751d127884ff52df` |
| 基础包GPU镜像 | `llm-serving-lab-worker:r5-21a63bfeed1d`，image/index digest `9fac91354607a61b1d48aa9f4c4999a73d49e99f2ea5dbc3035f1a9095058c2e` |
| 当前云B控制镜像 | `docker.io/library/llm-serving-lab-control@sha256:f6aa020d46a6361ed6851144c6630ce47a7363ebdb9e6d578e5c6bb3c8fb8b1d` |
| 当前云B GPU镜像 | `docker.io/library/llm-serving-lab-worker@sha256:24c5476d51446805dfd8614cfb41280b0b29e4743890d24b82c6b5940e4cdb45` |
| C0参数工件 | `1ee83b8562404b56bdb208a675bb5d4e4c83b827156df8f6b1e73d30842989de` |
| 应用证据文件 | `19a80f72ecd2c7db1573b0de7f756d564b39d8b601d1e8bd39306a70dbee0d14` |
| Kubernetes发行包 | K3s `v1.35.9+k3s1`，二进制及离线镜像按上游checksum核验 |
| GPU设备插件 | NVIDIA device plugin `v0.20.1`，镜像digest见`deployment/images.lock.json` |

镜像导出器区分OCI index、运行manifest和config，沿linux/amd64描述符核验内容摘要。运行镜像归档、源模型、应用证据和部署文件分别留有身份记录。[OCI index规范](https://github.com/opencontainers/image-spec/blob/v1.1.1/image-index.md)、[K3s离线导入](https://docs.k3s.io/installation/airgap)

角色交付包位于当前包的`transfer-final/`。`control.tar`为325,365,760字节，`worker.tar`为9,401,436,160字节，供两个GPU节点分别使用，`loadgen.tar`为52,602,880字节。每个归档内有`TRANSFER-SHA256SUMS`，全部成员已流式核验，并从包内脚本完成节点计划渲染及Linux shell语法检查。CA签名私钥保留在本地`pki/ca.key`，角色包仅包含该角色需要的叶证书和访问材料。

早期A包的错误格式化曾把本地测试凭据带入诊断日志。A包现已标记退役，普通消费入口拒绝加载。B包重新生成了CA、全部叶证书私钥和访问密钥，旧客户端证书在B的CA下验证失败。当前清单同时校验文件形式与Kubernetes Secret/ConfigMap内容的一致性。

## 本地验证结果

| 验证 | 已观察结果 | 证据入口 |
|---|---|---|
| 真实K3s API与CPU网络演练 | 3个本地容器节点就绪，13项生产资源的服务端dry-run通过，mTLS/DNS/访问控制通过，重建一个worker后其身份变化，另一worker身份保持 | `artifacts/r5-local-k3s-20261104-e/summary.json` |
| 实际GPU容器 | 两个C0容器共享本机RTX4080 Laptop GPU，各完成2次bootstrap和8项资格，4条应用请求通过mTLS网关，排空后观察到另一副本接收请求 | `artifacts/r5-gpu-containers-20261104-b/summary.json` |
| 原始记录复核 | 正式B轮24生成、20条网关到达均completed/confirmed，全部lease释放，三个容器正常退出 | `artifacts/r5-gpu-containers-20261104-b/raw-audit-v2.json` |
| 完整成本 | GPU A轮24生成，B轮24生成，总48次，工作包上限64次 | 两轮`reservations.jsonl`与原始release events |
| 镜像导出 | 控制镜像52,538,845字节，GPU镜像5,616,027,718字节，设备插件55,552,855字节，OCI身份和归档SHA均已核验 | 当前包`images/images.json` |

GPU A轮所有20条网关记录最终已完成并释放，但验证脚本在SSE终段到达后立即读取pending计数而报错。B轮按服务生命周期有界等待原租约收尾，再检查空闲状态。K3s演练保留A至D的失败，修复了配置导出、UTF-8读取和只读接口就绪等待。网络就绪等待发生在CPU替身的身份探测阶段，生成调用保持一次尝试的记录方式。

两台独立GPU主机的设备注册、实际调度及本轮跨节点故障观察见上方结果。正式目标负载表现继续单独验收。单控制者的持久状态由控制机保存。节点永久失联时，旧请求保持unknown/隔离，原监督环境中的准确释放或所选云平台的有效停机证明决定后续处置。

最终只读审查提出负载容器名称映射和诊断异常阻断清理两项重要问题，均已修复。负载机Docker检查也已补齐。对应反例先失败后通过，真实Docker检查确认日志文件写入失败后两个自有容器仍被清理，生成命令中的名称映射在实际容器内解析正确。最终Windows223项通过/6项平台跳过、Linux229项通过。镜像的6个运行入口脚本及35个包模块与当前运行源码逐字节一致。

原始消费核验在`artifacts/r5-delivery-consumption-20261104/result.json`，资源核验在`artifacts/r5-final-resource-check-20261104.json`。本包容器/网络已清理，16443与19383空闲，原有六个业务容器保持原身份。测试展开目录及旧角色包草稿已移入回收站，验证回执和原始失败保留。

## 清洁节点复现顺序

1. 按采购记录中的四台实际主机完成角色级只读前检。基础SSH、硬件身份和私网ICMP已检查，继续核验GPU容器运行时及各角色软件条件。根据实际主机网段确定Pod/Service网段。
2. 上传基础角色包，在新目录解包并执行`sha256sum -c TRANSFER-SHA256SUMS`。用当前源码中的`deployment/render_node_plan.py`渲染各角色计划，并记录补丁脚本摘要。当前生成器包含设备插件摘要导入及CRI核验。原包校验记录保持可追溯。
3. 先执行控制机的setup，再执行两个GPU节点的setup。确认三个节点Ready、NVIDIA运行时可用，再应用设备插件并确认两个GPU节点各有`nvidia.com/gpu`资源。
4. 将云B的`images/control-full.tar.gz`和`images/worker-full.tar.gz`按角色导入K3s内容库，并按`images/*-transfer.json`核对完整归档摘要、索引与配置身份。核对命名引用的实际索引摘要后建立对应仓库摘要引用，再用CRI核验。增量归档要求先核实其列出的基础内容全部存在。应用当前源码生成的Service配置与新镜像清单，当前实例绑定清单为云B的`runtime-manifests.json`。新运行使用独立状态目录，worker使用只读模型目录。
5. 通过管理入口逐个qualify，再按当前route generation执行promote。每次新worker代际的重新绑定与资格单独记录。
6. 启动负载机，检查管理入口和负载入口，按预先限定请求数、速率和预算执行窗口，再进入故障矩阵。导出镜像/节点/请求/释放/指标记录后收口。

实际地址取得后，可在角色包根目录生成计划。示例使用示意私网地址。

```bash
python3 deployment/render_node_plan.py --bundle . --role control \
  --node-ip 10.10.0.10 --loadgen-ip 10.10.0.13 --network-cidr 10.10.0.0/24 \
  --output node-plan
python3 deployment/node_preflight.py --role control --output host-preflight.json
```

GPU节点分别选择`--role gpu-a`和`--role gpu-b`并提供`--server-ip`。负载机选择`--role loadgen`，生成的setup设置客户端证书读取权限，`load-command.sh`通过`--add-host`为容器设置网关名称映射，并给出12请求/1 req/s的首个窗口。生成的shell文件统一使用LF行尾。

管理端使用文件中的凭据。通过SSH端口转发将gateway映射到本机19383后，普通命令如下。

```bash
python deployment/cluster_admin.py --endpoint https://localhost:19383 --secrets secrets/gateway status
python deployment/cluster_admin.py --endpoint https://localhost:19383 --secrets secrets/gateway qualify a
python deployment/cluster_admin.py --endpoint https://localhost:19383 --secrets secrets/gateway qualify b
python deployment/cluster_admin.py --endpoint https://localhost:19383 --secrets secrets/gateway \
  route promote --version qwen3-17b-coldstart-candidate --generation 0
```

提交路由变更前核对返回的实际generation。K3s原生模型端口位于worker容器loopback，外部调用经过mTLS桥和发布控制。worker替换后，控制者对旧代际的固定绑定仍需显式处理，再设置针对该副本的allow_rebind并重新资格。

## 云端验收矩阵

| 场景 | 检查对象 |
|---|---|
| 清洁节点部署 | 固定镜像与模型摘要、两个GPU UUID、实际Pod节点、运行版本、worker资格 |
| 公开服务与负载 | 普通/流式响应、实际版本/副本/代际、有效输出、到达延迟、首内容时间、拒绝与失败分母 |
| 副本排空 | 新请求转给可用副本，已有绑定按原实例收尾，租约释放后再退役 |
| worker进程故障 | 准确旧代际隔离、原监督侧释放/终止证明、健康副本继续接纳 |
| Pod替换 | 新实例身份、旧绑定拒绝、显式重绑定和重新资格 |
| 节点网络中断与恢复 | 失联实例隔离、健康节点行为、恢复原代际后的准确核对 |
| 控制者重启 | 原runtime中的路由、去重和未确定调用恢复 |
| 上游对照 | 将CUDA/模型/profile/负载对齐后再比较，原上游cu129镜像作为待对齐来源 |

研究下一检查点继续冻结新增技能/规则变化任务流和同信息量的强记忆对照。冷启动C0、运行交付与持续学习各自保留验收责任。
