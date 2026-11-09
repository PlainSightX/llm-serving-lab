# scripts 索引

脚本按用途分组。大部分脚本会启动真实 vLLM 或容器、只操作本次运行自己创建的进程，并把输入、源码快照与原始结果写进新的输出目录；审计脚本只读已有原件。

| 用途 | 脚本 |
|---|---|
| 启动引擎与网关 | `start_engine.py`（受控 vLLM 实例，记录进程与参数身份）、`serve_releases.py`（多版本网关）、`serve_replicas.py`（副本池网关）、`serve_worker.py` / `run_worker.py`（远端 worker 桥）、`serve_measurement.py` |
| 真实验收（含故障注入） | `probe_serving.py`（直连资格）、`probe_online.py`（单实例 14 类场景）、`probe_releases.py`（两版本发布与故障矩阵）、`probe_replicas.py`、`probe_application_release.py`、`probe_parameter_publication.py`（两参数版本灰度与热回滚）、`probe_cluster_local.py`、`probe_cluster_gpu.py` |
| 网关性能测量 | `run_gateway_ab.py` / `report_gateway_ab.py`（基线与优化网关 A/B）、`bench_lease_count.py`、`run_benchmark.py` / `report_benchmark.py`（R3/R4）、`diagnose_benchmark.py`、`instrument_gateway.py`、`compare_runtime_baseline.py`、`storage_benchmark.py` |
| 集群部署与测量 | `prepare_cluster.py`、`prepare_cluster_transfer.py`、`build_cluster_images.py`、`export_cluster_images.py`、`fetch_cluster_assets.py`、`fetch_runtime_wheels.py`、`cluster_node_preflight.py`、`cluster_health.py`、`collect_worker_manifests.py`、`load_cluster.py`、`benchmark_cluster.py`、`report_cluster_benchmark.py`、`cluster_cpu_fixture.py`、`owned_containers.py` |
| 上游对照（production-stack） | `prepare_upstream.py`、`render_upstream_engines.py`、`probe_upstream_router.py` |
| 量化（N2） | `n2_quantize.py`、`n2_quant_compat.py`、`n2_quality.py`、`n2_quality_audit.py`、`n2_benchmark.py`、`n2_benchmark_audit.py`、`n2_profile.py`、`n2_trace_summary.py`、`n2_report.py` |
| 分布式训练（N4） | `n4_prepare.py`、`n4_train.py`、`n4_compare.py`、`n4_audit.py`、`n4_export.py`、`n4_qualify.py`、`n4_serving_audit.py`、`n4_restore_archive.py` |
| 训练产物发布 | `qualify_release_model.py`、`import_coldstart_release.py`、`import_n1_release.py` |
| 持续学习研究 | `q1_*.py`、`q2_*.py`、`n1_study.py`、`n1_audit.py`、`n5_*.py` |
| 离线审计 | `audit_releases.py`、`audit_replicas.py`、`audit_application_release.py`、`audit_parameter_publication.py`、`audit_cluster_gpu.py` |
