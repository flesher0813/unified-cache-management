# Grafana Linker 精简后移出的指标

Linker 总览对齐 vLLM Connector，保留聚合带宽、接口耗时、逐层加载和传输失败。下表记录从 Grafana 总览移出、目前仍未展示的指标及其排障用途。采集和 Toolkit 配置仍可供单独查询。指标名省略直方图的 `_bucket`、`_sum` 和 `_count`。

## UCM 后台工作与 Store Adapter

| 指标 | 用途 |
| --- | --- |
| `ucm:linker_lookup_requests_total`、`ucm:linker_lookup_skipped_total` | 区分实际向 Store 查询的 owner 次数和非 owner 的跳过次数。 |
| `ucm:linker_lookup_resumed_total`、`ucm:linker_lookup_fallback_total` | 判断查询是否复用前次命中边界、是否因原边界消失而从第 0 页重查。 |
| `ucm:linker_{load,offload,prefetch}_{operations,started,completed}_total` | 对照三类后台任务的入队、开始和完成速率，排查提交与处理节奏。 |
| `ucm:linker_prefetch_queue_duration_ms` | 测量任务在 UCM 队列里从入队到被 worker 取出的等待。 |
| `ucm:linker_prefetch_prepare_duration_ms` | 判断后台任务准备阶段是否耗时。 |
| `ucm:linker_prefetch_duration_ms` | 测量后台任务从入队到 worker 完成的总耗时，包括排队和执行。 |
| `ucm:linker_offload_io_duration_ms` | 测量写回批次从提交到 worker 观察完成的时间，包含 Store 排队和完成检测延迟。 |
| `ucm:linker_load_requests_total`、`ucm:linker_load_cancelled_requests_total` | 统计进入待加载集合的请求与成批前被移除的请求，用于排查取消。 |
| `ucm:linker_load_ready_layers_total`、`ucm:linker_offload_skipped_total` | 统计发布就绪的逻辑层和非 owner 跳过 I/O 的写回调用。 |
| `ucm:linker_load_indices_sync_duration_ms` | 测量 load worker 等待索引就绪的时间。 |
| `ucm:linker_adapter_interface_{register_device_pools,prepare_load,submit_prepared_load_layer,submit_offload}_duration_ms` | 分别观测设备池注册、加载准备、逐层加载提交和写回提交的同步调用耗时，包括异常退出。 |

## SGLang 原生 Linker

| 指标 | 用途 |
| --- | --- |
| `sglang:linker_lookup_tokens_total`、`sglang:linker_hit_tokens_total` | 计算外部查询 token 速率、所有 rank 均可恢复的 token 速率及其比例。重复查询也会计数。 |
| `sglang:linker_load_tokens_total` | 按 `mode` 查看逐层加载和 prefetch 成功恢复的 token/s。 |
| `sglang:linker_offload_tokens_total` | 查看进入写回队列的 token/s。 |
| `sglang:linker_prefetch_requests_total` | 按 `outcome` 统计发布、策略停止、提交拒绝、释放和失败。 |
| `sglang:linker_failures_total` | 按 `stage` 查看 SGLang 感知的查询、加载和写回失败。 |

## SGLang 总览对齐 vLLM 时移出的面板

| 原面板 | 指标 | 用途 |
| --- | --- | --- |
| Request Rate | `sglang:num_requests_total`、`sglang:num_aborted_requests_total` | 查看每秒完成和中止的请求量，用于判断流量变化和中止是否集中发生。 |
| Token Generation Throughput | `sglang:gen_throughput` | 查看 SGLang 调度器上报的实例及 rank 级生成吞吐。总览保留基于 token 计数计算的输入和输出吞吐。 |
| Retracted Request Rate | `sglang:num_retracted_requests_total` | 查看每秒被调度器撤回的请求数，用于排查调度回退。 |
| Cache Hit Rate | `sglang:cache_hit_rate` | 查看调度器按实例及 rank 上报的缓存命中率。新版总览使用请求侧 token 计数展示缓存命中来源及占比。 |
