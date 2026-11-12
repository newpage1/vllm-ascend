# dsv41_layerwise v0.30.0rc2 镜像运行套件（t71）

## 目的

让 vllm-ascend `dsv41_layerwise` 分支（基于 vLLM main）能在
`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`（vLLM 0.30.0rc2）容器内完整跑通
DSV4.1 layerwise KV 迁移 smoke：服务就绪 + 真实单请求前向 + admission
生命周期证据。本套件 = t64 复盘出的 B1–B10 backport + hc_pre 路由根因修复。

## hc_pre 根因（t64 blocker 的修复）

t64 首个真实前向（profile run）崩溃于
`RuntimeError: aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so`：
镜像 vllm_ascend 树自带 custom-op vendor 包
`_cann_ops_custom/vendors/custom_transformer/`（全镜像唯一带 `aclnnHcPre`
符号的库），stock 的 `bootstrap_custom_op_env()` 把它写进
`ASCEND_CUSTOM_OPP_PATH`，扩展按绝对路径 dlopen 命中；分支影子树缺这个
vendor 目录 → bootstrap 早退 → dlsym 落到 CANN 分库（无该符号）→ 崩溃。
修法：把镜像 vendor 包整份复制进分支影子树（`overlay-0.30.0rc2.sh`
自动完成，幂等，复制后 `nm -D` 校验符号）。零 env hack，与 stock 完全
同路径。

## 文件

| 文件 | 作用 |
| --- | --- |
| `overlay-0.30.0rc2.sh` | 容器内一次性应用：hc_pre vendor 修复 + B1–B10 + 自检（幂等） |
| `start-deps-0.30.0rc2.sh` | 容器内装 LMCache/LMCache-Ascend（幂等）+ 起 lmc MP server（127.0.0.1:6555） |
| `serve-layerwise-0.30.0rc2.sh` | t64 attempt13 serve 配置 + `lmcache.mp.server_urls` 修正 + B5 CANN python 路径 |
| `smoke-and-collect-0.30.0rc2.sh` | 就绪轮询 + 双请求（cold store / 命中 retrieve）+ 证据收集 |
| `payload/` | 上游 backport 文件（见下） |

## payload 清单（B1–B10）

| 层 | 文件 | 说明 |
| --- | --- | --- |
| B1+B10 | `payload/kv_cache_planning.py` → 镜像 `vllm/v1/core/kv_cache_planning.py` | vLLM #61183 迁移后的模块别名（0.30 仍住在 kv_cache_utils）+ `VllmConfig.aux_output_config` 类属性注入 |
| B2 | `payload/engram_append.py` → 追加进镜像 `vllm/models/deepseek_v41/common/engram.py` | `engram_head_shard_rank()` 上游原文；`get_engram_dp_group` 缺失时回退 `get_dp_group` |
| B3 | `payload/deepseek_v4_utils.py` → 镜像 `vllm/transformers_utils/` | 上游整文件（812B，`get_compress_ratios`） |
| B4 | （无） | serve 参数已不带 `enable_dsa_cp` |
| B5 | overlay 探测 → serve 脚本消费 | CANN python site-packages 进 PYTHONPATH（`import acl`） |
| B6 | `payload/mm_input_norm.py` → 镜像 `vllm/model_executor/layers/fusion/` | 上游整文件（422 行，`FusedMMInputNorm`） |
| B7+B8 | `payload/target_dependent_ar/` → 镜像 `vllm/v1/worker/gpu/spec_decode/target_dependent_ar/` | shim 包；`SpeculatorCudaGraphManager` 从镜像 `autoregressive` 真类再导出；`TargetDependentARSpeculator` 占位（v1 worker lane 不执行） |
| B9 | `payload/gumbel_append.py` → 追加进镜像 `vllm/v1/worker/gpu/sample/gumbel.py` | `_log1p_neg_stable` 上游原文（自带 triton import，镜像 gumbel 无） |

## 源码内兼容层（B11/B12/B13，随分支树生效）

| 层 | 位置 | 说明 |
| --- | --- | --- |
| B11 | `vllm_ascend/worker/v1/model_runner.py` `profile_run` | 0.30 `GPUModelRunner.profile_run()` 无 `randomize_inputs` 形参；try 原调用，TypeError 文案含该 kwarg 才回退无参调用 |
| B13 | 同文件 `_check_and_update_cudagraph_mode` | 0.30 `adjust_cudagraph_sizes_for_spec_decode(uniform_decode_query_len, tensor_parallel_size)` 仍是双参（vLLM #60517 才删第二参）；TypeError 文案含 `tensor_parallel_size` 才回退双参调用 |
| B12 | `payload/kv_cache_planning.py` 内 `_MirroringModule` | 分支（main 系）把 7 个 planner 钩子打在 `_kv_cache_planning` 上，0.30 vllm 内部从 `kv_cache_utils` 命名空间查找 → shim 模块类替换为镜像模块，后续赋值同步到 `kv_cache_utils`；`_pool_bytes_per_block` 单/双参形态不同，走 `_pool_bytes_per_block_030_adapter` 适配（DSV4.1/GLM 路径不读 `vllm_config`，传 None；仅 TypeError 回退 stock 记账） |
| B14 | `vllm_ascend/worker/utils.py` `select_common_block_size` 调用点 | main 版三参 `(block_size, backends, kv_cache_specs)`，0.30 仍两参；TypeError 文案含函数名才回退两参调用（DSV4.1 层视图分组路径实际触发） |

## 冷首步空 metadata 修复（t71 forward 崩溃的根因）

0.30 stock scheduler 对**完全冷的第一步**（引擎 warmup / 首个 prefill 没有
RETRIEVE 也没有 STORE op）产出 `LMCacheMPConnectorMetadata(requests=[])`。
修复前的 `schedule_step_retrieves` 收空映射 → 一层都不开 window →
fail-closed 前向在 `layers.0.self_attn.v41_attn` 抛
`has never opened an admission window`（t71 起始日志 9 条，引擎随即整体退出）。

修法（`lmcache_ascend/integration/vllm/dsv41_mp_adapter.py`，
`schedule_step_retrieves`）：对 `plan.layer_order` 的每一层，当整步
`requests` 为空时开一个**步级空窗口**——哨兵 request id
`__cold_step__`、空 required/空 specs、登记即准入。语义边界：

- 步被显式声明并立即结算，forward 的 `wait_for_layer_load` /
  `check_admitted` 都能通过；
- **不伪造 hit**：不产生 RETRIEVE/STORE spec、不写 store range、不走 wire；
- 只覆盖「mapping 存在但为空」；metadata 缺失（无 `requests` 属性）仍按
  engine integration pending 拒绝（fail-closed 不变）；
- 真实请求仍走各自窗口，冷步窗口不并入、不放宽真实请求的等待语义。

配套的引擎初始化期旁路（`vllm_ascend/attention/dsv41_layerwise_admission.py`
的 `set_init_phase`/`synthetic_forward`）：profile_run、graph capture、
idle-DP dummy batch 没有 scheduler step，前向钩子直接旁路而不是报错；
真实请求步始终在该窗口之外，仍 fail-closed。

## 容器内执行顺序

```bash
docker run -d --name dsh-apf-c01-t71 --net=host --ipc=host --privileged \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /home/z00919396/zr/code:/workspace/code \
  -v /home/z00919396/zr/logs:/workspace/logs \
  -v /opt/data:/opt/data:ro quay.io/ascend/vllm-ascend:v0.30.0rc2-a3 sleep infinity

docker exec dsh-apf-c01-t71 bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/overlay-0.30.0rc2.sh
docker exec dsh-apf-c01-t71 bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/start-deps-0.30.0rc2.sh
docker exec -d dsh-apf-c01-t71 bash -c 'mkdir -p /workspace/logs/t71 && \
  bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/serve-layerwise-0.30.0rc2.sh \
  > /workspace/logs/t71/serve-t71.log 2>&1'
docker exec dsh-apf-c01-t71 bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/smoke-and-collect-0.30.0rc2.sh
```

（sortedcontainers wheel 需先放进容器 `/tmp/`；LMCache 源码树挂在
`/workspace/code/LMCache-dsv41_layerwise`、LMCache-Ascend 挂在
`/workspace/code/LMCache-Ascend-dsv41_layerwise`。）

## 与 t64 attempt13 的配置差异（仅两处 + 两处依赖修正）

1. `kv_connector_extra_config` 增加 `"lmcache.mp.server_urls":"tcp://127.0.0.1:6555"`
   ——t64 缺省连 `localhost:5555`（死端口，zmq 懒连接不报错但传输永远
   不可达）；S0 传输缝需要活 server。
2. PYTHONPATH 追加 CANN python site-packages（B5，实际路径
   `/usr/local/Ascend/cann-9.2.0/python/site-packages`）。

依赖层修正（lmc server 启动）：
- `--disable-observability`：镜像缺 `opentelemetry.exporter.prometheus`，
  默认 Prometheus 回退 import 直接崩；smoke 不需要观测面。
- 就绪探测用 `python3 socket` 而非 `ss`（镜像无 ss）。

## admission 执行证据（仓内配套改动）

`vllm_ascend/attention/dsv41_layerwise_admission.py`（本仓文件）为四个
生命周期点各加一条首见 INFO 日志（registry bound / retrieved KV ready /
admission granted / KV produced，逐层去重），smoke 脚本据此收集验收②
证据。stock 路径（registry=None）行为逐位不变。

## 风险与回退

- ACL graph 捕获阶段（FULL_DECODE_ONLY）的 admission 窗口缺失已由引擎
  初始化期旁路（`set_init_phase`/`synthetic_forward`）解决：捕获/profile/
  idle-DP dummy 前向不进 registry，真实请求步不受影响。若仍遇到捕获期
  fail-closed，兜底方案=serve 改 `cudagraph_mode=NONE` 并在报告注明偏差。
- lmc server 单请求约束（S0）：smoke 一次只发一个请求，规避
  `one request per step` 的 fail-closed 边界。
- 现有 smoke 的 lookup 会话可见 `has no lookup ipc key, skipping touch`：
  lookup 已真实到达 MP server，但 IPC key 尚未接线（C2 边界/codec 属后续
  里程碑），因此本轮不产生真实 prefix 命中；t71 验收只要求前向跑通与
  admission 生命周期证据，不要求命中。

## S1 C2 capture/restore hooks（t78）

`vllm_ascend/attention/dsv41_layerwise_c2.py` 是 S1 的引擎侧接缝：它把
C2 ring page 的真实请求视图交给 transport 侧 codec，并把 restore/capture
挂到 `dsa_v41_forward` 的生命周期上。默认关闭——没有 connector 安装
codec 时所有钩子立即返回，stock 路径、graph capture、profile dummy
逐位不变。

- 视图（引擎提供真实事实）：`DSV41BoundaryRequestView`（物理 ring page、
  绝对 prefix_end、本步写入行、5 个位置控制字、compression phase/ratio）、
  `DSV41BoundaryTensorView`（真实 compressor state 页、stride、两个 plane
  的布局）、`DSV41StepRequestView`（scheduler step 与批次 request 顺序）。
- 接缝（codec 实现）：`capture_layer_boundary(request_view, tensor_view)`
  与 `restore_layer_boundary(...)`；wire 描述符（`BoundaryStateDescriptor`）
  仍由 LMCache 侧 codec 构建，本仓不发布、不租约、不搬字节。
- 时机：`dsa_v41_forward` 中 restore 在 `guard_layer_admission` 之前、
  capture 在 `notify_layer_produced` 之前（"produced" 蕴含 "captured"）；
  `NPUModelRunner._model_forward` 在进入模型前绑定 step 视图并跑一次 TP
  一致步 gate（SUM all-reduce，只要任一 rank 报告失败或步号不一致，所有
  rank 在下一个算子/collective 之前一起抛出）。
- fail-closed：装了 codec 但真实前向没有绑定 step、ratio-2 层缺
  compressor metadata、批大小与 metadata 不一致、步内行不属于本请求的
  物理页、codec 抛错，一律 `DSV41LayerwiseC2Error`；不伪造命中、不跳过
  已跳过的 token。

### 与真实 transport codec 的接线（t81）

t77 把 C2 边界数据面做在 transport 侧（`lmcache_ascend/integration/vllm/
dsv41_boundary_codec.py` 的 `encode_boundary_rows` / `decode_boundary_rows`
＋ `DSV41LayerwiseMPWorkerAdapter` 内的 boundary pass），引擎侧只欠一份
"这一步到底写了哪些行"的真实事实。t81 因此补的是 **provider**，不是第二个
codec：

- `provide_boundary_rows(resource_id, snapshot_id, boundary_token)`：
  按资源返回 `(ring_cursor, compression_phase, row_indices)`；cursor =
  `prefix_end % ring_rows`，phase = `prefix_end % ratio`，rows = 本步
  state slot_mapping 命中本请求物理页的行（绝不 hash/modulo 猜）。资源
  没有捕获、token 对不上、或本步没写行时返回 `None` —— snapshot 对该资源
  保持 cold-only，不伪造。
- `install_boundary_provider()`：在 `NPUModelRunner` 完成
  `register_kv_caches` 之后调用一次。只有带
  `worker_adapter.set_boundary_provider` 的 layerwise connector 才安装；
  其它 connector（含 stock serving）是没有该方法的对象，本调用直接
  no-op，行为零差异。
- 捕获钩子现在由 `s1_capture_active()`（装了 push codec **或** provider）
  驱动；push codec 路径（`set_boundary_state_codec`）保留，供需要引擎侧
  直接搬字节的 transport 使用。
- 镜像内自检 `image_self_check_t81.py`（挂载三仓，`--network none`）用真实
  `lmcache` 协议与真实 codec 走通：provider → encode → descriptor 字段
  （resource_id = `<execution_layer>.compressor.state_cache`、cursor、phase、
  rows）→ 协议 checksum 复算一致 → decode → 换页 rebase 后逐字节相等 →
  陈旧 token/无捕获保持 cold → restore 未命中 fail-closed。

已知边界（t81 交 t82）：

- transport 侧 restore 目前把 payload 写回**捕获时的行号**，且只复制本步
  写入的行；plan §6 要求的是"整页复制 + rebase 到新请求自己的页"。跨请求
  复用同一个 prefix 时两者不等价（会写到别的请求的页、ring 尾部留下旧
  slot）。这是 transport 仓的改动，本仓只在 EVIDENCE-t81.md 记录为 finding。
- `resource_id` 与 `execution_layer` 的映射是 `<execution_layer>` +
  `.compressor.state_cache`，由 `vllm_impl.compressor_state_prefix` 提供，
  不接受字符串改写推导。
