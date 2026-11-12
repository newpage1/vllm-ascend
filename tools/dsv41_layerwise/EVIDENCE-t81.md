# t81 S1 接缝与真实 transport codec 接线

日期：2026-11-13（Asia/Shanghai） · 角色：layerwise-ds · 任务：t81（implementation, round 2）
范围：只改 `repos/vllm-ascend/`。未触碰 `repos/LMCache/`、`repos/LMCache-Ascend/`、`ascend-perf/`、`code/`。

## 1. 结论

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| ① 用 `set_boundary_state_codec` 安装**真实** codec（非 fake），核对两方法签名与 `BoundaryStateDescriptor` 字段（resource_id 由 execution_layer 映射、ring_cursor = `prefix_end % ring_length`、row_indices = `view.written_rows`） | 通过 | 新增 `tests/ut/attention/test_dsv41_layerwise_c2_transport.py`：按路径加载 `repos/LMCache` 的 `layerwise_protocol.py` 与 `repos/LMCache-Ascend` 的 `dsv41_boundary_codec.py`，`_TransportBoundaryCodec` 直接调用真实 `encode_boundary_rows` / `decode_boundary_rows`；断言 `descriptor.resource_id == "language_model.model.layers.0.attn.compressor.state_cache"`、`ring_cursor == 70 % 32`、`row_indices == (38,39,40)`、`schema_version == protocol.BOUNDARY_STATE_SCHEMA_VERSION`、`checksum == compute_boundary_checksum(...)`（§3.1） |
| ② 真实 codec 下 capture→restore 整页往返成功、未命中 fail-closed；t78 的 IMAGE_SELF_CHECK 改用真实 codec 后仍过 | 通过（整页语义有 finding） | 真实字节往返：capture 页 2 的 3 行 → decode → rebase 到页 3 → 逐字节相等（§3.1 测试 2）。**整页**：transport 目前只编码本步写入的行（`written_rows`），ring 尾部不随快照复制，见 finding F1。镜像内 `image_self_check_t81.py` 用真实协议+真实 codec 跑出 `IMAGE_SELF_CHECK_T81_PASSED`（§3.3） |
| ③ t78 UT(14) + t71 UT(6) + 新增真实 codec 集成用例全过；把钩子改回 S0 直返时对应用例必须失败 | 通过 | 三文件同跑 **26 passed**（§3.2）；把 `capture_layer_boundary_state` 改成 S0 直返 → 新用例 **3 failed**（§3.4），恢复后 md5 回到 `b29f460a172e291784f6de42ad6e01d3` |
| ④ 提交并推送 `dsv41_layerwise`（English Conventional Commits + Signed-off-by），changedPaths 与 EVIDENCE 更新，`diff --check` 干净 | 通过 | 提交见 §5；`git -C repos/vllm-ascend diff --check` exit 0 |
| ⑤ 只改 `repos/vllm-ascend/` | 通过 | `git status --porcelain` 只列本仓路径（§5） |

## 2. 实现

t77 的数据面在 transport 侧（`encode_boundary_rows` / `decode_boundary_rows` +
adapter 内的 boundary pass），引擎侧缺的是"这一步写了哪些行"的真值。t81 补的
是 **provider**，不是第二个 codec：

### 2.1 `vllm_ascend/attention/dsv41_layerwise_c2.py`

- `DSV41BoundaryRequestView` / `DSV41BoundaryTensorView` 增加 `resource_id`
  （= `v41_impl.compressor_state_prefix`，即
  `<execution_layer>.compressor.state_cache`，不接受由 `v41_layer_name`
  字符串改写推导）。
- `_BoundaryRecord`（frozen dataclass）+ `_boundary_records[resource_id]`：
  每个 ratio-2 层在 capture 钩子里记录本次真实事实
  （`prefix_end`、`end_token = prefix_end + step_tokens`、
  `ring_cursor = prefix_end % ring_rows`、`compression_phase`、
  `row_indices = view.written_rows`）。`reset_step_state()` 清空。
- `provide_boundary_rows(resource_id, snapshot_id, boundary_token)`：
  无记录 / token 对不上 / 本步没写行 → 返回 `None` 并打 warning（snapshot
  对该资源保持 cold-only，绝不伪造）；否则返回三元组。
- `install_boundary_provider(connector=None)`：缺省取
  `get_kv_transfer_group()`，要求 `connector.worker_adapter.set_boundary_provider`
  可调用；其它 connector（stock、offloading、MultiConnector）没有该方法，
  直接返回 `False` → no-op。
- `s1_capture_active()` = 装了 push codec **或** 装了 provider；
  `capture_layer_boundary_state` 由它驱动，restore 路径仍只走 push codec
  （生产 restore 由 adapter 在 `wait_for_layer_load` 内部完成，时序在
  `guard_layer_admission` 之前）。

### 2.2 `vllm_ascend/worker/v1/model_runner.py`

`register_kv_caches(kv_caches)` 之后调用一次 `install_boundary_provider()`，
注释说明其它 connector no-op。初始化期只安装，不捕获、不搬字节。

## 3. 验证

### 3.1 真实 codec 集成用例（CPU，无 lmcache 安装）

`tests/ut/attention/test_dsv41_layerwise_c2_transport.py` 用 namespace 包手术
按路径加载两个**真实**模块（`msgspec 0.22.0`，venv 无 lmcache），6 个用例：

1. `test_real_descriptor_fields_and_protocol_checksum`：descriptor 的
   resource_id / boundary_token="73" / ring_cursor=6 / phase=0 /
   row_indices=(38,39,40) / schema_version / checksum 与
   `compute_boundary_checksum(...)` 复算一致。
2. `test_real_capture_restore_moves_bytes_to_the_new_page`：页 2 写 3 行 →
   capture → 目标页 3 填 7.0 → restore（rebase）→ 写入行逐字节等于源行；
   未触碰的行仍是 7.0（记录 F1）。
3. `test_restore_without_capture_fails_closed`：无捕获 restore →
   `DSV41LayerwiseC2Error` 且 `step_failure_count() == 1`。
4. `test_provider_returns_the_real_rows_and_stays_cold_on_mismatch`：
   provider 返回 `(6, 0, (38,39,40))`；token 不符 / 资源不存在 → `None`；
   整数 token 也能对齐。
5. `test_provider_is_installed_on_the_adapter_and_noop_elsewhere`：
   `install_boundary_provider` 装的是 `provide_boundary_rows` 本身；没有
   `worker_adapter` 的对象返回 `False`。
6. `test_s0_stub_records_nothing_and_the_provider_stays_cold`：无 codec 且
   无 provider 时 capture 不记录、provider 返回 `None`（S0 行为）。

### 3.2 全套 UT

```
python -m pytest --noconftest -q \
  tests/ut/attention/test_dsv41_layerwise_c2_transport.py \
  tests/ut/attention/test_dsv41_layerwise_c2.py \
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
→ 26 passed
```

### 3.3 镜像内自检（0.30.0rc2，真实 vllm/torch，`--network none`）

三仓只读挂载，`PYTHONPATH=/ws/repos/vllm-ascend`，真实
`from vllm_ascend.attention import dsv41_layerwise_c2`（走平台插件激活）：

```
IMPORT_OK module
PROVIDER_OK
CAPTURE_OK (96, 97, 98) 0 0
DESCRIPTOR_OK language_model.model.layers.0.attn.compressor.state_cache 67 (96, 97, 98)
ROUND_TRIP_OK 3
COLD_ON_MISMATCH_OK
FAIL_CLOSED_OK
IMAGE_SELF_CHECK_T81_PASSED
```

含：provider 安装/拒绝、真实形状 capture、真实 codec 编码 + 协议 checksum
复算、decode + 换页 rebase 逐字节相等、陈旧 token 与无捕获保持 cold、restore
未命中 fail-closed 并打印 "the next TP gate rejects the batch on every rank"。

### 3.4 旧实现对照（必失败）

把 `capture_layer_boundary_state` 改回 S0 直返（`return`），同一集成用例 →
**3 failed**：

- `test_real_descriptor_fields_and_protocol_checksum`
- `test_real_capture_restore_moves_bytes_to_the_new_page`
- `test_provider_returns_the_real_rows_and_stays_cold_on_mismatch`

恢复后 md5 `b29f460a172e291784f6de42ad6e01d3`，复跑 26 passed。

### 3.5 静态

- `git -C repos/vllm-ascend diff --check` → exit 0
- `python3.11 -m compileall -q` 两个改动文件 + 两个新增文件 → exit 0

## 4. findings（交 t82；均在 transport 仓，不在本仓 scope）

| id | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| F1 | `repos/LMCache-Ascend/lmcache_ascend/integration/vllm/dsv41_mp_adapter.py::_restore_resource_payload` 的 `spec.boundary is not None` 分支 | 只把 `decoded["rows"]` 写回**捕获时的行号**，且只含本步写入的行；plan §6 要求整页复制并 rebase 到新请求自己的 ring page。跨请求复用同一 prefix 时会写错页、ring 尾部保留旧 slot | restore 时按 `target_first_page - source_first_page` 重定位，并复制整页 32 行 |
| F2 | `dsv41_boundary_codec.py::encode_boundary_rows` | 只接受 2D tensor + 行下标；ring 尾部（未写 slot）无 schema 字段描述，接收端无法判断"整页是否完整" | descriptor 增加 page/row-range 或 `rings_complete` 字段（schema v2） |

## 5. 变更清单

- 修改 `vllm_ascend/attention/dsv41_layerwise_c2.py`（resource_id、boundary
  record、provider、`s1_capture_active`、capture 在无 codec 时也记录）
- 修改 `vllm_ascend/worker/v1/model_runner.py`（+6：`install_boundary_provider()`）
- 新增 `tests/ut/attention/test_dsv41_layerwise_c2_transport.py`（6 个用例，
  真实协议 + 真实 codec）
- 新增 `tools/dsv41_layerwise/image_self_check_t81.py`（镜像内真实自检）
- 文档：本文件 + `tools/dsv41_layerwise/README.md` 的「与真实 transport codec
  的接线（t81）」
- 提交：见 `git log -1 --format=%H`
