# t78 S1 vllm-ascend hooks：C2 capture/restore + TP admission/request view

日期：2026-11-13（Asia/Shanghai） · 角色：layerwise-ds · 任务：t78（implementation, round 1）
范围：只改 `repos/vllm-ascend/`。未触碰 `repos/LMCache/`、`repos/LMCache-Ascend/`、`ascend-perf/`、`code/`。

## 1. 结论

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| ① hooks 真实接入 C2 capture/restore 与 TP admission/request view；0.30.0rc2 镜像内可 import/构造；默认关闭时 stock 路径不变 | 通过 | 镜像自检 `IMAGE_SELF_CHECK_PASSED`（§3.1）；无 codec 时全部钩子立即返回（§3.2 测试 1） |
| ② C2 restore 命中 t77 产出的 prefix 复用；未命中/未就绪 fail-closed | **未闭环** | t77 的 C2 codec 在本次收口时尚未落地（`repos/LMCache-Ascend` HEAD `b8553fb`，无 codec 文件）。本仓以契约一致的 fake codec 验证了 capture→restore 整页往返与未命中 fail-closed（§3.2 测试 8/9）；真实 prefix 命中需 t77 落地后由 t79 复审 |
| ③ 当前字节聚合/UT 通过，含旧实现应失败的对照；`git diff --check` 干净 | 通过 | 本仓 UT 14 passed；t71 UT 6 passed（合计 20）；旧实现（钩子直返的 S0 stub）对照 7 failed（§3.3）；`git diff --check` exit 0 |
| ④ 只改 `repos/vllm-ascend/` | 通过 | `git status --porcelain` 仅本仓 4 个路径（§5） |

## 2. 实现

### 2.1 新增接缝 `vllm_ascend/attention/dsv41_layerwise_c2.py`

引擎侧只提供**真实事实**，wire 描述符（`BoundaryStateDescriptor`）仍由
LMCache 侧 codec 构建；本仓不发布、不租约、不搬字节。

- `DSV41BoundaryRequestView`：物理 ring page（来自 `c2_ring_metadata` row4）、
  绝对 `prefix_end`（row0）、本步写入 token 数（row1）、5 个位置控制字、
  `next_slot = prefix_end % ring_rows`、`compression_phase = prefix_end % ratio`、
  本步真实写入行 `written_rows`（来自 state `slot_mapping`，按物理页过滤）。
- `DSV41BoundaryTensorView`：真实 compressor state 页
  （`state_cache.kv_cache[0].squeeze(-2)`，FP32、连续、32 行 × 2*width）、
  `page_ids`（`block_table[:, 0]`）、`planes()` 给出 kv/scores 两个逻辑 plane
  的 dtype/shape/行偏移/行跨距/页跨距。
- `DSV41StepRequestView`：scheduler step（引擎前向序号）、批次 request 顺序、
  generation、`synthetic`。
- `DSV41BoundaryStateCodec`（runtime_checkable Protocol）：
  `capture_layer_boundary(request_view, tensor_view, *, stream)`、
  `restore_layer_boundary(...)`。安装时校验两个方法齐全，缺一即拒绝。
- 生命周期函数：`set_boundary_state_codec` / `clear_boundary_state_codec` /
  `bind_step_request_view` / `clear_step_request_view` /
  `bind_model_forward_step`（runner 入口，含 TP gate）/
  `note_step_failure` / `step_failure_count` / `reset_step_state`。

### 2.2 调用点

- `vllm_ascend/attention/dsa_v41.py::dsa_v41_forward`：
  `wait_for_layer_load` → **`restore_layer_boundary_state`** →
  `guard_layer_admission` → `v41_impl.forward` →
  **`capture_layer_boundary_state`** → `notify_layer_produced`。
  init-phase 旁路分支保持原样（不 capture、不 restore）。
- `vllm_ascend/worker/v1/model_runner.py::_model_forward`：进入模型前
  `bind_model_forward_step(self, decoder_replay_dummy=...)`，绑定失败/门禁
  拒绝时在下一个算子前抛错；`finally` 中 `clear_step_request_view` 复原。

### 2.3 TP 一致步 gate（0.30 异步调度下的时机）

每个前向在 `_model_forward` 入口做**一次** SUM all-reduce（`[step_id+1,
failure_count]`，int64，TP device group）。归约值在所有 rank 上相同，因此判定
一致：步号不是 `world_size × 自己步号` → 所有 rank 抛；任一步内失败被记录
（`note_step_failure`）→ 所有 rank 抛。TP=1 时不发 collective。这是 S1 唯一
新增的集合通信，且只在装了 codec 时存在。

## 3. 验证

### 3.1 镜像内自检（0.30.0rc2，真实 vllm/torch，`--network none` 一次性容器）

```
IMPORT_OK dsv41_layerwise type
CAPTURE_OK 64 3 (96, 97) kv 32
FAIL_CLOSED_OK layer model.layers.0.self_attn reached the S1 C2 hook withou
IMAGE_SELF_CHECK_PASSED
```

自检内容：以包路径 `from vllm_ascend.attention import dsv41_layerwise_c2`
导入（走真实 `vllm_ascend/__init__.py` 平台插件激活）；构造真实形状的
compressor state（4 页 × 32 行 × 1 × 16 FP32）与 `c2_ring_metadata`；
断言 `isinstance(codec, DSV41BoundaryStateCodec)`；capture 得到
`prefix_end=64, ring_page=3, written_rows=(96, 97)`，两个 plane 的行跨距
64 字节、score 平面行偏移 32 字节；restore 未命中抛
`DSV41LayerwiseC2Error`。容器为临时自检，跑完即删；影子树已恢复原 md5
（`dsa_v41.py f0f5db80…`、`model_runner.py 9fcf8c01…`），临时
`dsv41_layerwise_c2.py` 已删除。

### 3.2 本仓 UT（CPU，无 torch_npu/vllm 安装）

`python -m pytest --noconftest -q tests/ut/attention/test_dsv41_layerwise_c2.py`
→ **14 passed**。覆盖：

1. 无 codec 时钩子全惰性（stock/S0 行为）；
2. 装了 codec 但真实前向没有 step 视图 → fail-closed；
3. 无 ring plane 的层跳过；
4. 缺 compressor metadata → fail-closed；
5. 步视图与 metadata 批大小不一致 → fail-closed；
6. 请求视图字段逐项等于真实 metadata（prefix_end/page/rows/phase/位置控制）；
7. 写入行不属于本请求物理页 → fail-closed（拒绝 hash/modulo 猜行）；
8. capture → restore 整页往返（不同物理页，目标页字节完全一致）；
9. restore 未命中 → fail-closed 且 step 失败计数 +1，下一次 gate 拒绝；
10. decoder-replay dummy 绑定 synthetic 视图并跳过 codec；
11. `bind_model_forward_step` 绑定真实 `input_batch.req_ids` 并可复原；
12. TP gate：步号不一致 → 所有 rank 抛；任一 rank 失败 → 所有 rank 抛；
    TP=1 不发 collective。

### 3.3 旧实现对照（必失败）

把 `_run_hook` 开头改成 `return`（等价 S0：钩子直返、不 capture/restore），
同一组 UT → **7 failed**：`test_real_forward_without_step_view_fails_closed`、
`test_missing_compressor_metadata_fails_closed`、
`test_batch_mismatch_between_view_and_metadata_fails_closed`、
`test_request_view_is_built_from_real_metadata`、
`test_rows_outside_the_request_page_are_rejected`、
`test_capture_then_restore_round_trips_the_whole_page`、
`test_restore_miss_raises_and_marks_the_step_failed`。恢复后 md5 一致
（`5d8e6bfe7e4090dac2bff0a82c97a83b`），复跑 14 passed。

### 3.4 既有测试与静态检查

- t71 UT：`tests/ut/attention/test_dsv41_layerwise_init_phase.py` → 6 passed；
  与 t78 UT 同跑 20 passed。
- `python3.11 -m compileall -q` 三个改动/新增文件 → exit 0。
- `git -C repos/vllm-ascend diff --check` → exit 0。

## 4. 失败关闭矩阵（装 codec 后）

| 情况 | 行为 |
| --- | --- |
| 真实前向没有 step 视图 | 抛 `DSV41LayerwiseC2Error`（不静默跳过 capture） |
| ratio-2 层缺 compressor metadata / 无 `c2_ring_metadata` | 抛 |
| `num_reqs` 与绑定批大小不一致 | 抛 |
| 本步写入行不属于本请求物理页 | 抛（拒绝伪造行） |
| codec restore 未命中/损坏/超时 | 包装后抛，`note_step_failure`，下一次 TP gate 所有 rank 停止 |
| TP 步号不一致 / 任一 rank 已失败 | 所有 rank 在下一算子/collective 前抛 |
| engine init（capture/profile）或 decoder-replay dummy | 不 capture/restore（inert 或 synthetic） |
| 未装 codec（stock） | 全部立即返回，零行为差异 |

## 5. 变更清单

- 新增 `vllm_ascend/attention/dsv41_layerwise_c2.py`
- 修改 `vllm_ascend/attention/dsa_v41.py`（+12：两个钩子调用）
- 修改 `vllm_ascend/worker/v1/model_runner.py`（+13：绑定 + 复原 + 失败门）
- 新增 `tests/ut/attention/test_dsv41_layerwise_c2.py`（14 个 CPU 测试）
- 文档：本文件 + `tools/dsv41_layerwise/README.md` 的「S1 C2」章节

## 6. 已知边界（交给 t79）

1. **t77 codec 未落地**：真实 prefix 命中无法在本 attempt 闭环；本仓以契约
   fake codec 验证往返。t79 需在 t77 落地后复查 `DSV41BoundaryStateCodec`
   的方法签名与 `BoundaryStateDescriptor` 字段能否一一对上。
2. **capture 路径的 D2H**：每层每请求读 5 个位置控制字 + 本步写入行
   （`.tolist()`），是 S1 capture 的真实成本；stock 路径无此开销。
3. **TP gate 只做一致性**：它保证所有 rank 同一步、同一失败状态，不做跨
   rank 数据校验；数据完整性仍由 codec 的 checksum/position/phase 校验负责。
4. **错误恢复**：S1 失败后的恢复仍是受控重启（首版不承诺 mid-forward
   rewind），与 t61 §6 的约定一致。
