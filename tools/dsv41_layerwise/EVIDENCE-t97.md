# EVIDENCE t97 — 首个真实 forward 的 S1 绑定竞态

c01 容器 `dsh-apf-c01-t97`（镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，
`ce42d65b2403`，16 davinci + 驱动挂载，`--network none` 只用于静态自检；
serve 走真实 NPU）。三仓字节：vllm-ascend = 本次提交，LMCache `3e6ab3e`，
LMCache-Ascend `9fea0dc`。

## 结论

| 项 | 结果 |
| --- | --- |
| t96 的首请求崩溃（`without a bound step view`） | **已消除**（真机 0 次；37-token 单请求 200、输出 92 字符） |
| 绑定条件 | 由 `_codec is None` 改为 `not s1_capture_active()`，provider-only 也绑定 |
| 过渡窗口 | 绑定早于激活代数 → 该步 warning + cold，不再整组 worker 崩；真未绑定仍 fail-closed |
| compressor-state 写行 | 按 ring 语义（`tail_count = min(used, 32)`）推导，不再读 PAD 的 token slot |
| boundary store 真正点亮 | 8k 请求：`boundary store dispatched over the wire` = 3（`full_page_rows=32 written_rows=32 bytes=131320 token=8192`） |
| boundary restore | **0**，被 transport 的两个问题挡住（F4、F5），本轮未达成 |
| 单请求 / QPS16 稳定 | 512-token 连续 3 请求 3/3 成功；8k 请求与 QPS16 仍崩在 transport（F4） |

## 两次真机崩溃与修复顺序

1. **`... reached the S1 C2 hook without a bound step view`**（t96 遗留）：
   `bind_model_forward_step` 早退看 `_codec is None`，而生产路径没有
   `set_boundary_state_codec` 的调用者（grep 本仓只命中 tests/ut 与 tools；
   LMCache-Ascend 全树 0 命中），S1 只靠 `install_boundary_provider`。
   修：早退条件改为 `not s1_capture_active()`；`admit_step` 内部仍只对 codec
   做 TP 门，provider-only 不新增 collective。
2. **`layer ...layers.2... wrote 39 token(s) ... but no state row belongs to
   page 2`**：compressor-state 的 token slot buffer 是 PAD
   （`dsa_v41.py:1098` "State writes use ring ownership metadata; this buffer
   stays PAD"），用 `slot_mapping` 推写行恒为空。修：`_request_view` 对
   `is_compressor_state` 改走 kernel 语义
   （`compressor_triton._cache_update_kernel`）：
   `tail_count = min(used, ring_rows)`、
   `first_token = prefix_end + used - tail_count`、
   `rows = page * ring_rows + (first_token + i) % ring_rows`。

## 代码改动（repos/vllm-ascend）

- `vllm_ascend/attention/dsv41_layerwise_c2.py`
  - `DSV41StepRequestView` 增 `request_pages`；模块态增
    `_request_pages`、`_activation_generation`、`_bound_generation`、
    `_bind_count`、`_logged_transitions`。
  - `bind_model_forward_step`：条件改 `s1_capture_active()`；绑定前
    `_request_page_plan(runner, req_ids)` 记 `(prefix_end, first_block)`。
  - `_require_step_view` → `DSV41StepRequestView | None`：激活发生在绑定之后时
    一次性 warning + cold；否则照旧抛错。
  - `restore_target`：先走 `_planned_target`（计划页），miss 再走 live
    metadata；两处都要求 `prefix_end == boundary_token`、`page > 0`、几何已知。
  - `_request_view`：compressor-state 走 ring 算术行。
- `vllm_ascend/worker/v1/model_runner.py`：调用不变（`install_boundary_provider
  (model=self.model)`），竞态修复不需要新的安装点。
- `tests/ut/attention/test_dsv41_layerwise_c2_timing.py`（新增，7 用例）：
  provider-only 首 forward 绑定并 capture、绑定早于激活时 cold、真未绑定仍
  fail-closed、forward 后按计划页 restore、多请求计划不借页、forward 内 live
  metadata 优先、compressor-state 写行（`used=3 → 134/135/136`；
  `used=39 → 最后 32 个槽位，32 行互不相同）。
- `tools/dsv41_layerwise/overlay-0.30.0rc2.sh`：hc_pre 符号校验改为先取
  `nm -D` 输出再 `grep -q`，避免 `set -o pipefail` 下 grep 提前退出造成
  SIGPIPE 误判（首次 setup 就是被这个假失败挡住）。

## 本地测试

```
pytest --noconftest -q tests/ut/attention/test_dsv41_layerwise_c2_timing.py \
  tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py \
  tests/ut/attention/test_dsv41_layerwise_c2.py \
  tests/ut/attention/test_dsv41_layerwise_c2_transport.py \
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
# 46 passed

cd dsv41-layerwise && .venv/bin/python -m pytest -o addopts="" -q
# 177 passed
```

旧实现反证：`git stash push -- vllm_ascend/attention/dsv41_layerwise_c2.py`
后跑 timing 套件 → `5 failed, 1 passed`（唯一通过的是"armed 但未绑定仍
fail-closed"），`git stash pop` 后 7 passed。

## 真机证据（c01，`/home/z00919396/zr/logs/t97/`）

- 第一次（修复①之后）：`/v1/models` 200（第 33 次轮询），16 worker
  `admission registry bound`；单请求 500，16 条
  `wrote 39 token(s) ... no state row belongs to page 2`。
- 第二次（修复②之后，本文件主证据）：`/v1/models` 200（第 23 次轮询），
  单请求 `http=200 bytes=801`、`finish=length carrier_len=92 completion_tokens=24`；
  随后 QPS16 48 请求在 transport 崩（见下）。
- 第三次（512-token × 3，串行）：3/3 成功；
  `store dispatched over the wire` = 72、`store snapshot committed` = 5、
  `DSV41LayerwiseC2Error` = 0、`without a bound step view` = 0、
  `has never opened an admission window` = 0、`NotImplementedError` = 0。
  boundary 侧 15 条 `provider returned no rows`，原文
  `... ends at token 644 but the transport asked for 512`（见 F5）。
- 8k 单请求：崩溃前已出现 3 条
  `boundary store dispatched over the wire ... full_page_rows=32 written_rows=32
  bytes=131320 token=8192`，证明 boundary 编码/发送在真实 serve 里成立；随后崩在
  `RuntimeError: recorded block groups (12) do not cover engine group 6 at
  index 12`。

## Findings（transport 仓，出 t97 scope，只上报）

- **F4（阻断验收②）**：`lmcache_ascend/integration/vllm/dsv41_mp_adapter.py`
  的 `_step_spec_for` → `_blocks_for_engine_group` 对"本步 STORE op 未记录到
  该资源所属 engine group"直接 `RuntimeError`，而同一个文件里
  `_store_covers_resource` 已经承认 stock 调度器会省略没有分配 token 的 group。
  8k 前缀（单请求也会）触发：`recorded block groups (12) do not cover engine
  group 6 at index 12`，导致整组 worker 退出、API `EngineDeadError`。
  建议：STORE 走 `_store_covers_resource` 的同一判定，未覆盖则跳过该资源，
  与 README/注释里"no bytes to move this step; callers skip"一致。
- **F5（让收益打折）**：transport 以**调度 chunk 终点**请求 boundary
  （本机为 512），而引擎的 C2 捕获只存在于**真实 forward 的步终点**
  （本例一步覆盖 644）。plan §6 要求"无准确 C2 的 chunk 不 publish"，所以
  引擎保持 cold 并告警是正确行为；要让收益出现，需要在 transport 侧只对
  "步终点 == chunk 终点"的 chunk 发 boundary，或由 scheduler 侧把 chunk 边界
  对齐到步边界。
- 这两个都不是引擎侧能单方面修好的：改 transport 不在 t97 scope 内。

## 复现

```bash
# 1) 起容器（t97 脚本已固定在 c01 /home/z00919396/zr/code/t97/）
bash /home/z00919396/zr/code/t97/setup-t97.sh
# 2) 单请求 + QPS16（drive 脚本自带日志计数）
docker exec dsh-apf-c01-t97 bash /workspace/code/t97/drive-t97.sh
# 3) 计数
docker exec dsh-apf-c01-t97 grep -ac "boundary store dispatched over the wire" \
  /workspace/logs/t97/serve-t97-s1.log
```
