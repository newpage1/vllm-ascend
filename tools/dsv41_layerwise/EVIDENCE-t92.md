# EVIDENCE-t92 — 多请求 boundary page 归属（引擎侧）

任务：t92（repair round 2，派生自 t88 复审 findings）
范围：只改 `repos/vllm-ascend/`（transport 仓 LMCache / LMCache-Ascend 未改）
基线：t87 `0d450c6`（整页 store/restore 的引擎侧几何）

## 1. 结论

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| ① 按请求解析 boundary page；两请求同 step 各自 rebase | 通过 | 引擎 UT 13 项（含 `test_restore_target_names_each_request_its_own_page`、`test_two_requests_store_their_own_pages_without_overwriting`）；设备自检 `CAPTURE_OK`/`RESTORE_OK`/`ISOLATION_OK` |
| ② boundary record 按 (request_id, resource_id) 存储、按请求取记录、无覆盖 | 通过 | UT `test_two_requests_store_their_own_pages_without_overwriting`、`test_concurrent_requests_keep_isolated_boundaries[2/4/8]`；设备自检两请求行号 `(70,71,72)` 与 `(103,104,105)` |
| ③ 真实 runner/connector 集成或 c01 真实路径 | 部分通过（见 §5 边界） | 设备自检在 c01 镜像内用**真实** `build_c2_ring_metadata` 产出 metadata、装进**真实** `ForwardContext`、走**真实** `provide_boundary_rows` + 真实 codec + 真 NPU ring；未跑 transport 的 `start_load_kv`（该类在 LMCache-Ascend，属 t90/t91 的覆盖） |

反向验证：把两个钩子摘掉（模拟 t87 之前的实现）后，`test_without_the_hooks_the_whole_page_contract_is_impossible`
按预期失败——此时的 codec 只能产 schema 1 稀疏包，transport 拒收。

## 2. 实现

`vllm_ascend/attention/dsv41_layerwise_c2.py`

- 记录按请求：`_boundary_records: dict[(step_id, request_id, resource_id), _BoundaryRecord]`，
  `_BOUNDARY_RECORD_LIMIT = 4096` 按插入序淘汰最旧；`_record_boundary(view)`
  写三元组键；`boundary_record(resource_id, request_id=None, step_id=None)`
  取最新匹配。
- `_matching_records(resource_id, boundary_token, request_id)`：命中同一 token
  的所有请求记录。
- `provide_boundary_rows(..., request_id=None)`：唯一记录直接返回；多条记录
  的 `(ring_cursor, phase, row_indices)` 相同则返回（同页同位置无需区分），
  否则告警并 `None`；有记录但 token 不符沿用 `"... ends at token ..."` 文案
  （外部语料断言用），完全无记录用新文案。
- `restore_target(..., request_id=None)`：点名取该请求行，未点名扫描本步
  所有 `prefix_end == token` 的行；候选页多于一个 → `None`；`prefix_end`
  与 token 不符 → `None`；**候选行 `used <= 0`（kernel 不写 page）→ 剔除，
  全部剔除则 `None`**；几何未知 → `None`。返回 `(page, ring_rows)`。
- 语义：一个请求无法归属只让**该** boundary 回 cold，同批其它请求照常恢复。

`tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py`

- 新增 `_MultiMeta` / `_bind_multi_step` / `_capture_multi` / `_frame` harness。
- 新增 6 个用例：两请求各自页不覆盖、同 prefix 歧义与点名、按 token/按 id
  精确解析、单请求失配不拖累整批、2/4/8 并发隔离、无 token 请求不可命名页。
- 既有"多请求步"断言从"整体 fail-closed"改为"歧义 → `None`，点名 → 精确页"。

`tools/dsv41_layerwise/npu_check_t92.py`

- c01 镜像内 `--network none` 跑：真实 `build_c2_ring_metadata`（NPU）产
  metadata → 真实 `ForwardContext` → `install_boundary_provider(model=...)` →
  逐请求 capture/整页编码/解码 → 第二/第三步验证按 token/按 id 的页解析、
  跨页整页写回与尾部归零、同 token 歧义与 `used == 0` 的 fail-closed。

## 3. 验证

本机（Python 3.11 虚拟环境）：

```
pytest --noconftest -q \
  tests/ut/attention/test_dsv41_layerwise_c2.py \
  tests/ut/attention/test_dsv41_layerwise_c2_transport.py \
  tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py \
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
# 39 passed

cd dsv41-layerwise && .venv/bin/python -m pytest -o addopts="" -q
# 170 passed（t91 基线）
```

c01 设备自检（镜像 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，16 davinci
设备 + 驱动挂载，`--network none`，只读挂载三仓）：

```
IMPORT_OK vllm_ascend.attention.dsv41_layerwise_c2 codec lmcache_ascend.integration.vllm.dsv41_boundary_codec
PROVIDER_OK 32 (256, 16) hooks ['restore_target', 'ring_rows']
CONTEXT_OK ForwardContext
CAPTURE_OK [('req-a', 73, (70, 71, 72)), ('req-b', 74, (103, 104, 105))]
RESTORE_OK 4 32
ISOLATION_OK 4 5
UNSET_PAGE_FAIL_CLOSED_OK [73, 74]
NPU_CHECK_T92_PASSED
```

复现命令（c01）：

```
docker run --rm --network none --privileged \
  --device=/dev/davinci0 ... --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf -v /var/log/npu:/var/log/npu \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v <workspace>:/ws:ro -e PYTHONPATH=/ws/repos/vllm-ascend \
  quay.io/ascend/vllm-ascend:v0.30.0rc2-a3 bash -lc \
  'P=/ws/repos/vllm-ascend/tools/dsv41_layerwise/payload;
   cp $P/kv_cache_planning.py /vllm-workspace/vllm/vllm/v1/core/kv_cache_planning.py;
   cp $P/deepseek_v4_utils.py /vllm-workspace/vllm/vllm/transformers_utils/deepseek_v4_utils.py;
   source /usr/local/Ascend/ascend-toolkit/set_env.sh;
   python3 /ws/repos/vllm-ascend/tools/dsv41_layerwise/npu_check_t92.py'
```

（两个 `cp` 是镜像缺口的 B1/B3 垫片，与生产 overlay 脚本相同；自检脚本不
修改仓库，只在一次性容器内补镜像缺的模块。）

## 4. 设备侧新事实

`build_c2_ring_metadata` 只在请求本步有 token 时写 page：`row4 =
where(used > 0, block_table[b, 0], 0)`。设备实测 `seq_lens=[73,74], used=0`
时 `ring_metadata[4] == [0, 0]`，`prefix_end` 仍然正确（`[73, 74]`）。因此
"按 token 找到行"不足以命名页，必须同时要求该行 `used > 0`；t92 已按此
fail-closed，并新增用例 `test_a_request_without_tokens_cannot_name_a_page`。

## 5. 边界与 finding（transport 仓，出本任务范围）

- F3：`dsv41_mp_adapter._store_step_boundaries` 只取 `entries[0]` 的
  `(request_id, start_token, end_token)`，provider 调用不带 request id，因此
  **服务端每次 commit 只携带一个请求的 boundary**。引擎侧已按请求记录、并
  支持 `request_id` 关键字；transport 改为逐请求发 spec 后即可直接生效。
- F4（已知）：`test_s1_protocol_v2_real_provider.py` 等 dsv41-layerwise 语料
  指向克隆树，不覆盖本仓文件；本仓的新逻辑由 `tests/ut/attention/` 下的
  四套用例覆盖。
- F5：`VllmConfig` 无法在镜像内为 DeepSeek-V4.1（Engram 段要求 CUDA）构造，
  设备自检改用 vllm 自己的 `ForwardContext` + `override_forward_context`；
  生产路径（serve）由 `--engram-config` 走完整引擎初始化，本自检不改变它。
