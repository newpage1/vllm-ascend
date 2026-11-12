# t87 引擎侧整页 store/restore 几何（ring_rows / restore_target）

日期：2026-11-13（Asia/Shanghai） · 角色：layerwise-ds · 任务：t87（implementation, round 1）
范围：只改 `repos/vllm-ascend/`。未触碰 `repos/LMCache/`、`repos/LMCache-Ascend/`、`ascend-perf/`、`code/`。

## 1. 结论

| 验收项 | 结论 | 证据 |
| --- | --- | --- |
| ① `ring_rows` / `restore_target` 原生实现并在安装时附着到真实 provider；cold-only 不崩、不伪造 | 通过 | `install_boundary_provider(connector, model=...)` 在 `register_kv_caches` 之后被调用，`setattr(provide_boundary_rows, "ring_rows", ring_rows)` / `"restore_target"`；几何从 `compressor.state_cache.kv_cache[0].shape[1]` 绑定（`test_install_binds_real_geometry_and_row_space_views`）。几何未知时 `ring_rows` 抛 `DSV41LayerwiseC2Error`；无捕获/未安装返回 `None`（§3.1 测试 4、§3.3 的 `COLD_ON_MISMATCH`） |
| ② 真实 full-page：整页编码、未写行归零、跨请求 rebase、ring 尾部无旧 slot；失配/无钩子/越界/异 token fail-closed | 通过 | `test_store_frames_the_whole_page_and_zeroes_unwritten_rows`（schema 2、`page_offset == 64`、32 行、未写行全 0、checksum 复算一致）、`test_restore_target_rebases_the_page_and_leaves_no_stale_tail`（页 2→页 4，尾行无旧 slot）、`test_restore_target_fails_closed_on_every_mismatch`（token/双请求/synthetic/无 metadata/未知几何/未安装）、`test_without_the_hooks_the_whole_page_contract_is_impossible`（删钩子后只能产 schema 1 稀疏包，transport 拒收） |
| ③ 170 tests 基线不回归 + 新增引擎集成用例通过 + 旧实现对照必失败 + 0.30.0rc2 镜像自检 | 通过 | 聚合 `170 passed`；引擎侧四套 UT `31 passed`；旧实现（`git stash` 掉两个文件）下新用例 `5 failed`；镜像内 `IMAGE_SELF_CHECK_T87_PASSED`（§3.4） |
| ④ 提交推送 `dsv41_layerwise`（English Conventional Commits + `-s`）、EVIDENCE 更新、`diff --check` 干净 | 通过 | 提交见 §5；`git -C repos/vllm-ascend diff --check` exit 0 |
| ⑤ 只改 `repos/vllm-ascend/` | 通过 | `git status --porcelain` 只列本仓路径（§5） |

## 2. 实现

transport（LMCache-Ascend `ef2d992` / LMCache `3e6ab3e`）已把 boundary 升到
schema v2：payload 是**整个 ring page**，以绝对行寻址，restore 端 rebase 到新
请求自己的页；它需要引擎提供页几何与目标页。t87 补的正是这两件事。

### 2.1 `vllm_ascend/attention/dsv41_layerwise_c2.py`

- `BOUNDARY_STATE_CACHE_SUFFIX = ".compressor.state_cache"`；
  `boundary_state_cache_name(name)` 判定资源是否为 C2 环。
- `_ring_geometry: dict[str, int]` + `bind_ring_geometry(model)`：从
  `model.modules()` 里带 `v41_impl.compressor_state_prefix` 的层读真实
  `kv_cache[0].shape[1]`（4-D `[pages, 32, 1, 2*head]` 时即 32），**不写常量**。
- `ring_rows(resource_id) -> int`：只认绑定几何；未知即
  `DSV41LayerwiseC2Error("... ring geometry is not bound ...")`。
- `_live_ring_metadata(resource_id)`：从 live `get_forward_context().attn_metadata
  [resource_id].c2_ring_metadata` 取本步控制字（校验 2-D、至少 5 行）。
- `restore_target(resource_id, boundary_token)`：要求 provider 已安装、本步是
  单请求的非 synthetic 真实 forward、token 可解析、`prefix_end == token`、
  几何已知；返回 `(int(metadata[4, 0]), ring_rows)`（row4 = 该请求
  `block_table[:,0]`，即 ring 物理页）。任何不符 → warning + `None`。
- `row_space_view(value)` / `use_row_space_views(worker)`：把 adapter 里的
  boundary 张量换成同一 storage 的 2-D 行视图（4-D 要求 `dim2 == 1` 且
  contiguous；`data_ptr` 不变，注册哈希仍按 4-D 计算），非 boundary 条目不动。
- `install_boundary_provider(connector=None, model=None)`：安装 provider 后
  附着 `ring_rows` / `restore_target` 两个属性；有 model 时先绑几何，再做行
  视图替换，INFO 记录数量。

### 2.2 `vllm_ascend/worker/v1/model_runner.py`

`has_kv_transfer_group()` 分支里，`register_kv_caches(kv_caches)` **之后**调用
`install_boundary_provider(model=self.model)`。其它 connector 没有
`worker_adapter.set_boundary_provider`，本调用返回 `False`，stock 路径零差异。

## 3. 验证

### 3.1 新增 `tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py`（5 用例）

按路径只读加载 `repos/LMCache` 的 `layerwise_protocol.py` 与
`repos/LMCache-Ascend` 的 `dsv41_boundary_codec.py`，用真实 provider 路径
（非 fake 类）跑：安装与几何、整页 store、跨页 restore、六种失配 fail-closed、
旧实现（无钩子）对照。

### 3.2 套件

- `pytest --noconftest -q tests/ut/attention/test_dsv41_layerwise_c2.py
  tests/ut/attention/test_dsv41_layerwise_c2_transport.py
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
  tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py` → **31 passed**
- `cd dsv41-layerwise && .venv/bin/python -m pytest -o addopts="" -q` →
  **170 passed**（t91 基线）

### 3.3 旧实现对照

`git stash push -- vllm_ascend/attention/dsv41_layerwise_c2.py
vllm_ascend/worker/v1/model_runner.py` 后跑新用例 → **5 failed**
（`TypeError: install_boundary_provider() got an unexpected keyword argument
'model'`、缺 `ring_rows`/`restore_target`）；`git stash pop` 后 **5 passed**。

### 3.4 镜像自检

`tools/dsv41_layerwise/image_self_check_t87.py`，在
`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` 内以 `--network none` 挂载三仓
（`/tmp/t87ws`，只含三个包树）运行：

```
IMPORT_OK vllm_ascend.attention.dsv41_layerwise_c2
PROVIDER_OK ['restore_target', 'ring_rows']
GEOMETRY_OK 32 (192, 16)
CAPTURE_OK 6 0 (70, 71, 72)
STORE_OK 2 (32, 16) 2210
RESTORE_OK 4 32
FAIL_CLOSED_OK
IMAGE_SELF_CHECK_T87_PASSED
EXIT=0
```

`GEOMETRY_OK 32 (192, 16)` = 6 页 × 32 行、每行 2×8 float32；`STORE_OK` =
schema 2、32×16 整页、2210 字节；`CAPTURE_OK 6 0 (70, 71, 72)` = cursor 70%32、
phase 0、页 2 的三个绝对行。

### 3.5 静态检查

`git -C repos/vllm-ascend diff --check` exit 0；`compileall`（python 3.11，
本仓 PEP 646 文件在系统 3.9 下本就不过）对四个改动/新增文件 exit 0。

## 4. Findings / 已知边界

- F1（已在 t84 修复，本仓只消费）：transport restore 曾按捕获行号写回、只复制
  本步写入行；v2 改为整页 + rebase 后，本仓只需给出目标页与页几何。
- F2（设计约束）：`restore_target` 只支持单请求步。boundary page 属于恰好一个
  请求，多请求步无法判定归属，按 fail-closed 回 cold（不猜页）。
- F3（跨仓）：长 KV / index / SWA 的恢复仍走各自资源，不由本钩子覆盖；C2 页
  只承载 compressor state。

## 5. 变更清单

```
vllm_ascend/attention/dsv41_layerwise_c2.py          （+ 钩子与行视图几何）
vllm_ascend/worker/v1/model_runner.py                （install(..., model=self.model)）
tests/ut/attention/test_dsv41_layerwise_c2_fullpage.py  （新增）
tools/dsv41_layerwise/image_self_check_t87.py        （新增）
tools/dsv41_layerwise/README.md                       （新增 t87 章节）
tools/dsv41_layerwise/EVIDENCE-t87.md                 （本文件）
```
