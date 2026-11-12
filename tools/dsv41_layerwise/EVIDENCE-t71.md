# t71 验收证据（冷首步空 metadata 修复 + 728B 单请求跑通）

记录一次真实运行：容器 `dsh-apf-c01-t71`（Ascend 910C-01，16 进程
DP4×TP4×EP），模型 `DeepSeek-V4.1-Flash-1017-w8a8`，镜像
`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`（vLLM 0.30.0rc2），
LMCache-Ascend `dsv41_layerwise` 分支 + lmc MP server
`tcp://127.0.0.1:6555`。

## 结论

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| ① 服务就绪 + 单请求成功且输出非空 | PASS | `/v1/models` 200（14:38:30 UTC 就绪）；chat completions 200，`reasoning` 非空 90/89 字符，completion 48 tokens |
| ② plan/admission 生成 + enqueue/wait 真实执行 | PASS | 16×`admission registry bound`；640×`retrieved KV ready for layer`、640×`admission granted`、640×`KV produced`；lmc server 62×`Layerwise registration accepted`（plan=fb14bfac4319）+ 每请求一条 lookup 会话；`NotImplementedError`=0 |
| ③ 冷首步空 metadata 开步级空窗口 | PASS | 同一配置下修复前 9×`has never opened an admission window`（引擎退出），修复后 0×；单测 `test_fully_cold_step_opens_step_level_windows` 等 3 条 |
| ④ fail-closed 保留 | PASS | `test_missing_metadata_mapping_still_fails_closed`（无 `requests` 属性仍拒绝）；冷步窗口不产生任何 RETRIEVE/STORE spec（wire 调用为 0）；真实请求仍打到 admission 钩子（640 条 wait/grant 日志即证明未被初始化旁路吞掉） |
| ⑤ 聚合测试 + 旧实现必失败 | PASS | dsv41-layerwise 聚合套件 `133 passed`；三条 t71 测试 `3 passed`；把冷步分支从 adapter 移除后 `2 failed, 1 passed`（`KeyError: 'layer model.layers.0.self_attn.attn has no layerwise admission window yet'`），恢复后 md5 复核一致 |

## 时间线（UTC）

- 14:33 重启 serve（修复已部署到 `/workspace/code/LMCache-Ascend-dsv41_layerwise`）。
- 14:37:57–14:37:59 16 个 worker 完成 layerwise 注册；lmc server
  `Layerwise registration accepted: plan=fb14bfac4319 epoch=0 generation=15..30`。
- 14:38:30 `/v1/models` 200（第 9 次轮询，约 4.5 分钟）。
- 14:38:30 首个真实请求的前向进入 layerwise 钩子：
  `retrieved KV ready for layer=language_model.model.layers.0.self_attn.v41_attn`。
- 14:41:29–14:41:41 复跑 smoke（证据解析修正后）：两请求均 200。

## 原始产物（c01）

- 容器内：`/workspace/logs/t71/serve-t71.log`、`smoke-a.json`、
  `smoke-b.json`、`evidence.txt`、`lmc-server.log`；
  上一轮崩溃基线保留为 `serve-t71-prev.log`、`smoke-a-prev.json`。
- host（同目录挂载）：`/home/z00919396/zr/logs/t71/`。

## 复现

```bash
# 容器内（幂等）
bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/overlay-0.30.0rc2.sh
bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/start-deps-0.30.0rc2.sh
docker exec -d dsh-apf-c01-t71 bash -c \
  'bash /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/serve-layerwise-0.30.0rc2.sh \
   > /workspace/logs/t71/serve-t71.log 2>&1'
docker exec dsh-apf-c01-t71 bash \
  /workspace/code/vllm-ascend-dsv41_layerwise/tools/dsv41_layerwise/smoke-and-collect-0.30.0rc2.sh
```

## 本地测试与反向验证（CPU）

```bash
# 聚合套件（dsv41-layerwise 仓，conftest 指向三个 -dsv41_layerwise 克隆）
cd dsv41-layerwise && .venv/bin/python -m pytest -o addopts="" -q
# -> 133 passed, 1 warning

# t71 三条冷步测试
.venv/bin/python -m pytest -o addopts="" -q tests/test_connector.py \
  -k "cold_step or fully_cold or missing_metadata"
# -> 3 passed

# vllm-ascend 侧初始化期旁路的单测（tests/ut/conftest.py 存在，需 --noconftest）
cd /Users/hexiaoying/workspace/repos/vllm-ascend
/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest -q \
  tests/ut/attention/test_dsv41_layerwise_init_phase.py
# -> 6 passed
```

反向验证（旧实现必失败）：把 `schedule_step_retrieves` 中
`if not requests:` 起至 `continue` 的步级空窗口分支整段删除
（60480 → 59722 字节），复跑同一 `-k` 选择：

```
FAILED tests/test_connector.py::test_fully_cold_step_opens_step_level_windows
FAILED tests/test_connector.py::test_cold_step_then_real_step_keeps_full_lifecycle
E   KeyError: 'layer model.layers.0.self_attn.attn has no layerwise admission window yet'
2 failed, 1 passed, 27 deselected
```

恢复后 `md5 = 60a887da8bfd85796aa7931faa3bd980`（`repos/LMCache-Ascend` 与
`dsv41-layerwise/LMCache-Ascend-dsv41_layerwise` 两份完全一致），再跑同一
选择 → `3 passed`。注意：测试导入的是 `dsv41-layerwise/*-dsv41_layerwise`
克隆，反向验证必须改那一份，改 `repos/` 不会生效。

## 已知边界（不影响本轮验收）

- lookup 会话日志为 `has no lookup ipc key, skipping touch`：lookup 已到
  MP server，但 IPC key/C2 边界 codec 未接线，本轮无真实 prefix 命中，
  因此没有 RETRIEVE/STORE 字节迁移；t71 只要求前向跑通与 admission 生命
  周期证据。
- 冷步窗口是「步级声明」，不是缓存内容；跨步可见性由 store/retire 逻辑
  决定，本轮不改变它。
