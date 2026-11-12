# t75 证据：S0 真实 STORE 闭环（c01，2026-11-12 17:44Z）

三仓字节：vllm-ascend dsv41_layerwise=3e0379bc6（store hook + 防御读取），
LMCache dsv41_layerwise=0258e4d（不变），LMCache-Ascend dsv41_layerwise=b8553fb。

## run13 最终证据（serve-t75-run13.log，3450 行）

| 指标 | 值 |
| --- | --- |
| store submitted（wait-time 路径） | 96（4 worker × 24 层资源） |
| store dispatched over the wire | 24（单 owner 当选，tokens=[0,3328)） |
| snapshot committed | 3（snap-fb14bfac-0-1-g3/g4/g5，worker 唯一 id） |
| 落选跳过（lost the single-owner election） | 72（WARNING，不 raise） |
| 真实字节总量 | 64,131,223 |
| Traceback | 0 |
| 请求结果 | 200，prompt 3547 / completion 48（reasoning），finish=length |
| 请求后引擎 | /v1/models=200 存活 |
| smoke 判定 | SMOKE_COLD_ONLY_STORE_FAIL=0 |

证据快照：c01 /home/z00919396/zr/logs/t75/t75-run13-evidence.txt。

## 实现链（全在两仓 dsv41_layerwise）

1. vllm-ascend：dsa_v41_forward 在 notify 后调用
   maybe_save_kv_layer_to_connector（V4.1 此前是唯一不调 save hook 的 attention 家族）。
2. LMCache-Ascend 适配器：wait-time store pass（0.30 异步调度下 metadata 在
   forward 后才绑定，hook 时机拿不到 store range——由 wait_for_pending_stores
   权威补派）；_op_owner=g{server_generation}- 前缀（op id / 快照 id 进程唯一）；
   server 按 (epoch,request,step,resource,direction) 单 owner 选主，落选 worker
   WARNING 跳过不 raise；快照 id 追加 owner 标签避免"reused with different
   content"；全链 INFO/WARNING 观测日志。
3. 测试：dsv41-layerwise 聚合 138 passed（含 S0 真实 store 5 用例，旧字节
   2 failed 反证）；vllm-ascend UT test_dsv41_layerwise_store_hook.py 5 passed。

## 已知边界（诚实记录）

- S0 每 (request, step, resource) 单 worker 存储当选：TP sibling 的分片不重复存
  （同一 key 服务端拒绝重复），落选者显式 WARNING。
- 快照 boundaries=()（C2 codec 缺席）：S0 快照 lookup 不可命中，仅存储闭环。
