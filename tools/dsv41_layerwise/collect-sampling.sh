#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# t120 采样汇总（容器内或宿主上跑均可；输入=sampling/*.jsonl）。
#
# 输出（stdout，写回日志目录由调用方重定向）：
#   1. 每 tag 的事件计数表（event -> count）
#   2. store_payload / boundary_store / restore_* 的 bytes 统计
#      （payload 纯 tensor bytes 与 wire bytes 分开）
#   3. 层覆盖表：swa_write 的 layer 去重计数
#   4. 请求覆盖表：step_schedule 的 request_id 去重计数
#   5. 失败面：bridge drop / boundary_cold / lookup_miss 计数
set -uo pipefail
DIR="${1:-/workspace/logs/t120/sampling}"
python3 - "$DIR" <<'PYEOF'
import collections
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("sampling-*.jsonl"))
events = collections.Counter()
bytes_payload = []
bytes_wire = []
layers = collections.Counter()
requests = collections.Counter()
fail = collections.Counter()
for path in files:
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            events[rec.get("event", "?")] += 1
            ev = rec.get("event")
            if ev == "store_payload":
                bytes_payload.append(int(rec.get("tensor_bytes", 0)))
                bytes_wire.append(int(rec.get("wire_bytes", 0)))
            elif ev == "boundary_store":
                bytes_payload.append(int(rec.get("wire_bytes", 0)))
                bytes_wire.append(int(rec.get("wire_bytes", 0)))
            elif ev == "swa_write":
                layers[rec.get("layer", "")] += 1
            elif ev == "step_schedule" and rec.get("request_id"):
                requests[rec["request_id"]] += 1
            elif ev in ("bridge_resource",) and rec.get("outcome") not in (
                "loaded",
                "already_served",
            ):
                fail[f"bridge:{rec.get('outcome')}"] += 1
            elif ev == "boundary_cold":
                fail["boundary_cold"] += 1
            elif ev == "lookup_miss":
                fail["lookup_miss"] += 1

print("== files ==")
for f in files:
    print(f"  {f.name}: {sum(1 for _ in f.open())} lines")
print("== events ==")
for name, count in events.most_common():
    print(f"  {name}: {count}")
if bytes_payload:
    print("== bytes ==")
    print(f"  payload sum: {sum(bytes_payload)}")
    print(f"  wire    sum: {sum(bytes_wire)}")
    print(f"  payload avg: {sum(bytes_payload)//len(bytes_payload)}")
    print(f"  wire    avg: {sum(bytes_wire)//len(bytes_wire)}")
print("== swa_write layers ==")
for name, count in sorted(layers.items()):
    print(f"  {name or '(unnamed)'}: {count}")
print("== step requests ==")
for name, count in requests.most_common(20):
    print(f"  {name}: {count}")
print("== failure surface ==")
for name, count in fail.most_common():
    print(f"  {name}: {count}")
PYEOF
