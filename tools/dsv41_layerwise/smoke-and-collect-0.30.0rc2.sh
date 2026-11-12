#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# dsv41_layerwise t71 smoke：就绪轮询 + 双请求 + 证据收集
#
# 证据模式（验收②）：
#   [dsv41_layerwise] admission registry bound     —— 注册链完成（plan/admission 结构就绪）
#   [dsv41_layerwise] retrieved KV ready for layer —— wait_for_layer_load 真实执行
#   [dsv41_layerwise] admission granted for layer  —— guard 真实执行
#   [dsv41_layerwise] KV produced for layer        —— notify/store 调度点真实执行
#   无 NotImplementedError / 无 aclnnHcPre 报错     —— 前两层验收的否定证据
set -euo pipefail

# SMOKE_COLD_ONLY=1 (t75): one long unique cold request whose prompt spans at
# least one full LMCache chunk, so the stock scheduler emits a real STORE op
# and the S0 wire must carry real KV bytes. Evidence then asserts: store
# submitted -> dispatched (bytes>0) -> snapshot committed -> wait completed.
SMOKE_COLD_ONLY="${SMOKE_COLD_ONLY:-0}"
LOG_DIR="${LOG_DIR:-/workspace/logs/t71}"
SERVE_LOG="${SERVE_LOG:-$LOG_DIR/serve-t71.log}"
if [ "$SMOKE_COLD_ONLY" = 1 ]; then
    # ~700 distinct numeric tokens: comfortably above the default chunk size
    # (256), unique per run (time-seeded), no shared prefix with anything the
    # server may have seen.
    COLD_PROMPT="请逐个记住下面这批实验编号，稍后我会抽查复述。编号清单："
    COLD_PROMPT="$COLD_PROMPT$(python3 - "$LOG_DIR" <<'PYGEN'
import random
import sys

seed = int(__import__("time").time())
random.seed(seed)
nums = [f"{random.randint(10000, 99999)}-{i:03d}" for i in range(700)]
with open(f"{sys.argv[1]}/cold-prompt-seed.txt", "w") as fh:
    fh.write(str(seed))
print(" ".join(nums))
PYGEN
)"
    PROMPT="$COLD_PROMPT"
else
    PROMPT='用一句话介绍你自己'
fi
export LOG_DIR

echo "[smoke] 轮询 /v1/models（最长 40 分钟）..."
READY=0
for i in $(seq 1 80); do
    if curl -s -o /dev/null -w '%{http_code}' --max-time 30 http://127.0.0.1:18000/v1/models | grep -q 200; then
        READY=1
        echo "[smoke] 服务就绪（第 $i 次轮询，$(date +%H:%M:%S)）"
        break
    fi
    # 提前发现致命错误
    if grep -qE "aclnnHcPre|NotImplementedError|DSV41LayerwiseAdmissionError|ValueError|RuntimeError" "$SERVE_LOG" 2>/dev/null; then
        if grep -qE "aclnnHcPre|NotImplementedError|DSV41LayerwiseAdmissionError" "$SERVE_LOG"; then
            echo "[smoke][FATAL] serve 日志出现致命错误，提前终止" >&2
            grep -m5 -E "aclnnHcPre|NotImplementedError|DSV41LayerwiseAdmissionError" "$SERVE_LOG" >&2 || true
            exit 2
        fi
    fi
    sleep 30
done
[ "$READY" = 1 ] || { echo "[smoke][FATAL] 40 分钟未就绪" >&2; exit 1; }

echo "[smoke] 请求 A（cold：store 路径）..."
curl -s --max-time 300 http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":48,\"temperature\":0}" \
  | tee "$LOG_DIR/smoke-a.json"
echo

if [ "$SMOKE_COLD_ONLY" = 1 ]; then
    echo "[smoke] SMOKE_COLD_ONLY=1：跳过请求 B（本模式只验证单冷请求真实 STORE 链）"
else
    sleep 5
    echo "[smoke] 请求 B（同 prompt：prefix 命中，retrieve 路径）..."
    curl -s --max-time 300 http://127.0.0.1:18000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":48,\"temperature\":0}" \
      | tee "$LOG_DIR/smoke-b.json"
    echo
fi

# ---- 证据收集 ----
{
    echo "===== EVIDENCE $(date) ====="
    echo "--- 验收①：两请求输出非空 ---"
    python3 - <<'PY'
import json

# The 0.30 lane's deepseek_v41 chat template puts the generated tokens in
# ``reasoning`` (a reasoning model), with ``content`` staying empty while the
# budget is consumed by thoughts. Probe every text carrier the server may use
# instead of assuming the OpenAI ``content`` field (t71 evidence fix).
import os

TEXT_KEYS = ("reasoning", "reasoning_content", "content", "text")
tags = ("a",) if os.environ.get("SMOKE_COLD_ONLY") == "1" else ("a", "b")
for tag in tags:
    try:
        with open(f"{os.environ['LOG_DIR']}/smoke-{tag}.json") as f:
            d = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"smoke-{tag}: FAILED to parse: {exc}")
        continue
    if "error" in d:
        print(f"smoke-{tag}: FAILED server error: {d['error']}")
        continue
    choice = d["choices"][0]
    message = choice.get("message") or {}
    carrier, text = None, ""
    for key in TEXT_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            carrier, text = key, value
            break
    usage = d.get("usage") or {}
    verdict = "PASS" if text.strip() else "FAIL"
    print(
        f"smoke-{tag}: {verdict} id={d.get('id')} finish={choice.get('finish_reason')} "
        f"carrier={carrier} len(text)={len(text)} "
        f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}"
    )
    print(f"  text[:80]={text[:80]!r}")
PY
    echo
    echo "--- 验收②：admission 生命周期证据（四类首见日志计数） ---"
    for pat in "admission registry bound" "retrieved KV ready for layer" "admission granted for layer" "KV produced for layer"; do
        echo "  [$pat] = $(grep -c "$pat" "$SERVE_LOG" || true)"
    done
    echo
    echo "--- 否定证据（应为 0） ---"
    for pat in "NotImplementedError" "aclnnHcPre" "DSV41LayerwiseAdmissionError" "has never opened an admission window"; do
        echo "  [$pat] = $(grep -c "$pat" "$SERVE_LOG" || true)"
    done
    echo
    echo "--- 样本日志行 ---"
    grep -m3 "admission registry bound" "$SERVE_LOG" || true
    grep -m3 "retrieved KV ready" "$SERVE_LOG" || true
    grep -m3 "admission granted" "$SERVE_LOG" || true
    grep -m3 "KV produced" "$SERVE_LOG" || true
    echo
    echo "--- lmc server 侧 plan/registration 证据（验收②：plan 生成） ---"
    LMC_LOG="$LOG_DIR/lmc-server.log"
    for pat in "Layerwise registration accepted" "Registered KV cache"; do
        echo "  [$pat] = $(grep -c "$pat" "$LMC_LOG" 2>/dev/null || true)"
    done
    grep "Layerwise registration accepted" "$LMC_LOG" 2>/dev/null | tail -2 \
        | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-200 || true
    echo
    echo "--- lmc server 侧请求级 RPC 活动（验收②：enqueue 真实到达 server） ---"
    grep -iE "lookup|retrieve|transfer|has no lookup ipc key" "$LMC_LOG" 2>/dev/null | tail -5 \
        | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-200 || true
    echo
    if [ "$SMOKE_COLD_ONLY" = 1 ]; then
        echo "--- t75：S0 真实 STORE 链证据（store submitted -> dispatched -> committed） ---"
        SUBMITTED=$(grep -c "store submitted" "$SERVE_LOG" || true)
        DISPATCHED=$(grep -c "store dispatched over the wire" "$SERVE_LOG" || true)
        COMMITTED=$(grep -c "store snapshot committed" "$SERVE_LOG" || true)
        BYTES_SUM=$(grep -o "bytes=[0-9]*" "$SERVE_LOG" | cut -d= -f2 \
            | awk '{s+=$1} END {print s+0}')
        NO_SPEC_HOLE=$(grep -c "has no pending transfers to dispatch" "$SERVE_LOG" || true)
        echo "  [store submitted]        = $SUBMITTED"
        echo "  [store dispatched]       = $DISPATCHED"
        echo "  [store snapshot committed] = $COMMITTED"
        echo "  [store bytes total]      = $BYTES_SUM"
        echo "  [no-pending debug lines] = $NO_SPEC_HOLE (informational)"
        echo "  样本行："
        grep -m3 "store dispatched over the wire" "$SERVE_LOG" || true
        grep -m2 "store snapshot committed" "$SERVE_LOG" || true
        STORE_FAIL=0
        [ "$DISPATCHED" -ge 1 ] || { echo "  [FAIL] 无任何 store 帧过线"; STORE_FAIL=1; }
        [ "$BYTES_SUM" -gt 0 ] || { echo "  [FAIL] store 字节总数为 0"; STORE_FAIL=1; }
        [ "$COMMITTED" -ge 1 ] || { echo "  [FAIL] 无 snapshot commit"; STORE_FAIL=1; }
        if [ "$STORE_FAIL" = 0 ]; then
            echo "  t75 STORE 链：PASS"
        else
            echo "  t75 STORE 链：FAIL"
        fi
        echo "SMOKE_COLD_ONLY_STORE_FAIL=$STORE_FAIL" > "$LOG_DIR/store-verdict.txt"
    fi
} | tee "$LOG_DIR/evidence.txt"

# The evidence block above runs inside the tee pipeline, so its verdict is
# carried through the file; surface it as this script's exit code.
if [ "$SMOKE_COLD_ONLY" = 1 ]; then
    STORE_FAIL="$(cut -d= -f2 "$LOG_DIR/store-verdict.txt" 2>/dev/null || echo 1)"
    echo "[smoke] SMOKE_COLD_ONLY store verdict: $STORE_FAIL (0=PASS)"
    exit "$STORE_FAIL"
fi
