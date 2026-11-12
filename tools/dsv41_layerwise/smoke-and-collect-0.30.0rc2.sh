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

LOG_DIR="${LOG_DIR:-/workspace/logs/t71}"
SERVE_LOG="$LOG_DIR/serve-t71.log"
PROMPT='用一句话介绍你自己'

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

sleep 5
echo "[smoke] 请求 B（同 prompt：prefix 命中，retrieve 路径）..."
curl -s --max-time 300 http://127.0.0.1:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":48,\"temperature\":0}" \
  | tee "$LOG_DIR/smoke-b.json"
echo

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
TEXT_KEYS = ("reasoning", "reasoning_content", "content", "text")
for tag in ("a", "b"):
    try:
        with open(f"/workspace/logs/t71/smoke-{tag}.json") as f:
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
} | tee "$LOG_DIR/evidence.txt"
