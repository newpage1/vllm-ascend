#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# t120 诊断采样三阶段运行（容器 dsh-apf-c01-t120 内执行）。
#
# 前置：serve-with-sampling-0.30.0rc2.sh 已起服（DSV41_SAMPLING=1），
#       18000 /v1/models 返回 200。
#
# 阶段（t119 §4.3）：
#   A cold    ：一条 ≥5k token 的 prompt（跨多个 256-token chunk 边界），
#               全部 chunk 为全新计算 → STORE 侧采样；
#   B warm    ：同前缀 + 短后缀 → scheduler lookup / RETRIEVE 采样；
#   C realloc ：长 decode（max_tokens=512）触发 SWA 滑窗/页重分配。
#
# 采样文件不落 prompt 正文：dsv41_sampling 只记 request_id 与几何量。
# 产物：/workspace/logs/t120/{phaseA,phaseB,phaseC}-response.json 与 sampling/。
set -uo pipefail

LOG=/workspace/logs/t120
mkdir -p "$LOG"

# 用 python 生成确定性 prompt；段落重复体天然跨 chunk 边界。
make_prompt() {
    python3 - "$1" <<'PYEOF'
import sys
mode = sys.argv[1]
para = ("昇腾推理性能工作台采样段落。模型在长前缀下逐层写入 KV 缓存，"
        "压缩层每两 token 聚合一行边界状态；跨 chunk 边界时页与滑窗重排，"
        "采样记录写入行与目标页。")
if mode == "long":
    text = para * 220          # ~6k+ tokens（中文≈1-2 char/token，保守取倍数）
elif mode == "suffix":
    text = para * 220 + "请用一句话总结上述内容。"
else:
    text = para
sys.stdout.write(text)
PYEOF
}

request() {  # request <phase> <max_tokens> <mode>
    local phase="$1" max_tokens="$2" mode="$3"
    make_prompt "$mode" > /tmp/t120-prompt.txt
    echo "$max_tokens" > /tmp/t120-max
    local body
    body=$(python3 - <<'PYEOF'
import json, pathlib
text = pathlib.Path("/tmp/t120-prompt.txt").read_text()
mt = int(pathlib.Path("/tmp/t120-max").read_text().strip())
print(json.dumps({
    "model": "dsv41",
    "messages": [{"role": "user", "content": text}],
    "max_tokens": mt,
    "temperature": 0,
}))
PYEOF
)
    local t0=$(date +%s)
    local code
    code=$(curl -s -o "$LOG/${phase}-response.json" -w "%{http_code}" \
        --max-time 600 \
        -H "Content-Type: application/json" \
        -d "$body" \
        http://127.0.0.1:18000/v1/chat/completions) || code="curl_err"
    local t1=$(date +%s)
    local chars=0
    [ -f "$LOG/${phase}-response.json" ] && chars=$(wc -c < "$LOG/${phase}-response.json")
    echo "[$phase] http=$code bytes=$chars elapsed=$((t1-t0))s"
    echo "$phase http=$code bytes=$chars elapsed=$((t1-t0))s" >> "$LOG/phases.txt"
}

echo "[t120] sampling dir: ${DSV41_SAMPLING_DIR:-/tmp/dsv41-sampling}"
echo "[t120] phase A (cold, >=5k tokens)"
request phaseA 64 long
sleep 5

echo "[t120] phase B (warm, same prefix + short suffix)"
request phaseB 64 suffix
sleep 5

echo "[t120] phase C (long decode -> SWA/page reassignment)"
request phaseC 512 long

echo "[t120] done; sampling files:"
ls -la "${DSV41_SAMPLING_DIR:-/tmp/dsv41-sampling}" | head -40
