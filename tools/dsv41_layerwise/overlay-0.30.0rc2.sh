#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# dsv41_layerwise v0.30.0rc2 镜像 overlay（t71）
#
# 用途：把 vllm-ascend dsv41_layerwise 分支（基于 vLLM main）装进
#       quay.io/ascend/vllm-ascend:v0.30.0rc2-a3 镜像容器内运行。
#       t64 的 10 层 backport（B1–B10）+ hc_pre 路由修复（vendor 包补齐）。
#
# 修复的根因（t64 blocker）：镜像 vllm_ascend 树自带
#   _cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so
#   （唯一带 aclnnHcPre 符号的库），stock 的 bootstrap_custom_op_env() 会把它
#   写进 ASCEND_CUSTOM_OPP_PATH；分支影子树缺这个 vendor 目录 → bootstrap 早退
#   → 扩展 dlsym 走 CANN 分库/libopapi.so → 无符号 → 首个真实前向
#   RuntimeError("aclnnHcPre or aclnnHcPreGetWorkspaceSize not in libopapi.so, ...")。
#   修法=把镜像 vendor 包整份复制进影子分支树，bootstrap 与 stock 完全同路。
#
# 全部步骤幂等：重复执行直接跳过已应用层。
set -euo pipefail

BRANCH_DIR="${BRANCH_DIR:-/workspace/code/vllm-ascend-dsv41_layerwise}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
IMAGE_VA_DIR="${IMAGE_VA_DIR:-/vllm-workspace/vllm-ascend/vllm_ascend}"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$KIT_DIR/payload"

log() { echo "[overlay] $*"; }
die() { echo "[overlay][FATAL] $*" >&2; exit 1; }

[ -d "$BRANCH_DIR/vllm_ascend" ] || die "分支树不存在: $BRANCH_DIR（export BRANCH_DIR=...）"
[ -d "$VLLM_DIR/vllm" ] || die "镜像 vllm 树不存在: $VLLM_DIR（export VLLM_DIR=...）"
[ -d "$IMAGE_VA_DIR" ] || die "镜像 vllm_ascend 树不存在: $IMAGE_VA_DIR（export IMAGE_VA_DIR=...）"
[ -d "$PAYLOAD" ] || die "payload 目录不存在: $PAYLOAD"

# ------------------------------------------------ B11/B13: 源码内签名兼容 --
# vLLM 0.30 的 GPUModelRunner.profile_run() 无 randomize_inputs 形参、
# CompilationConfig.adjust_cudagraph_sizes_for_spec_decode() 仍是双参形态；
# 两处兼容已直接落在仓库源码 vllm_ascend/worker/v1/model_runner.py
# （try/except TypeError 窄回退），随分支树 rsync 生效，无需容器内打补丁。

# ---------------------------------------------------------------- hc_pre 修复 --
VENDOR_SRC="$IMAGE_VA_DIR/_cann_ops_custom/vendors/custom_transformer"
VENDOR_DST="$BRANCH_DIR/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
if [ -d "$VENDOR_DST/op_api/lib" ] && nm -D "$VENDOR_DST/op_api/lib/libcust_opapi.so" 2>/dev/null | grep -q aclnnHcPre; then
    log "hc_pre 修复已应用（vendor 包已在分支树）"
else
    [ -f "$VENDOR_SRC/op_api/lib/libcust_opapi.so" ] || die "镜像 vendor 库缺失: $VENDOR_SRC"
    mkdir -p "$BRANCH_DIR/vllm_ascend/_cann_ops_custom/vendors"
    cp -a "$VENDOR_SRC" "$VENDOR_DST"
    nm -D "$VENDOR_DST/op_api/lib/libcust_opapi.so" | grep -q aclnnHcPre \
        || die "vendor 复制完成但 aclnnHcPre 符号校验失败"
    log "hc_pre 修复完成：vendor custom_transformer 已复制进分支树并校验 aclnnHcPre"
fi

# ------------------------------------------------------------------- B1 + B10 --
B1_DST="$VLLM_DIR/vllm/v1/core/kv_cache_planning.py"
if [ -f "$B1_DST" ]; then
    log "B1 已应用: $B1_DST"
else
    cp "$PAYLOAD/kv_cache_planning.py" "$B1_DST"
    log "B1+B10 完成: kv_cache_planning 别名 + VllmConfig.aux_output_config 注入 → $B1_DST"
fi

# ------------------------------------------------------------------------ B2 --
B2_DST="$VLLM_DIR/vllm/models/deepseek_v41/common/engram.py"
if grep -q "def engram_head_shard_rank" "$B2_DST"; then
    log "B2 已应用: engram_head_shard_rank 已存在"
else
    grep -q "get_tensor_model_parallel_rank" "$B2_DST" \
        || die "镜像 engram.py 缺 get_tensor_model_parallel_rank（结构变化，需人工核对）"
    cat "$PAYLOAD/engram_append.py" >> "$B2_DST"
    log "B2 完成: engram_head_shard_rank 追加 → $B2_DST"
fi

# ------------------------------------------------------------------------ B3 --
B3_DST="$VLLM_DIR/vllm/transformers_utils/deepseek_v4_utils.py"
if [ -f "$B3_DST" ]; then
    log "B3 已应用: deepseek_v4_utils.py 已存在"
else
    cp "$PAYLOAD/deepseek_v4_utils.py" "$B3_DST"
    log "B3 完成: get_compress_ratios → $B3_DST"
fi

# ------------------------------------------------------------------------ B6 --
B6_DST="$VLLM_DIR/vllm/model_executor/layers/fusion/mm_input_norm.py"
if [ -f "$B6_DST" ]; then
    log "B6 已应用: mm_input_norm.py 已存在"
else
    cp "$PAYLOAD/mm_input_norm.py" "$B6_DST"
    log "B6 完成: FusedMMInputNorm → $B6_DST"
fi

# --------------------------------------------------------------------- B7 + B8 --
B7_DST="$VLLM_DIR/vllm/v1/worker/gpu/spec_decode/target_dependent_ar"
if [ -f "$B7_DST/speculator.py" ]; then
    log "B7+B8 已应用: target_dependent_ar 已存在"
else
    mkdir -p "$B7_DST"
    cp "$PAYLOAD/target_dependent_ar/__init__.py" "$B7_DST/__init__.py"
    cp "$PAYLOAD/target_dependent_ar/speculator.py" "$B7_DST/speculator.py"
    cp "$PAYLOAD/target_dependent_ar/cudagraph_utils.py" "$B7_DST/cudagraph_utils.py"
    log "B7+B8 完成: target_dependent_ar shim 包（autoregressive 真类再导出）→ $B7_DST"
fi

# ------------------------------------------------------------------------ B9 --
B9_DST="$VLLM_DIR/vllm/v1/worker/gpu/sample/gumbel.py"
if grep -q "_log1p_neg_stable" "$B9_DST"; then
    log "B9 已应用: _log1p_neg_stable 已存在"
else
    cat "$PAYLOAD/gumbel_append.py" >> "$B9_DST"
    log "B9 完成: _log1p_neg_stable（自带 triton import）→ $B9_DST"
fi

# ------------------------------------------------------------------- B5 探测 --
# CANN python3 site-packages/acl 路径（serve 脚本消费）
CANN_PY="$(find /usr/local/Ascend/cann-* -maxdepth 4 -type d -path "*python*/site-packages" 2>/dev/null | head -1 || true)"
if [ -n "$CANN_PY" ]; then
    echo "$CANN_PY" > "$KIT_DIR/.cann-python-path"
    log "B5 探测: CANN python 路径 = $CANN_PY（serve 脚本会追加进 PYTHONPATH）"
else
    log "B5 探测: 未找到 CANN python site-packages（若 import acl 失败需人工定位）"
fi

# ------------------------------------------------------------------ 自检 probes --
# 只做文件/字节级校验，不 import vllm（避免无 NPU 环境误报）。
log "自检：字节级校验"
grep -q "aclnnHcPre" <(nm -D "$VENDOR_DST/op_api/lib/libcust_opapi.so") || die "自检失败: vendor 库无 aclnnHcPre"
for f in "$B1_DST" "$B3_DST" "$B6_DST" "$B7_DST/speculator.py" "$B7_DST/cudagraph_utils.py"; do
    [ -s "$f" ] || die "自检失败: $f 为空"
done
grep -q "def engram_head_shard_rank" "$B2_DST" || die "自检失败: B2"
grep -q "_log1p_neg_stable" "$B9_DST" || die "自检失败: B9"
grep -q "DSV41LayerwiseMPConnector" "$BRANCH_DIR/vllm_ascend/models/deepseek_v41/cache_config.py" \
    || log "提示: cache_config.py 未引用 DSV41LayerwiseMPConnector（kv_transfer 由 lmcache_ascend 侧提供，正常）"
log "overlay 全部完成 ✔"
