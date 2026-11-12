#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# dsv41_layerwise t71 smoke：镜像 v0.30.0rc2-a3 容器内全链启动
#
# 前置（容器内已完成）：
#   1. overlay-0.30.0rc2.sh 已应用（B1–B10 + hc_pre vendor 修复）
#   2. LMCache / LMCache-Ascend 已 pip install -e（setup 脚本）
#   3. lmcache MP server 已在 127.0.0.1:6555 监听（start-lmc-server.sh）
#
# 本脚本 = t64 attempt13 serve 配置 + 两处修正：
#   - kv_connector_extra_config 增加 lmcache.mp.server_urls=tcp://127.0.0.1:6555
#     （t64 缺省连 localhost:5555 死端口；S0 传输缝需要活 server）
#   - PYTHONPATH 追加 CANN python site-packages（B5，import acl）
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export PYTHONPATH="/workspace/code/vllm-ascend-dsv41_layerwise:$PYTHONPATH"
# B5：CANN python 绑定路径（overlay 脚本探测产物；存在才追加）
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$KIT_DIR/.cann-python-path" ]; then
    CANN_PY="$(cat "$KIT_DIR/.cann-python-path")"
    export PYTHONPATH="$CANN_PY:$PYTHONPATH"
    echo "[serve] PYTHONPATH += $CANN_PY"
fi

export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S=36000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV

MODEL_PATH=/opt/data/models/DeepSeek-V4.1-Flash-1017-w8a8

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18000 \
  --max-model-len 150000 --max-num-batched-tokens 8192 \
  --served-model-name dsv41 \
  --gpu-memory-utilization 0.9 --max-num-seqs 32 \
  --data-parallel-size 4 --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --safetensors-load-strategy lazy --quantization ascend --block-size 128 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
  --async-scheduling \
  --engram-config "{\"cpu_offload\":true,\"dp_shared_memory\":true}" \
  --additional-config "{\"ascend_compilation_config\":{\"enable_npugraph_ex\":true,\"enable_static_kernel\":false},\"enable_cpu_binding\":true,\"enable_fused_mc2\":1,\"enable_flashcomm1\":true,\"enable_shared_expert_dp\":true}" \
  --kv-transfer-config "{\"kv_connector\":\"DSV41LayerwiseMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_module_path\":\"lmcache_ascend.integration.vllm.dsv41_mp_connector\",\"kv_connector_extra_config\":{\"dsv41.layerwise\":true,\"lmcache.mp.server_urls\":\"tcp://127.0.0.1:6555\"}}"
