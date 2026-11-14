#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# t120 采样启动：复用 serve-layerwise-0.30.0rc2.sh 的全部配置，
# 仅注入采样环境（DSV41_SAMPLING=1 + 目录）。
#
# 采样目录挂到 /workspace/logs/t120/sampling（宿主 zr/logs/t120/sampling），
# 容器内三个 tag 的 jsonl（engine/adapter/server）都会落在这里。
set -euo pipefail

export DSV41_SAMPLING=1
export DSV41_SAMPLING_DIR=/workspace/logs/t120/sampling
mkdir -p "$DSV41_SAMPLING_DIR"

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$KIT_DIR/serve-layerwise-0.30.0rc2.sh"
