#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# dsv41_layerwise t71：容器内依赖安装 + lmcache MP server 启动
#
# 幂等：包已可导入则跳过安装；6555 已监听则跳过 server 启动。
set -euo pipefail

P=/usr/local/python3.12.13/bin/pip
LOG_DIR="${LOG_DIR:-/workspace/logs/t71}"
mkdir -p "$LOG_DIR"

# ---- sortedcontainers（镜像缺失，LMCache 依赖；wheel 由主机注入 /tmp） ----
if ! python3 -c "import sortedcontainers" 2>/dev/null; then
    WHEEL="$(ls /tmp/sortedcontainers-*.whl 2>/dev/null | head -1 || true)"
    if [ -n "$WHEEL" ]; then
        $P install "$WHEEL" >> "$LOG_DIR/setup.log" 2>&1
    else
        $P install sortedcontainers >> "$LOG_DIR/setup.log" 2>&1 || true
    fi
    echo "[setup] sortedcontainers 安装完成"
else
    echo "[setup] sortedcontainers 已存在"
fi

# ---- LMCache（editable 0.4.0） ----
if ! python3 -c "import lmcache" 2>/dev/null; then
    SETUPTOOLS_SCM_PRETEND_VERSION=0.4.0 $P install --no-build-isolation --no-deps \
        -e /workspace/code/LMCache-dsv41_layerwise >> "$LOG_DIR/setup.log" 2>&1
    echo "[setup] LMCache 安装完成"
else
    echo "[setup] LMCache 已可导入"
fi

# ---- LMCache-Ascend（editable 0.3.12，cmake 构建 c_ops） ----
if ! python3 -c "import lmcache_ascend" 2>/dev/null; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    rm -rf /workspace/code/LMCache-Ascend-dsv41_layerwise/build
    cd /workspace/code/LMCache-Ascend-dsv41_layerwise
    SETUPTOOLS_SCM_PRETEND_VERSION=0.3.12 $P install -v --no-build-isolation --no-deps \
        -e . >> "$LOG_DIR/install-lmc-ascend.log" 2>&1
    echo "[setup] LMCache-Ascend 安装完成"
else
    echo "[setup] LMCache-Ascend 已可导入"
fi

# ---- lmcache MP server（127.0.0.1:6555，chunk 256，L1 100GB，LRU） ----
if python3 -c "import socket; socket.create_connection(('127.0.0.1', 6555), 3).close()" 2>/dev/null; then
    echo "[setup] lmc server 已在 6555 监听"
else
    nohup python3 -m lmcache.v1.multiprocess.http_server \
        --instance-id t71 --port 6555 --http-port 8080 \
        --chunk-size 256 --l1-size-gb 100 --eviction-policy LRU \
        --disable-observability \
        >> "$LOG_DIR/lmc-server.log" 2>&1 &
    for i in $(seq 1 90); do
        sleep 2
        if python3 -c "import socket; socket.create_connection(('127.0.0.1', 6555), 3).close()" 2>/dev/null; then
            echo "[setup] lmc server 就绪（127.0.0.1:6555，日志 $LOG_DIR/lmc-server.log）"
            exit 0
        fi
    done
    echo "[setup][FATAL] lmc server 120s 未就绪，查 $LOG_DIR/lmc-server.log" >&2
    exit 1
fi
