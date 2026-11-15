# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side JSONL timeline for the layerwise v2 group-head seam.

This is the vllm-ascend half of the default-off mask timeline. It mirrors
``lmcache_ascend.integration.vllm.dsv41_lw_v2.mask_trace`` but cannot import
it (the engine package must stay importable without LMCache-Ascend), so it
writes its own ``mask-trace-<pid>-engine.jsonl`` next to the transport file.
Both sinks share the same record schema, so B01 can merge them by
``(request_id, step_id, group_id)`` or simply order by ``ts_ns``.

Only ``forward_start`` / ``forward_end`` are produced here: the connector owns
the load/store/wait events because it has the request and window facts. The
span between ``wait_exit`` and ``forward_end`` is the mask this tracing
measures -- how much of a group load's wait is hidden behind the previous
layer's forward.

The switch is read once per process and the sink is inert unless
``DSV41_LW_V2_MASK_TRACE`` is exactly ``1``. A missing/unwritable
``DSV41_LW_V2_MASK_TRACE_DIR`` disables tracing with one warning; nothing here
raises into the forward path, calls a device synchronisation primitive or a
blocking RPC.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_ENABLED = "DSV41_LW_V2_MASK_TRACE"
ENV_DIR = "DSV41_LW_V2_MASK_TRACE_DIR"
ENV_MAX_LINES = "DSV41_LW_V2_MASK_TRACE_MAX_LINES"
ENV_MAX_LINES_PER_REQUEST = "DSV41_LW_V2_MASK_TRACE_MAX_LINES_PER_REQUEST"

DEFAULT_DIR = "/tmp/dsv41-lw-v2-mask-trace"
DEFAULT_MAX_LINES = 2_000_000
DEFAULT_MAX_LINES_PER_REQUEST = 20_000

_lock = threading.Lock()
_enabled: bool | None = None
_stream = None
_warned = False
_seq = 0
_lines = 0
_per_request: dict[str, int] = {}
_truncated: set[str] = set()
_stats: dict[str, int] = {
    "lines": 0,
    "dropped": 0,
    "truncated_requests": 0,
}


def reset_state() -> None:
    """Forget the cached switch and close the sink (tests)."""
    global _enabled, _stream, _warned, _seq, _lines
    with _lock:
        stream = _stream
        _stream = None
        _enabled = None
        _warned = False
        _seq = 0
        _lines = 0
        _per_request.clear()
        _truncated.clear()
        for name in _stats:
            _stats[name] = 0
    if stream is not None:
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - teardown only
            pass


def enabled() -> bool:
    """Whether the process opted into the timeline (single env read)."""
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get(ENV_ENABLED, "") == "1"
    return _enabled


def now_ns() -> int:
    """Monotonic timestamp used by every trace record."""
    return time.monotonic_ns()


def _limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _open_locked() -> bool:
    global _stream, _warned, _enabled

    directory = Path(os.environ.get(ENV_DIR, DEFAULT_DIR))
    if not directory.is_dir():
        if not _warned:
            _warned = True
            logger.warning(
                "dsv41 lw v2 engine mask trace disabled: %s=%s is not an "
                "existing directory",
                ENV_DIR,
                directory,
            )
        _enabled = False
        return False
    path = directory / f"mask-trace-{os.getpid()}-engine.jsonl"
    try:
        _stream = path.open("a", encoding="utf-8", buffering=1)
    except OSError as exc:
        if not _warned:
            _warned = True
            logger.warning(
                "dsv41 lw v2 engine mask trace disabled: cannot open %s: %s",
                path,
                exc,
            )
        _enabled = False
        return False
    return True


def emit(event: str, **fields) -> None:
    """Append one record when tracing is enabled, else return immediately."""
    global _seq, _lines, _warned

    if not enabled():
        return
    with _lock:
        if _stream is None and not _open_locked():
            return
        request_id = fields.get("request_id")
        request_key = request_id if isinstance(request_id, str) else None
        if request_key is not None:
            limit = _limit(
                ENV_MAX_LINES_PER_REQUEST, DEFAULT_MAX_LINES_PER_REQUEST
            )
            count = _per_request.get(request_key, 0)
            if count >= limit:
                if request_key not in _truncated:
                    _truncated.add(request_key)
                    _stats["truncated_requests"] += 1
                    _write_locked(
                        {
                            "ts_ns": time.monotonic_ns(),
                            "seq": _seq,
                            "pid": os.getpid(),
                            "event": "trace_truncated",
                            "phase": "trace",
                            "request_id": request_key,
                            "limit": limit,
                        }
                    )
                    _seq += 1
                _stats["dropped"] += 1
                return
            _per_request[request_key] = count + 1
        total = _limit(ENV_MAX_LINES, DEFAULT_MAX_LINES)
        if _lines >= total:
            if not _warned:
                _warned = True
                logger.warning(
                    "dsv41 lw v2 engine mask trace reached %s=%s lines; "
                    "further records are dropped",
                    ENV_MAX_LINES,
                    total,
                )
            _stats["dropped"] += 1
            return
        record = {
            "ts_ns": time.monotonic_ns(),
            "seq": _seq,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        _seq += 1
        _lines += 1
        _stats["lines"] += 1
        _write_locked(record)


def _write_locked(record: dict) -> None:
    global _stream
    stream = _stream
    if stream is None:
        return
    try:
        stream.write(json.dumps(record, separators=(",", ":"), default=str))
        stream.write("\n")
    except Exception:  # noqa: BLE001 - tracing must never break serving
        try:
            stream.close()
        finally:
            _stream = None
