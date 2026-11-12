#!/usr/bin/env python3
"""In-image self check for the t81 S1 wiring (real transport codec).

Runs inside ``quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`` with the three trees
mounted read-only:

    /ws/repos/vllm-ascend          the engine under test (PYTHONPATH)
    /ws/repos/LMCache              lmcache (wire protocol, no install needed)
    /ws/repos/LMCache-Ascend       lmcache_ascend (real boundary codec)

What it proves, on the real package import path:

1. ``vllm_ascend.attention.dsv41_layerwise_c2`` imports and the provider is
   installed on a layerwise-shaped connector (and is a no-op elsewhere).
2. A capture from a real-shaped ratio-2 layer records the rows the forward
   wrote; the provider returns ``(prefix_end % ring_rows, phase, rows)``.
3. The real ``encode_boundary_rows`` accepts those rows and produces a
   ``BoundaryStateDescriptor`` whose resource id is the compressor state
   cache of the execution layer and whose protocol checksum verifies.
4. ``decode_boundary_rows`` round-trips the real bytes, and rebasing them onto
   another request's ring page reproduces the captured rows bit for bit.
5. A stale boundary token and a missing capture stay cold (``None``), and a
   restore miss raises instead of silently continuing.

Exit code 0 and ``IMAGE_SELF_CHECK_T81_PASSED`` on success.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

WS = Path("/ws")
LMCACHE_TREE = WS / "repos" / "LMCache" / "lmcache"
ASCEND_TREE = WS / "repos" / "LMCache-Ascend" / "lmcache_ascend"
CODEC_PATH = ASCEND_TREE / "integration" / "vllm" / "dsv41_boundary_codec.py"

RING_ROWS = 32
HEAD_DIM = 8
EXECUTION_LAYER = "language_model.model.layers.0.attn"
STATE_PREFIX = f"{EXECUTION_LAYER}.compressor.state_cache"
LAYER_NAME = "language_model.model.layers.0.self_attn.v41_attn"


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def _load_codec() -> tuple[types.ModuleType, types.ModuleType]:
    _namespace("lmcache", LMCACHE_TREE)
    _namespace("lmcache.v1", LMCACHE_TREE / "v1")
    _namespace("lmcache.v1.multiprocess", LMCACHE_TREE / "v1" / "multiprocess")
    protocol = importlib.import_module("lmcache.v1.multiprocess.layerwise_protocol")
    _namespace("lmcache_ascend", ASCEND_TREE)
    _namespace("lmcache_ascend.integration", ASCEND_TREE / "integration")
    _namespace("lmcache_ascend.integration.vllm", ASCEND_TREE / "integration" / "vllm")
    spec = importlib.util.spec_from_file_location(
        "lmcache_ascend.integration.vllm.dsv41_boundary_codec", CODEC_PATH
    )
    assert spec is not None and spec.loader is not None
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    return protocol, codec


class _Meta:
    def __init__(self, prefix_end: int, used: int, page: int) -> None:
        control = torch.zeros(5, 1, dtype=torch.int32)
        control[0, 0] = prefix_end
        control[1, 0] = used
        control[4, 0] = page
        self.c2_ring_metadata = control
        self.block_table = torch.tensor([[page]], dtype=torch.int64)
        self.num_reqs = 1
        self.compress_ratio = 2
        self.slot_mapping = torch.tensor(
            [page * RING_ROWS + (prefix_end + i) % RING_ROWS for i in range(used)],
            dtype=torch.int64,
        )


class _StateCache:
    def __init__(self, pages: int = 4) -> None:
        self.kv_cache = [torch.zeros(pages, RING_ROWS, 1, 2 * HEAD_DIM, dtype=torch.float32)]


class _Compressor:
    def __init__(self) -> None:
        self.state_cache = _StateCache()


class _Impl:
    layer_name = "language_model.model.layers.0.self_attn"
    compressor_state_prefix = STATE_PREFIX


class _Attn:
    layerwise_execution_layer = EXECUTION_LAYER

    def __init__(self) -> None:
        self.v41_impl = _Impl()
        self.compressor = _Compressor()


class _Worker:
    def __init__(self) -> None:
        self.installed = None

    def set_boundary_provider(self, provider) -> None:
        self.installed = provider


class _Connector:
    def __init__(self) -> None:
        self.worker_adapter = _Worker()


def main() -> int:
    from vllm_ascend.attention import dsv41_layerwise_c2 as c2

    print("IMPORT_OK", type(c2).__name__)

    protocol, codec = _load_codec()
    connector = _Connector()
    assert c2.install_boundary_provider(connector) is True
    assert connector.worker_adapter.installed is c2.provide_boundary_rows
    assert c2.install_boundary_provider(object()) is False
    print("PROVIDER_OK")

    meta = _Meta(prefix_end=64, used=3, page=3)
    attn = _Attn()
    for offset, slot in enumerate(range(0, 3)):
        attn.compressor.state_cache.kv_cache[0][3, slot, 0, :] = torch.arange(
            2 * HEAD_DIM, dtype=torch.float32
        ) + offset

    import vllm.forward_context as forward_context

    context = types.SimpleNamespace(attn_metadata={STATE_PREFIX: meta})
    original = forward_context.get_forward_context
    forward_context.get_forward_context = lambda: context
    try:
        c2.reset_step_state()
        c2.bind_step_request_view(
            c2.DSV41StepRequestView(
                step_id=5, request_ids=("req-a",), generation=0, synthetic=False
            )
        )
        c2.capture_layer_boundary_state(LAYER_NAME, attn)
    finally:
        pass

    rows = tuple(3 * RING_ROWS + slot for slot in range(3))
    answer = c2.provide_boundary_rows(STATE_PREFIX, "snapshot-5", 67)
    assert answer == (64 % RING_ROWS, 64 % 2, rows), answer
    ring_cursor, phase, row_indices = answer
    print("CAPTURE_OK", row_indices, ring_cursor, phase)

    flat = attn.compressor.state_cache.kv_cache[0].reshape(
        attn.compressor.state_cache.kv_cache[0].shape[0] * RING_ROWS, 2 * HEAD_DIM
    )
    descriptor, payload = codec.encode_boundary_rows(
        flat,
        list(row_indices),
        snapshot_id="snapshot-5",
        resource_id=STATE_PREFIX,
        boundary_token="67",
        ring_cursor=ring_cursor,
        compression_phase=phase,
    )
    assert descriptor.resource_id == STATE_PREFIX, descriptor
    assert descriptor.ring_cursor == 0 and descriptor.compression_phase == 0, descriptor
    assert descriptor.checksum == protocol.compute_boundary_checksum(
        descriptor.snapshot_id,
        descriptor.resource_id,
        descriptor.boundary_token,
        descriptor.ring_cursor,
        descriptor.compression_phase,
        descriptor.row_indices,
        descriptor.schema_version,
    )
    print("DESCRIPTOR_OK", descriptor.resource_id, descriptor.boundary_token, descriptor.row_indices)

    decoded = codec.decode_boundary_rows(payload)
    captured = flat[list(row_indices)].clone()
    assert torch.equal(decoded["rows"].to(torch.float32), captured)

    restored = torch.full_like(flat, -1.0)
    # Rebase onto page 0 of the same engine (a different request's ring page).
    target_rows = [index - 3 * RING_ROWS for index in row_indices]
    restored[target_rows] = decoded["rows"].to(torch.float32)
    assert torch.equal(restored[target_rows], captured)
    print("ROUND_TRIP_OK", len(row_indices))

    # Cold on a stale token, cold on a resource with no capture.
    assert c2.provide_boundary_rows(STATE_PREFIX, "snapshot-5", 68) is None
    assert c2.provide_boundary_rows("other.compressor.state_cache", "s", 67) is None
    print("COLD_ON_MISMATCH_OK")

    class _MissCodec:
        def capture_layer_boundary(self, request_view, tensor_view, *, stream=None):
            raise KeyError("capture unavailable")

        def restore_layer_boundary(self, request_view, tensor_view, *, stream=None):
            raise KeyError("no leased boundary for this request")

    c2.set_boundary_state_codec(_MissCodec())
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=6, request_ids=("req-b",), generation=0, synthetic=False
        )
    )
    failed = False
    try:
        c2.restore_layer_boundary_state(LAYER_NAME, attn)
    except c2.DSV41LayerwiseC2Error:
        failed = True
    finally:
        c2.clear_boundary_state_codec()
        c2.reset_step_state()
        forward_context.get_forward_context = original
    assert failed, "a restore miss must fail closed"
    print("FAIL_CLOSED_OK")

    print("IMAGE_SELF_CHECK_T81_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
