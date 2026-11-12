#!/usr/bin/env python3
"""In-image self check for the t87 whole-page store/restore wiring.

Runs inside ``quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`` with the three trees
mounted read-only:

    /ws/repos/vllm-ascend          the engine under test (PYTHONPATH)
    /ws/repos/LMCache              lmcache (wire protocol, no install needed)
    /ws/repos/LMCache-Ascend       lmcache_ascend (real boundary codec)

What it proves, on the real package import path and with the real codec:

1. ``install_boundary_provider(connector, model=...)`` installs the real
   provider, attaches the two geometry hooks the transport looks up, and is a
   no-op for a connector that is not the layerwise transport.
2. The model's real ring geometry is bound (``ring_rows`` returns the cache's
   rows per page) and the adapter's boundary caches are pointed at absolute
   row views of the *same* storage; non-boundary caches keep their layout.
3. A store frames the whole ring page: written rows keep their bytes,
   unwritten rows travel as zero, the descriptor spans the page and its
   protocol checksum verifies.
4. ``restore_target`` names the new request's own page only when the live
   forward resumes exactly at the captured boundary; the decoded page lands
   there with no stale tail slots.
5. Every mismatch (token, batch, synthetic step, unknown geometry) stays
   fail-closed; nothing is fabricated.

Exit code 0 and ``IMAGE_SELF_CHECK_T87_PASSED`` on success.
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
LONG_KV = f"{EXECUTION_LAYER}.long_kv_cache"
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
    def __init__(self, prefix_end: int, used: int, page: int, num_reqs: int = 1) -> None:
        control = torch.zeros(5, num_reqs, dtype=torch.int32)
        control[0, :] = prefix_end
        control[1, :] = used
        control[4, :] = page
        self.c2_ring_metadata = control
        self.block_table = torch.tensor([[page]], dtype=torch.int64)
        self.num_reqs = num_reqs
        self.compress_ratio = 2
        self.max_query_len = used
        self.slot_mapping = torch.tensor(
            [
                page * RING_ROWS + (prefix_end + index) % RING_ROWS
                for index in range(max(used, 0))
            ],
            dtype=torch.int64,
        )


class _Ring:
    def __init__(self, pages: int = 6) -> None:
        self.kv_cache = [
            torch.zeros(pages, RING_ROWS, 1, 2 * HEAD_DIM, dtype=torch.float32)
        ]


class _Compressor:
    def __init__(self) -> None:
        self.state_cache = _Ring()


class _Impl:
    compressor_state_prefix: str | None = STATE_PREFIX
    layer_name = "language_model.model.layers.0.self_attn"


class _Attn:
    def __init__(self) -> None:
        self.v41_impl = _Impl()
        self.compressor = _Compressor()
        self.layerwise_execution_layer = EXECUTION_LAYER


class _Model:
    def __init__(self, layers: list[object]) -> None:
        self._layers = layers

    def modules(self):
        return iter(self._layers)


class _Worker:
    def __init__(self, kv_caches: dict[str, object]) -> None:
        self.kv_caches = kv_caches
        self.providers: list[object] = []

    def set_boundary_provider(self, provider: object) -> None:
        self.providers.append(provider)


class _Connector:
    def __init__(self, worker: _Worker) -> None:
        self.worker_adapter = worker


def _bind_step(c2: types.ModuleType, request_id: str, step_id: int, synthetic: bool = False) -> None:
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=step_id,
            request_ids=(request_id,),
            generation=0,
            synthetic=synthetic,
        )
    )


def main() -> int:
    protocol, codec = _load_codec()
    from vllm_ascend.attention import dsv41_layerwise_c2 as c2

    print("IMPORT_OK", c2.__name__)

    attn = _Attn()
    original = attn.compressor.state_cache.kv_cache[0]
    worker = _Worker({STATE_PREFIX: original, LONG_KV: torch.zeros(6, 128, 1, 2 * HEAD_DIM)})
    connector = _Connector(worker)
    assert c2.install_boundary_provider(connector, model=_Model([attn])) is True
    assert worker.providers == [c2.provide_boundary_rows]
    assert c2.provide_boundary_rows.ring_rows is c2.ring_rows
    assert c2.provide_boundary_rows.restore_target is c2.restore_target
    print("PROVIDER_OK", sorted(name for name in ("ring_rows", "restore_target") if hasattr(c2.provide_boundary_rows, name)))

    assert c2.ring_rows(STATE_PREFIX) == RING_ROWS
    row_space = worker.kv_caches[STATE_PREFIX]
    assert isinstance(row_space, torch.Tensor) and row_space.dim() == 2
    assert tuple(row_space.shape) == (6 * RING_ROWS, 2 * HEAD_DIM)
    assert row_space.data_ptr() == original.view(-1, 2 * HEAD_DIM).data_ptr()
    assert worker.kv_caches[LONG_KV].dim() == 4
    print("GEOMETRY_OK", RING_ROWS, tuple(row_space.shape))

    assert c2.install_boundary_provider(object()) is False

    for offset, slot in enumerate(range(6, 9)):
        original[2, slot, 0, :] = torch.arange(2 * HEAD_DIM) + offset
    meta = _Meta(prefix_end=70, used=3, page=2)
    import vllm.forward_context as forward_context

    forward_context.get_forward_context = lambda: types.SimpleNamespace(  # type: ignore[assignment]
        attn_metadata={STATE_PREFIX: meta}
    )
    _bind_step(c2, "req-a", 3)
    c2.capture_layer_boundary_state(LAYER_NAME, attn)
    record = c2.boundary_record(STATE_PREFIX)
    assert record is not None and record.end_token == 73
    print("CAPTURE_OK", record.ring_cursor, record.compression_phase, record.row_indices)

    descriptor, payload = codec.encode_boundary_rows(
        row_space,
        list(record.row_indices),
        snapshot_id="snap-70",
        resource_id=STATE_PREFIX,
        boundary_token=str(record.end_token),
        ring_cursor=record.ring_cursor,
        compression_phase=record.compression_phase,
        ring_length=c2.ring_rows(STATE_PREFIX),
    )
    assert descriptor.schema_version == 2
    assert descriptor.row_indices[0] == 2 * RING_ROWS
    assert len(descriptor.row_indices) == RING_ROWS
    assert descriptor.checksum == protocol.compute_boundary_checksum(
        descriptor.snapshot_id,
        descriptor.resource_id,
        descriptor.boundary_token,
        descriptor.ring_cursor,
        descriptor.compression_phase,
        descriptor.row_indices,
        descriptor.schema_version,
    )
    decoded = codec.decode_boundary_rows(payload)["rows"]
    assert torch.equal(decoded[6:9], original[2, 6:9, 0, :])
    assert torch.count_nonzero(decoded[:6]) == 0 and torch.count_nonzero(decoded[9:]) == 0
    print("STORE_OK", descriptor.schema_version, tuple(decoded.shape), len(payload))

    target_meta = _Meta(prefix_end=73, used=0, page=4)
    forward_context.get_forward_context = lambda: types.SimpleNamespace(  # type: ignore[assignment]
        attn_metadata={STATE_PREFIX: target_meta}
    )
    _bind_step(c2, "req-b", 4)
    assert c2.restore_target(STATE_PREFIX, "73") == (4, RING_ROWS)
    row_space[4 * RING_ROWS : 5 * RING_ROWS] = decoded
    assert torch.equal(original[4, 6:9, 0, :], original[2, 6:9, 0, :])
    assert torch.count_nonzero(original[4, :6, 0, :]) == 0
    assert torch.count_nonzero(original[4, 9:, 0, :]) == 0
    print("RESTORE_OK", 4, RING_ROWS)

    # Fail-closed: the snapshot does not apply to a request that resumes
    # elsewhere, a multi-request step owns no single page, and a synthetic
    # forward never restores.
    forward_context.get_forward_context = lambda: types.SimpleNamespace(  # type: ignore[assignment]
        attn_metadata={STATE_PREFIX: _Meta(prefix_end=71, used=2, page=5)}
    )
    _bind_step(c2, "req-c", 5)
    assert c2.restore_target(STATE_PREFIX, "73") is None
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=6, request_ids=("req-c", "req-d"), generation=0, synthetic=False
        )
    )
    assert c2.restore_target(STATE_PREFIX, "73") is None
    _bind_step(c2, "req-c", 7, synthetic=True)
    assert c2.restore_target(STATE_PREFIX, "73") is None
    try:
        c2.ring_rows("model.layers.7.self_attn.compressor.state_cache")
    except c2.DSV41LayerwiseC2Error:
        pass
    else:  # pragma: no cover - the contract requires a refusal here
        raise AssertionError("unknown ring geometry must raise")

    c2.reset_step_state()
    print("FAIL_CLOSED_OK")
    print("IMAGE_SELF_CHECK_T87_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
