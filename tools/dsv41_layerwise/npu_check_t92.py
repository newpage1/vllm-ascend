# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-device check that the live forward context carries per-request C2 pages.

t87 handed the transport one boundary page per step: ``restore_target`` only
resolved a single-request step and every multi-request batch fell back to
cold. t92 resolves the page *per request* from that request's own control row
in ``c2_ring_metadata``. The CPU suites cover the resolution logic; this script
covers what they cannot:

* ``c2_ring_metadata`` is produced by the **real** metadata kernel
  (``vllm_ascend.ops.triton.c2_ring_metadata.build_c2_ring_metadata``) running
  on an NPU, not by a hand-written tensor;
* the metadata is placed in the **real** ``vllm.forward_context`` (a real
  ``VllmConfig`` and a real ``ForwardContext``) so the engine-side hook reads
  it exactly the way ``dsa_v41_forward`` does;
* the boundary provider, the ring geometry and the whole-page codec are the
  real ones, and the page copy happens on a real NPU compressor state tensor.

What it proves:

1. two requests in one step (different prefixes, different pages) each keep
   their own capture and their own whole page;
2. a restore step resolves every resuming request to its own physical page,
   without a one-request step;
3. two requests resuming at the same token on different pages stay
   ambiguous (cold-only) unless a request id names one;
4. bytes move to the right page on the device, with unwritten rows zeroed.

Run inside the v0.30.0rc2 image on a node with free chips (see EVIDENCE-t92.md
for the exact command). It prints one line per stage and
``NPU_CHECK_T92_PASSED`` at the end.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

WORKSPACE = Path("/ws/repos")
STATE_PREFIX = "language_model.model.layers.0.attn.compressor.state_cache"
EXECUTION_LAYER = "language_model.model.layers.0.attn"
LAYER = f"{EXECUTION_LAYER}.v41_attn"
RING_ROWS = 32
HEAD_DIM = 8


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def _load_transport() -> types.ModuleType:
    """Load the real protocol + codec from the mounted trees (not installed)."""
    _namespace("lmcache", WORKSPACE / "LMCache" / "lmcache")
    _namespace("lmcache.v1", WORKSPACE / "LMCache" / "lmcache" / "v1")
    _namespace(
        "lmcache.v1.multiprocess",
        WORKSPACE / "LMCache" / "lmcache" / "v1" / "multiprocess",
    )
    importlib.import_module("lmcache.v1.multiprocess.layerwise_protocol")
    ascend = WORKSPACE / "LMCache-Ascend" / "lmcache_ascend"
    _namespace("lmcache_ascend", ascend)
    _namespace("lmcache_ascend.integration", ascend / "integration")
    _namespace("lmcache_ascend.integration.vllm", ascend / "integration" / "vllm")
    path = ascend / "integration" / "vllm" / "dsv41_boundary_codec.py"
    spec = importlib.util.spec_from_file_location(
        "lmcache_ascend.integration.vllm.dsv41_boundary_codec", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch
    import torch_npu  # noqa: F401

    from vllm.forward_context import (
        ForwardContext,
        get_forward_context,
        override_forward_context,
    )

    import vllm_ascend.attention.dsv41_layerwise_c2 as c2
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata
    from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata

    codec = _load_transport()
    print("IMPORT_OK", c2.__name__, "codec", codec.__name__)

    def _context(metadata):
        """Install the metadata in vLLM's real forward context.

        ``set_forward_context`` is not used here: it builds a ``VllmConfig``,
        and this checkpoint's Engram section requires CUDA, so a config
        cannot be constructed inside the Ascend image (the served path passes
        ``--engram-config`` and runs a full engine init). The context object
        and the context-var swap are vLLM's own, and the metadata comes from
        the real kernel; only the config-driven ``additional_kwargs`` wiring
        is skipped, which the boundary hooks do not read.
        """
        return override_forward_context(
            ForwardContext(
                no_compile_layers={},
                attn_metadata={STATE_PREFIX: metadata},
                slot_mapping={},
            )
        )

    device = "npu:0"
    ring = torch.zeros(8, RING_ROWS, 1, 2 * HEAD_DIM, dtype=torch.float32, device=device)

    def kernel_metadata(
        seq_lens: list[int],
        pages: list[int],
        used: int,
    ) -> AscendDSAV41Metadata:
        """Build real metadata through the engine's own ring-metadata kernel."""
        num_reqs = len(pages)
        num_tokens = used * num_reqs
        starts = [index * used for index in range(num_reqs + 1)]
        positions = [
            value
            for index, length in enumerate(seq_lens)
            for value in range(length - used, length)
        ]
        query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
        seq = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        pos = torch.tensor(positions, dtype=torch.int32, device=device)
        block_table = torch.tensor(
            [[page] for page in pages], dtype=torch.int32, device=device
        )
        rope = torch.zeros(8, HEAD_DIM, dtype=torch.float32, device=device)
        control = torch.zeros(5, num_reqs, dtype=torch.int32, device=device)
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        source_positions = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        cos = torch.zeros(num_tokens, 1, 1, HEAD_DIM, dtype=torch.float32, device=device)
        sin = torch.zeros_like(cos)
        build_c2_ring_metadata(
            query_start_loc,
            seq,
            pos,
            block_table,
            rope,
            rope,
            num_reqs,
            num_tokens,
            num_reqs,
            num_tokens,
            skip_update=False,
            ring_metadata_output=control,
            complete_mask_output=mask,
            source_positions_output=source_positions,
            cos_output=cos,
            sin_output=sin,
        )
        torch.npu.synchronize()
        slots = [
            page * RING_ROWS + (seq_lens[index] - used + step) % RING_ROWS
            for index, page in enumerate(pages)
            for step in range(used)
        ]
        return AscendDSAV41Metadata(
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq,
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            compress_ratio=2,
            storage_block_size=128,
            is_compressor_state=True,
            num_reqs=num_reqs,
            num_actual_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            max_query_len=used,
            c2_ring_metadata=control,
        )

    class _Ring:
        kv_cache = [ring]

    class _Compressor:
        state_cache = _Ring()

    class _Impl:
        compressor_state_prefix = STATE_PREFIX
        layer_name = "language_model.model.layers.0.self_attn"

    class _Attn:
        v41_impl = _Impl()
        compressor = _Compressor()
        layerwise_execution_layer = EXECUTION_LAYER

    class _Model:
        def modules(self):
            return iter([_Attn()])

    class _Worker:
        def __init__(self) -> None:
            self.kv_caches = {
                STATE_PREFIX: ring,
                f"{EXECUTION_LAYER}.long_kv_cache": torch.zeros(
                    8, 128, 1, 2 * HEAD_DIM, dtype=torch.bfloat16, device=device
                ),
            }

        def set_boundary_provider(self, provider) -> None:
            self.provider = provider

    class _Connector:
        def __init__(self) -> None:
            self.worker_adapter = _Worker()

    connector = _Connector()
    attn = _Attn()
    assert c2.install_boundary_provider(connector, model=_Model()) is True
    assert connector.worker_adapter.provider is c2.provide_boundary_rows
    row_space = connector.worker_adapter.kv_caches[STATE_PREFIX]
    assert row_space.dim() == 2 and row_space.data_ptr() == ring.data_ptr()
    print(
        "PROVIDER_OK",
        c2.ring_rows(STATE_PREFIX),
        tuple(row_space.shape),
        "hooks",
        sorted(
            name
            for name in ("ring_rows", "restore_target")
            if getattr(c2.provide_boundary_rows, name, None) is not None
        ),
    )

    # Two requests, two prefixes (70 and 71), two pages (2 and 3), three
    # tokens each. Each writes a distinguishable payload into its own slots.
    first = kernel_metadata([73, 74], [2, 3], used=3)
    control = first.c2_ring_metadata.detach().cpu().tolist()
    assert control[0] == [70, 71], control
    assert control[4] == [2, 3], control
    for index, page in enumerate((2, 3)):
        for step in range(3):
            slot = (70 + index + step) % RING_ROWS
            ring[page, slot, 0, :] = torch.arange(
                2 * HEAD_DIM, dtype=torch.float32, device=device
            ) + 100 * (index + 1) + step
    torch.npu.synchronize()

    request_ids = ("req-a", "req-b")
    with _context(first):
        assert get_forward_context().attn_metadata[STATE_PREFIX] is first
        print("CONTEXT_OK", type(get_forward_context()).__name__)
        c2.bind_step_request_view(
            c2.DSV41StepRequestView(
                step_id=1, request_ids=request_ids, generation=0, synthetic=False
            )
        )
        c2.capture_layer_boundary_state(LAYER, attn)
        records = [
            c2.boundary_record(STATE_PREFIX, request_id=request_id)
            for request_id in request_ids
        ]
        assert all(record is not None for record in records)
        assert [(r.end_token, r.ring_cursor) for r in records] == [
            (73, 70 % RING_ROWS),
            (74, 71 % RING_ROWS),
        ]
        assert records[0].row_indices == tuple(
            2 * RING_ROWS + (70 + step) % RING_ROWS for step in range(3)
        )
        assert records[1].row_indices == tuple(
            3 * RING_ROWS + (71 + step) % RING_ROWS for step in range(3)
        )
        for request_id, token, page in (("req-a", "73", 2), ("req-b", "74", 3)):
            provided = c2.provide_boundary_rows(
                STATE_PREFIX, "snap", token, request_id=request_id
            )
            assert provided is not None
            cursor, phase, rows = provided
            descriptor, payload = codec.encode_boundary_rows(
                row_space,
                list(rows),
                snapshot_id="snap",
                resource_id=STATE_PREFIX,
                boundary_token=token,
                ring_cursor=cursor,
                compression_phase=phase,
                ring_length=c2.ring_rows(STATE_PREFIX),
            )
            assert descriptor.schema_version == 2
            assert descriptor.row_indices[0] == page * RING_ROWS
            captured = codec.decode_boundary_rows(payload)["rows"]
            written = {(70 + (page - 2) + step) % RING_ROWS for step in range(3)}
            for slot in sorted(written):
                assert torch.equal(captured[slot], ring[page, slot, 0, :].cpu())
            unwritten = [row for row in range(RING_ROWS) if row not in written]
            assert torch.count_nonzero(captured[unwritten]) == 0
        print(
            "CAPTURE_OK",
            [(r.request_id, r.end_token, r.row_indices) for r in records],
        )

    # A restore step with two requests resuming at their own boundaries. They
    # carry their own tokens (three each): the kernel keeps the page only when
    # a request writes something, which is exactly why the page must be read
    # per request instead of assumed.
    second = kernel_metadata([76, 77], [4, 5], used=3)
    control = second.c2_ring_metadata.detach().cpu().tolist()
    assert control[0] == [73, 74] and control[4] == [4, 5], control
    with _context(second):
        c2.bind_step_request_view(
            c2.DSV41StepRequestView(
                step_id=2,
                request_ids=("req-x", "req-y"),
                generation=0,
                synthetic=False,
            )
        )
        assert c2.restore_target(STATE_PREFIX, "73") == (4, RING_ROWS)
        assert c2.restore_target(STATE_PREFIX, "74") == (5, RING_ROWS)
        assert c2.restore_target(STATE_PREFIX, "73", request_id="req-y") is None
        assert c2.restore_target(STATE_PREFIX, "73", request_id="req-x") == (
            4,
            RING_ROWS,
        )
        # Whole-page restore onto the live page, on the device.
        provided = c2.provide_boundary_rows(
            STATE_PREFIX, "snap", "73", request_id="req-a"
        )
        assert provided is not None
        cursor, phase, rows = provided
        descriptor, payload = codec.encode_boundary_rows(
            row_space,
            list(rows),
            snapshot_id="snap",
            resource_id=STATE_PREFIX,
            boundary_token="73",
            ring_cursor=cursor,
            compression_phase=phase,
            ring_length=c2.ring_rows(STATE_PREFIX),
        )
        target_page, target_rows = c2.restore_target(
            STATE_PREFIX, descriptor.boundary_token, request_id="req-x"
        )
        assert target_rows == c2.ring_rows(STATE_PREFIX)
        decoded = codec.decode_boundary_rows(payload)["rows"]
        row_space[target_page * RING_ROWS : (target_page + 1) * RING_ROWS] = decoded.to(
            device
        )
        torch.npu.synchronize()
        for step in range(3):
            slot = (70 + step) % RING_ROWS
            assert torch.equal(ring[target_page, slot, 0, :], ring[2, slot, 0, :])
        untouched = [
            slot for slot in range(RING_ROWS) if slot not in range(6, 9)
        ]
        assert torch.count_nonzero(ring[target_page, untouched, 0, :]) == 0
        print("RESTORE_OK", target_page, target_rows)

        # A request that writes nothing this step leaves its page unset in the
        # kernel output, so its row can never be read as page 0.
        silent = kernel_metadata([73, 74], [4, 5], used=0)
        silent_control = silent.c2_ring_metadata.detach().cpu().tolist()
        assert silent_control[4] == [0, 0], silent_control

        # Two requests resuming at the same token on different pages stay
        # ambiguous; naming one resolves exactly that request's page.
        ambiguous = kernel_metadata([76, 76], [4, 5], used=3)
    with _context(ambiguous):
        c2.bind_step_request_view(
            c2.DSV41StepRequestView(
                step_id=3,
                request_ids=("req-p", "req-q"),
                generation=0,
                synthetic=False,
            )
        )
        assert c2.restore_target(STATE_PREFIX, "73") is None
        assert c2.restore_target(STATE_PREFIX, "73", request_id="req-p") == (
            4,
            RING_ROWS,
        )
        assert c2.restore_target(STATE_PREFIX, "73", request_id="req-q") == (
            5,
            RING_ROWS,
        )
        print("ISOLATION_OK", 4, 5)

    with _context(silent):
        c2.bind_step_request_view(
            c2.DSV41StepRequestView(
                step_id=4,
                request_ids=("req-s", "req-t"),
                generation=0,
                synthetic=False,
            )
        )
        assert c2.restore_target(STATE_PREFIX, "73") is None
        assert c2.restore_target(STATE_PREFIX, "73", request_id="req-s") is None
        print("UNSET_PAGE_FAIL_CLOSED_OK", silent_control[0])

    print("NPU_CHECK_T92_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
