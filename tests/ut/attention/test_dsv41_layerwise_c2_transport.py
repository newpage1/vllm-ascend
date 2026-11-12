#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""CPU tests wiring the S1 C2 seam to the real transport codec (t81).

The t78 seam was verified against a stand-in codec. This suite replaces the
stand-in with the *real* t77 wire codec:

- ``repos/LMCache/lmcache/v1/multiprocess/layerwise_protocol.py`` owns
  ``BoundaryStateDescriptor`` and ``compute_boundary_checksum``;
- ``repos/LMCache-Ascend/.../dsv41_boundary_codec.py`` owns the real
  ``encode_boundary_rows`` / ``decode_boundary_rows`` byte format.

Both are loaded by path through namespace packages, so no lmcache install is
required. What must hold:

- the descriptor the engine builds from a real forward carries the real
  resource id (``<execution_layer>.compressor.state_cache``), the real ring
  cursor (``prefix_end % ring_rows``), the real compression phase and the
  rows this step actually wrote (never a hash/modulo guess);
- the protocol checksum recomputed from those fields matches the descriptor;
- capture -> restore moves real bytes into the *new* request's own ring page
  and the restored rows are bit-identical to the captured rows;
- a miss, a stale boundary token or a resource with no capture stays
  fail-closed / cold-only and never fabricates a hit;
- the provider the runner installs is the engine function itself, and a
  connector without ``worker_adapter.set_boundary_provider`` is a no-op.

Known transport gap (reported to the transport owner, out of this repo): the
t77 adapter restores payload rows at the *captured* row indices and copies
only the written rows, while section 6 of the plan asks for a whole-page copy
rebased onto the new request's page. The bridge codec below rebases so this
suite can prove the byte format end to end; the transport-side rebase is a
finding, not something this repo can fix.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

RING_ROWS = 32
HEAD_DIM = 8
EXECUTION_LAYER = "language_model.model.layers.0.attn"
STATE_PREFIX = f"{EXECUTION_LAYER}.compressor.state_cache"
WORKSPACE = Path(__file__).resolve().parents[5]
LMCACHE_TREE = WORKSPACE / "repos" / "LMCache" / "lmcache"
ASCEND_TREE = WORKSPACE / "repos" / "LMCache-Ascend" / "lmcache_ascend"
CODEC_PATH = ASCEND_TREE / "integration" / "vllm" / "dsv41_boundary_codec.py"


class _Meta:
    """Stand-in for AscendDSAV41Metadata of one ratio-2 layer."""

    def __init__(self, prefix_end: int, used: int, page: int, num_reqs: int = 1) -> None:
        control = torch.zeros(5, num_reqs, dtype=torch.int32)
        control[0, :] = prefix_end
        control[1, :] = used
        control[2, :] = 0
        control[3, :] = 0
        control[4, :] = page
        self.c2_ring_metadata = control
        self.block_table = torch.tensor([[page]], dtype=torch.int64)
        self.num_reqs = num_reqs
        self.compress_ratio = 2
        self.slot_mapping = torch.tensor(
            [
                page * RING_ROWS + (prefix_end + i) % RING_ROWS
                for i in range(max(used, 0))
            ],
            dtype=torch.int64,
        )


class _StateCache:
    def __init__(self, pages: int = 4) -> None:
        self.kv_cache = [
            torch.zeros(pages, RING_ROWS, 1, 2 * HEAD_DIM, dtype=torch.float32)
        ]


class _Compressor:
    def __init__(self) -> None:
        self.state_cache = _StateCache()


class _Impl:
    def __init__(self, prefix: str | None) -> None:
        self.compressor_state_prefix = prefix
        self.layer_name = "language_model.model.layers.0.self_attn"


class _Attn:
    def __init__(self, prefix: str | None, meta: _Meta | None) -> None:
        self.v41_impl = _Impl(prefix)
        self.compressor = _Compressor()
        self.layerwise_execution_layer = EXECUTION_LAYER
        self._meta = meta


class _TransportBoundaryCodec:
    """Engine-side codec over the transport's real wire codec.

    Capture encodes the rows the forward wrote with ``encode_boundary_rows``;
    restore decodes the real payload and rebases the rows onto the *target*
    request's ring page (the plan's section 6 rule) before writing them.
    """

    def __init__(self, protocol: types.ModuleType, codec: types.ModuleType) -> None:
        self._protocol = protocol
        self._codec = codec
        self.captures: dict[str, tuple[Any, bytes, int]] = {}

    def capture_layer_boundary(
        self, request_view: Any, tensor_view: Any, *, stream: Any = None
    ) -> None:
        if request_view.resource_id != STATE_PREFIX:
            raise ValueError(
                f"resource id {request_view.resource_id!r} is not the compressor "
                f"state cache of execution layer {request_view.execution_layer!r}"
            )
        if not request_view.written_rows:
            raise ValueError("refusing to capture a boundary without written rows")
        descriptor, payload = self._codec.encode_boundary_rows(
            _flat_ring(tensor_view.ring),
            list(request_view.written_rows),
            snapshot_id=f"snapshot-{request_view.step_id}",
            resource_id=request_view.resource_id,
            boundary_token=str(request_view.prefix_end + request_view.step_tokens),
            ring_cursor=request_view.prefix_end % request_view.ring_rows,
            compression_phase=request_view.compression_phase,
        )
        self.captures[request_view.resource_id] = (
            descriptor,
            payload,
            request_view.ring_page,
        )

    def restore_layer_boundary(
        self, request_view: Any, tensor_view: Any, *, stream: Any = None
    ) -> None:
        entry = self.captures.get(request_view.resource_id)
        if entry is None:
            raise KeyError(f"no captured boundary for {request_view.resource_id}")
        descriptor, payload, source_page = entry
        decoded = self._codec.decode_boundary_rows(payload)
        delta = (request_view.ring_page - source_page) * request_view.ring_rows
        rows = [index + delta for index in descriptor.row_indices]
        _flat_ring(tensor_view.ring)[rows] = decoded["rows"].to(torch.float32)


def _flat_ring(ring: torch.Tensor) -> torch.Tensor:
    """Flatten [pages, rows, 2 * head] into the transport's [rows, 2 * head]."""
    if ring.dim() == 4:
        ring = ring.squeeze(-2)
    return ring.reshape(ring.shape[0] * ring.shape[1], ring.shape[2])


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def transport() -> tuple[types.ModuleType, types.ModuleType]:
    if not LMCACHE_TREE.is_dir() or not CODEC_PATH.is_file():
        pytest.skip("the transport trees are not checked out next to this repo")
    _namespace("lmcache", LMCACHE_TREE)
    _namespace("lmcache.v1", LMCACHE_TREE / "v1")
    _namespace("lmcache.v1.multiprocess", LMCACHE_TREE / "v1" / "multiprocess")
    protocol = importlib.import_module(
        "lmcache.v1.multiprocess.layerwise_protocol"
    )
    _namespace("lmcache_ascend", ASCEND_TREE)
    _namespace("lmcache_ascend.integration", ASCEND_TREE / "integration")
    _namespace("lmcache_ascend.integration.vllm", ASCEND_TREE / "integration" / "vllm")
    codec = _load_module(
        "lmcache_ascend.integration.vllm.dsv41_boundary_codec", CODEC_PATH
    )
    return protocol, codec


@pytest.fixture(scope="module")
def c2() -> types.ModuleType:
    vllm = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    forward_context_mod = types.ModuleType("vllm.forward_context")
    context: dict[str, Any] = {"attn_metadata": None}

    class _Logger:
        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def warning(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def error(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    logger_mod.init_logger = lambda _name: _Logger()
    forward_context_mod.get_forward_context = lambda: types.SimpleNamespace(
        attn_metadata=context["attn_metadata"]
    )
    forward_context_mod._test_context = context
    vllm.logger = logger_mod
    vllm.forward_context = forward_context_mod
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm.forward_context"] = forward_context_mod

    top_pkg = types.ModuleType("vllm_ascend")
    attention_pkg = types.ModuleType("vllm_ascend.attention")
    top_pkg.__path__ = []  # type: ignore[attr-defined]
    attention_pkg.__path__ = []  # type: ignore[attr-defined]
    top_pkg.attention = attention_pkg
    sys.modules["vllm_ascend"] = top_pkg
    sys.modules["vllm_ascend.attention"] = attention_pkg

    root = Path(__file__).resolve().parents[3] / "vllm_ascend" / "attention"
    admission = _load_module(
        "dsv41_layerwise_admission_transport_under_test",
        root / "dsv41_layerwise_admission.py",
    )
    sys.modules["vllm_ascend.attention.dsv41_layerwise_admission"] = admission
    attention_pkg.dsv41_layerwise_admission = admission
    module = _load_module(
        "dsv41_layerwise_c2_transport_under_test", root / "dsv41_layerwise_c2.py"
    )
    module._test_forward_context = forward_context_mod
    module._test_context = context
    return module


@pytest.fixture(autouse=True)
def _clean(c2: types.ModuleType):
    c2.clear_boundary_state_codec()
    c2.reset_step_state()
    c2._provider_installed = False
    yield
    c2.clear_boundary_state_codec()
    c2.reset_step_state()
    c2._provider_installed = False


def _bind_step(c2: types.ModuleType, request_id: str = "req-a", step_id: int = 3) -> None:
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=step_id,
            request_ids=(request_id,),
            generation=0,
            synthetic=False,
        )
    )


def _run_capture(
    c2: types.ModuleType,
    attn: _Attn,
    meta: _Meta,
    *,
    request_id: str = "req-a",
    step_id: int = 3,
) -> None:
    c2._test_context["attn_metadata"] = {STATE_PREFIX: meta}
    _bind_step(c2, request_id=request_id, step_id=step_id)
    c2.capture_layer_boundary_state(
        "language_model.model.layers.0.self_attn.v41_attn", attn
    )


def _run_restore(
    c2: types.ModuleType,
    attn: _Attn,
    meta: _Meta,
    *,
    request_id: str = "req-b",
    step_id: int = 4,
) -> None:
    c2._test_context["attn_metadata"] = {STATE_PREFIX: meta}
    _bind_step(c2, request_id=request_id, step_id=step_id)
    c2.restore_layer_boundary_state(
        "language_model.model.layers.0.self_attn.v41_attn", attn
    )


def test_real_descriptor_fields_and_protocol_checksum(transport, c2) -> None:
    protocol, codec = transport
    bridge = _TransportBoundaryCodec(protocol, codec)
    c2.set_boundary_state_codec(bridge)

    meta = _Meta(prefix_end=70, used=3, page=1)
    attn = _Attn(STATE_PREFIX, meta)
    _run_capture(c2, attn, meta)

    descriptor, payload, page = bridge.captures[STATE_PREFIX]
    rows = tuple(RING_ROWS + 6 + offset for offset in range(3))
    assert page == 1
    assert descriptor.resource_id == STATE_PREFIX
    assert descriptor.boundary_token == "73"
    assert descriptor.ring_cursor == 70 % RING_ROWS == 6
    assert descriptor.compression_phase == 0
    assert descriptor.row_indices == rows
    assert descriptor.schema_version == protocol.BOUNDARY_STATE_SCHEMA_VERSION
    assert descriptor.checksum == protocol.compute_boundary_checksum(
        descriptor.snapshot_id,
        descriptor.resource_id,
        descriptor.boundary_token,
        descriptor.ring_cursor,
        descriptor.compression_phase,
        descriptor.row_indices,
        descriptor.schema_version,
    )
    assert payload


def test_real_capture_restore_moves_bytes_to_the_new_page(transport, c2) -> None:
    protocol, codec = transport
    bridge = _TransportBoundaryCodec(protocol, codec)
    c2.set_boundary_state_codec(bridge)

    source_meta = _Meta(prefix_end=70, used=3, page=2)
    source = _Attn(STATE_PREFIX, source_meta)
    ring = source.compressor.state_cache.kv_cache[0]
    for offset, row in enumerate(range(6, 9)):
        ring[2, row, 0, :] = torch.arange(2 * HEAD_DIM) + offset
    expected = ring[2].clone()
    _run_capture(c2, source, source_meta)

    target_meta = _Meta(prefix_end=70, used=3, page=3)
    target = _Attn(STATE_PREFIX, target_meta)
    target.compressor.state_cache.kv_cache[0][3].fill_(7.0)
    _run_restore(c2, target, target_meta)

    restored = target.compressor.state_cache.kv_cache[0][3]
    assert torch.equal(restored[6:9], expected[6:9])
    # The transport copies the written rows only; untouched rows keep their
    # previous content, which is the whole-page gap reported to the owner.
    assert torch.all(restored[:6] == 7.0)


def test_restore_without_capture_fails_closed(transport, c2) -> None:
    protocol, codec = transport
    c2.set_boundary_state_codec(_TransportBoundaryCodec(protocol, codec))

    meta = _Meta(prefix_end=70, used=3, page=2)
    attn = _Attn(STATE_PREFIX, meta)
    with pytest.raises(c2.DSV41LayerwiseC2Error):
        _run_restore(c2, attn, meta)
    assert c2.step_failure_count() == 1


def test_provider_returns_the_real_rows_and_stays_cold_on_mismatch(c2) -> None:
    # The runner installs the provider, not a push-mode codec: capture must
    # still record the rows for it.
    c2._provider_installed = True
    meta = _Meta(prefix_end=70, used=3, page=1)
    attn = _Attn(STATE_PREFIX, meta)
    _run_capture(c2, attn, meta)

    assert c2.s1_capture_active() is True
    rows = tuple(RING_ROWS + 6 + offset for offset in range(3))
    assert c2.provide_boundary_rows(STATE_PREFIX, "snapshot-3", "73") == (6, 0, rows)
    # Stale boundary, foreign resource and integer token all resolve strictly.
    assert c2.provide_boundary_rows(STATE_PREFIX, "s", "74") is None
    assert c2.provide_boundary_rows("other.compressor.state_cache", "s", "73") is None
    assert c2.provide_boundary_rows(STATE_PREFIX, "s", 73) == (6, 0, rows)


def test_provider_is_installed_on_the_adapter_and_noop_elsewhere(c2) -> None:
    installed: list[Any] = []

    class _Worker:
        def set_boundary_provider(self, provider: Any) -> None:
            installed.append(provider)

    class _Connector:
        worker_adapter = _Worker()

    assert c2.install_boundary_provider(_Connector()) is True
    assert installed == [c2.provide_boundary_rows]
    assert c2.s1_capture_active() is True

    class _Other:
        pass

    assert c2.install_boundary_provider(_Other()) is False


def test_s0_stub_records_nothing_and_the_provider_stays_cold(c2) -> None:
    """Contrast: with the S1 capture removed the provider cannot answer."""
    meta = _Meta(prefix_end=70, used=3, page=1)
    attn = _Attn(STATE_PREFIX, meta)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: meta}
    _bind_step(c2)
    # No codec and no provider installed == the S0 stub behaviour.
    c2.capture_layer_boundary_state(
        "language_model.model.layers.0.self_attn.v41_attn", attn
    )
    assert c2.boundary_record(STATE_PREFIX) is None
    assert c2.provide_boundary_rows(STATE_PREFIX, "s", "73") is None
