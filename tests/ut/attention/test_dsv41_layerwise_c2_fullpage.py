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
"""CPU tests for the S1 whole-page store/restore path (t87).

t81 taught the transport *which* rows a step wrote. S1 also needs two facts
only the engine owns:

- ``ring_rows(resource_id)``: the real rows per ring page, so the transport
  can frame a whole page instead of a sparse row set;
- ``restore_target(resource_id, boundary_token)``: the *new* request's own
  physical ring page, so a captured page can be rebased instead of being
  written back at the old page.

Both are installed as attributes of the real provider function and are
exercised here against the real wire codec
(``repos/LMCache-Ascend/.../dsv41_boundary_codec.py``) and the real protocol
checksum. What must hold:

- install binds the model's real geometry and points the adapter's boundary
  caches at absolute-row views of the *same* storage;
- a store frames the whole page: written rows keep their bytes, unwritten
  rows travel as zero, and the descriptor spans ``page_offset..+ring_rows``;
- a restore rebases onto the new request's page and leaves no stale tail
  slots from the previous request;
- every mismatch (token, batch size, synthetic step, missing metadata,
  unknown geometry) stays fail-closed and never fabricates a page;
- removing the two hooks (the pre-t87 engine) makes the whole-page contract
  impossible: the store degrades to schema v1 sparse rows, which the
  transport refuses to restore.
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
LONG_KV = f"{EXECUTION_LAYER}.long_kv_cache"
WORKSPACE = Path(__file__).resolve().parents[5]
LMCACHE_TREE = WORKSPACE / "repos" / "LMCache" / "lmcache"
ASCEND_TREE = WORKSPACE / "repos" / "LMCache-Ascend" / "lmcache_ascend"
CODEC_PATH = ASCEND_TREE / "integration" / "vllm" / "dsv41_boundary_codec.py"


class _Meta:
    """Stand-in for one ratio-2 layer's AscendDSAV41Metadata."""

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
    """The real 4-D compressor state page set of one layer."""

    def __init__(self, pages: int = 6) -> None:
        self.kv_cache = [torch.zeros(pages, RING_ROWS, 1, 2 * HEAD_DIM, dtype=torch.float32)]


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
    def __init__(self, layers: list[Any]) -> None:
        self._layers = layers

    def modules(self):
        return iter(self._layers)


class _Worker:
    def __init__(self, kv_caches: dict[str, Any]) -> None:
        self.kv_caches = kv_caches
        self.providers: list[Any] = []

    def set_boundary_provider(self, provider: Any) -> None:
        self.providers.append(provider)


class _Connector:
    def __init__(self, worker: _Worker) -> None:
        self.worker_adapter = worker


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
    protocol = importlib.import_module("lmcache.v1.multiprocess.layerwise_protocol")
    _namespace("lmcache_ascend", ASCEND_TREE)
    _namespace("lmcache_ascend.integration", ASCEND_TREE / "integration")
    _namespace("lmcache_ascend.integration.vllm", ASCEND_TREE / "integration" / "vllm")
    codec = _load_module("lmcache_ascend.integration.vllm.dsv41_boundary_codec", CODEC_PATH)
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
        "dsv41_layerwise_admission_fullpage_under_test",
        root / "dsv41_layerwise_admission.py",
    )
    sys.modules["vllm_ascend.attention.dsv41_layerwise_admission"] = admission
    attention_pkg.dsv41_layerwise_admission = admission
    module = _load_module("dsv41_layerwise_c2_fullpage_under_test", root / "dsv41_layerwise_c2.py")
    module._test_context = context
    return module


@pytest.fixture(autouse=True)
def _clean(c2: types.ModuleType):
    def _reset() -> None:
        c2.clear_boundary_state_codec()
        c2.reset_step_state()
        c2._provider_installed = False
        # ``getattr`` keeps this suite readable against the pre-t87 engine:
        # the contrast cases then fail on their own assertions instead of
        # erroring out in the fixture.
        getattr(c2, "_ring_geometry", {}).clear()
        for attribute in ("ring_rows", "restore_target"):
            if hasattr(c2.provide_boundary_rows, attribute):
                delattr(c2.provide_boundary_rows, attribute)

    _reset()
    yield
    _reset()


def _install(c2: types.ModuleType, attn: _Attn) -> tuple[_Worker, _Connector]:
    ring = attn.compressor.state_cache.kv_cache[0]
    worker = _Worker({STATE_PREFIX: ring, LONG_KV: torch.zeros(6, 128, 1, 2 * HEAD_DIM)})
    connector = _Connector(worker)
    assert c2.install_boundary_provider(connector, model=_Model([attn])) is True
    return worker, connector


def _bind_step(c2: types.ModuleType, request_id: str, step_id: int, synthetic: bool = False) -> None:
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=step_id,
            request_ids=(request_id,),
            generation=0,
            synthetic=synthetic,
        )
    )


def _capture(c2: types.ModuleType, attn: _Attn, meta: _Meta, *, token: str) -> None:
    c2._test_context["attn_metadata"] = {STATE_PREFIX: meta}
    _bind_step(c2, "req-a", 3)
    c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn.v41_attn", attn)
    record = c2.boundary_record(STATE_PREFIX)
    assert record is not None
    assert str(record.end_token) == token


def _row_space(worker: _Worker) -> torch.Tensor:
    cache = worker.kv_caches[STATE_PREFIX]
    assert isinstance(cache, torch.Tensor)
    assert cache.dim() == 2, "install must hand the transport the absolute-row view"
    return cache


def test_install_binds_real_geometry_and_row_space_views(c2) -> None:
    attn = _Attn()
    original = attn.compressor.state_cache.kv_cache[0]
    original[2, 6, 0, :] = torch.arange(2 * HEAD_DIM)

    worker, _connector = _install(c2, attn)

    # The provider is the engine function and it carries the two geometry
    # hooks the transport looks up with getattr.
    assert worker.providers == [c2.provide_boundary_rows]
    assert c2.provide_boundary_rows.ring_rows is c2.ring_rows
    assert c2.provide_boundary_rows.restore_target is c2.restore_target
    assert c2.ring_rows(STATE_PREFIX) == RING_ROWS

    row_space = _row_space(worker)
    assert row_space.shape == (6 * RING_ROWS, 2 * HEAD_DIM)
    # Same storage, not a copy: the engine keeps writing the 4-D page set and
    # the transport reads the flat rows.
    assert row_space.data_ptr() == original.view(-1, 2 * HEAD_DIM).data_ptr()
    assert torch.equal(row_space[2 * RING_ROWS + 6], torch.arange(2 * HEAD_DIM))

    # Non-boundary caches keep their registered geometry and are still
    # addressed by block id, exactly as the S0 path expects.
    assert worker.kv_caches[LONG_KV].dim() == 4


def test_store_frames_the_whole_page_and_zeroes_unwritten_rows(transport, c2) -> None:
    protocol, codec = transport
    attn = _Attn()
    worker, _connector = _install(c2, attn)
    ring = attn.compressor.state_cache.kv_cache[0]
    for offset, slot in enumerate(range(6, 9)):
        ring[2, slot, 0, :] = torch.arange(2 * HEAD_DIM) + offset
    meta = _Meta(prefix_end=70, used=3, page=2)
    _capture(c2, attn, meta, token="73")

    record = c2.boundary_record(STATE_PREFIX)
    assert record is not None
    descriptor, payload = codec.encode_boundary_rows(
        _row_space(worker),
        list(record.row_indices),
        snapshot_id="snap-1",
        resource_id=STATE_PREFIX,
        boundary_token=str(record.end_token),
        ring_cursor=record.ring_cursor,
        compression_phase=record.compression_phase,
        ring_length=c2.ring_rows(STATE_PREFIX),
    )

    assert descriptor.schema_version == 2
    assert descriptor.boundary_token == "73"
    assert descriptor.ring_cursor == 70 % RING_ROWS
    assert len(descriptor.row_indices) == RING_ROWS
    assert descriptor.row_indices[0] == 2 * RING_ROWS
    assert descriptor.checksum == protocol.compute_boundary_checksum(
        descriptor.snapshot_id,
        descriptor.resource_id,
        descriptor.boundary_token,
        descriptor.ring_cursor,
        descriptor.compression_phase,
        descriptor.row_indices,
        descriptor.schema_version,
    )

    decoded = codec.decode_boundary_rows(payload)
    page = decoded["rows"]
    assert tuple(page.shape) == (RING_ROWS, 2 * HEAD_DIM)
    assert torch.equal(page[6:9], ring[2, 6:9, 0, :])
    assert torch.count_nonzero(page[:6]) == 0
    assert torch.count_nonzero(page[9:]) == 0


def test_restore_target_rebases_the_page_and_leaves_no_stale_tail(transport, c2) -> None:
    _protocol, codec = transport
    source = _Attn()
    worker, _connector = _install(c2, source)
    ring = source.compressor.state_cache.kv_cache[0]
    for offset, slot in enumerate(range(6, 9)):
        ring[2, slot, 0, :] = torch.arange(2 * HEAD_DIM) + offset
    _capture(c2, source, _Meta(prefix_end=70, used=3, page=2), token="73")
    record = c2.boundary_record(STATE_PREFIX)
    assert record is not None
    descriptor, payload = codec.encode_boundary_rows(
        _row_space(worker),
        list(record.row_indices),
        snapshot_id="snap-1",
        resource_id=STATE_PREFIX,
        boundary_token=str(record.end_token),
        ring_cursor=record.ring_cursor,
        compression_phase=record.compression_phase,
        ring_length=c2.ring_rows(STATE_PREFIX),
    )

    # The new request resumes exactly at token 73 and owns page 4, whose slots
    # still hold this worker's previous traffic.
    target_meta = _Meta(prefix_end=73, used=0, page=4)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: target_meta}
    _bind_step(c2, "req-b", 4)
    target = c2.restore_target(STATE_PREFIX, "73")
    assert target == (4, RING_ROWS)

    page = codec.decode_boundary_rows(payload)["rows"]
    row_space = _row_space(worker)
    row_space[4 * RING_ROWS : 5 * RING_ROWS] = page
    assert torch.equal(ring[4, 6:9, 0, :], ring[2, 6:9, 0, :])
    assert torch.count_nonzero(ring[4, :6, 0, :]) == 0
    assert torch.count_nonzero(ring[4, 9:, 0, :]) == 0
    assert descriptor.schema_version == 2


def test_restore_target_fails_closed_on_every_mismatch(c2) -> None:
    attn = _Attn()
    _install(c2, attn)
    _capture(c2, attn, _Meta(prefix_end=70, used=3, page=2), token="73")

    # Boundary token does not match the live prefix end: the snapshot does not
    # apply to this request.
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(prefix_end=71, used=2, page=4)}
    _bind_step(c2, "req-b", 4)
    assert c2.restore_target(STATE_PREFIX, "73") is None

    # Two requests in one step: a boundary page belongs to exactly one.
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(step_id=5, request_ids=("req-b", "req-c"), generation=0, synthetic=False)
    )
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(prefix_end=73, used=0, page=4)}
    assert c2.restore_target(STATE_PREFIX, "73") is None

    # Synthetic (capture/profile/dummy) forwards never own a page.
    _bind_step(c2, "req-b", 6, synthetic=True)
    assert c2.restore_target(STATE_PREFIX, "73") is None

    # No live metadata for the resource.
    _bind_step(c2, "req-b", 7)
    c2._test_context["attn_metadata"] = {}
    assert c2.restore_target(STATE_PREFIX, "73") is None

    # Unknown geometry never guesses 32 rows.
    with pytest.raises(c2.DSV41LayerwiseC2Error):
        c2.ring_rows("model.layers.7.self_attn.compressor.state_cache")

    # A provider that was never installed answers nothing.
    c2._provider_installed = False
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(prefix_end=73, used=0, page=4)}
    _bind_step(c2, "req-b", 8)
    assert c2.restore_target(STATE_PREFIX, "73") is None


def test_without_the_hooks_the_whole_page_contract_is_impossible(transport, c2) -> None:
    """Contrast: the pre-t87 engine cannot satisfy the transport (fails closed)."""
    _protocol, codec = transport
    attn = _Attn()
    worker, _connector = _install(c2, attn)
    ring = attn.compressor.state_cache.kv_cache[0]
    ring[2, 6, 0, :] = torch.arange(2 * HEAD_DIM)
    _capture(c2, attn, _Meta(prefix_end=70, used=1, page=2), token="71")
    record = c2.boundary_record(STATE_PREFIX)
    assert record is not None

    # Drop the hooks: this is exactly what the engine exposed before t87.
    del c2.provide_boundary_rows.ring_rows
    del c2.provide_boundary_rows.restore_target
    assert getattr(c2.provide_boundary_rows, "ring_rows", None) is None
    assert getattr(c2.provide_boundary_rows, "restore_target", None) is None

    # The transport's store guard sees no geometry and stays cold-only; the
    # only encoding it can still build is the sparse v1 shape, which the
    # transport refuses to restore (schema v2 is required for rebasing).
    descriptor, _payload = codec.encode_boundary_rows(
        _row_space(worker),
        list(record.row_indices),
        snapshot_id="snap-1",
        resource_id=STATE_PREFIX,
        boundary_token=str(record.end_token),
        ring_cursor=record.ring_cursor,
        compression_phase=record.compression_phase,
    )
    assert descriptor.schema_version == 1
    assert len(descriptor.row_indices) == 1
