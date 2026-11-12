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
"""CPU tests for the S1 layerwise C2 capture/restore hooks (t78).

The module under test stays import-light: torch plus ``vllm.logger`` plus the
already CPU-safe admission module (whose only vllm import is ``vllm.logger``)
and a lazy ``vllm.forward_context`` read inside the hook. These tests stub the
two vllm modules and load every file by path, so the suite runs on a plain CPU
Python with no torch_npu/vllm/vllm_ascend install.

What must hold:

- With no codec installed every hook is inert: stock serving, fake forwards
  and graph capture stay bit-identical (S0 behaviour).
- With a codec installed a real forward without a bound step view fails
  closed instead of silently skipping capture (the S0 hole S1 closes).
- The request view carries *real* metadata: the physical ring page, the
  absolute prefix end and the rows this step actually wrote from the state
  slot mapping (no hash or modulo guessing).
- capture -> restore round-trips a whole ring page into the new request's
  own page, and a restore miss raises instead of leaving a partial page.
- The per-step TP gate stops every rank on a step disagreement or on any
  rank's recorded failure, and adds no collective when the group is TP=1.
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
            [page * RING_ROWS + (prefix_end + i) % RING_ROWS for i in range(max(used, 0))],
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
        self.layerwise_execution_layer = "language_model.model.layers.0.attn"
        self._meta = meta


class _FakeCodec:
    def __init__(self) -> None:
        self.captures: dict[tuple[str, str], Any] = {}
        self.captures_views: list[Any] = []
        self.restores: list[Any] = []

    def capture_layer_boundary(self, request_view, tensor_view, *, stream=None) -> None:
        self.captures_views.append(request_view)
        self.captures[(request_view.layer_name, request_view.request_id)] = (
            request_view,
            tensor_view.ring[request_view.ring_page].clone(),
        )

    def restore_layer_boundary(self, request_view, tensor_view, *, stream=None) -> None:
        key = (request_view.layer_name, request_view.request_id)
        if key not in self.captures:
            raise KeyError(f"no lease/capture for {key}")
        _, page = self.captures[key]
        target = tensor_view.ring[request_view.ring_page]
        if tuple(target.shape) != tuple(page.shape):
            raise ValueError("boundary page shape mismatch")
        target.copy_(page)
        self.restores.append(request_view)


class _FakeTpGroup:
    def __init__(self, result: list[int], world_size: int = 2) -> None:
        self._result = list(result)
        self.world_size = world_size
        self.calls = 0

    def all_reduce(self, tensor, op=None) -> None:
        self.calls += 1
        tensor.copy_(torch.tensor(self._result, dtype=tensor.dtype))


@pytest.fixture(scope="module")
def c2() -> types.ModuleType:
    """Load the C2 module by path with stubbed vllm modules."""
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

    attention_pkg = types.ModuleType("vllm_ascend.attention")
    attention_pkg.__path__ = []  # type: ignore[attr-defined]
    top_pkg = types.ModuleType("vllm_ascend")
    top_pkg.__path__ = []  # type: ignore[attr-defined]
    top_pkg.attention = attention_pkg
    sys.modules["vllm_ascend"] = top_pkg
    sys.modules["vllm_ascend.attention"] = attention_pkg

    root = Path(__file__).resolve().parents[3] / "vllm_ascend" / "attention"
    admission = _load_module(
        "dsv41_layerwise_admission_under_test",
        root / "dsv41_layerwise_admission.py",
    )
    sys.modules["vllm_ascend.attention.dsv41_layerwise_admission"] = admission
    attention_pkg.dsv41_layerwise_admission = admission

    module = _load_module("dsv41_layerwise_c2_under_test", root / "dsv41_layerwise_c2.py")
    module._test_forward_context = forward_context_mod
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean(c2: types.ModuleType):
    c2.clear_boundary_state_codec()
    c2.reset_step_state()
    yield
    c2.clear_boundary_state_codec()
    c2.reset_step_state()


def _bind(c2: types.ModuleType, *req_ids: str, synthetic: bool = False, step_id: int = 1):
    return c2.bind_step_request_view(
        c2.DSV41StepRequestView(
            step_id=step_id,
            request_ids=tuple(req_ids),
            generation=0,
            synthetic=synthetic,
        )
    )


def _publish_metadata(c2: types.ModuleType, meta: _Meta | None, prefix: str | None) -> None:
    c2._test_forward_context._test_context["attn_metadata"] = (
        None if (meta is None or prefix is None) else {prefix: meta}
    )


def test_hooks_are_inert_without_a_codec(c2: types.ModuleType) -> None:
    assert c2.boundary_state_codec_is_set() is False
    # No codec, no step view, no metadata: the stock path must not care.
    c2.restore_layer_boundary_state("language_model.model.layers.0.self_attn", object())
    c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", object())
    assert (
        c2.bind_model_forward_step(_Runner(("r0",)), decoder_replay_dummy=False)
        is c2.NOT_BOUND
    )
    c2.admit_step(torch.device("cpu"))


def test_real_forward_without_step_view_fails_closed(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    meta = _Meta(prefix_end=64, used=2, page=2)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="without a bound step view"):
        c2.restore_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="without a bound step view"):
        c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)


def test_layers_without_a_ring_plane_are_skipped(c2: types.ModuleType) -> None:
    codec = _FakeCodec()
    c2.set_boundary_state_codec(codec)
    _bind(c2, "r0")
    _publish_metadata(c2, None, None)
    attn = _Attn(None, None)
    c2.restore_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    assert codec.captures == {}


def test_missing_compressor_metadata_fails_closed(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "r0")
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, None, prefix)
    attn = _Attn(prefix, _Meta(prefix_end=0, used=1, page=1))
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="no compressor metadata"):
        c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)


def test_batch_mismatch_between_view_and_metadata_fails_closed(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "r0", "r1")
    meta = _Meta(prefix_end=0, used=1, page=1, num_reqs=1)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="disagree on the batch"):
        c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)


def test_request_view_is_built_from_real_metadata(c2: types.ModuleType) -> None:
    codec = _FakeCodec()
    c2.set_boundary_state_codec(codec)
    _bind(c2, "req-0")
    meta = _Meta(prefix_end=70, used=3, page=3)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)

    view = codec.captures_views[0]
    assert view.request_id == "req-0"
    assert view.prefix_end == 70
    assert view.step_tokens == 3
    assert view.ring_page == 3
    assert view.next_slot == 70 % RING_ROWS
    assert view.compression_ratio == 2
    assert view.compression_phase == 70 % 2
    assert view.position_control == (70, 3, 0, 0, 3)
    # Rows come from the state slot mapping for this page, not from a formula
    # over the prefix end.
    assert view.written_rows == tuple(range(3 * RING_ROWS + 6, 3 * RING_ROWS + 9))
    assert view.execution_layer == "language_model.model.layers.0.attn"
    planes = codec.captures_views[0]
    assert planes.ring_length == RING_ROWS


def test_rows_outside_the_request_page_are_rejected(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "req-0")
    meta = _Meta(prefix_end=0, used=2, page=1)
    meta.slot_mapping = torch.tensor([7 * RING_ROWS, 7 * RING_ROWS + 1], dtype=torch.int64)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="no state row belongs to page 1"):
        c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)


def test_capture_then_restore_round_trips_the_whole_page(c2: types.ModuleType) -> None:
    codec = _FakeCodec()
    c2.set_boundary_state_codec(codec)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    layer = "language_model.model.layers.0.self_attn"

    # Request A writes page 1 and captures it.
    _bind(c2, "A")
    meta_a = _Meta(prefix_end=32, used=32, page=1)
    _publish_metadata(c2, meta_a, prefix)
    attn_a = _Attn(prefix, meta_a)
    ring = attn_a.compressor.state_cache.kv_cache[0].squeeze(-2)
    ring[1] = torch.arange(RING_ROWS * 2 * HEAD_DIM, dtype=torch.float32).reshape(
        RING_ROWS, 2 * HEAD_DIM
    )
    expected = ring[1].clone()
    c2.capture_layer_boundary_state(layer, attn_a)
    assert codec.captures_views[0].prefix_end == 32

    # Request B owns page 2 (different physical page) and restores there.
    c2.reset_step_state()
    _bind(c2, "B")
    meta_b = _Meta(prefix_end=32, used=1, page=2)
    _publish_metadata(c2, meta_b, prefix)
    attn_b = _Attn(prefix, meta_b)
    ring_b = attn_b.compressor.state_cache.kv_cache[0].squeeze(-2)
    ring_b[2] = -1.0
    codec.captures[(layer, "B")] = codec.captures[(layer, "A")]
    c2.restore_layer_boundary_state(layer, attn_b)
    assert torch.equal(ring_b[2], expected)
    assert codec.restores[0].ring_page == 2


def test_restore_miss_raises_and_marks_the_step_failed(c2: types.ModuleType) -> None:
    codec = _FakeCodec()
    c2.set_boundary_state_codec(codec)
    _bind(c2, "cold")
    meta = _Meta(prefix_end=16, used=1, page=2)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="restore_layer_boundary failed"):
        c2.restore_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    assert c2.step_failure_count() == 1
    c2._tp_group = lambda: _FakeTpGroup([1, 0], world_size=1)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="failed earlier in this process"):
        c2.admit_step(torch.device("cpu"))


def test_synthetic_replay_binds_but_skips_the_codec(c2: types.ModuleType) -> None:
    codec = _FakeCodec()
    c2.set_boundary_state_codec(codec)
    runner = _Runner(("r0",))
    token = c2.bind_model_forward_step(runner, decoder_replay_dummy=True)
    assert c2.current_step_request_view().synthetic is True
    meta = _Meta(prefix_end=8, used=2, page=1)
    prefix = "language_model.model.layers.0.compressor.state_cache"
    _publish_metadata(c2, meta, prefix)
    attn = _Attn(prefix, meta)
    c2.restore_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    c2.capture_layer_boundary_state("language_model.model.layers.0.self_attn", attn)
    assert codec.captures == {} and codec.restores == []
    c2.clear_step_request_view(token)
    assert c2.current_step_request_view() is None


def test_bind_model_forward_step_binds_the_input_batch(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    c2._tp_group = lambda: _FakeTpGroup([1, 0], world_size=1)
    runner = _Runner(("a", "b"))
    token = c2.bind_model_forward_step(runner, decoder_replay_dummy=False)
    view = c2.current_step_request_view()
    assert view.request_ids == ("a", "b")
    assert view.synthetic is False
    assert view.step_id >= 1
    c2.clear_step_request_view(token)
    assert c2.current_step_request_view() is None


def test_tp_gate_requires_a_uniform_step(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "r0", step_id=4)
    c2._tp_group = lambda: _FakeTpGroup([2 * 5 + 1, 0], world_size=2)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="disagree on the layerwise step"):
        c2.admit_step(torch.device("cpu"))
    c2._tp_group = lambda: _FakeTpGroup([2 * 5, 0], world_size=2)
    c2.admit_step(torch.device("cpu"))


def test_tp_gate_stops_every_rank_on_any_failure(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "r0", step_id=2)
    c2._tp_group = lambda: _FakeTpGroup([2 * 3, 1], world_size=2)
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="marked failed"):
        c2.admit_step(torch.device("cpu"))


def test_tp_gate_is_skipped_for_a_single_rank(c2: types.ModuleType) -> None:
    c2.set_boundary_state_codec(_FakeCodec())
    _bind(c2, "r0")
    group = _FakeTpGroup([1, 0], world_size=1)
    c2._tp_group = lambda: group
    c2.admit_step(torch.device("cpu"))
    assert group.calls == 0


class _InputBatch:
    def __init__(self, req_ids: tuple[str, ...]) -> None:
        self.req_ids = list(req_ids)


class _Runner:
    def __init__(self, req_ids: tuple[str, ...]) -> None:
        self.input_batch = _InputBatch(req_ids)
