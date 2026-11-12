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
"""CPU tests for the S1 binding timing (t97).

A real serve crashed on its first request with ``... reached the S1 C2 hook
without a bound step view``: the connector installs its boundary provider at
``register_kv_caches`` time while a codec may arrive later, but the engine
only bound a step when a codec was already installed. The same run showed
``boundary dispatched 0`` / ``restore 0`` — with async KV loads the transport
restores *after* the forward, when the step's ring metadata is gone.

What must hold now:

- a provider-only process (the real serve) binds every real forward and
  captures the rows it wrote;
- a step that was bound before S1 was armed serves once cold instead of
  failing every rank, and the next step is bound normally;
- a step that is active but never bound still fails closed;
- the page plan taken from the input batch keeps each request's target page
  addressable after the forward, so ``restore_target`` can name it without
  live ring metadata — and still refuses a token that does not match.
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
LAYER_NAME = "language_model.model.layers.0.self_attn.v41_attn"


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


class _MultiMeta:
    """Metadata of a step that carries several requests at once."""

    def __init__(self, prefix_ends: list[int], used: list[int], pages: list[int]) -> None:
        count = len(pages)
        control = torch.zeros(5, count, dtype=torch.int32)
        control[0, :] = torch.tensor(prefix_ends, dtype=torch.int32)
        control[1, :] = torch.tensor(used, dtype=torch.int32)
        control[4, :] = torch.tensor(pages, dtype=torch.int32)
        self.c2_ring_metadata = control
        self.block_table = torch.tensor([[page] for page in pages], dtype=torch.int64)
        self.num_reqs = count
        self.compress_ratio = 2
        self.max_query_len = max(used) if used else 0
        self.slot_mapping = torch.tensor(
            [
                page * RING_ROWS + (prefix_end + step) % RING_ROWS
                for prefix_end, page, written in zip(prefix_ends, pages, used)
                for step in range(max(written, 0))
            ],
            dtype=torch.int64,
        )


class _GroupTable:
    """One KV cache group's block table, in host memory like vLLM's."""

    def __init__(self, pages: list[int]) -> None:
        # Two more columns so "first block" is a real column read, not the
        # only value in the row.
        self._cpu = torch.tensor(
            [[page, 11, 12] for page in pages], dtype=torch.int32
        )

    def get_cpu_tensor(self) -> torch.Tensor:
        return self._cpu


class _MultiGroupBlockTable:
    def __init__(self, pages: list[int]) -> None:
        self.block_tables = [_GroupTable(pages)]


class _InputBatch:
    def __init__(self, req_ids: list[str], computed: list[int], pages: list[int]) -> None:
        self.req_ids = list(req_ids)
        self.num_computed_tokens_cpu = torch.tensor(computed, dtype=torch.int64)
        self.block_table = _MultiGroupBlockTable(pages)


class _Runner:
    def __init__(self, req_ids: list[str], computed: list[int], pages: list[int]) -> None:
        self.input_batch = _InputBatch(req_ids, computed, pages)


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
        "dsv41_layerwise_admission_timing_under_test",
        root / "dsv41_layerwise_admission.py",
    )
    sys.modules["vllm_ascend.attention.dsv41_layerwise_admission"] = admission
    attention_pkg.dsv41_layerwise_admission = admission
    module = _load_module(
        "dsv41_layerwise_c2_timing_under_test", root / "dsv41_layerwise_c2.py"
    )
    module._test_context = context
    return module


@pytest.fixture(autouse=True)
def _clean(c2: types.ModuleType):
    def _reset() -> None:
        c2.clear_boundary_state_codec()
        c2.reset_step_state()
        c2._provider_installed = False
        c2._test_context["attn_metadata"] = None
        getattr(c2, "_ring_geometry", {}).clear()
        for attribute in ("ring_rows", "restore_target"):
            if hasattr(c2.provide_boundary_rows, attribute):
                delattr(c2.provide_boundary_rows, attribute)

    _reset()
    yield
    _reset()


def _install(c2: types.ModuleType, attn: _Attn) -> tuple[_Worker, _Connector]:
    ring = attn.compressor.state_cache.kv_cache[0]
    worker = _Worker({STATE_PREFIX: ring})
    connector = _Connector(worker)
    assert c2.install_boundary_provider(connector, model=_Model([attn])) is True
    return worker, connector


def _forward(c2: types.ModuleType, runner: _Runner) -> None:
    """One real forward: bind, run the layers, then unbind (as the runner does)."""
    previous = c2.bind_model_forward_step(runner, decoder_replay_dummy=False)
    c2.clear_step_request_view(previous)


def test_a_provider_only_forward_is_bound_and_captures_its_rows(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(70, 3, 4)}
    runner = _Runner(["req-a"], [70], [4])

    previous = c2.bind_model_forward_step(runner, decoder_replay_dummy=False)
    view = c2.current_step_request_view()
    assert view is not None and view.request_ids == ("req-a",)
    assert view.request_pages == ((70, 4),)
    c2.capture_layer_boundary_state(LAYER_NAME, attn)
    c2.clear_step_request_view(previous)

    record = c2.boundary_record(STATE_PREFIX, request_id="req-a")
    assert record is not None
    assert record.end_token == 73
    assert record.row_indices == (134, 135, 136)
    assert c2.provide_boundary_rows(STATE_PREFIX, "snapshot", "73", "req-a") == (
        6,
        0,
        (134, 135, 136),
    )


def test_a_step_bound_before_arming_serves_cold_instead_of_crashing(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    runner = _Runner(["req-a"], [70], [4])
    # Nothing is armed yet: this forward cannot carry rows and says so.
    assert c2.bind_model_forward_step(runner, decoder_replay_dummy=False) is c2.NOT_BOUND

    # The connector arms S1 in the middle of this very step.
    _install(c2, attn)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(70, 3, 4)}
    assert c2.capture_layer_boundary_state(LAYER_NAME, attn) is None
    assert c2.boundary_record(STATE_PREFIX) is None

    # The next forward is bound and captures normally.
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(73, 1, 4)}
    _forward(c2, _Runner(["req-a"], [73], [4]))
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(73, 1, 4)}
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(step_id=9, request_ids=("req-a",), generation=0)
    )
    c2.capture_layer_boundary_state(LAYER_NAME, attn)
    assert c2.boundary_record(STATE_PREFIX, request_id="req-a") is not None


def test_an_armed_but_unbound_forward_still_fails_closed(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _Meta(70, 3, 4)}
    with pytest.raises(c2.DSV41LayerwiseC2Error, match="without a bound step view"):
        c2.capture_layer_boundary_state(LAYER_NAME, attn)


def test_the_page_plan_names_the_target_after_the_forward(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    runner = _Runner(["req-a"], [70], [4])
    _forward(c2, runner)

    # The transport loads after the forward: no forward context, no step view.
    c2._test_context["attn_metadata"] = None
    assert c2.restore_target(STATE_PREFIX, "70", "req-a") == (4, RING_ROWS)
    assert c2.restore_target(STATE_PREFIX, "70") == (4, RING_ROWS)
    # A token the plan never ended at is refused rather than guessed.
    assert c2.restore_target(STATE_PREFIX, "71", "req-a") is None
    assert c2.restore_target(STATE_PREFIX, "71") is None


def test_the_page_plan_still_separates_concurrent_requests(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    runner = _Runner(["req-a", "req-b"], [70, 70], [4, 5])
    _forward(c2, runner)

    c2._test_context["attn_metadata"] = None
    # Two requests resume at the same token with different pages: unnamed
    # restores stay fail-closed, named ones get their own page.
    assert c2.restore_target(STATE_PREFIX, "70") is None
    assert c2.restore_target(STATE_PREFIX, "70", "req-a") == (4, RING_ROWS)
    assert c2.restore_target(STATE_PREFIX, "70", "req-b") == (5, RING_ROWS)
    # Only the requests of the bound step are addressable.
    assert c2.restore_target(STATE_PREFIX, "70", "req-c") is None


def test_live_metadata_still_wins_when_the_forward_is_in_flight(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _MultiMeta([73, 74], [3, 3], [4, 5])}
    previous = c2.bind_model_forward_step(
        _Runner(["req-a", "req-b"], [73, 74], [4, 5]),
        decoder_replay_dummy=False,
    )
    assert c2.restore_target(STATE_PREFIX, "74", "req-b") == (5, RING_ROWS)
    assert c2.restore_target(STATE_PREFIX, "73") == (4, RING_ROWS)
    c2.clear_step_request_view(previous)


class _StateMeta:
    """Compressor-state metadata: token slots stay PAD, only the ring counts."""

    def __init__(self, prefix_end: int, used: int, page: int) -> None:
        control = torch.zeros(5, 1, dtype=torch.int32)
        control[0, 0] = prefix_end
        control[1, 0] = used
        control[4, 0] = page
        self.c2_ring_metadata = control
        self.block_table = torch.tensor([[page]], dtype=torch.int64)
        self.num_reqs = 1
        self.compress_ratio = 2
        self.max_query_len = used
        self.is_compressor_state = True
        # Deliberately unrelated to the ring: the engine must not read these.
        self.slot_mapping = torch.full((max(used, 0),), 7, dtype=torch.int64)


def test_a_compressor_state_layer_derives_the_rows_the_kernel_wrote(
    c2: types.ModuleType,
) -> None:
    attn = _Attn()
    _install(c2, attn)
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _StateMeta(70, 3, 4)}
    _forward(c2, _Runner(["req-a"], [70], [4]))

    c2._test_context["attn_metadata"] = {STATE_PREFIX: _StateMeta(70, 3, 4)}
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(step_id=11, request_ids=("req-a",), generation=0)
    )
    c2.capture_layer_boundary_state(LAYER_NAME, attn)
    record = c2.boundary_record(STATE_PREFIX, request_id="req-a")
    assert record is not None
    assert record.row_indices == (134, 135, 136)

    # The kernel keeps the last ``ring_rows`` tokens of a longer step.
    c2._test_context["attn_metadata"] = {STATE_PREFIX: _StateMeta(70, 39, 4)}
    c2.bind_step_request_view(
        c2.DSV41StepRequestView(step_id=12, request_ids=("req-a",), generation=0)
    )
    c2.capture_layer_boundary_state(LAYER_NAME, attn)
    record = c2.boundary_record(STATE_PREFIX, request_id="req-a", step_id=12)
    assert record is not None
    expected = tuple(
        4 * RING_ROWS + ((70 + 39 - 32 + index) % RING_ROWS) for index in range(32)
    )
    assert record.row_indices == expected
    assert len(set(record.row_indices)) == RING_ROWS
