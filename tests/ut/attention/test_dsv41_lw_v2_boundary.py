# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the DSV4.1 layerwise v2 boundary / step / ring seam.

``dsa_v41.py`` and ``model_runner.py`` need a vLLM install, so the topology,
the group seam and the boundary module are loaded by path inside lightweight
stubs. ``vllm.forward_context`` is stubbed per test, which lets the hook see a
fully synthetic live ring set built from real ``SimpleNamespace`` facts rather
than mocked engine methods.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_PATH = REPO_ROOT / "vllm_ascend/models/deepseek_v41/topology.py"
SEAM_PATH = REPO_ROOT / "vllm_ascend/attention/dsv41_lw_v2.py"
BOUNDARY_PATH = REPO_ROOT / "vllm_ascend/attention/dsv41_lw_v2_boundary.py"
MODEL_RUNNER_PATH = REPO_ROOT / "vllm_ascend/worker/v1/model_runner.py"

BOUNDARY_LAYERS = (2, 8, 14)
ALL_GROUPS = ("g2", "g8", "g14", "g20")
POSITION_CONTROL_WORDS = 5


def _dispatch(layer: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{layer}.self_attn.v41_attn"


def _execution(layer: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{layer}.self_attn.attn"


def _resource(layer: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{layer}.self_attn.compressor.state_cache"


_ENGINE_MODULES = (
    ("vllm_ascend.models.deepseek_v41.topology", TOPOLOGY_PATH),
    ("vllm_ascend.attention.dsv41_lw_v2", SEAM_PATH),
    ("vllm_ascend.attention.dsv41_lw_v2_boundary", BOUNDARY_PATH),
)


def _load_engine_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load topology, the group seam and the boundary seam by path.

    The dotted leaf names are registered directly so the modules' own
    ``from vllm_ascend....<leaf> import ...`` statements resolve without
    executing the real ``vllm_ascend`` package (it needs vLLM). Every entry
    this loader added is removed afterwards.
    """
    loaded: dict[str, ModuleType] = {}
    added: list[str] = []
    try:
        for module_name, module_path in _ENGINE_MODULES:
            existing = sys.modules.get(module_name)
            if existing is not None:
                loaded[module_name] = existing
                continue
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            added.append(module_name)
            loaded[module_name] = module
            spec.loader.exec_module(module)
    finally:
        for module_name in reversed(added):
            if sys.modules.get(module_name) is loaded.get(module_name):
                del sys.modules[module_name]
    return (
        loaded["vllm_ascend.models.deepseek_v41.topology"],
        loaded["vllm_ascend.attention.dsv41_lw_v2"],
        loaded["vllm_ascend.attention.dsv41_lw_v2_boundary"],
    )


TOPOLOGY, SEAM, BOUNDARY = _load_engine_modules()

BoundaryRequestView = BOUNDARY.BoundaryRequestView
BoundaryTensorView = BOUNDARY.BoundaryTensorView
DSV41BoundaryBindingError = BOUNDARY.DSV41BoundaryBindingError
DSV41BoundaryNotBoundError = BOUNDARY.DSV41BoundaryNotBoundError
DSV41BoundaryStateError = BOUNDARY.DSV41BoundaryStateError
DSV41LwV2BoundaryHook = BOUNDARY.DSV41LwV2BoundaryHook


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Fresh engine binding, stubbed forward context and no codec per test."""
    BOUNDARY.uninstall_dsv41_lw_v2_codec()
    SEAM.unbind_lw_v2_engine()
    forward_context = SimpleNamespace(no_compile_layers={}, attn_metadata={}, capturing=False)
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    forward_context_module = ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: forward_context
    vllm_module.forward_context = forward_context_module
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)
    yield forward_context
    BOUNDARY.uninstall_dsv41_lw_v2_codec()
    SEAM.unbind_lw_v2_engine()


@pytest.fixture
def text_config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        index_topk=512,
        engram_layer_ids=[1, 14],
    )


# --------------------------------------------------------------------------
# Fakes: a real-shaped connector, codec, model module tree and runner.
# --------------------------------------------------------------------------


class _RecordingCodec:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.fail_on = fail_on

    def capture_layer_boundary(self, request_view, tensor_view, *, stream=None):
        self.calls.append(("capture", request_view, tensor_view))
        if self.fail_on == "capture":
            raise RuntimeError("capture boom")

    def restore_layer_boundary(self, request_view, tensor_view, *, stream=None):
        self.calls.append(("restore", request_view, tensor_view))
        if self.fail_on == "restore":
            raise RuntimeError("restore boom")


class _FakeConnector:
    """Structural double of DSV41LwV2Connector's public seam."""

    def __init__(self, *, step=None, codec=None, prefix="model", plan=True) -> None:
        self._step = step
        self._codec = codec
        self._prefix = prefix
        self._plan = plan
        self.bound_hook = None
        self.events: list[tuple[str, str]] = []

    @property
    def grouped_plan(self):
        if not self._plan:
            return None
        # The production plan derives from the registered cache names, so its
        # first layer carries the real model root (not a hard-coded "model").
        return SimpleNamespace(
            base=SimpleNamespace(layer_order=(_execution(0, self._prefix),))
        )

    @property
    def current_step(self):
        return self._step

    @property
    def boundary_codec(self):
        return self._codec

    def group_for_layer(self, layer_name):
        return None

    def bind_boundary_hook(self, hook):
        self.bound_hook = hook

    def wait_group_load(self, group_id):
        self.events.append(("wait_group_load", group_id))

    def restore_group_boundary(self, group_id):
        self.events.append(("restore_group_boundary", group_id))

    def capture_group_boundary(self, group_id):
        self.events.append(("capture_group_boundary", group_id))

    def submit_group_store(self, group_id):
        self.events.append(("submit_group_store", group_id))


def _ring(*, pages: int = 6, rows: int = 8, head: int = 2, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros((pages, rows, 2 * head), dtype=dtype)


def _compressor_module(layer: int, ring: torch.Tensor, prefix: str = "model"):
    return SimpleNamespace(
        v41_impl=SimpleNamespace(
            compressor_state_prefix=_resource(layer, prefix),
            layer_name=_execution(layer, prefix),
        ),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[ring])),
    )


def _attention(layer: int, ring: torch.Tensor, prefix: str = "model"):
    return SimpleNamespace(
        v41_impl=SimpleNamespace(
            compressor_state_prefix=_resource(layer, prefix),
            layer_name=_execution(layer, prefix),
        ),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[ring])),
        v41_layer_name=_dispatch(layer, prefix),
        layerwise_execution_layer=_execution(layer, prefix),
    )


class _FakeModel:
    def __init__(self, modules, config=None) -> None:
        self._modules = tuple(modules)
        self.config = config

    def modules(self):
        return iter(self._modules)


class _FakeBlockTable:
    def __init__(self, pages) -> None:
        self._tensor = torch.tensor([[page, 0] for page in pages], dtype=torch.int32)

    def get_cpu_tensor(self):
        return self._tensor


class _FakeInputBatch:
    def __init__(self, req_ids, computed, pages) -> None:
        self.req_ids = list(req_ids)
        self.num_reqs = len(req_ids)
        self.num_computed_tokens_cpu = np.asarray(computed, dtype=np.int32)
        self.block_table = SimpleNamespace(block_tables=[_FakeBlockTable(pages)])


class _FakeRunner:
    def __init__(self, req_ids, computed, pages, query_lens) -> None:
        self.input_batch = _FakeInputBatch(req_ids, computed, pages)
        self.query_lens = torch.tensor(query_lens, dtype=torch.int32)


def _transport_request(request_id, *, load: int = 0, store: int = 0):
    return SimpleNamespace(
        request_id=request_id, load_prefix_end=load, store_prefix_end=store
    )


def _transport_step(requests, *, epoch: int = 4, generation: int = 7, synthetic: bool = False):
    return SimpleNamespace(
        step_id=0,
        registration_epoch=epoch,
        server_generation=generation,
        synthetic=synthetic,
        requests=tuple(requests),
    )


def _metadata(pages, *, ratio: int = 2, compressor: bool = True):
    block_table = torch.tensor([[page, 0] for page in pages], dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(pages),
        compress_ratio=ratio,
        block_table=block_table,
        is_compressor_state=compressor,
    )


def _bind_engine(text_config):
    connector = _FakeConnector()
    SEAM.bind_lw_v2_engine(connector, topology=TOPOLOGY.build_layer_plan(text_config))
    return connector


def _install(text_config, forward_context, *, codec=None, layers=BOUNDARY_LAYERS, pages=(3, 7)):
    connector = _bind_engine(text_config)
    rings = {layer: _ring() for layer in layers}
    model = _FakeModel([_compressor_module(layer, rings[layer]) for layer in layers])
    forward_context.no_compile_layers = {
        _dispatch(layer): _attention(layer, rings[layer]) for layer in layers
    }
    forward_context.attn_metadata = {
        _resource(layer): _metadata(list(pages)) for layer in layers
    }
    codec = codec or _RecordingCodec()
    BOUNDARY.install_dsv41_lw_v2_codec(codec, model=model)
    return connector, codec, rings


# --------------------------------------------------------------------------
# Install: fail closed on codec, resource identity and plane geometry.
# --------------------------------------------------------------------------


def test_install_requires_a_bound_engine_seam(text_config):
    model = _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS])
    with pytest.raises(DSV41BoundaryBindingError, match="engine seam is not bound"):
        BOUNDARY.install_dsv41_lw_v2_codec(_RecordingCodec(), model=model)


def test_install_binds_three_boundary_resources_and_one_hook(text_config, _isolated):
    connector, codec, _rings = _install(text_config, _isolated)
    assert BOUNDARY.lw_v2_boundary_resource_ids() == tuple(
        _resource(layer) for layer in BOUNDARY_LAYERS
    )
    assert BOUNDARY.lw_v2_boundary_codec() is codec
    assert isinstance(connector.bound_hook, DSV41LwV2BoundaryHook)
    # Same instance is idempotent, a different one conflicts.
    BOUNDARY.install_dsv41_lw_v2_codec(codec, model=model_of(connector))
    with pytest.raises(DSV41BoundaryBindingError, match="different layerwise v2 boundary codec"):
        BOUNDARY.install_dsv41_lw_v2_codec(_RecordingCodec(), model=model_of(connector))


def model_of(connector):
    """Rebuild the three-ring model tree the connector was installed with."""
    return _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS])


def test_install_rejects_a_codec_without_capture(text_config, _isolated):
    connector = _bind_engine(text_config)
    codec = SimpleNamespace(
        restore_layer_boundary=lambda request_view, tensor_view, *, stream=None: None
    )
    model = _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS])
    with pytest.raises(DSV41BoundaryBindingError, match="capture_layer_boundary"):
        BOUNDARY.install_dsv41_lw_v2_codec(codec, model=model)


def test_install_rejects_a_wrong_codec_signature(text_config, _isolated):
    _bind_engine(text_config)

    class _BadCodec:
        def capture_layer_boundary(self, request_view):
            pass

        def restore_layer_boundary(self, request_view, tensor_view, *, stream=None):
            pass

    model = _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS])
    with pytest.raises(DSV41BoundaryBindingError, match="must accept"):
        BOUNDARY.install_dsv41_lw_v2_codec(_BadCodec(), model=model)


def test_install_rejects_a_missing_boundary_resource(text_config, _isolated):
    _bind_engine(text_config)
    model = _FakeModel(
        [_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS[:2]]
    )
    with pytest.raises(DSV41BoundaryBindingError, match="do not match the bound plan"):
        BOUNDARY.install_dsv41_lw_v2_codec(_RecordingCodec(), model=model)


@pytest.mark.parametrize(
    "break_ring",
    [
        pytest.param(lambda: _ring(dtype=torch.float16), id="not-float32"),
        pytest.param(lambda: torch.zeros((6, 8, 3), dtype=torch.float32), id="odd-width"),
        pytest.param(lambda: torch.zeros((6, 8, 16), dtype=torch.float32)[:, ::2, :], id="non-contiguous"),
    ],
)
def test_install_rejects_a_ring_that_cannot_hold_two_full_planes(
    text_config, _isolated, break_ring
):
    _bind_engine(text_config)
    rings = [_ring() for _ in BOUNDARY_LAYERS]
    rings[0] = break_ring()
    model = _FakeModel(
        [_compressor_module(layer, ring) for layer, ring in zip(BOUNDARY_LAYERS, rings)]
    )
    with pytest.raises(DSV41BoundaryBindingError):
        BOUNDARY.install_dsv41_lw_v2_codec(_RecordingCodec(), model=model)


def test_install_from_connector_leaves_stock_connectors_alone(text_config):
    model = _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS])
    model.config = text_config
    assert BOUNDARY.install_from_connector(model=model, connector=object()) is False
    assert SEAM.lw_v2_engine() is None
    assert BOUNDARY.lw_v2_boundary_codec() is None


def test_install_from_connector_binds_engine_and_installs_codec(text_config):
    codec = _RecordingCodec()
    connector = _FakeConnector(codec=codec)
    model = _FakeModel(
        [_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS],
        config=text_config,
    )
    assert BOUNDARY.install_from_connector(model=model, connector=connector) is True
    binding = SEAM.lw_v2_engine()
    assert binding is not None and binding.hooks is connector
    assert BOUNDARY.lw_v2_boundary_codec() is codec
    assert isinstance(connector.bound_hook, DSV41LwV2BoundaryHook)


def test_install_from_connector_uses_the_registered_model_prefix(
    text_config, _isolated
):
    prefix = "language_model.model"
    codec = _RecordingCodec()
    connector = _FakeConnector(codec=codec, prefix=prefix)
    rings = {layer: _ring() for layer in BOUNDARY_LAYERS}
    model = _FakeModel(
        [
            _compressor_module(layer, rings[layer], prefix=prefix)
            for layer in BOUNDARY_LAYERS
        ],
        config=text_config,
    )

    assert BOUNDARY.install_from_connector(model=model, connector=connector) is True
    binding = SEAM.lw_v2_engine()
    assert binding is not None and binding.prefix == prefix
    assert all(group.readiness_layer.startswith(prefix) for group in binding.groups)

    _isolated.no_compile_layers = {
        _dispatch(layer, prefix): _attention(layer, rings[layer], prefix=prefix)
        for layer in BOUNDARY_LAYERS
    }
    _isolated.attn_metadata = {
        _resource(layer, prefix): _metadata([3, 7])
        for layer in BOUNDARY_LAYERS
    }
    runner = _FakeRunner(
        req_ids=["r0", "r1"], computed=[5, 9], pages=[3, 7], query_lens=[2, 3]
    )
    token = BOUNDARY.bind_forward_step(runner)
    try:
        connector._step = _transport_step([_transport_request("r0", store=5)])
        connector.bound_hook.capture_group_boundary("g2")
    finally:
        BOUNDARY.clear_forward_step(token)

    assert [call[0] for call in codec.calls] == ["capture"]
    view, tensor_view = codec.calls[0][1], codec.calls[0][2]
    assert view.resource_id == _resource(2, prefix)
    assert view.dispatch_layer == _dispatch(2, prefix)
    assert view.execution_layer == _execution(2, prefix)
    assert tensor_view.resource_id == _resource(2, prefix)


def test_install_from_connector_derives_the_prefix_without_a_plan(
    text_config, _isolated
):
    prefix = "language_model.model"
    connector = _FakeConnector(prefix=prefix, plan=False)
    model = _FakeModel(
        [_compressor_module(layer, _ring(), prefix=prefix) for layer in BOUNDARY_LAYERS],
        config=text_config,
    )

    # No grouped plan yet: the live compressor-state resource names are the
    # only registered truth available, and guessing "model" would be wrong.
    assert BOUNDARY.install_from_connector(model=model, connector=connector) is False
    binding = SEAM.lw_v2_engine()
    assert binding is not None and binding.prefix == prefix


def test_install_from_connector_without_a_codec_stays_inert(text_config, caplog):
    connector = _FakeConnector(codec=None)
    model = _FakeModel(
        [_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS],
        config=text_config,
    )
    with caplog.at_level("WARNING"):
        assert BOUNDARY.install_from_connector(model=model, connector=connector) is False
    assert "no boundary codec" in caplog.text
    assert SEAM.lw_v2_engine() is not None
    assert connector.bound_hook is None
    assert BOUNDARY.lw_v2_boundary_codec() is None


def test_install_from_connector_rejects_a_non_v2_topology():
    connector = _FakeConnector(codec=_RecordingCodec())
    config = SimpleNamespace(
        num_hidden_layers=2,
        compress_ratios=[0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        index_topk=512,
        engram_layer_ids=[1, 14],
    )
    model = _FakeModel([_compressor_module(layer, _ring()) for layer in BOUNDARY_LAYERS], config=config)
    with pytest.raises(DSV41BoundaryBindingError, match="rejected the model topology"):
        BOUNDARY.install_from_connector(model=model, connector=connector)


# --------------------------------------------------------------------------
# Forward step binding: host facts only, synthetic steps carry no requests.
# --------------------------------------------------------------------------


def test_step_binding_is_inert_until_a_codec_is_installed():
    token = BOUNDARY.bind_forward_step(None)
    assert BOUNDARY.forward_step() is None
    BOUNDARY.clear_forward_step(token)


def test_step_binding_reads_real_host_facts(text_config, _isolated):
    _install(text_config, _isolated)
    runner = _FakeRunner(
        req_ids=["r0", "r1"], computed=[5, 9], pages=[3, 7], query_lens=[2, 3]
    )
    token = BOUNDARY.bind_forward_step(runner)
    step = BOUNDARY.forward_step()
    assert step is not None and not step.synthetic
    first, second = step.requests
    assert (first.request_id, first.request_index) == ("r0", 0)
    assert (first.prefix_end, first.step_tokens, first.out_base, first.in_base) == (5, 2, 0, 0)
    assert first.ring_page == 3
    assert (second.request_id, second.request_index) == ("r1", 1)
    assert (second.prefix_end, second.step_tokens, second.out_base, second.in_base) == (9, 3, 2, 2)
    assert second.ring_page == 7
    BOUNDARY.clear_forward_step(token)


def test_step_binding_keeps_scheduler_arrays_on_the_host(text_config, _isolated):
    _install(text_config, _isolated)
    runner = _FakeRunner(req_ids=["r0"], computed=[5], pages=[3], query_lens=[2])
    runner.query_lens = torch.empty(1, dtype=torch.int32, device="meta")
    with pytest.raises(DSV41BoundaryStateError, match="must stay on the host"):
        BOUNDARY.bind_forward_step(runner)


def test_step_binding_rejects_unset_or_duplicate_request_ids(text_config, _isolated):
    _install(text_config, _isolated)
    unset = _FakeRunner(req_ids=[None, "r1"], computed=[5, 9], pages=[3, 7], query_lens=[2, 3])
    with pytest.raises(DSV41BoundaryStateError, match="unset request id"):
        BOUNDARY.bind_forward_step(unset)
    duplicate = _FakeRunner(req_ids=["r0", "r0"], computed=[5, 9], pages=[3, 7], query_lens=[2, 3])
    with pytest.raises(DSV41BoundaryStateError, match="duplicate request ids"):
        BOUNDARY.bind_forward_step(duplicate)


def test_synthetic_steps_carry_no_requests_and_never_read_the_runner(text_config, _isolated):
    _install(text_config, _isolated)
    BOUNDARY.bind_forward_step(None, synthetic=True)
    step = BOUNDARY.forward_step()
    assert step is not None and step.synthetic and step.requests == ()
    BOUNDARY.clear_forward_step(None)
    BOUNDARY.bind_forward_step(None, decoder_replay_dummy=True)
    assert BOUNDARY.forward_step().synthetic is True


def test_step_ids_advance_and_clearing_restores_the_previous_binding(text_config, _isolated):
    _install(text_config, _isolated)
    runner = _FakeRunner(req_ids=["r0"], computed=[5], pages=[3], query_lens=[2])
    first = BOUNDARY.bind_forward_step(runner, synthetic=True)
    first_step_id = BOUNDARY.forward_step().step_id
    previous = BOUNDARY.bind_forward_step(runner, synthetic=True)
    assert previous is not None and previous.step_id == first_step_id
    assert BOUNDARY.forward_step().step_id == first_step_id + 1
    BOUNDARY.clear_forward_step(previous)
    assert BOUNDARY.forward_step().step_id == first_step_id
    BOUNDARY.clear_forward_step(None)
    assert BOUNDARY.forward_step() is None


# --------------------------------------------------------------------------
# Group hooks: real DTOs, live page relocation, hit/store filtering.
# --------------------------------------------------------------------------


def _hook_setup(text_config, forward_context, *, pages=(3, 7), codec=None):
    connector, codec, _rings = _install(text_config, forward_context, codec=codec)
    runner = _FakeRunner(
        req_ids=["r0", "r1"], computed=[5, 9], pages=list(pages), query_lens=[2, 3]
    )
    token = BOUNDARY.bind_forward_step(runner)
    return connector, codec, token


def test_restore_hands_the_codec_real_live_facts(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step(
        [_transport_request("r0", load=5, store=5), _transport_request("r1", load=9)]
    )
    hook = connector.bound_hook
    hook.restore_group_boundary("g2")
    assert [call[0] for call in codec.calls] == ["restore", "restore"]
    action, view, tensor_view = codec.calls[0]
    assert action == "restore"
    step = BOUNDARY.forward_step()
    assert view.request_id == "r0" and view.request_index == 0
    assert view.step_id == step.step_id
    assert (view.registration_epoch, view.server_generation) == (4, 7)
    assert view.resource_id == _resource(2)
    assert view.dispatch_layer == _dispatch(2)
    assert view.execution_layer == _execution(2)
    assert (view.prefix_end, view.step_tokens, view.boundary_end) == (5, 2, 7)
    assert view.ring_length == 8 and view.ring_page == 3
    assert view.next_slot == 5 and view.compression_phase == 1
    assert view.position_control == (5, 2, 0, 0, 3)
    assert len(view.position_control) == POSITION_CONTROL_WORDS
    assert view.written_rows == (3 * 8 + 5, 3 * 8 + 6)
    assert tensor_view.resource_id == _resource(2)
    assert tensor_view.page_for(view) == 3
    assert tensor_view.ring.shape == (6, 8, 4)
    assert tensor_view.ring_rows == 8 and tensor_view.ring_head_size == 2
    kv_plane, scores_plane = tensor_view.planes()
    assert kv_plane["row_offset_bytes"] == 0
    assert scores_plane["row_offset_bytes"] == 2 * tensor_view.element_size
    assert kv_plane["shape"] == scores_plane["shape"] == (8, 2)
    BOUNDARY.clear_forward_step(token)


def test_restore_relocates_to_the_live_page_not_the_capture_page(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated, pages=(3, 7))
    connector._step = _transport_step(
        [_transport_request("r0", store=5), _transport_request("r1")]
    )
    hook = connector.bound_hook
    hook.capture_group_boundary("g2")
    capture_view = codec.calls[0][1]
    assert capture_view.ring_page == 3
    BOUNDARY.clear_forward_step(token)

    runner = _FakeRunner(req_ids=["r0", "r1"], computed=[5, 9], pages=[7, 3], query_lens=[2, 3])
    token = BOUNDARY.bind_forward_step(runner)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    hook.restore_group_boundary("g2")
    restore_view = codec.calls[-1][1]
    assert restore_view.request_id == "r0"
    assert restore_view.ring_page == 7
    assert restore_view.written_rows == (7 * 8 + 5, 7 * 8 + 6)
    assert restore_view.written_rows != capture_view.written_rows
    assert restore_view.position_control[4] == 7
    BOUNDARY.clear_forward_step(token)


def test_each_request_keeps_its_own_page_and_index(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated, pages=(3, 7))
    connector._step = _transport_step(
        [_transport_request("r0", load=5), _transport_request("r1", load=9)]
    )
    connector.bound_hook.restore_group_boundary("g14")
    views = [call[1] for call in codec.calls]
    assert [view.request_id for view in views] == ["r0", "r1"]
    assert [view.request_index for view in views] == [0, 1]
    assert [view.ring_page for view in views] == [3, 7]
    BOUNDARY.clear_forward_step(token)


def test_hit_and_store_sets_come_from_the_transport_step(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step(
        [_transport_request("r0", store=7), _transport_request("r1", load=9)]
    )
    hook = connector.bound_hook
    hook.restore_group_boundary("g8")
    assert [call[1].request_id for call in codec.calls] == ["r1"]
    codec.calls.clear()
    hook.capture_group_boundary("g8")
    assert [call[1].request_id for call in codec.calls] == ["r0"]
    BOUNDARY.clear_forward_step(token)


def test_a_step_without_hits_or_stores_is_an_explicit_no_op(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step([])
    connector.bound_hook.restore_group_boundary("g2")
    connector.bound_hook.capture_group_boundary("g2")
    assert codec.calls == []
    BOUNDARY.clear_forward_step(token)


def test_g20_and_unknown_groups_have_no_boundary(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step([_transport_request("r0", store=5)])
    hook = connector.bound_hook
    with pytest.raises(DSV41BoundaryBindingError, match="declares no boundary resource"):
        hook.capture_group_boundary("g20")
    with pytest.raises(DSV41BoundaryBindingError, match="unknown layerwise v2 group"):
        hook.capture_group_boundary("g99")
    BOUNDARY.clear_forward_step(token)


def test_a_real_hit_without_a_codec_raises_boundary_not_bound(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    hook = connector.bound_hook
    BOUNDARY.uninstall_dsv41_lw_v2_codec()
    with pytest.raises(DSV41BoundaryNotBoundError, match="BOUNDARY_NOT_BOUND") as error:
        hook.restore_group_boundary("g2")
    assert error.value.error_code == "BOUNDARY_NOT_BOUND"
    BOUNDARY.clear_forward_step(token)


def test_a_hook_without_a_real_forward_step_fails_closed(text_config, _isolated):
    connector, _codec, _token = _hook_setup(text_config, _isolated)
    BOUNDARY.clear_forward_step(None)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="without a real forward step"):
        connector.bound_hook.restore_group_boundary("g2")


def test_a_synthetic_step_emits_no_boundary_operation(text_config, _isolated):
    connector, codec, token = _hook_setup(text_config, _isolated)
    BOUNDARY.clear_forward_step(token)
    runner = _FakeRunner(req_ids=["r0"], computed=[5], pages=[3], query_lens=[2])
    token = BOUNDARY.bind_forward_step(runner, synthetic=True)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="without a real forward step"):
        connector.bound_hook.restore_group_boundary("g2")
    assert codec.calls == []
    BOUNDARY.clear_forward_step(token)


def test_a_missing_transport_step_fails_closed(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    connector._step = None
    with pytest.raises(DSV41BoundaryStateError, match="without a current transport step"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


def test_forward_and_metadata_batches_must_agree(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    _isolated.attn_metadata[_resource(2)] = _metadata([3, 7, 1])
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="disagree on the batch"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


def test_non_compressor_metadata_cannot_build_a_schema2_payload(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    _isolated.attn_metadata[_resource(2)] = _metadata([3, 7], compressor=False)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="not a compressor-state"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


def test_live_ring_geometry_must_match_install(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    _isolated.no_compile_layers[_dispatch(2)] = _attention(2, _ring(rows=16))
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="geometry changed"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


def test_a_transport_request_outside_the_forward_is_rejected(text_config, _isolated):
    connector, _codec, token = _hook_setup(text_config, _isolated)
    connector._step = _transport_step([_transport_request("r9", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="not part of this forward"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


def test_codec_failures_are_wrapped_and_never_swallowed(text_config, _isolated):
    codec = _RecordingCodec(fail_on="restore")
    connector, _codec, token = _hook_setup(text_config, _isolated, codec=codec)
    connector._step = _transport_step([_transport_request("r0", load=5)])
    with pytest.raises(DSV41BoundaryStateError, match="failed for request 'r0'"):
        connector.bound_hook.restore_group_boundary("g2")
    BOUNDARY.clear_forward_step(token)


# --------------------------------------------------------------------------
# DTO validators: schema2 full-page payloads only.
# --------------------------------------------------------------------------


def _request_view_kwargs(**overrides):
    kwargs = {
        "request_id": "r0",
        "request_index": 0,
        "step_id": 3,
        "registration_epoch": 4,
        "server_generation": 7,
        "resource_id": _resource(2),
        "dispatch_layer": _dispatch(2),
        "execution_layer": _execution(2),
        "prefix_end": 5,
        "step_tokens": 2,
        "ring_length": 8,
        "ring_page": 3,
        "next_slot": 5,
        "compression_ratio": 2,
        "compression_phase": 1,
        "position_control": (5, 2, 0, 0, 3),
        "written_rows": (29, 30),
    }
    kwargs.update(overrides)
    return kwargs


def test_request_view_accepts_one_page_aligned_schema2_record():
    view = BoundaryRequestView(**_request_view_kwargs())
    assert view.boundary_end == 7


@pytest.mark.parametrize(
    "overrides, message",
    [
        pytest.param({"written_rows": (32 + 29, 32 + 30)}, "inside this request's live ring page", id="off-page-row"),
        pytest.param({"position_control": (5, 2, 0, 0)}, "position_control", id="short-control"),
        pytest.param({"compression_phase": 2}, "compression_phase", id="phase-outside-ratio"),
        pytest.param({"next_slot": 6}, "next_slot", id="wrong-next-slot"),
        pytest.param({"step_tokens": 0}, "written_rows", id="rows-must-match-step"),
        pytest.param({"ring_page": True}, "ring_page", id="bool-is-not-an-int"),
    ],
)
def test_request_view_rejects_non_schema2_facts(overrides, message):
    with pytest.raises(ValueError, match=message):
        BoundaryRequestView(**_request_view_kwargs(**overrides))


def test_tensor_view_describes_both_planes_and_rejects_float_page_ids():
    tensor_view = BoundaryTensorView(
        resource_id=_resource(2),
        dispatch_layer=_dispatch(2),
        execution_layer=_execution(2),
        ring=_ring(),
        page_ids=torch.tensor([3, 7], dtype=torch.int32),
        ring_rows=8,
        ring_head_size=2,
    )
    kv_plane, scores_plane = tensor_view.planes()
    assert kv_plane["row_offset_bytes"] == 0
    assert scores_plane["row_offset_bytes"] == kv_plane["shape"][1] * tensor_view.element_size
    assert tensor_view.page_for(BoundaryRequestView(**_request_view_kwargs())) == 3
    with pytest.raises(ValueError, match="1-D integer tensor"):
        BoundaryTensorView(
            resource_id=_resource(2),
            dispatch_layer=_dispatch(2),
            execution_layer=_execution(2),
            ring=_ring(),
            page_ids=torch.tensor([3.0], dtype=torch.float32),
            ring_rows=8,
            ring_head_size=2,
        )


# --------------------------------------------------------------------------
# Static checks: no legacy layerwise import and the model runner wires it.
# --------------------------------------------------------------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_boundary_module_does_not_import_legacy_layerwise_state():
    modules = _imported_modules(BOUNDARY_PATH)
    legacy = {name for name in modules if name.startswith("vllm_ascend.attention.dsv41_layerwise")}
    assert legacy == set()
    source = BOUNDARY_PATH.read_text()
    assert "admit_step" not in source
    for forbidden in (".tolist(", ".cpu(", ".item("):
        assert forbidden not in source


def test_model_runner_wires_binds_and_clears_the_v2_step():
    source = MODEL_RUNNER_PATH.read_text()
    assert "from vllm_ascend.attention import dsv41_lw_v2_boundary" in source
    assert "dsv41_lw_v2_boundary.install_from_connector(model=self.model)" in source
    assert "dsv41_lw_v2_boundary.bind_forward_step(" in source
    assert "dsv41_lw_v2_boundary.clear_forward_step(" in source
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert {"bind_forward_step", "clear_forward_step", "install_from_connector"} <= calls
