# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the DeepSeek V4.1 layerwise v2 engine seam.

``dsa_v41.py`` needs a vLLM install, so the seam and the topology it derives
are loaded by path inside lightweight stub packages. The seam's behaviour is
then exercised with a recording connector facade, and the dispatch that
``dsa_v41_forward`` performs is checked statically.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_PATH = REPO_ROOT / "vllm_ascend/models/deepseek_v41/topology.py"
SEAM_PATH = REPO_ROOT / "vllm_ascend/attention/dsv41_lw_v2.py"
DSA_V41_PATH = REPO_ROOT / "vllm_ascend/attention/dsa_v41.py"

HEAD_LAYERS = (2, 8, 14, 20)
BOUNDARY_HEAD_LAYERS = (2, 8, 14)
GROUP_HOOK_NAMES = (
    "wait_group_load",
    "restore_group_boundary",
    "capture_group_boundary",
    "submit_group_store",
)


_ENGINE_MODULES = (
    ("vllm_ascend.models.deepseek_v41.topology", TOPOLOGY_PATH),
    ("vllm_ascend.attention.dsv41_lw_v2", SEAM_PATH),
)


def _load_engine_modules() -> tuple[ModuleType, ModuleType]:
    """Load topology and the seam by path without executing the real package.

    The dotted leaf names are registered directly, so the seam's
    ``from vllm_ascend.models.deepseek_v41.topology import ...`` resolves
    without importing the vllm_ascend ``__init__`` modules (they need vLLM).
    Every entry this test added is removed again afterwards, so a real
    ``vllm_ascend`` install is never shadowed by a duplicate module object.
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
    )


TOPOLOGY, SEAM = _load_engine_modules()

bind_lw_v2_engine = SEAM.bind_lw_v2_engine
unbind_lw_v2_engine = SEAM.unbind_lw_v2_engine
run_group_forward = SEAM.run_group_forward
DSV41LwV2BindingError = SEAM.DSV41LwV2BindingError
DSV41LwV2EngineBinding = SEAM.DSV41LwV2EngineBinding


def _dispatch(layer: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{layer}.self_attn.v41_attn"


class RecordingHooks:
    """Connector facade double that records the group protocol it receives."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.fail_on: str | None = None

    def _record(self, name: str, group_id: str) -> None:
        self.events.append((name, group_id))
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed for {group_id}")

    def wait_group_load(self, group_id: str) -> None:
        self._record("wait_group_load", group_id)

    def restore_group_boundary(self, group_id: str) -> None:
        self._record("restore_group_boundary", group_id)

    def capture_group_boundary(self, group_id: str) -> None:
        self._record("capture_group_boundary", group_id)

    def submit_group_store(self, group_id: str) -> None:
        self._record("submit_group_store", group_id)


def _record_forward(events: list[tuple[str, str]], layer: int):
    def forward() -> None:
        events.append(("forward", f"layer{layer}"))

    return forward


@pytest.fixture(autouse=True)
def _clean_binding():
    unbind_lw_v2_engine()
    yield
    unbind_lw_v2_engine()


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


@pytest.fixture
def topology(text_config):
    return TOPOLOGY.build_layer_plan(text_config)


@pytest.fixture
def bound(topology):
    hooks = RecordingHooks()
    binding = bind_lw_v2_engine(hooks, topology=topology)
    return binding, hooks


def test_engine_is_off_until_a_connector_binds():
    assert SEAM.lw_v2_engine() is None
    assert SEAM.lw_v2_active() is False


def test_group_heads_follow_the_contract_order(bound):
    binding, hooks = bound
    for head in BOUNDARY_HEAD_LAYERS:
        hooks.events.clear()
        run_group_forward(binding, _dispatch(head), _record_forward(hooks.events, head))
        assert hooks.events == [
            ("wait_group_load", f"g{head}"),
            ("restore_group_boundary", f"g{head}"),
            ("forward", f"layer{head}"),
            ("capture_group_boundary", f"g{head}"),
            ("submit_group_store", f"g{head}"),
        ]


def test_g20_has_no_boundary_restore_or_capture(bound):
    binding, hooks = bound
    run_group_forward(binding, _dispatch(20), _record_forward(hooks.events, 20))
    assert hooks.events == [
        ("wait_group_load", "g20"),
        ("forward", "layer20"),
        ("submit_group_store", "g20"),
    ]


def test_non_group_heads_never_touch_the_transport(bound):
    binding, hooks = bound
    ran = []
    for layer in range(40):
        if layer in HEAD_LAYERS:
            continue
        hooks.events.clear()
        seen = []
        run_group_forward(binding, _dispatch(layer), lambda layer=layer: seen.append(layer))
        assert seen == [layer]
        assert hooks.events == []
        assert binding.group_for_layer(_dispatch(layer)) is None
        ran.append(layer)
    assert len(ran) == 36


def test_a_token_step_transfers_only_the_four_group_heads(bound):
    binding, hooks = bound
    for layer in range(40):
        run_group_forward(binding, _dispatch(layer), _record_forward(hooks.events, layer))
    names = [name for name, _ in hooks.events]
    assert names.count("wait_group_load") == 4
    assert names.count("restore_group_boundary") == 3
    assert names.count("capture_group_boundary") == 3
    assert names.count("submit_group_store") == 4
    assert names.count("forward") == 40
    assert [group_id for name, group_id in hooks.events if name == "restore_group_boundary"] == [
        "g2",
        "g8",
        "g14",
    ]
    assert set(names) <= {*GROUP_HOOK_NAMES, "forward"}


def test_group_load_failure_fails_the_step_without_local_forward(bound):
    binding, hooks = bound
    hooks.fail_on = "wait_group_load"
    with pytest.raises(RuntimeError, match="wait_group_load failed"):
        run_group_forward(binding, _dispatch(8), _record_forward(hooks.events, 8))
    assert hooks.events == [("wait_group_load", "g8")]


def test_boundary_restore_failure_blocks_the_forward(bound):
    binding, hooks = bound
    hooks.fail_on = "restore_group_boundary"
    with pytest.raises(RuntimeError, match="restore_group_boundary failed"):
        run_group_forward(binding, _dispatch(14), _record_forward(hooks.events, 14))
    assert hooks.events == [
        ("wait_group_load", "g14"),
        ("restore_group_boundary", "g14"),
    ]


def test_capture_failure_skips_the_store(bound):
    binding, hooks = bound
    hooks.fail_on = "capture_group_boundary"
    with pytest.raises(RuntimeError, match="capture_group_boundary failed"):
        run_group_forward(binding, _dispatch(2), _record_forward(hooks.events, 2))
    assert hooks.events == [
        ("wait_group_load", "g2"),
        ("restore_group_boundary", "g2"),
        ("forward", "layer2"),
        ("capture_group_boundary", "g2"),
    ]
    assert not any(name == "submit_group_store" for name, _ in hooks.events)


def test_store_submit_failure_propagates_after_the_forward(bound):
    binding, hooks = bound
    hooks.fail_on = "submit_group_store"
    with pytest.raises(RuntimeError, match="submit_group_store failed"):
        run_group_forward(binding, _dispatch(2), _record_forward(hooks.events, 2))
    assert ("forward", "layer2") in hooks.events


def test_bind_requires_all_four_group_facades(topology):
    class PartialHooks:
        def wait_group_load(self, group_id: str) -> None: ...

        def restore_group_boundary(self, group_id: str) -> None: ...

        def capture_group_boundary(self, group_id: str) -> None: ...

    with pytest.raises(DSV41LwV2BindingError, match="submit_group_store"):
        bind_lw_v2_engine(PartialHooks(), topology=topology)


def test_bind_rejects_a_non_v2_topology(text_config):
    text_config.num_hidden_layers = 2
    text_config.compress_ratios = [0, 0]
    text_config.kv_source_layer_ids = []
    text_config.index_source_layer_ids = []
    with pytest.raises(ValueError, match="backbone layers"):
        bind_lw_v2_engine(RecordingHooks(), topology=TOPOLOGY.build_layer_plan(text_config))


def test_bind_is_idempotent_for_the_same_connector(topology):
    hooks = RecordingHooks()
    first = bind_lw_v2_engine(hooks, topology=topology)
    second = bind_lw_v2_engine(hooks, topology=topology)
    assert first is second
    assert SEAM.lw_v2_active() is True
    assert SEAM.lw_v2_engine() is first
    assert first.group_ids == ("g2", "g8", "g14", "g20")
    assert first.head_layers == tuple(_dispatch(layer) for layer in HEAD_LAYERS)
    unbind_lw_v2_engine()
    assert SEAM.lw_v2_engine() is None


def test_bind_rejects_a_second_connector_instance(topology):
    bind_lw_v2_engine(RecordingHooks(), topology=topology)
    with pytest.raises(DSV41LwV2BindingError, match="already bound"):
        bind_lw_v2_engine(RecordingHooks(), topology=topology)


def test_binding_rejects_duplicate_dispatch_layers(topology):
    hooks = RecordingHooks()
    bound = bind_lw_v2_engine(hooks, topology=topology)
    with pytest.raises(DSV41LwV2BindingError, match="two groups"):
        DSV41LwV2EngineBinding(hooks=hooks, groups=(bound.groups[0], bound.groups[0]), prefix="model")


def test_prefix_is_applied_to_dispatch_and_consumer_names(topology):
    prefix = "language_model.model"
    binding = bind_lw_v2_engine(RecordingHooks(), topology=topology, prefix=prefix)
    group = binding.group_for_layer(_dispatch(8, prefix))
    assert group is not None and group.group_id == "g8"
    assert group.consumer_layers[-1] == f"{prefix}.layers.13.self_attn.attn"


def _function_calls(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = sorted(
        (node for node in ast.walk(function) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    names = []
    for call in calls:
        callee = call.func
        if isinstance(callee, ast.Attribute):
            names.append(callee.attr)
        elif isinstance(callee, ast.Name):
            names.append(callee.id)
    return names


def _v2_branch(path: Path, function_name: str, marker: str) -> ast.If:
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    branches = [node for node in function.body if isinstance(node, ast.If) and marker in ast.unparse(node.test)]
    assert len(branches) == 1
    return branches[0]


def test_dsa_v41_forward_dispatches_v2_after_synthetic_bypass_and_before_legacy_hooks():
    order = _function_calls(DSA_V41_PATH, "dsa_v41_forward")
    assert order.index("init_phase_bypass") < order.index("lw_v2_engine")
    assert order.index("lw_v2_engine") < order.index("run_group_forward")
    assert order.index("run_group_forward") < order.index("wait_for_layer_load")
    for legacy in (
        "restore_layer_boundary_state",
        "guard_layer_admission",
        "capture_layer_boundary_state",
        "notify_layer_produced",
        "maybe_save_kv_layer_to_connector",
    ):
        assert legacy in order


def test_dsa_v41_v2_branch_returns_and_never_saves_swa_per_layer():
    branch = _v2_branch(DSA_V41_PATH, "dsa_v41_forward", "lw_v2_binding")
    assert ast.unparse(branch.test).endswith("is not None")
    assert isinstance(branch.body[-1], ast.Return)
    body_source = "\n".join(ast.unparse(statement) for statement in branch.body)
    assert "run_group_forward" in body_source
    assert "maybe_save_kv_layer_to_connector" not in body_source
    assert "swa" not in body_source.lower()


def test_v2_seam_does_not_import_legacy_layerwise_runtime_or_sync_on_host():
    tree = ast.parse(SEAM_PATH.read_text())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
    assert imported_modules
    assert all("dsv41_layerwise" not in module for module in imported_modules)
    source = SEAM_PATH.read_text()
    for forbidden in (".item()", ".tolist()", ".cpu()"):
        assert forbidden not in source
