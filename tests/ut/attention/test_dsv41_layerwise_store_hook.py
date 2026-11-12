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
"""Source-level wiring test for the DSV4.1 layerwise store hook (t75).

``dsa_v41.py`` imports torch/torch_npu/vllm internals, so a CPU-only test
cannot import it. What *can* be verified without an NPU is the wiring the
t65 gap was about: ``dsa_v41_forward`` must hand the freshly written KV to
the connector via ``maybe_save_kv_layer_to_connector`` -- the same hook
``dsa_v1``/``sfa_v1``/``mla_v1`` already call. Before t75 the V4.1 lane was
the only attention family that never stored, so ``save_kv_layer`` never ran
and the S0 wire never carried a single STORE frame.

Assertions (old implementation fails every one of them):

- the symbol is imported from ``vllm_ascend.attention.utils``;
- ``dsa_v41_forward`` calls it exactly once, in the main path *after*
  ``notify_layer_produced`` (so the producer window exists when the store
  submits) and *outside* the init-phase bypass branch (engine-init dummies
  keep skipping both admission and stores);
- the call passes the layer's ``long_kv_cache`` tensors;
- the graph-capture fake path (``dsa_v41_forward_fake``) does not double-store.

Behavioral evidence (real bytes over the wire) comes from the c01 cold-only
smoke; this test pins the source contract that makes that smoke pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "attention"
    / "dsa_v41.py"
)

MAIN_PATH_LAYER = "model.layers.0.self_attn.v41_attn"


def _load() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {SOURCE.name}")


def _calls(node: ast.AST, func_name: str) -> list[ast.Call]:
    found: list[ast.Call] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == func_name
        ):
            found.append(child)
    return found


def test_store_hook_is_imported_from_attention_utils() -> None:
    tree = _load()
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "vllm_ascend.attention.utils"
        for alias in node.names
    ]
    assert "maybe_save_kv_layer_to_connector" in imports, (
        "dsa_v41 must import maybe_save_kv_layer_to_connector from "
        "vllm_ascend.attention.utils so the V4.1 lane stores like "
        "dsa_v1/sfa_v1/mla_v1 (t65 gap)"
    )


def test_forward_calls_the_store_hook_once_after_notify() -> None:
    tree = _load()
    forward = _function(tree, "dsa_v41_forward")

    hook_calls = _calls(forward, "maybe_save_kv_layer_to_connector")
    assert len(hook_calls) == 1, (
        "dsa_v41_forward must call maybe_save_kv_layer_to_connector exactly "
        "once per layer forward"
    )

    # Statement order in the main path: wait -> guard -> forward -> notify ->
    # save. The store submit reads the producer window notify just opened, so
    # the save call must be ordered after it.
    order: list[str] = []
    for stmt in forward.body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    order.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    order.append(child.func.attr)
    assert "notify_layer_produced" in order, "notify hook missing"
    assert "maybe_save_kv_layer_to_connector" in order, "store hook missing"
    assert order.index("notify_layer_produced") < order.index(
        "maybe_save_kv_layer_to_connector"
    ), "store must submit after notify_layer_produced opened the window"
    assert order.index("guard_layer_admission") < order.index(
        "maybe_save_kv_layer_to_connector"
    ), "store must run after admission"


def test_store_hook_is_outside_the_init_phase_bypass() -> None:
    tree = _load()
    forward = _function(tree, "dsa_v41_forward")

    # The bypass branch must return early without reaching the store hook;
    # engine-init dummies (capture/profile) must not store anything.
    bypass_branches = [
        stmt
        for stmt in forward.body
        if isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Call)
        and isinstance(stmt.test.func, ast.Name)
        and stmt.test.func.id == "init_phase_bypass"
    ]
    assert bypass_branches, "init-phase bypass branch not found"
    for branch in bypass_branches:
        assert not _calls(branch, "maybe_save_kv_layer_to_connector"), (
            "the init-phase bypass must not store: engine-init forwards carry "
            "no scheduler step"
        )
        # and the hook must live in the main path (after the last bypass if)
    hook_calls = _calls(forward, "maybe_save_kv_layer_to_connector")
    assert hook_calls, "store hook missing from the main path"


def test_store_hook_passes_the_layers_long_kv_cache() -> None:
    tree = _load()
    forward = _function(tree, "dsa_v41_forward")
    hook = _calls(forward, "maybe_save_kv_layer_to_connector")[0]

    assert hook.args, "store hook must pass (layer_name, kv tensors)"
    first = hook.args[0]
    assert isinstance(first, ast.Name) and first.id == "layer_name", (
        "first argument must be the engine layer name"
    )
    rendered = ast.unparse(hook.args[1])
    assert "long_kv_cache" in rendered and "kv_cache" in rendered, (
        f"second argument must read the layer's long KV cache tensors, got: {rendered}"
    )


def test_capture_fake_path_does_not_double_store() -> None:
    tree = _load()
    fake = _function(tree, "dsa_v41_forward_fake")
    assert not _calls(fake, "maybe_save_kv_layer_to_connector"), (
        "the graph-capture fake must not submit a second store for the same layer"
    )
