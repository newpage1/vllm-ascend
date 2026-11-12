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
"""CPU tests for the DSV4.1 layerwise admission init-phase bypass (t71).

The admission module under ``vllm_ascend/attention/dsv41_layerwise_admission.py``
stays import-light (stdlib plus ``vllm.logger`` only). These tests stub
``vllm.logger`` before importing it, so they run on a plain CPU Python with no
torch/torch_npu/vllm installation -- the same pattern the aggregator suite
uses for stock-source checks.

What must hold (captain directive for t71, mirrored by t72 re-verification):

- Engine-init forwards (NPU graph capture incl. warmup dummies, memory profile
  runs) carry no scheduler step, so no admission window can exist: the forward
  hook must bypass admission there instead of raising.
- Real request steps keep the fail-closed contract: with a registry bound and
  no admission window, ``wait_for_layer_load`` still raises.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class _RecordingRegistry:
    """Minimal AdmissionRegistry double recording every hook invocation."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def check_admitted(self, dispatch_layer_name: str) -> None:
        self.calls.append(f"check:{dispatch_layer_name}")

    def wait_for_layer(self, dispatch_layer_name: str, timeout: float) -> None:
        self.calls.append(f"wait:{dispatch_layer_name}")
        raise RuntimeError(
            f"layer {dispatch_layer_name} has never opened an admission window"
        )

    def notify_layer_produced(self, dispatch_layer_name: str) -> None:
        self.calls.append(f"notify:{dispatch_layer_name}")


@pytest.fixture(scope="module")
def admission() -> types.ModuleType:
    """Load the admission module directly from its file (CPU-safe).

    The module is import-light (stdlib plus ``vllm.logger``); stubbing that
    one import and loading via ``spec_from_file_location`` keeps the test
    runnable on a plain CPU Python with no torch/torch_npu/vllm/vllm_ascend
    installation, mirroring the aggregator suite's stock-source pattern.
    """
    stub = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")

    class _Logger:
        def info(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def warning(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def _init_logger(_name: str) -> _Logger:
        return _Logger()

    logger_mod.init_logger = _init_logger
    stub.logger = logger_mod
    sys.modules.setdefault("vllm", stub)
    sys.modules.setdefault("vllm.logger", logger_mod)

    source = (
        Path(__file__).resolve().parents[3]
        / "vllm_ascend"
        / "attention"
        / "dsv41_layerwise_admission.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dsv41_layerwise_admission_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_phase(admission: types.ModuleType):
    """Every test starts from a cleared registry and a cleared init flag."""
    admission.clear_admission_registry()
    admission.set_init_phase(False)
    yield
    admission.clear_admission_registry()
    admission.set_init_phase(False)


def test_init_phase_defaults_to_inactive(admission: types.ModuleType) -> None:
    assert admission.init_phase_active() is False
    assert admission.init_phase_bypass("model.layers.0.self_attn.v41_attn") is False


def test_init_phase_bypasses_without_touching_the_registry(
    admission: types.ModuleType,
) -> None:
    registry = _RecordingRegistry()
    admission.set_admission_registry(registry)

    admission.set_init_phase(True)
    assert admission.init_phase_active() is True
    # Bypass must short-circuit before wait_for_layer_load: the recording
    # registry raises on wait, so an empty call list proves the hooks never
    # ran (no fake admission, no silent waiting).
    assert admission.init_phase_bypass("model.layers.0.self_attn.v41_attn") is True
    assert registry.calls == []


def test_init_phase_flag_is_idempotent_and_reversible(admission: types.ModuleType) -> None:
    admission.set_init_phase(True)
    admission.set_init_phase(True)
    assert admission.init_phase_active() is True
    admission.set_init_phase(False)
    assert admission.init_phase_active() is False


def test_synthetic_forward_cm_covers_idle_dp_dummy_batches(
    admission: types.ModuleType,
) -> None:
    registry = _RecordingRegistry()
    admission.set_admission_registry(registry)

    # Outside the window: fail-closed still applies.
    assert admission.init_phase_active() is False
    # Inside the window (idle-DP dummy batch / capture / profile): bypass.
    with admission.synthetic_forward():
        assert admission.init_phase_active() is True
        assert admission.init_phase_bypass("model.layers.0.self_attn.v41_attn") is True
        assert registry.calls == []
    # Leaving the window restores the fail-closed contract, even on error.
    assert admission.init_phase_active() is False
    with pytest.raises(RuntimeError, match="never opened an admission window"):
        admission.wait_for_layer_load("model.layers.0.self_attn.v41_attn")

    with pytest.raises(ValueError, match="boom"):
        with admission.synthetic_forward():
            assert admission.init_phase_active() is True
            raise ValueError("boom")
    assert admission.init_phase_active() is False


def test_real_steps_stay_fail_closed_without_a_window(
    admission: types.ModuleType,
) -> None:
    admission.set_admission_registry(_RecordingRegistry())
    # Outside the init phase a missing admission window must still raise:
    # the fail-closed contract only tolerates engine-init forwards.
    with pytest.raises(RuntimeError, match="never opened an admission window"):
        admission.wait_for_layer_load("model.layers.0.self_attn.v41_attn")


def test_stock_path_without_registry_is_always_a_noop(
    admission: types.ModuleType,
) -> None:
    admission.set_init_phase(False)
    admission.wait_for_layer_load("model.layers.0.self_attn.v41_attn")
    admission.guard_layer_admission("model.layers.0.self_attn.v41_attn")
    admission.notify_layer_produced("model.layers.0.self_attn.v41_attn")
