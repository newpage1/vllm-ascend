# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-side half of the default-off mask timeline.

The seam is loaded by path because ``vllm_ascend/__init__`` needs a real vLLM
install. A duck-typed binding drives ``run_group_forward`` so the tests pin
only the seam's own contract: group heads emit ``forward_start`` /
``forward_end`` around the wait/restore/forward/capture/store sequence, while
non-head layers and a disabled switch emit nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_PATH = REPO_ROOT / "vllm_ascend/models/deepseek_v41/topology.py"
SEAM_PATH = REPO_ROOT / "vllm_ascend/attention/dsv41_lw_v2.py"
TRACE_PATH = REPO_ROOT / "vllm_ascend/attention/dsv41_lw_v2_trace.py"

HEAD_LAYER = "model.layers.2.self_attn.v41_attn"
OTHER_LAYER = "model.layers.3.self_attn.v41_attn"


def _install_packages() -> None:
    packages = (
        ("vllm_ascend", REPO_ROOT / "vllm_ascend"),
        ("vllm_ascend.attention", REPO_ROOT / "vllm_ascend/attention"),
        ("vllm_ascend.models", REPO_ROOT / "vllm_ascend/models"),
        (
            "vllm_ascend.models.deepseek_v41",
            REPO_ROOT / "vllm_ascend/models/deepseek_v41",
        ),
    )
    for name, path in packages:
        if name in sys.modules:
            continue
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package


def _load_by_path(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_seam() -> tuple[ModuleType, ModuleType, ModuleType]:
    _install_packages()
    topology = _load_by_path(
        "vllm_ascend.models.deepseek_v41.topology", TOPOLOGY_PATH
    )
    trace = _load_by_path("vllm_ascend.attention.dsv41_lw_v2_trace", TRACE_PATH)
    seam = _load_by_path("vllm_ascend.attention.dsv41_lw_v2", SEAM_PATH)
    return seam, trace, topology


seam, trace, topology = _load_seam()


class _RecordingHooks:
    def __init__(self, *, step_id: int = 7) -> None:
        self.calls: list[str] = []
        request = SimpleNamespace(request_id="r1")
        self.current_step = SimpleNamespace(step_id=step_id, requests=(request,))

    def wait_group_load(self, group_id: str) -> None:
        self.calls.append(f"wait:{group_id}")

    def restore_group_boundary(self, group_id: str) -> None:
        self.calls.append(f"restore:{group_id}")

    def capture_group_boundary(self, group_id: str) -> None:
        self.calls.append(f"capture:{group_id}")

    def submit_group_store(self, group_id: str) -> None:
        self.calls.append(f"store:{group_id}")


def _binding(hooks, group):
    return SimpleNamespace(
        hooks=hooks,
        group_for_layer=lambda name: group if name == HEAD_LAYER else None,
    )


@pytest.fixture(autouse=True)
def _clean_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(trace.ENV_ENABLED, raising=False)
    monkeypatch.delenv(trace.ENV_DIR, raising=False)
    trace.reset_state()
    yield
    trace.reset_state()


def _records(directory: Path) -> list[dict]:
    files = list(directory.glob("mask-trace-*-engine.jsonl"))
    assert len(files) == 1, files
    return [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_disabled_switch_emits_no_file_and_keeps_the_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()
    hooks = _RecordingHooks()
    group = SimpleNamespace(group_id="g2", has_boundary=True)
    forwarded: list[str] = []

    seam.run_group_forward(
        _binding(hooks, group), HEAD_LAYER, lambda: forwarded.append("f")
    )

    assert forwarded == ["f"]
    assert hooks.calls == [
        "wait:g2",
        "restore:g2",
        "capture:g2",
        "store:g2",
    ]
    assert list(tmp_path.iterdir()) == []


def test_group_head_emits_forward_events_with_step_and_request(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()
    hooks = _RecordingHooks(step_id=11)
    group = SimpleNamespace(group_id="g2", has_boundary=True)

    seam.run_group_forward(_binding(hooks, group), HEAD_LAYER, lambda: None)

    records = _records(tmp_path)
    assert [record["event"] for record in records] == [
        "forward_start",
        "compute_start",
        "compute_end",
        "forward_end",
    ]
    assert [record["ts_ns"] for record in records] == sorted(
        record["ts_ns"] for record in records
    )
    start, compute_start, compute_end, end = records
    assert start["phase"] == "forward"
    assert start["group_id"] == "g2"
    assert start["layer"] == HEAD_LAYER
    assert start["step_id"] == 11
    assert start["request_ids"] == ["r1"]
    assert compute_start["phase"] == "compute"
    assert compute_end["elapsed_ns"] >= 0
    assert end["ok"] is True
    assert end["elapsed_ns"] >= 0


def test_non_head_layer_emits_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()
    hooks = _RecordingHooks()
    group = SimpleNamespace(group_id="g2", has_boundary=True)
    forwarded: list[str] = []

    seam.run_group_forward(
        _binding(hooks, group), OTHER_LAYER, lambda: forwarded.append("f")
    )

    assert forwarded == ["f"]
    assert hooks.calls == []
    assert list(tmp_path.iterdir()) == []


def test_forward_failure_still_records_a_terminal_span(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()
    hooks = _RecordingHooks()
    group = SimpleNamespace(group_id="g2", has_boundary=True)

    def _boom() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        seam.run_group_forward(_binding(hooks, group), HEAD_LAYER, _boom)

    records = _records(tmp_path)
    assert [record["event"] for record in records] == [
        "forward_start",
        "compute_start",
        "forward_end",
    ]
    assert records[-1]["ok"] is False
