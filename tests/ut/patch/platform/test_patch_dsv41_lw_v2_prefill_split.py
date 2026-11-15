# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the DSV4.1 layerwise v2 prefill chunk-boundary split.

On a CPU-only host (no vLLM installation) the patch module is loaded against a
minimal scheduler stub, so the chunk arithmetic and the dispatch rules can be
verified without the engine. The committed CI job imports the real module.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PATCH_PATH = _REPO_ROOT / "vllm_ascend/patch/platform/patch_dsv41_lw_v2_prefill_split.py"


def _stub_scheduler_class():
    class _StubScheduler:
        def __init__(self, vllm_config=None) -> None:
            self.vllm_config = vllm_config
            self.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
            self.has_mamba_layers = False

        def _mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens=0,
            num_external_computed_tokens=0,
        ):
            return num_new_tokens

    return _StubScheduler


def _load_patch_with_stubs():
    """Load the patch against a minimal scheduler stand-in for CPU hosts."""
    scheduler_class = _stub_scheduler_class()
    added: list[str] = []
    packages = (
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm_ascend",
        "vllm_ascend.patch",
        "vllm_ascend.patch.platform",
    )
    for name in packages:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        added.append(name)
    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = scheduler_class
    sys.modules["vllm.v1.core.sched.scheduler"] = scheduler_module
    added.append("vllm.v1.core.sched.scheduler")
    request_module = types.ModuleType("vllm.v1.request")
    request_module.Request = object
    sys.modules["vllm.v1.request"] = request_module
    added.append("vllm.v1.request")
    mamba_module = types.ModuleType("vllm_ascend.patch.platform.patch_mamba_block_aligned_split")
    sys.modules["vllm_ascend.patch.platform.patch_mamba_block_aligned_split"] = mamba_module
    added.append("vllm_ascend.patch.platform.patch_mamba_block_aligned_split")
    name = "tests._stub_dsv41_lw_v2_prefill_split"
    spec = importlib.util.spec_from_file_location(name, _PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for added_name in added:
            sys.modules.pop(added_name, None)
    return module, scheduler_class


def _load_patch():
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError:
        return _load_patch_with_stubs()
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_ascend.patch.platform import patch_dsv41_lw_v2_prefill_split

    return patch_dsv41_lw_v2_prefill_split, Scheduler


MOD, SCHEDULER = _load_patch()


def _request(
    *,
    computed: int,
    num_prompt_tokens: int,
    num_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=num_prompt_tokens,
        num_tokens=num_tokens if num_tokens is not None else num_prompt_tokens,
    )


def _split(request, num_new_tokens: int, chunk: int = 256) -> int:
    return MOD._aligned_prefill_tokens(request, num_new_tokens, 0, 0, chunk)


# --------------------------------------------------------------------------
# Engine detection: only a layerwise v2 connector opts in.
# --------------------------------------------------------------------------


def _engine(*, connector="DSV41LwV2Connector", extra_config=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_connector_extra_config=extra_config or {},
        )
    )


def test_detection_requires_the_v2_connector_name():
    config = _engine(connector="LMCacheMPConnector", extra_config={"dsv41.lw_v2": True})
    assert MOD._lw_v2_boundary_chunk(config) == 0


@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_detection_requires_the_json_true_flag(flag):
    extra_config = {} if flag is None else {"dsv41.lw_v2": flag}
    assert MOD._lw_v2_boundary_chunk(_engine(extra_config=extra_config)) == 0


def test_detection_reads_the_explicit_chunk_override():
    config = _engine(extra_config={"dsv41.lw_v2": True, "dsv41.lw_v2.boundary_chunk_size": 128})
    assert MOD._lw_v2_boundary_chunk(config) == 128


@pytest.mark.parametrize("override", [0, -256, True, "256", 12.5])
def test_detection_ignores_an_invalid_chunk_override(override):
    config = _engine(extra_config={"dsv41.lw_v2": True, "dsv41.lw_v2.boundary_chunk_size": override})
    assert MOD._lw_v2_boundary_chunk(config) == 0


def test_detection_without_a_kv_transfer_config():
    assert MOD._lw_v2_boundary_chunk(SimpleNamespace(kv_transfer_config=None)) == 0


def test_connector_chunk_comes_from_the_scheduler_adapter():
    scheduler = SimpleNamespace(
        connector=SimpleNamespace(scheduler_adapter=SimpleNamespace(lmcache_tokens_per_chunk=256))
    )
    assert MOD._connector_boundary_chunk(scheduler) == 256
    scheduler.connector.scheduler_adapter.lmcache_tokens_per_chunk = 0
    assert MOD._connector_boundary_chunk(scheduler) == 0


def test_explicit_chunk_wins_over_the_connector():
    scheduler = SimpleNamespace(
        vllm_config=_engine(extra_config={"dsv41.lw_v2": True, "dsv41.lw_v2.boundary_chunk_size": 128}),
        connector=SimpleNamespace(scheduler_adapter=SimpleNamespace(lmcache_tokens_per_chunk=256)),
    )
    assert MOD._resolve_boundary_chunk(scheduler) == 128


# --------------------------------------------------------------------------
# Chunk arithmetic.
# --------------------------------------------------------------------------


def test_first_short_prompt_chunk_stops_at_the_last_boundary():
    # 543 prompt tokens: the first forward stops at 512, leaving a 31 token tail.
    assert _split(_request(computed=0, num_prompt_tokens=543), 543) == 512


def test_the_short_tail_chunk_is_not_split_again():
    assert _split(_request(computed=512, num_prompt_tokens=543), 31) == 31


def test_a_prompt_shorter_than_one_chunk_runs_unchanged():
    assert _split(_request(computed=0, num_prompt_tokens=200), 200) == 200


def test_a_prompt_of_exactly_one_chunk_runs_unchanged():
    assert _split(_request(computed=0, num_prompt_tokens=256), 256) == 256


def test_a_prompt_just_over_one_chunk_leaves_a_short_tail():
    assert _split(_request(computed=0, num_prompt_tokens=300), 300) == 256


def test_a_two_chunk_prompt_runs_unchanged():
    assert _split(_request(computed=0, num_prompt_tokens=512), 512) == 512


def test_long_prefills_stop_at_every_boundary():
    # 9690 prompt tokens with a 4096 token budget.
    assert _split(_request(computed=0, num_prompt_tokens=9690), 4096) == 4096
    assert _split(_request(computed=4096, num_prompt_tokens=9690), 4096) == 4096
    assert _split(_request(computed=8192, num_prompt_tokens=9690), 1498) == 1280
    assert _split(_request(computed=9472, num_prompt_tokens=9690), 218) == 218


def test_a_short_final_chunk_stays_short():
    # One chunk budget left: the prompt end is not a store window either.
    assert _split(_request(computed=9472, num_prompt_tokens=9690), 218) == 218


def test_a_warm_replay_window_is_not_extended():
    # Warm hit at 512, a 128 token replay window starts the forward at 384.
    assert _split(_request(computed=384, num_prompt_tokens=543), 159) == 159


def test_a_warm_long_prefill_stops_at_the_next_boundary():
    # Warm hit at 8192, replay starts at 8064, the rest is aligned at 9472.
    assert _split(_request(computed=8064, num_prompt_tokens=9690), 1626) == 1408


def test_decode_steps_are_never_touched():
    assert _split(_request(computed=543, num_prompt_tokens=543), 1) == 1


def test_a_resumed_request_uses_the_num_tokens_barrier():
    # A resumed request replays one output token: the barrier moves with it.
    request = _request(computed=1024, num_prompt_tokens=1024, num_tokens=1030)
    assert _split(request, 6) == 6


def test_zero_chunk_is_a_no_op():
    assert _split(_request(computed=0, num_prompt_tokens=543), 543, chunk=0) == 543


# --------------------------------------------------------------------------
# Scheduler dispatch.
# --------------------------------------------------------------------------


def _fake_scheduler(*, chunk: int, has_mamba_layers: bool = False) -> SimpleNamespace:
    scheduler = SimpleNamespace(has_mamba_layers=has_mamba_layers)
    if chunk:
        scheduler._dsv41_lw_v2_boundary_chunk = chunk
    return scheduler


def test_split_wrapper_aligned_prefill_on_a_v2_scheduler(monkeypatch):
    delegated = []
    monkeypatch.setattr(
        MOD, "_original_block_aligned_split", lambda *args: delegated.append(args) or 7
    )
    token_count = MOD._mamba_block_aligned_split(
        _fake_scheduler(chunk=256), _request(computed=0, num_prompt_tokens=543), 543
    )
    assert token_count == 512
    assert delegated == []


@pytest.mark.parametrize("chunk", [0])
def test_split_wrapper_delegates_without_a_v2_chunk(monkeypatch, chunk):
    delegated = []
    monkeypatch.setattr(
        MOD, "_original_block_aligned_split", lambda *args: delegated.append(args) or 7
    )
    token_count = MOD._mamba_block_aligned_split(
        _fake_scheduler(chunk=chunk), _request(computed=0, num_prompt_tokens=543), 543
    )
    assert token_count == 7
    assert len(delegated) == 1


def test_split_wrapper_delegates_for_mamba_models(monkeypatch):
    delegated = []
    monkeypatch.setattr(
        MOD, "_original_block_aligned_split", lambda *args: delegated.append(args) or 7
    )
    token_count = MOD._mamba_block_aligned_split(
        _fake_scheduler(chunk=256, has_mamba_layers=True),
        _request(computed=0, num_prompt_tokens=543),
        543,
    )
    assert token_count == 7
    assert len(delegated) == 1


def test_scheduler_init_records_the_chunk_and_enables_the_split(monkeypatch):
    monkeypatch.setattr(MOD, "_original_scheduler_init", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(MOD, "_resolve_boundary_chunk", lambda self: 256)
    scheduler = SimpleNamespace(
        has_mamba_layers=False,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    MOD._scheduler_init(scheduler, None)
    assert scheduler._dsv41_lw_v2_boundary_chunk == 256
    assert scheduler.need_mamba_block_aligned_split is True


def test_scheduler_init_leaves_other_engines_alone(monkeypatch):
    monkeypatch.setattr(MOD, "_original_scheduler_init", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(MOD, "_resolve_boundary_chunk", lambda self: 0)
    scheduler = SimpleNamespace(
        has_mamba_layers=False,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    MOD._scheduler_init(scheduler, None)
    assert not hasattr(scheduler, "_dsv41_lw_v2_boundary_chunk")
    assert not hasattr(scheduler, "need_mamba_block_aligned_split")


def test_scheduler_init_leaves_mamba_models_alone(monkeypatch):
    monkeypatch.setattr(MOD, "_original_scheduler_init", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(MOD, "_resolve_boundary_chunk", lambda self: 256)
    scheduler = SimpleNamespace(
        has_mamba_layers=True,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    MOD._scheduler_init(scheduler, None)
    assert not hasattr(scheduler, "_dsv41_lw_v2_boundary_chunk")


def test_scheduler_init_requires_chunked_prefill(monkeypatch):
    """An engine that disabled chunked prefill must not receive partial prefills."""
    monkeypatch.setattr(MOD, "_original_scheduler_init", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(MOD, "_resolve_boundary_chunk", lambda self: 256)
    scheduler = SimpleNamespace(
        has_mamba_layers=False,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
    )
    MOD._scheduler_init(scheduler, None)
    assert not hasattr(scheduler, "_dsv41_lw_v2_boundary_chunk")


def test_the_patch_is_installed_on_the_scheduler_class():
    assert SCHEDULER.__init__ is MOD._scheduler_init
    assert SCHEDULER._mamba_block_aligned_split is MOD._mamba_block_aligned_split
