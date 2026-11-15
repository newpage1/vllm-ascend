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

"""Aligned prefill chunks for DSV4.1 layerwise v2 boundary capture.

The layerwise v2 compressor ring page only holds the last ``ring_rows`` tokens
of a forward. A boundary-state payload describes one LMCache chunk endpoint and
is captured at the end of a forward, so a forward that ends between two chunk
boundaries can never produce the payload that the store window needs: the rows
of the preceding endpoint are already overwritten.

A short prefill (for example 543 prompt tokens with a 4096 token budget) runs
in one forward, so the only capture lands after the prompt and no chunk-aligned
endpoint is ever captured. Warm hits then find token markers but no boundary
marker for the endpoint and report a zero visible prefix.

This patch keeps the prefill chunk arithmetic of vLLM but caps a prefill chunk
at the last boundary chunk that still leaves at least one whole chunk to the
prompt end. The remainder is scheduled as its own final chunk, which is not
boundary capturable but also not a store window. Every other vLLM behavior,
including non-v2 engines, is delegated unchanged.

The boundary chunk comes from the layerwise v2 connector (the LMCache chunk
size the manifest is written with). ``dsv41.lw_v2.boundary_chunk_size`` in
``kv_connector_extra_config`` overrides it, which keeps the patch usable when a
connector does not expose the negotiated chunk size yet.
"""

import functools
import inspect
from typing import Any, Mapping

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

# Importing the mamba patch first guarantees it wraps the upstream split before
# this module captures the current implementation: mamba models keep their own
# alignment and layerwise v2 only adds the chunk-boundary stop.
import vllm_ascend.patch.platform.patch_mamba_block_aligned_split  # noqa: F401

#: Connector name that opts an engine into layerwise v2.
_DSV41_LW_V2_CONNECTOR = "DSV41LwV2Connector"
#: JSON boolean flag that enables layerwise v2 for that connector.
_DSV41_LW_V2_FLAG = "dsv41.lw_v2"
#: Optional explicit boundary chunk size (tokens) for the patch.
_DSV41_LW_V2_CHUNK_KEY = "dsv41.lw_v2.boundary_chunk_size"
#: Split signature this patch is written against.
_EXPECTED_SPLIT_PARAMETERS = (
    "self",
    "request",
    "num_new_tokens",
    "num_new_local_computed_tokens",
    "num_external_computed_tokens",
)

_original_scheduler_init = Scheduler.__init__
_original_block_aligned_split = Scheduler._mamba_block_aligned_split


def _lw_v2_engine_config(vllm_config: Any) -> Any | None:
    """Return the KV transfer config of a layerwise v2 engine, or ``None``.

    This is the single eligibility gate of the patch, mirroring the connector's
    own opt-in: the connector name and the JSON ``true`` flag must both match.
    Callers must check it *before* reading any chunk size, so a stock LMCache
    deployment that happens to negotiate ``lmcache_tokens_per_chunk`` can never
    be split.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None
    if getattr(kv_transfer_config, "kv_connector", None) != _DSV41_LW_V2_CONNECTOR:
        return None
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, Mapping):
        return None
    if extra_config.get(_DSV41_LW_V2_FLAG) is not True:
        return None
    return kv_transfer_config


def _lw_v2_boundary_chunk(vllm_config: Any) -> int:
    """Return the configured layerwise v2 boundary chunk override, or 0.

    Only a layerwise v2 engine is eligible (see ``_lw_v2_engine_config``). A
    non-integer or non-positive override is ignored (0 disables the patch).
    """
    kv_transfer_config = _lw_v2_engine_config(vllm_config)
    if kv_transfer_config is None:
        return 0
    configured = kv_transfer_config.kv_connector_extra_config.get(_DSV41_LW_V2_CHUNK_KEY)
    if isinstance(configured, bool) or not isinstance(configured, int):
        return 0
    return configured if configured > 0 else 0


def _connector_boundary_chunk(scheduler: Any) -> int:
    """Return the chunk size the bound layerwise v2 connector negotiated."""
    connector = getattr(scheduler, "connector", None)
    scheduler_adapter = getattr(connector, "scheduler_adapter", None)
    chunk = getattr(scheduler_adapter, "lmcache_tokens_per_chunk", 0)
    if isinstance(chunk, bool) or not isinstance(chunk, int):
        return 0
    return chunk if chunk > 0 else 0


def _resolve_boundary_chunk(scheduler: Any) -> int:
    """Pick the boundary chunk for ``scheduler``; 0 means "leave vLLM alone".

    The engine must be layerwise v2 *before* any chunk is read: the connector
    fallback exists only for a v2 engine whose connector negotiated the chunk
    (``DSV41LwV2Connector`` keeps the stock attribute name). A stock LMCache
    engine therefore resolves to 0 even when its connector reports a chunk.
    """
    vllm_config = getattr(scheduler, "vllm_config", None)
    if _lw_v2_engine_config(vllm_config) is None:
        return 0
    override = _lw_v2_boundary_chunk(vllm_config)
    if override > 0:
        return override
    return _connector_boundary_chunk(scheduler)


def _aligned_prefill_tokens(
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int,
    num_external_computed_tokens: int,
    chunk: int,
) -> int:
    """Cap a prefill chunk at the last boundary chunk before the prompt end.

    ``start`` is where this forward starts computing. When the prompt still has
    more than one chunk left, the chunk ends at the last chunk boundary at or
    before ``start + num_new_tokens``; the remaining tokens are scheduled as the
    final (short) chunk. A prompt with at most one chunk left is left untouched,
    because its own end is not a store window either.
    """
    start = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    # ``num_tokens - 1`` extends this to resumed requests replaying output
    # tokens, mirroring the upstream mamba boundary definition.
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens
    if chunk <= 0 or prefill_end - start <= chunk:
        return num_new_tokens
    end = start + num_new_tokens
    aligned_end = end // chunk * chunk
    if aligned_end <= start:
        return num_new_tokens
    return min(num_new_tokens, aligned_end - start)


@functools.wraps(_original_block_aligned_split)
def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Apply the layerwise v2 chunk-boundary stop, then the mamba logic."""
    chunk = getattr(self, "_dsv41_lw_v2_boundary_chunk", 0)
    if chunk > 0 and not getattr(self, "has_mamba_layers", False):
        return _aligned_prefill_tokens(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
            chunk,
        )
    return _original_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
    )


@functools.wraps(_original_scheduler_init)
def _scheduler_init(self: Scheduler, *args: Any, **kwargs: Any) -> None:
    """Opt a layerwise v2 scheduler into the chunk-boundary split."""
    _original_scheduler_init(self, *args, **kwargs)
    chunk = _resolve_boundary_chunk(self)
    if chunk <= 0:
        return
    if getattr(self, "has_mamba_layers", False):
        # Mamba models already stop at their own cacheable positions; a second
        # alignment would fight the recurrent-state checkpoint contract.
        return
    if not getattr(getattr(self, "scheduler_config", None), "enable_chunked_prefill", False):
        # Splitting a request across steps is chunked prefill; an engine that
        # disabled it must not be handed partial prefills by this patch.
        return
    self._dsv41_lw_v2_boundary_chunk = chunk
    # Reuse the scheduler's single per-request split hook: forcing the flag on a
    # non-mamba engine only adds the call above, which handles the v2 case.
    self.need_mamba_block_aligned_split = True


current_parameters = tuple(inspect.signature(_original_block_aligned_split).parameters)
if current_parameters != _EXPECTED_SPLIT_PARAMETERS:
    raise RuntimeError(
        "Cannot apply the layerwise v2 prefill split patch: unexpected "
        "Scheduler._mamba_block_aligned_split signature "
        f"{current_parameters}"
    )

Scheduler.__init__ = _scheduler_init
Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
