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

"""Fail-closed admission guard for DSV4.1 layerwise KV transfer.

This module stays import-light (stdlib plus ``vllm.logger`` only) so it can
be unit tested on CPU and imported by ``dsa_v41`` without pulling LMCache or
torch_npu.

Lifecycle and concurrency of the module-level registry handle
(AGENTS.md requires an explicit statement for mutable module state):

- The registry handle is written exactly once per worker process, by the
  DSV4.1 layerwise MP connector's worker adapter inside
  ``register_kv_caches`` (before the first forward), and cleared on
  connector ``shutdown``. There is a single writer thread (the vLLM worker
  main thread) and the handle is swapped by rebinding the module attribute,
  so readers on the forward path only ever observe ``None`` or one fully
  constructed registry object (CPython attribute rebind is atomic).
- When the handle is ``None`` (stock deployment: the layerwise connector is
  not enabled), ``guard_layer_admission`` returns immediately and the custom
  op path is bit-for-bit the stock behavior, including fake/profiling mode.
- When the handle is set, every ``dsa_v41_forward`` call must be admitted by
  the registry; any unknown layer, missing step admission, or incomplete
  transfer state raises instead of computing silently on stale caches
  (fail-closed contract of the integration design, sections 4.5/5.4).

The registry object itself is supplied by LMCache-Ascend
(``DSV41LayerwiseMPWorkerAdapter``) and implements the three hooks of
``AdmissionRegistry``: ``check_admitted`` (synchronous gate),
``wait_for_layer`` (block until the layer's retrieves are ready), and
``notify_layer_produced`` (record that the layer finished writing).

Observable execution evidence (t71): binding the registry and the first
wait / grant / produce of every dispatch layer each log one INFO line, so
an integration run proves the admission lifecycle really executed --
settling real transfers, not a silent return or ``NotImplementedError``.
The ``_logged_*`` dedupe sets are append-only mutable module state: they
are written only from the single worker main thread that owns the registry
handle and are never read for control flow (logging dedupe only), so the
stock-path behavior stays bit-for-bit unchanged.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

from vllm.logger import init_logger

DSV41_LAYERWISE_ADMISSION_TAG = "dsv41_layerwise"

_LOGGER = init_logger(__name__)

# Name under which the DSV4.1 layerwise connector is registered in the
# KVConnectorFactory (see vllm_ascend/distributed/kv_transfer/__init__.py).
DSV41_LAYERWISE_CONNECTOR_NAME = "DSV41LayerwiseMPConnector"


@runtime_checkable
class AdmissionRegistry(Protocol):
    """Minimal contract the worker-side adapter must satisfy."""

    def check_admitted(self, dispatch_layer_name: str) -> None:
        """Raise if the step executing ``dispatch_layer_name`` is not admitted."""
        ...

    def wait_for_layer(self, dispatch_layer_name: str, timeout: float) -> None:
        """Block until the layer's retrieved KV resources are ready.

        Raises on timeout or transfer failure; never returns silently while
        a required resource is still in flight (fail-closed).
        """
        ...

    def notify_layer_produced(self, dispatch_layer_name: str) -> None:
        """Record that the layer's KV write completed (producer side)."""
        ...


class DSV41LayerwiseAdmissionError(RuntimeError):
    """A layerwise forward ran without a complete admission decision.

    Raised fail-closed: the DSV4.1 layerwise connector is enabled, but the
    layer's request step has not been admitted (resources missing, transfer
    failed, or admission state never recorded). The step must fail loudly
    rather than compute on incomplete/stale KV state.
    """


# Module-level registry handle; see the module docstring for the documented
# single-writer lifecycle and the stock-path ``None`` invariant.
_admission_registry: AdmissionRegistry | None = None

# Engine-synthetic forward declaration (t71): the worker runner sets this
# around forwards that carry no scheduler step -- NPU graph capture
# (including its warmup dummy runs), memory-profile runs, and idle-DP
# dummy batches (``execute_dummy_batch``). No admission window can exist
# for such forwards, so the hook must bypass admission instead of failing
# closed.
# Single writer (worker main thread), read-only on the forward path; see
# ``init_phase_bypass``.
_init_phase: bool = False

# First-seen logging dedupe (t71 evidence): append-only sets written by the
# same single thread that owns the registry handle; never control flow.
_logged_waits: set[str] = set()
_logged_grants: set[str] = set()
_logged_produces: set[str] = set()
_logged_bypasses: set[str] = set()


def set_init_phase(active: bool) -> None:
    """Declare whether the worker is currently running synthetic forwards.

    Called by the worker runner around graph capture / warmup / profile runs
    and around idle-DP dummy batches (contexts that never carry a scheduled
    request step). Real request steps always execute with the flag cleared,
    so the fail-closed contract below is untouched.

    Args:
        active: ``True`` while the current forwards are engine-synthetic.
    """
    global _init_phase
    _init_phase = active


@contextmanager
def synthetic_forward() -> Iterator[None]:
    """Context-manager form of :func:`set_init_phase` for synthetic forwards.

    Marks one engine-synthetic forward window (no scheduler step attached:
    idle-DP dummy batch, warmup dummy, or capture/profile run) so the
    admission hooks bypass instead of failing closed. Restores the previous
    state even on error.
    """
    global _init_phase
    previous = _init_phase
    _init_phase = True
    try:
        yield
    finally:
        _init_phase = previous


def init_phase_active() -> bool:
    """Report whether engine-synthetic forwards are currently executing."""
    return _init_phase


def init_phase_bypass(layer_name: str) -> bool:
    """Return whether ``dsa_v41_forward`` must bypass admission for this call.

    Args:
        layer_name: The custom-op dispatch key of the attention object.

    Behavior:
        - Outside synthetic windows: always ``False`` -- request steps keep
          the full wait/guard/notify lifecycle (fail-closed contract).
        - During engine-synthetic forwards (graph capture, warmup dummies,
          memory profile, idle-DP dummy batches): ``True``. These forwards
          have no scheduler step, hence no admission window can exist and
          none may be faked; the hooks are skipped outright and the bypass
          logs once per layer so integration evidence distinguishes
          synthetic dummies from real request steps.
    """
    if not _init_phase:
        return False
    if layer_name not in _logged_bypasses:
        _logged_bypasses.add(layer_name)
        _LOGGER.info(
            "[%s] synthetic engine forward bypassed admission for layer=%s "
            "(capture / profile / idle-DP dummy carries no scheduler step)",
            DSV41_LAYERWISE_ADMISSION_TAG,
            layer_name,
        )
    return True


def set_admission_registry(registry: AdmissionRegistry) -> None:
    """Bind the worker adapter's admission registry (once per registration)."""
    global _admission_registry
    _admission_registry = registry
    _LOGGER.info(
        "[%s] admission registry bound: layerwise KV transfer path is ACTIVE "
        "(registration completed; forward hooks now gated)",
        DSV41_LAYERWISE_ADMISSION_TAG,
    )


def clear_admission_registry() -> None:
    """Drop the registry handle; the forward path returns to stock behavior."""
    global _admission_registry
    _admission_registry = None


def admission_registry_is_set() -> bool:
    """Report whether a layerwise registry is bound (diagnostics/tests)."""
    return _admission_registry is not None


def guard_layer_admission(layer_name: str) -> None:
    """Guard one ``dsa_v41_forward`` dispatch.

    Args:
        layer_name: The custom-op dispatch key (``...self_attn.v41_attn``)
            of the attention object about to execute.

    Behavior:
        - No registry bound (stock deployment): return immediately, no side
          effects, so fake/profiling/capture paths are unaffected.
        - Registry bound: delegate to
          ``registry.check_admitted(layer_name)``; a rejection propagates
          :class:`DSV41LayerwiseAdmissionError` (or the adapter's own
          fail-closed error) and the forward must not continue.
    """
    registry = _admission_registry
    if registry is None:
        return
    registry.check_admitted(layer_name)
    if layer_name not in _logged_grants:
        _logged_grants.add(layer_name)
        _LOGGER.info(
            "[%s] admission granted for layer=%s (check_admitted ok)",
            DSV41_LAYERWISE_ADMISSION_TAG,
            layer_name,
        )


def wait_for_layer_load(layer_name: str, timeout: float = 60.0) -> None:
    """Block until ``layer_name``'s retrieved KV resources are ready.

    Called by ``dsa_v41_forward`` before the layer reads its caches. With no
    registry bound this is a no-op, preserving the stock execution path.

    Args:
        layer_name: The custom-op dispatch key of the attention object.
        timeout: Maximum seconds to wait for the retrieves to settle.

    Raises:
        DSV41LayerwiseAdmissionError: Or the adapter's own fail-closed error
            when the layer never becomes ready (timeout or failure).
    """
    registry = _admission_registry
    if registry is None:
        return
    registry.wait_for_layer(layer_name, timeout)
    if layer_name not in _logged_waits:
        _logged_waits.add(layer_name)
        _LOGGER.info(
            "[%s] retrieved KV ready for layer=%s (wait_for_layer_load "
            "settled)",
            DSV41_LAYERWISE_ADMISSION_TAG,
            layer_name,
        )


def notify_layer_produced(layer_name: str) -> None:
    """Record that ``layer_name`` finished writing its KV output.

    Called by ``dsa_v41_forward`` after the layer's attention computes. With
    no registry bound this is a no-op.
    """
    registry = _admission_registry
    if registry is None:
        return
    registry.notify_layer_produced(layer_name)
    if layer_name not in _logged_produces:
        _logged_produces.add(layer_name)
        _LOGGER.info(
            "[%s] KV produced for layer=%s (notify_layer_produced recorded)",
            DSV41_LAYERWISE_ADMISSION_TAG,
            layer_name,
        )
