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

This module is deliberately dependency-free (stdlib only) so it can be unit
tested on CPU and imported by ``dsa_v41`` without pulling LMCache or torch_npu.

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
"""

from typing import Protocol, runtime_checkable

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


def set_admission_registry(registry: AdmissionRegistry) -> None:
    """Bind the worker adapter's admission registry (once per registration)."""
    global _admission_registry
    _admission_registry = registry


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


def notify_layer_produced(layer_name: str) -> None:
    """Record that ``layer_name`` finished writing its KV output.

    Called by ``dsa_v41_forward`` after the layer's attention computes. With
    no registry bound this is a no-op.
    """
    registry = _admission_registry
    if registry is None:
        return
    registry.notify_layer_produced(layer_name)
