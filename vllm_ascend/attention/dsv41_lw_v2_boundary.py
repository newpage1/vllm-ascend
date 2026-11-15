# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side boundary / step / ring seam for DSV4.1 layerwise v2.

This module is the engine half of contract section 6. It provides the pure
boundary DTOs, the codec install point, and the real scheduler-step / ring
binding that the layerwise v2 connector's group boundary hooks need. The
transport codec itself (wire descriptor, payload staging, encode/decode) is
implemented on the LMCache-Ascend side; the engine only hands it real facts.

The seam is deliberately isolated from the legacy ``dsv41_layerwise_*``
modules: it imports none of them, owns no admission registry, no lease, no
snapshot lookup and no prefix key. It only knows what the current forward
really did:

* which requests the scheduler put in this step, where their prefix ends and
  which tokens this step computes (host-side facts from the input batch);
* which physical compressor-state ring page each request owns right now;
* the live ring tensor, its plane geometry and the compressor cursor /
  phase that the kernel consumes.

Boundary operations are only driven through the connector's group hooks, so a
token step that carries no real external hit or store never reaches this
module. A real hit with no installed codec raises ``BOUNDARY_NOT_BOUND``
instead of silently continuing, which is the contract's D3 fix.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from vllm_ascend.attention.dsv41_lw_v2 import bind_lw_v2_engine, lw_v2_engine
from vllm_ascend.models.deepseek_v41.topology import (
    DSV41_LW_V2_BOUNDARY_SOURCE_LAYERS,
    DeepseekV41TransferGroup,
    build_layer_plan,
)

logger = logging.getLogger(__name__)
_OBSERVED: set[tuple[str, int]] = set()

#: Boundary source layers are exactly the three groups that own a
#: ``compressor.state_cache`` resource (g2, g8, g14). g20 has no boundary.
DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT = len(DSV41_LW_V2_BOUNDARY_SOURCE_LAYERS)

#: ``position_control`` is the real compressor cursor tuple consumed by the
#: kernel: ``(prefix_end, step_tokens, out_base, in_base, ring_page)``.
POSITION_CONTROL_WORDS = 5

#: Marker between the model root and the per-layer attention modules.  The
#: real root is whatever the live model registered (for DSV4.1 under vLLM this
#: is ``language_model.model``); it is never assumed to be ``model``.
_MODEL_PREFIX_MARKER = ".layers."

#: Error code carried by :class:`DSV41BoundaryNotBoundError`.
BOUNDARY_NOT_BOUND = "BOUNDARY_NOT_BOUND"

_NOT_BOUND = object()


class DSV41BoundaryBindingError(RuntimeError):
    """The engine boundary seam cannot be bound as configured."""


class DSV41BoundaryNotBoundError(RuntimeError):
    """A real boundary hit reached the engine without an installed codec."""

    error_code = BOUNDARY_NOT_BOUND

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_code = BOUNDARY_NOT_BOUND


class DSV41BoundaryStateError(RuntimeError):
    """Live forward / ring facts needed by a boundary operation are missing."""


@dataclass(frozen=True)
class BoundaryRequestView:
    """One request's real forward facts for a single layer boundary.

    Every field is read from the live scheduler step and ring page set; none
    of them may be derived from a hash, a default page, or another request's
    first block. ``written_rows`` are absolute ring rows inside the request's
    own page and are computed with the same formula the kernel uses, so a
    codec can only ever stage a full, page-aligned schema2 payload.
    """

    request_id: str
    request_index: int
    step_id: int
    registration_epoch: int
    server_generation: int
    resource_id: str
    dispatch_layer: str
    execution_layer: str
    prefix_end: int
    step_tokens: int
    ring_length: int
    ring_page: int
    next_slot: int
    compression_ratio: int
    compression_phase: int
    position_control: tuple[int, ...]
    written_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("request_id", "resource_id", "dispatch_layer", "execution_layer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"BoundaryRequestView.{name} must be a non-empty string")
        for name in (
            "request_index",
            "step_id",
            "registration_epoch",
            "server_generation",
            "prefix_end",
            "step_tokens",
            "ring_length",
            "ring_page",
            "next_slot",
            "compression_ratio",
            "compression_phase",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"BoundaryRequestView.{name} must be an int")
        if self.request_index < 0 or self.step_id < 0:
            raise ValueError("BoundaryRequestView request/step indexes must be non-negative")
        if self.registration_epoch < 0 or self.server_generation < 0:
            raise ValueError("BoundaryRequestView generations must be non-negative")
        if self.prefix_end < 0 or self.step_tokens < 0:
            raise ValueError("BoundaryRequestView prefix_end/step_tokens must be non-negative")
        if self.ring_length <= 0:
            raise ValueError("BoundaryRequestView.ring_length must be positive")
        if self.ring_page < 0:
            raise ValueError("BoundaryRequestView.ring_page must be non-negative")
        if self.compression_ratio <= 0:
            raise ValueError("BoundaryRequestView.compression_ratio must be positive")
        if not 0 <= self.compression_phase < self.compression_ratio:
            raise ValueError("BoundaryRequestView.compression_phase must be inside one ratio span")
        if self.next_slot != self.prefix_end % self.ring_length:
            raise ValueError("BoundaryRequestView.next_slot must be prefix_end modulo ring_length")
        control = self.position_control
        if len(control) != POSITION_CONTROL_WORDS or any(type(word) is not int for word in control):
            raise ValueError(
                "BoundaryRequestView.position_control must be "
                f"{POSITION_CONTROL_WORDS} ints (the real compressor cursor)"
            )
        rows = self.written_rows
        if any(type(row) is not int for row in rows):
            raise ValueError("BoundaryRequestView.written_rows must be ints")
        if len(rows) != min(self.step_tokens, self.ring_length):
            raise ValueError(
                "BoundaryRequestView.written_rows must describe exactly this step's "
                "ring rows (min(step_tokens, ring_length))"
            )
        page_start = self.ring_page * self.ring_length
        page_end = page_start + self.ring_length
        if any(not page_start <= row < page_end for row in rows):
            raise ValueError(
                "BoundaryRequestView.written_rows must stay inside this request's "
                "live ring page (schema2 full-page payloads only)"
            )

    @property
    def boundary_end(self) -> int:
        """Logical prefix this step's tokens end at (the boundary payload end)."""
        return self.prefix_end + self.step_tokens


@dataclass(frozen=True)
class BoundaryTensorView:
    """The real compressor-state ring a boundary payload lives in.

    ``ring`` is the live float32 ``[pages, ring_rows, 2 * ring_head_size]``
    page set. Every row is one compressed boundary: the KV plane occupies the
    first ``ring_head_size`` floats, the gate scores plane the second half.
    Both planes belong to the same boundary payload; they are not two
    transfer resources.
    """

    resource_id: str
    dispatch_layer: str
    execution_layer: str
    ring: torch.Tensor
    page_ids: torch.Tensor
    ring_rows: int
    ring_head_size: int

    def __post_init__(self) -> None:
        for name in ("resource_id", "dispatch_layer", "execution_layer"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"BoundaryTensorView.{name} must be a non-empty string")
        if type(self.ring_rows) is not int or self.ring_rows <= 0:
            raise ValueError("BoundaryTensorView.ring_rows must be a positive int")
        if type(self.ring_head_size) is not int or self.ring_head_size <= 0:
            raise ValueError("BoundaryTensorView.ring_head_size must be a positive int")
        if not isinstance(self.ring, torch.Tensor) or not isinstance(self.page_ids, torch.Tensor):
            raise ValueError("BoundaryTensorView needs real torch ring/page tensors")
        if self.ring.dim() != 3:
            raise ValueError(
                "BoundaryTensorView.ring must be [pages, ring_rows, 2 * ring_head_size], "
                f"got shape {tuple(self.ring.shape)}"
            )
        if self.ring.shape[1] != self.ring_rows:
            raise ValueError("BoundaryTensorView.ring_rows disagrees with the live ring tensor")
        if self.ring.shape[2] != 2 * self.ring_head_size:
            raise ValueError("BoundaryTensorView must describe both KV and score planes of one row")
        if self.ring.dtype != torch.float32:
            raise ValueError(f"BoundaryTensorView.ring must be float32, got {self.ring.dtype}")
        if not self.ring.is_contiguous():
            raise ValueError("BoundaryTensorView.ring page set must be contiguous")
        if self.page_ids.dim() != 1 or self.page_ids.dtype.is_floating_point:
            raise ValueError("BoundaryTensorView.page_ids must be a 1-D integer tensor")

    @property
    def dtype_name(self) -> str:
        return str(self.ring.dtype).removeprefix("torch.")

    @property
    def element_size(self) -> int:
        return self.ring.element_size()

    def page_for(self, request_view: BoundaryRequestView) -> int:
        """Physical page of this request, exactly as the scheduler owns it."""
        return request_view.ring_page

    def planes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Describe the two logical planes of one ring row (KV, scores)."""
        width = self.ring_head_size
        row_stride = self.ring.stride(1) * self.element_size
        page_stride = self.ring.stride(0) * self.element_size
        return (
            {
                "name": "kv",
                "dtype": self.dtype_name,
                "shape": (self.ring_rows, width),
                "row_offset_bytes": 0,
                "row_stride_bytes": row_stride,
                "page_stride_bytes": page_stride,
            },
            {
                "name": "scores",
                "dtype": self.dtype_name,
                "shape": (self.ring_rows, width),
                "row_offset_bytes": width * self.element_size,
                "row_stride_bytes": row_stride,
                "page_stride_bytes": page_stride,
            },
        )


@runtime_checkable
class DSV41LwV2BoundaryCodec(Protocol):
    """Boundary codec protocol implemented by the layerwise v2 transport."""

    def capture_layer_boundary(
        self,
        request_view: BoundaryRequestView,
        tensor_view: BoundaryTensorView,
        *,
        stream: Any | None = None,
    ) -> None: ...

    def restore_layer_boundary(
        self,
        request_view: BoundaryRequestView,
        tensor_view: BoundaryTensorView,
        *,
        stream: Any | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _ForwardRequest:
    """Host-side facts of one request in this forward."""

    request_id: str
    request_index: int
    prefix_end: int
    step_tokens: int
    out_base: int
    in_base: int
    ring_page: int


@dataclass(frozen=True)
class _ForwardStep:
    """The real scheduler step a model forward belongs to."""

    step_id: int
    requests: tuple[_ForwardRequest, ...]
    synthetic: bool = False


@dataclass(frozen=True)
class _RingState:
    """Resolved live ring facts of one boundary group."""

    resource_id: str
    dispatch_layer: str
    execution_layer: str
    tensor_view: BoundaryTensorView
    ring_rows: int
    ring_head_size: int
    num_reqs: int
    ratio: int


# The boundary codec is a single process-wide binding, exactly like the engine
# seam it accompanies. It is installed from the model runner before the first
# real forward and never mutated per layer or per step.
_codec: DSV41LwV2BoundaryCodec | None = None
_hook: DSV41LwV2BoundaryHook | None = None
_boundary_resource_ids: tuple[str, ...] = ()
_resource_geometry: Mapping[str, tuple[int, int]] = {}
_forward_step: _ForwardStep | None = None
_next_forward_step_id = 0


def lw_v2_boundary_codec() -> DSV41LwV2BoundaryCodec | None:
    """Return the installed boundary codec, if any."""
    return _codec


def lw_v2_boundary_resource_ids() -> tuple[str, ...]:
    """Return the compressor-state resources this engine installed for."""
    return _boundary_resource_ids


def bind_forward_step(
    runner: Any,
    *,
    decoder_replay_dummy: bool = False,
    synthetic: bool = False,
) -> object:
    """Bind the running model forward to its real scheduler step.

    Called from ``NPUModelRunner._model_forward``; returns the previous
    binding for :func:`clear_forward_step`. Nothing is read from the input
    batch until a boundary codec is installed, so stock serving pays one
    attribute check.

    ``synthetic`` is set by the model runner from the engine-init phase flag
    (profile / warmup) and ``decoder_replay_dummy`` from the replay path;
    graph capture is detected here. A synthetic step carries no requests, so
    no boundary operation can be emitted for a dummy forward.
    """
    global _forward_step, _next_forward_step_id

    if _codec is None:
        return _NOT_BOUND
    previous = _forward_step
    step_id = _next_forward_step_id
    _next_forward_step_id += 1
    if synthetic or decoder_replay_dummy or _capturing_graph():
        _forward_step = _ForwardStep(step_id=step_id, requests=(), synthetic=True)
        return previous
    _forward_step = _read_forward_step(runner, step_id)
    return previous


def clear_forward_step(token: object) -> None:
    """Retire a forward step binding, even when the forward raised."""
    global _forward_step
    if token is _NOT_BOUND:
        return
    _forward_step = token if isinstance(token, _ForwardStep) else None


def forward_step() -> _ForwardStep | None:
    """Return the currently bound forward step (tests / diagnostics)."""
    return _forward_step


def install_dsv41_lw_v2_codec(
    codec: DSV41LwV2BoundaryCodec,
    *,
    model: Any,
) -> None:
    """Install the boundary codec and arm the connector's group hooks.

    Must run before the first real forward. Validates the codec methods, the
    boundary resource identity (exactly the three compressor-state resources
    of the bound plan) and the two-plane geometry of every live ring, then
    binds one group hook into the connector. Re-installing the same codec
    instance is idempotent; a different instance conflicts.
    """
    global _codec, _hook, _boundary_resource_ids, _resource_geometry

    _validate_codec(codec)
    binding = lw_v2_engine()
    if binding is None:
        raise DSV41BoundaryBindingError(
            "layerwise v2 engine seam is not bound; bind the connector before "
            "installing its boundary codec"
        )
    if _codec is codec:
        return
    if _codec is not None:
        raise DSV41BoundaryBindingError(
            "a different layerwise v2 boundary codec is already installed"
        )
    expected = _expected_boundary_resource_ids(binding)
    observed, geometry = _model_boundary_resources(model, expected)
    hook = DSV41LwV2BoundaryHook(codec=codec, connector=binding.hooks)
    _bind_boundary_hook(binding.hooks, hook)
    _codec = codec
    _hook = hook
    _boundary_resource_ids = observed
    _resource_geometry = geometry
    logger.info(
        "installed DSV4.1 layerwise v2 boundary codec for resources %s",
        ", ".join(observed),
    )


def uninstall_dsv41_lw_v2_codec() -> None:
    """Drop the installed codec (test/teardown only; not a runtime path)."""
    global _codec, _hook, _boundary_resource_ids, _resource_geometry, _forward_step
    _codec = None
    _hook = None
    _boundary_resource_ids = ()
    _resource_geometry = {}
    _forward_step = None


def install_from_connector(model: Any, connector: Any | None = None) -> bool:
    """Wire the engine seam to a live layerwise v2 connector.

    Called from ``initialize_kv_cache`` after the connector registered the KV
    caches. Stock connectors are left completely untouched (returns False).
    For a v2 connector this binds the engine seam (idempotent) with the model
    root resolved from the live plan/registration, then installs the boundary
    codec the connector exposes. A v2 connector without a codec
    logs and stays inert; the connector itself already fails closed with
    ``BOUNDARY_NOT_BOUND`` the moment a real boundary hit appears.
    """
    if connector is None:
        connector = _kv_transfer_group()
    if connector is None or not _is_lw_v2_connector(connector):
        return False
    config = getattr(model, "config", None)
    if config is None:
        raise DSV41BoundaryBindingError(
            "layerwise v2 boundary install needs the model config to rebuild "
            "the default DSV4.1 topology"
        )
    try:
        topology = build_layer_plan(config)
    except (TypeError, ValueError) as exc:
        raise DSV41BoundaryBindingError(
            f"layerwise v2 boundary install rejected the model topology: {exc}"
        ) from exc
    binding = lw_v2_engine()
    if binding is None or binding.hooks is not connector:
        bind_lw_v2_engine(
            connector,
            topology=topology,
            prefix=_resolve_model_prefix(connector, model),
        )
    codec = getattr(connector, "boundary_codec", None)
    if codec is None:
        logger.warning(
            "layerwise v2 connector exposes no boundary codec yet; boundary "
            "capture/restore stays unbound until it is installed"
        )
        return False
    install_dsv41_lw_v2_codec(codec, model=model)
    return True


class DSV41LwV2BoundaryHook:
    """Engine implementation of the connector's group boundary hook.

    The connector calls :meth:`restore_group_boundary` right after a group's
    async load is ready and :meth:`capture_group_boundary` after the source
    layer's forward. Each call resolves the live transport step, the live ring
    state and the live request page set, then hands the codec real DTOs. A
    missing codec is a hard ``BOUNDARY_NOT_BOUND`` error, never a no-op.
    """

    def __init__(self, *, codec: DSV41LwV2BoundaryCodec, connector: Any) -> None:
        self._codec = codec
        self._connector = connector

    def restore_group_boundary(self, group_id: str) -> None:
        self._run(group_id, action="restore")

    def capture_group_boundary(self, group_id: str) -> None:
        self._run(group_id, action="capture")

    def _run(self, group_id: str, *, action: str) -> None:
        codec = _codec
        if codec is None or self._codec is not codec:
            raise DSV41BoundaryNotBoundError(
                f"group {group_id} has a real boundary {action} but no boundary "
                f"codec is installed ({BOUNDARY_NOT_BOUND})"
            )
        binding = lw_v2_engine()
        if binding is None:
            raise DSV41BoundaryBindingError(
                "layerwise v2 engine seam is not bound while a boundary hook ran"
            )
        group = _group_by_id(binding, group_id)
        if group is None:
            raise DSV41BoundaryBindingError(f"unknown layerwise v2 group {group_id!r}")
        if not group.has_boundary:
            raise DSV41BoundaryBindingError(
                f"group {group_id!r} declares no boundary resource; the connector "
                "must not restore or capture one"
            )
        forward_step = _forward_step
        if forward_step is None or forward_step.synthetic:
            raise DSV41BoundaryStateError(
                f"group {group_id} reached its boundary {action} without a real "
                "forward step binding"
            )
        capture_prefill = False
        if action == "restore":
            step = _current_transport_step(self._connector, group_id)
            targets = _target_requests(step, action)
            registration_epoch = int(getattr(step, "registration_epoch", 0) or 0)
            server_generation = int(getattr(step, "server_generation", 0) or 0)
        else:
            step = getattr(self._connector, "current_step", None)
            synthetic = bool(getattr(step, "synthetic", False)) if step else False
            if step is not None and not synthetic:
                targets = _target_requests(step, action)
                registration_epoch = int(
                    getattr(step, "registration_epoch", 0) or 0
                )
                server_generation = int(
                    getattr(step, "server_generation", 0) or 0
                )
            else:
                # vLLM 0.30 builds store metadata only after the prefill
                # forward, so the cold prefill captures from the live forward
                # batch.  Only chunk-aligned endpoints are captured; the store
                # window that later matches the endpoint attaches the payload,
                # and an unaligned endpoint is never re-labelled (T25-1).
                capture_prefill = True
                targets = tuple(
                    (request.request_id, 0) for request in forward_step.requests
                )
                registration_epoch = int(
                    getattr(self._connector, "registration_epoch", 0) or 0
                )
                server_generation = int(
                    getattr(self._connector, "server_generation", 0) or 0
                )
        if not targets:
            return
        attn = _attention_for(group)
        state = _ring_state(group, attn)
        if len(forward_step.requests) != state.num_reqs:
            raise DSV41BoundaryStateError(
                f"group {group_id} forward step and attention metadata disagree on "
                f"the batch: {len(forward_step.requests)} bound request(s) vs "
                f"{state.num_reqs} metadata row(s)"
            )
        requests_by_id = {request.request_id: request for request in forward_step.requests}
        if len(requests_by_id) != len(forward_step.requests):
            raise DSV41BoundaryStateError(
                f"group {group_id} forward step carries duplicate request ids"
            )
        method = getattr(codec, f"{action}_layer_boundary", None)
        if not callable(method):
            raise DSV41BoundaryBindingError(
                f"boundary codec is missing {action}_layer_boundary"
            )
        for request_id, transport_endpoint in targets:
            forward_request = requests_by_id.get(request_id)
            if forward_request is None:
                raise DSV41BoundaryStateError(
                    f"group {group_id} boundary {action} targets request "
                    f"{request_id!r}, which is not part of this forward"
                )
            if capture_prefill:
                endpoint = forward_request.prefix_end + forward_request.step_tokens
                if (
                    forward_request.step_tokens <= 0
                    or endpoint % state.ring_rows != 0
                ):
                    continue
                observation = (group.group_id, endpoint)
                if observation not in _OBSERVED:
                    _OBSERVED.add(observation)
                    logger.warning(
                        "dsv41 lw v2 boundary capture group=%s request=%s "
                        "endpoint=%s step_tokens=%s ring_rows=%s",
                        group.group_id,
                        request_id,
                        endpoint,
                        forward_request.step_tokens,
                        state.ring_rows,
                    )
            view = _request_view(
                state,
                forward_request,
                step_id=forward_step.step_id,
                registration_epoch=registration_epoch,
                server_generation=server_generation,
                ring_page=int(state.tensor_view.page_ids[forward_request.request_index]),
                prefix_end=transport_endpoint if action == "restore" else None,
            )
            try:
                method(view, state.tensor_view, stream=None)
            except (
                DSV41BoundaryBindingError,
                DSV41BoundaryNotBoundError,
                DSV41BoundaryStateError,
            ):
                raise
            except Exception as exc:  # noqa: BLE001 - boundary failure must fail the step
                raise DSV41BoundaryStateError(
                    f"group {group_id} boundary {action} failed for request "
                    f"{request_id!r} on resource {state.resource_id!r}"
                ) from exc


def _validate_codec(codec: Any) -> None:
    if codec is None:
        raise DSV41BoundaryBindingError("a boundary codec instance is required")
    for name in ("capture_layer_boundary", "restore_layer_boundary"):
        method = getattr(codec, name, None)
        if not callable(method):
            raise DSV41BoundaryBindingError(
                f"boundary codec is missing the {name} method"
            )
        try:
            inspect.signature(method).bind(object(), object(), stream=None)
        except TypeError as exc:
            raise DSV41BoundaryBindingError(
                f"boundary codec {name} must accept (request_view, tensor_view, "
                "*, stream=None)"
            ) from exc


def _resolve_model_prefix(connector: Any, model: Any) -> str:
    """Derive the live model root instead of hard-coding ``"model"``.

    The connector's grouped plan is built from the names the worker really
    registered, so its first execution layer carries the real root (for
    example ``language_model.model``).  When no plan is bound yet, fall back
    to the live compressor-state resource names.  Anything else fails closed:
    binding the engine seam with a guessed root would make every real capture
    look up the wrong dispatch layer.
    """
    plan = getattr(connector, "grouped_plan", None)
    base = getattr(plan, "base", None)
    layer_order = getattr(base, "layer_order", None)
    if layer_order:
        layer = layer_order[0]
        if isinstance(layer, str) and _MODEL_PREFIX_MARKER in layer:
            prefix = layer.split(_MODEL_PREFIX_MARKER, 1)[0]
            if prefix:
                return prefix
    modules = getattr(model, "modules", None)
    if callable(modules):
        for module in modules():
            impl = getattr(module, "v41_impl", None)
            resource_id = getattr(impl, "compressor_state_prefix", None)
            if (
                isinstance(resource_id, str)
                and _MODEL_PREFIX_MARKER in resource_id
            ):
                prefix = resource_id.split(_MODEL_PREFIX_MARKER, 1)[0]
                if prefix:
                    return prefix
    raise DSV41BoundaryBindingError(
        "cannot resolve the layerwise v2 model prefix from the connector plan "
        "or the live compressor-state resources"
    )


def _expected_boundary_resource_ids(binding: Any) -> tuple[str, ...]:
    resource_ids = tuple(
        group.boundary_resource.cache_name
        for group in binding.groups
        if group.has_boundary
    )
    if len(resource_ids) != DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT:
        raise DSV41BoundaryBindingError(
            "a layerwise v2 engine binding must expose exactly "
            f"{DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT} boundary resources, got "
            f"{len(resource_ids)}"
        )
    return resource_ids


def _model_boundary_resources(
    model: Any, expected: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[str, tuple[int, int]]]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise DSV41BoundaryBindingError(
            "boundary install needs the real model module tree to locate its "
            "compressor-state rings"
        )
    observed: dict[str, tuple[int, int]] = {}
    for module in modules():
        impl = getattr(module, "v41_impl", None)
        resource_id = getattr(impl, "compressor_state_prefix", None)
        if not isinstance(resource_id, str) or not resource_id:
            continue
        if resource_id in observed:
            raise DSV41BoundaryBindingError(
                f"two model modules declare compressor state {resource_id!r}"
            )
        observed[resource_id] = _ring_geometry(resource_id, _raw_ring(module))
    if tuple(sorted(observed)) != tuple(sorted(expected)):
        raise DSV41BoundaryBindingError(
            "model compressor-state resources do not match the bound plan: "
            f"expected {sorted(expected)}, found {sorted(observed)}"
        )
    ordered = tuple(resource_id for resource_id in expected)
    return ordered, {resource_id: observed[resource_id] for resource_id in ordered}


def _raw_ring(module: Any) -> torch.Tensor:
    compressor = getattr(module, "compressor", None)
    state_cache = getattr(compressor, "state_cache", None)
    kv_cache = getattr(state_cache, "kv_cache", None)
    if not kv_cache:
        raise DSV41BoundaryBindingError(
            "a module declares compressor state but carries no state_cache kv_cache"
        )
    return kv_cache[0]


def _ring_geometry(resource_id: str, ring: torch.Tensor) -> tuple[int, int]:
    if not isinstance(ring, torch.Tensor):
        raise DSV41BoundaryBindingError(
            f"compressor state {resource_id!r} is not a torch tensor"
        )
    if ring.dim() == 4:  # [pages, rows, 1, 2 * head]
        ring = ring.squeeze(-2)
    if ring.dim() != 3:
        raise DSV41BoundaryBindingError(
            f"compressor state {resource_id!r} must be [pages, rows, 2 * head], "
            f"got shape {tuple(ring.shape)}"
        )
    if ring.dtype != torch.float32:
        raise DSV41BoundaryBindingError(
            f"compressor state {resource_id!r} must be float32, got {ring.dtype}"
        )
    if not ring.is_contiguous() or ring.stride(2) != 1:
        raise DSV41BoundaryBindingError(
            f"compressor state {resource_id!r} page set must be contiguous with "
            "packed planes"
        )
    rows = int(ring.shape[1])
    width = int(ring.shape[2])
    if rows <= 0 or width <= 0 or width % 2:
        raise DSV41BoundaryBindingError(
            f"compressor state {resource_id!r} must pack KV and score planes into "
            f"one even-width row, got rows={rows} width={width}"
        )
    return rows, width // 2


def _bind_boundary_hook(connector: Any, hook: DSV41LwV2BoundaryHook) -> None:
    binder = getattr(connector, "bind_boundary_hook", None)
    if not callable(binder):
        raise DSV41BoundaryBindingError(
            "layerwise v2 connector does not expose bind_boundary_hook"
        )
    binder(hook)


def _is_lw_v2_connector(connector: Any) -> bool:
    connector_type = type(connector)
    return (
        getattr(connector_type, "grouped_plan", None) is not None
        and getattr(connector_type, "group_for_layer", None) is not None
        and callable(getattr(connector, "bind_boundary_hook", None))
    )


def _kv_transfer_group() -> Any | None:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return None
    return get_kv_transfer_group()


def _group_by_id(binding: Any, group_id: str) -> DeepseekV41TransferGroup | None:
    for group in binding.groups:
        if group.group_id == group_id:
            return group
    return None


def _current_transport_step(connector: Any, group_id: str) -> Any:
    try:
        step = connector.current_step
    except AttributeError as exc:
        raise DSV41BoundaryBindingError(
            "layerwise v2 connector does not expose the current transport step"
        ) from exc
    if step is None:
        raise DSV41BoundaryStateError(
            f"group {group_id} boundary hook ran without a current transport step"
        )
    if getattr(step, "synthetic", False):
        raise DSV41BoundaryStateError(
            f"group {group_id} boundary hook ran inside a synthetic step"
        )
    return step


def _target_requests(step: Any, action: str) -> tuple[tuple[str, int], ...]:
    """Real hit (restore) or store (capture) requests of this transport step.

    Each entry carries the logical prefix end the transport step serves for
    that request: ``load_prefix_end`` for a hit, ``store_prefix_end`` for a
    store.  A restore presents the *hit* endpoint to the codec, because the
    boundary page was captured there, the payload is keyed there and the codec
    validates the page's ``boundary_end`` against it.  The forward's own
    prefix can be smaller when SWA bounded replay rewinds the hit; that replay
    start is an engine fact, not the boundary's endpoint.
    """
    field_name = "load_prefix_end" if action == "restore" else "store_prefix_end"
    targets: list[tuple[str, int]] = []
    for request in getattr(step, "requests", ()) or ():
        endpoint = int(getattr(request, field_name, 0) or 0)
        if endpoint <= 0:
            continue
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise DSV41BoundaryStateError(
                f"transport step carries a {action} request without an id"
            )
        targets.append((request_id, endpoint))
    return tuple(targets)


def _attention_for(group: DeepseekV41TransferGroup) -> Any:
    forward_context = _get_forward_context()
    layers = getattr(forward_context, "no_compile_layers", None)
    attn = layers.get(group.dispatch_layer) if isinstance(layers, Mapping) else None
    if attn is None:
        raise DSV41BoundaryStateError(
            f"group {group.group_id} boundary layer {group.dispatch_layer!r} is not "
            "part of the current forward"
        )
    layer_name = getattr(attn, "v41_layer_name", None)
    if layer_name != group.dispatch_layer:
        raise DSV41BoundaryStateError(
            f"dispatch layer {group.dispatch_layer!r} resolved to a module named "
            f"{layer_name!r}"
        )
    return attn


def _ring_state(group: DeepseekV41TransferGroup, attn: Any) -> _RingState:
    impl = getattr(attn, "v41_impl", None)
    resource_id = getattr(impl, "compressor_state_prefix", None)
    if not isinstance(resource_id, str) or not resource_id:
        raise DSV41BoundaryStateError(
            f"boundary group {group.group_id} layer {group.dispatch_layer!r} declares "
            "no compressor state resource"
        )
    boundary_resource = group.boundary_resource
    if boundary_resource is None or boundary_resource.cache_name != resource_id:
        raise DSV41BoundaryStateError(
            f"group {group.group_id} boundary resource "
            f"{getattr(boundary_resource, 'cache_name', None)!r} does not match the "
            f"live compressor state {resource_id!r}"
        )
    execution_layer = getattr(attn, "layerwise_execution_layer", None)
    if not isinstance(execution_layer, str) or not execution_layer:
        raise DSV41BoundaryStateError(
            f"boundary group {group.group_id} attention module has no execution layer"
        )
    ring = _raw_ring(attn)
    if ring.dim() == 4:
        ring = ring.squeeze(-2)
    geometry = _resource_geometry.get(resource_id)
    if geometry is None:
        raise DSV41BoundaryStateError(
            f"compressor state {resource_id!r} was not installed for boundaries"
        )
    rows, head = geometry
    if int(ring.shape[1]) != rows or int(ring.shape[2]) != 2 * head:
        raise DSV41BoundaryStateError(
            f"compressor state {resource_id!r} live ring geometry changed after "
            f"install: installed rows={rows} head={head}"
        )
    metadata = _compressor_metadata(resource_id)
    num_reqs = int(getattr(metadata, "num_reqs", 0) or 0)
    if num_reqs <= 0:
        raise DSV41BoundaryStateError(
            f"compressor metadata for {resource_id!r} reports num_reqs={num_reqs}"
        )
    ratio = int(getattr(metadata, "compress_ratio", 0) or 0)
    if ratio <= 0:
        raise DSV41BoundaryStateError(
            f"compressor metadata for {resource_id!r} reports compress_ratio={ratio}"
        )
    if not bool(getattr(metadata, "is_compressor_state", False)):
        raise DSV41BoundaryStateError(
            f"compressor metadata for {resource_id!r} is not a compressor-state "
            "boundary view; schema2 full-page payloads cannot be built"
        )
    block_table = getattr(metadata, "block_table", None)
    if block_table is None:
        raise DSV41BoundaryStateError(
            f"compressor metadata for {resource_id!r} carries no block table"
        )
    page_ids = block_table[:num_reqs, 0]
    tensor_view = BoundaryTensorView(
        resource_id=resource_id,
        dispatch_layer=group.dispatch_layer,
        execution_layer=execution_layer,
        ring=ring,
        page_ids=page_ids,
        ring_rows=rows,
        ring_head_size=head,
    )
    return _RingState(
        resource_id=resource_id,
        dispatch_layer=group.dispatch_layer,
        execution_layer=execution_layer,
        tensor_view=tensor_view,
        ring_rows=rows,
        ring_head_size=head,
        num_reqs=num_reqs,
        ratio=ratio,
    )


def _compressor_metadata(resource_id: str) -> Any:
    forward_context = _get_forward_context()
    attn_metadata = getattr(forward_context, "attn_metadata", None) or {}
    if not isinstance(attn_metadata, Mapping):
        raise DSV41BoundaryStateError(
            "forward attention metadata is not addressable by compressor resource"
        )
    metadata = attn_metadata.get(resource_id)
    if metadata is None:
        raise DSV41BoundaryStateError(
            f"no compressor metadata for {resource_id!r} in this forward"
        )
    return metadata


def _request_view(
    state: _RingState,
    forward_request: _ForwardRequest,
    *,
    step_id: int,
    registration_epoch: int,
    server_generation: int,
    ring_page: int | None = None,
    prefix_end: int | None = None,
) -> BoundaryRequestView:
    """Describe one request's boundary plane for a capture or a restore.

    ``prefix_end`` overrides the forward's computed-token prefix for a
    restore: a hit rewound by SWA bounded replay runs from an earlier prefix,
    while the boundary page it restores is the one captured at the hit
    endpoint.  The page geometry (page, rows, slot, phase) follows whichever
    prefix the view describes.
    """
    if prefix_end is None:
        prefix_end = forward_request.prefix_end
    step_tokens = forward_request.step_tokens
    if ring_page is None:
        ring_page = forward_request.ring_page
    if ring_page < 0:
        raise DSV41BoundaryStateError(
            f"request index {forward_request.request_index} has no live boundary page"
        )
    ring_rows = state.ring_rows
    control = (
        prefix_end,
        step_tokens,
        forward_request.out_base,
        forward_request.in_base,
        ring_page,
    )
    return BoundaryRequestView(
        request_id=forward_request.request_id,
        request_index=forward_request.request_index,
        step_id=step_id,
        registration_epoch=registration_epoch,
        server_generation=server_generation,
        resource_id=state.resource_id,
        dispatch_layer=state.dispatch_layer,
        execution_layer=state.execution_layer,
        prefix_end=prefix_end,
        step_tokens=step_tokens,
        ring_length=ring_rows,
        ring_page=ring_page,
        next_slot=prefix_end % ring_rows,
        compression_ratio=state.ratio,
        compression_phase=prefix_end % state.ratio,
        position_control=control,
        written_rows=_written_rows(
            ring_rows=ring_rows,
            prefix_end=prefix_end,
            step_tokens=step_tokens,
            ring_page=ring_page,
        ),
    )


def _written_rows(
    *, ring_rows: int, prefix_end: int, step_tokens: int, ring_page: int
) -> tuple[int, ...]:
    """Absolute ring rows this step writes, with the kernel's row formula."""
    tail = min(max(step_tokens, 0), ring_rows)
    first_token = prefix_end + max(step_tokens, 0) - tail
    page_start = ring_page * ring_rows
    return tuple(page_start + (first_token + offset) % ring_rows for offset in range(tail))


def _read_forward_step(runner: Any, step_id: int) -> _ForwardStep:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        raise DSV41BoundaryStateError(
            "a real forward step has no input batch to bind its boundary to"
        )
    raw_req_ids = tuple(getattr(input_batch, "req_ids", None) or ())
    num_reqs = getattr(input_batch, "num_reqs", None)
    if num_reqs is None:
        num_reqs = len([req_id for req_id in raw_req_ids if req_id is not None])
    num_reqs = int(num_reqs)
    if num_reqs <= 0:
        raise DSV41BoundaryStateError(
            f"a real forward step carries no requests (num_reqs={num_reqs})"
        )
    if len(raw_req_ids) < num_reqs:
        raise DSV41BoundaryStateError(
            "input batch request ids are shorter than its request count"
        )
    req_ids = raw_req_ids[:num_reqs]
    request_ids = []
    for req_id in req_ids:
        if not isinstance(req_id, str) or not req_id:
            raise DSV41BoundaryStateError(
                "input batch carries an unset request id in a real forward step"
            )
        request_ids.append(req_id)
    if len(set(request_ids)) != num_reqs:
        raise DSV41BoundaryStateError(
            "input batch carries duplicate request ids in a real forward step"
        )
    prefix_ends = _host_int_array(
        getattr(input_batch, "num_computed_tokens_cpu", None), "num_computed_tokens_cpu"
    )
    step_tokens = _host_int_array(getattr(runner, "query_lens", None), "query_lens")
    if prefix_ends.shape[0] < num_reqs or step_tokens.shape[0] < num_reqs:
        raise DSV41BoundaryStateError(
            "host scheduler arrays are shorter than the bound request count"
        )
    block_tables = _host_block_table(input_batch)
    requests = []
    out_base = 0
    for index in range(num_reqs):
        tokens = int(step_tokens[index])
        if tokens < 0:
            raise DSV41BoundaryStateError(
                f"request {request_ids[index]!r} schedules a negative token count"
            )
        page = int(block_tables[index, 0]) if tokens > 0 else 0
        requests.append(
            _ForwardRequest(
                request_id=request_ids[index],
                request_index=index,
                prefix_end=int(prefix_ends[index]),
                step_tokens=tokens,
                out_base=out_base,
                in_base=out_base,
                ring_page=page,
            )
        )
        out_base += tokens
    return _ForwardStep(step_id=step_id, requests=tuple(requests))


def _host_int_array(value: Any, name: str, *, ndim: int = 1) -> np.ndarray:
    """Read a host-side scheduler array without ever touching device memory."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise DSV41BoundaryStateError(
                f"{name} must stay on the host for layerwise v2 step binding"
            )
        array = value.numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise DSV41BoundaryStateError(f"forward step carries no host {name} array")
    if not np.issubdtype(array.dtype, np.integer):
        raise DSV41BoundaryStateError(f"host {name} must hold integer token counts")
    array = np.asarray(array, dtype=np.int64)
    if array.ndim != ndim:
        raise DSV41BoundaryStateError(
            f"host {name} must be {ndim}-dimensional, got shape {array.shape}"
        )
    return array


def _host_block_table(input_batch: Any) -> np.ndarray:
    block_table = getattr(input_batch, "block_table", None)
    tables = getattr(block_table, "block_tables", None) if block_table is not None else None
    if not tables:
        raise DSV41BoundaryStateError(
            "forward step carries no host block table for its compressor pages"
        )
    try:
        tensor = tables[0].get_cpu_tensor()
    except AttributeError as exc:
        raise DSV41BoundaryStateError(
            "forward step block table does not expose host CPU rows"
        ) from exc
    array = _host_int_array(tensor, "block_tables", ndim=2)
    if array.ndim != 2 or array.shape[1] < 1:
        raise DSV41BoundaryStateError(
            "forward step block table must be [requests, blocks] with at least one block"
        )
    return array


def _capturing_graph() -> bool:
    forward_context = _get_forward_context()
    if bool(getattr(forward_context, "capturing", False)):
        return True
    npu = getattr(torch, "npu", None)
    is_capturing = getattr(npu, "is_current_stream_capturing", None)
    if callable(is_capturing):
        try:
            return bool(is_capturing())
        except Exception:  # noqa: BLE001 - capability probe only
            return False
    return False


def _get_forward_context() -> Any:
    from vllm.forward_context import get_forward_context

    return get_forward_context()


__all__ = [
    "BOUNDARY_NOT_BOUND",
    "BoundaryRequestView",
    "BoundaryTensorView",
    "DSV41BoundaryBindingError",
    "DSV41BoundaryNotBoundError",
    "DSV41BoundaryStateError",
    "DSV41LwV2BoundaryCodec",
    "DSV41LwV2BoundaryHook",
    "bind_forward_step",
    "clear_forward_step",
    "forward_step",
    "install_dsv41_lw_v2_codec",
    "install_from_connector",
    "lw_v2_boundary_codec",
    "lw_v2_boundary_resource_ids",
    "uninstall_dsv41_lw_v2_codec",
]
