# SPDX-License-Identifier: Apache-2.0
"""S1 layerwise hooks on the engine side: C2 boundary capture / restore.

The DSV4.1 layerwise path keeps a compressor ring state per ratio-2 layer
(``compressor.state_cache``: 32 rows x ``2 * head_dim`` float32 per page).
S0 only moved the long KV / index / SWA token ranges; the C2 ring was
declared "cold only" because nothing captured or restored it, so a lookup
could never hit (see ``dsv41_mp_adapter`` module docstring). This module is
the vllm-ascend half of S1:

- a *request view* of the real forward metadata (physical ring page,
  absolute prefix end, rows written by this step, the five kernel
  position-control words) so the boundary codec never guesses a page from a
  hash or a modulo;
- a *tensor view* over the real compressor state page (layout + strides) so
  the codec copies payload without reaching into engine internals;
- per-layer ``restore`` before the layer's first ring read and ``capture``
  after its writes, wired into ``dsa_v41_forward``;
- a per-step TP gate: every TP rank must report the same step and no rank
  may have flagged a failure before any rank enters the next collective.

Everything here is inert unless a codec is installed by the layerwise
connector (``set_boundary_state_codec``). With no codec the hooks return
immediately, so stock serving, fake/profiling forwards and graph capture
are bit-identical to the pre-S1 engine.

Fail-closed rules (a codec is installed, i.e. S1 was requested):

- a real forward that carries no bound step view raises instead of silently
  skipping capture (that is the S0 hole this module exists to close);
- a ratio-2 layer whose compressor metadata is missing raises;
- a batch whose metadata request count does not match the bound request
  order raises;
- codec errors are wrapped and mark the step failed, so the next TP gate
  stops every rank before it enters an unmatched collective.

The wire descriptor stays in LMCache (``BoundaryStateDescriptor``); the
codec builds it from these views. Nothing in this module publishes, leases
or transfers bytes -- that is the transport's job (t77).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from vllm.logger import init_logger

from vllm_ascend.attention.dsv41_layerwise_admission import init_phase_active

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = init_logger(__name__)

DSV41_LAYERWISE_C2_TAG = "dsv41_layerwise"

#: Compressor ring rows per page; mirrored from ``compressor.STATE_RING_ROWS``
#: so the view can describe the page without importing the model layer.
RING_ROWS_PER_PAGE = 32

#: Words of kernel position control carried per request (row0..row4 of
#: ``c2_ring_metadata``): absolute prefix end, tokens written this step,
#: output base, input base and the physical ring page.
POSITION_CONTROL_WORDS = 5


class DSV41LayerwiseC2Error(RuntimeError):
    """The S1 layerwise C2 contract was violated (fail-closed)."""


@dataclass(frozen=True)
class DSV41BoundaryRequestView:
    """One request's real forward facts for a single layer boundary."""

    layer_name: str
    execution_layer: str
    request_id: str
    request_index: int
    step_id: int
    generation: int
    prefix_end: int
    step_tokens: int
    ring_rows: int
    ring_head_size: int
    ring_length: int
    ring_page: int
    next_slot: int
    compression_ratio: int
    compression_phase: int
    position_control: tuple[int, ...]
    written_rows: tuple[int, ...]


@dataclass(frozen=True)
class DSV41BoundaryTensorView:
    """The real compressor state page set a boundary payload lives in."""

    layer_name: str
    execution_layer: str
    ring: torch.Tensor
    page_ids: torch.Tensor
    ring_rows: int
    ring_head_size: int

    @property
    def dtype_name(self) -> str:
        return str(self.ring.dtype).removeprefix("torch.")

    @property
    def element_size(self) -> int:
        return self.ring.element_size()

    def page_for(self, request_view: DSV41BoundaryRequestView) -> int:
        """Physical page of this request (never derived from a hash)."""
        return request_view.ring_page

    def planes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Describe the two logical planes of one ring row (kv, scores).

        A row is ``2 * head_dim`` float32: the KV projection occupies the
        first ``head_dim`` floats and the gate scores the second half. Both
        planes share one row set; the wire schema counts them as
        ``COMPRESSOR_RING_PLANES`` planes of one boundary, not as two
        transfer resources.
        """
        width = self.ring_head_size
        row_stride = self.ring.stride(1) * self.element_size
        stride = self.ring.stride(0) * self.element_size
        return (
            {
                "name": "kv",
                "dtype": self.dtype_name,
                "shape": (self.ring_rows, width),
                "row_offset_bytes": 0,
                "row_stride_bytes": row_stride,
                "page_stride_bytes": stride,
            },
            {
                "name": "scores",
                "dtype": self.dtype_name,
                "shape": (self.ring_rows, width),
                "row_offset_bytes": width * self.element_size,
                "row_stride_bytes": row_stride,
                "page_stride_bytes": stride,
            },
        )


@dataclass(frozen=True)
class DSV41StepRequestView:
    """The scheduler step a model forward belongs to."""

    step_id: int
    request_ids: tuple[str, ...]
    generation: int
    synthetic: bool = False


@runtime_checkable
class DSV41BoundaryStateCodec(Protocol):
    """Engine-side C2 codec seam (implemented by the layerwise transport).

    The codec owns the wire descriptor, the lease state and the payload
    buffers. It must fail closed: when a request holds a lease but the
    capture is missing, stale or corrupt, the restore call raises instead
    of leaving a partially restored ring behind.
    """

    def capture_layer_boundary(
        self,
        request_view: DSV41BoundaryRequestView,
        tensor_view: DSV41BoundaryTensorView,
        *,
        stream: torch.Stream | None = None,
    ) -> None: ...

    def restore_layer_boundary(
        self,
        request_view: DSV41BoundaryRequestView,
        tensor_view: DSV41BoundaryTensorView,
        *,
        stream: torch.Stream | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _LayerRingState:
    metadata: Any
    request_view_base: tuple[str, str]
    tensor_view: DSV41BoundaryTensorView
    ratio: int
    num_reqs: int


#: Sentinel returned when a call did not bind a view, so callers can tell
#: "nothing to clear" apart from "the previous binding was None".
NOT_BOUND = object()

_codec: DSV41BoundaryStateCodec | None = None
_step_view: DSV41StepRequestView | None = None
_step_view_previous: list[DSV41StepRequestView | None] = []
_next_step_id = 0
_step_failures = 0
_validated_step_id: int | None = None


def boundary_state_codec() -> DSV41BoundaryStateCodec | None:
    return _codec


def boundary_state_codec_is_set() -> bool:
    return _codec is not None


def set_boundary_state_codec(codec: DSV41BoundaryStateCodec) -> None:
    """Install the S1 C2 codec for this process (connector startup only).

    Registering a codec switches the engine from S0 to S1 semantics: from
    then on every real forward must carry a bound step view and every ratio-2
    layer must reach the codec. A codec missing either callable is refused
    here rather than discovered mid-step.
    """
    if not isinstance(codec, DSV41BoundaryStateCodec):
        raise DSV41LayerwiseC2Error(
            "codec does not implement capture_layer_boundary/restore_layer_boundary: "
            f"{type(codec).__name__}"
        )
    global _codec, _step_view, _step_failures, _validated_step_id
    _codec = codec
    _step_view = None
    _step_failures = 0
    _validated_step_id = None
    logger.info(
        "%s: boundary state codec bound (%s): S1 C2 capture/restore is ACTIVE",
        DSV41_LAYERWISE_C2_TAG,
        type(codec).__name__,
    )


def clear_boundary_state_codec() -> None:
    global _codec, _step_view
    _codec = None
    _step_view = None


def note_step_failure(reason: str) -> None:
    """Record a step failure; the next TP gate stops every rank."""
    global _step_failures
    _step_failures += 1
    logger.error(
        "%s: layerwise step marked failed (%s); the next TP gate rejects the "
        "batch on every rank (fail-closed)",
        DSV41_LAYERWISE_C2_TAG,
        reason,
    )


def step_failure_count() -> int:
    return _step_failures


def bind_step_request_view(
    view: DSV41StepRequestView,
) -> DSV41StepRequestView | None | object:
    """Bind the step a forward belongs to; returns the previous binding."""
    global _step_view
    previous = _step_view
    _step_view_previous.append(previous)
    _step_view = view
    return previous


def clear_step_request_view(previous: DSV41StepRequestView | None | object) -> None:
    """Undo one ``bind_step_request_view``/``bind_model_forward_step`` call."""
    global _step_view
    if previous is NOT_BOUND:
        return
    if _step_view_previous:
        _step_view_previous.pop()
    _step_view = previous  # type: ignore[assignment]


def current_step_request_view() -> DSV41StepRequestView | None:
    return _step_view


def next_step_id() -> int:
    global _next_step_id
    _next_step_id += 1
    return _next_step_id


def step_id_counter() -> int:
    return _next_step_id


def reset_step_state() -> None:
    """Forget step bindings; used by tests and by connector teardown."""
    global _step_view, _next_step_id, _step_failures, _validated_step_id
    _step_view = None
    _step_view_previous.clear()
    _next_step_id = 0
    _step_failures = 0
    _validated_step_id = None


def _tp_group() -> Any:
    # Imported lazily: this module is imported by the attention op path, and
    # the distributed stack is only initialised inside a worker.
    from vllm.distributed.parallel_state import get_tp_group

    return get_tp_group()


def admit_step(device: torch.device | None = None) -> None:
    """TP uniform-step gate, run once per model forward when S1 is on.

    Each rank contributes ``(step_id + 1, failure_count)`` to one SUM
    all-reduce over the TP group. The reduced values are identical on every
    rank, so both decisions are uniform: either all ranks continue or all
    ranks raise before entering the next operator. This is the only
    collective added by S1 and it exists only while a codec is installed.
    """
    if _codec is None:
        return
    view = _step_view
    if view is None:
        raise DSV41LayerwiseC2Error(
            "layerwise C2 is active but this forward carries no step request view"
        )
    if view.synthetic:
        return
    group = _tp_group()
    world = int(getattr(group, "world_size", 1))
    if world <= 1:
        if _step_failures:
            raise DSV41LayerwiseC2Error(
                f"layerwise step {view.step_id} failed earlier in this process "
                f"({_step_failures} failure(s)); refusing to continue"
            )
        return
    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    packed = torch.tensor(
        [view.step_id + 1, _step_failures], dtype=torch.int64, device=device
    )
    _tp_all_reduce_sum(group, packed)
    step_sum, failure_sum = (int(v) for v in packed.tolist())
    if step_sum != world * (view.step_id + 1):
        raise DSV41LayerwiseC2Error(
            "TP ranks disagree on the layerwise step: expected "
            f"{world} x step {view.step_id + 1}, observed {step_sum}"
        )
    if failure_sum:
        raise DSV41LayerwiseC2Error(
            f"layerwise step {view.step_id + 1} was marked failed on "
            f"{failure_sum} TP rank(s); stopping every rank before the next "
            "collective"
        )


def _tp_all_reduce_sum(group: Any, tensor: torch.Tensor) -> None:
    reducer = getattr(group, "all_reduce", None)
    if callable(reducer):
        reducer(tensor, op=torch.distributed.ReduceOp.SUM)
        return
    device_group = getattr(group, "device_group", None)
    if device_group is None:
        raise DSV41LayerwiseC2Error(
            "cannot verify the TP step gate: the TP group exposes neither "
            "all_reduce nor device_group"
        )
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.SUM, group=device_group
    )


def bind_model_forward_step(
    runner: Any,
    *,
    decoder_replay_dummy: bool,
    device: torch.device | None = None,
) -> DSV41StepRequestView | None | object:
    """Bind one model forward to its scheduler step and run the TP gate.

    Called by ``NPUModelRunner._model_forward`` right before the model runs,
    which is the one point every TP rank reaches with the step's input batch
    already prepared. Returns the previous binding for ``clear_step_request_view``.
    """
    if _codec is None:
        return NOT_BOUND
    step_id = next_step_id()
    if init_phase_active() or decoder_replay_dummy:
        return bind_step_request_view(
            DSV41StepRequestView(
                step_id=step_id,
                request_ids=(),
                generation=_generation_of(runner),
                synthetic=True,
            )
        )
    input_batch = getattr(runner, "input_batch", None)
    req_ids = tuple(getattr(input_batch, "req_ids", ()) or ())
    view = DSV41StepRequestView(
        step_id=step_id,
        request_ids=req_ids,
        generation=_generation_of(runner),
    )
    previous = bind_step_request_view(view)
    try:
        admit_step(device)
    except Exception as exc:
        note_step_failure("TP step gate rejected the batch")
        raise DSV41LayerwiseC2Error(str(exc)) from exc
    return previous


def _generation_of(runner: Any) -> int:
    """The connector's registration generation, when the runner knows it."""
    connector = getattr(runner, "kv_connector", None)
    generation = getattr(connector, "registration_generation", None)
    if generation is None:
        generation = getattr(connector, "server_generation", None)
    return int(generation or 0)


def _require_step_view(layer_name: str) -> DSV41StepRequestView:
    view = _step_view
    if view is None:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} reached the S1 C2 hook without a bound step "
            "view; a real layerwise forward must be bound to its scheduler step"
        )
    return view


def _layer_ring_state(layer_name: str, attn: Any) -> _LayerRingState | None:
    """Resolve the real ring page set of one layer, or None when it has none."""
    impl = getattr(attn, "v41_impl", None)
    state_prefix = getattr(impl, "compressor_state_prefix", None)
    if state_prefix is None:
        return None
    compressor = getattr(attn, "compressor", None)
    state_cache = getattr(compressor, "state_cache", None)
    kv_cache = getattr(state_cache, "kv_cache", None)
    if not kv_cache:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} declares compressor state {state_prefix!r} but "
            "carries no state_cache kv_cache"
        )
    ring = kv_cache[0]
    if ring.dim() == 4:  # [pages, rows, 1, 2 * head]
        ring = ring.squeeze(-2)
    if ring.dim() != 3:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor state must be [pages, rows, 2*head], "
            f"got shape {tuple(ring.shape)}"
        )
    if ring.dtype != torch.float32:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor state must be float32, got {ring.dtype}"
        )
    if not ring.is_contiguous():
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor state page set is not contiguous"
        )
    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    attn_metadata = getattr(forward_context, "attn_metadata", None) or {}
    metadata = attn_metadata.get(state_prefix)
    if metadata is None:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} has no compressor metadata for {state_prefix!r} "
            "in this forward; the C2 boundary cannot be located"
        )
    ring_metadata = getattr(metadata, "c2_ring_metadata", None)
    if ring_metadata is None:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor metadata carries no c2_ring_metadata"
        )
    num_reqs = int(getattr(metadata, "num_reqs", 0) or 0)
    if num_reqs <= 0:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor metadata reports num_reqs={num_reqs}"
        )
    ratio = int(getattr(metadata, "compress_ratio", 0) or 0)
    if ratio <= 0:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor metadata reports compress_ratio={ratio}"
        )
    ring_rows = int(ring.shape[1])
    ring_head = int(ring.shape[2]) // 2
    if ring_rows != RING_ROWS_PER_PAGE:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} ring has {ring_rows} rows, expected "
            f"{RING_ROWS_PER_PAGE}"
        )
    block_table = getattr(metadata, "block_table", None)
    if block_table is None:
        raise DSV41LayerwiseC2Error(
            f"layer {layer_name} compressor metadata carries no block table; "
            "the physical ring page is unknown"
        )
    tensor_view = DSV41BoundaryTensorView(
        layer_name=str(getattr(impl, "layer_name", layer_name)),
        execution_layer=str(getattr(attn, "layerwise_execution_layer", layer_name)),
        ring=ring,
        page_ids=block_table[:num_reqs, 0],
        ring_rows=ring_rows,
        ring_head_size=ring_head,
    )
    return _LayerRingState(
        metadata=metadata,
        request_view_base=(tensor_view.layer_name, tensor_view.execution_layer),
        tensor_view=tensor_view,
        ratio=ratio,
        num_reqs=num_reqs,
    )


def _validate_bound_requests(step_view: DSV41StepRequestView, state: _LayerRingState) -> None:
    if len(step_view.request_ids) != state.num_reqs:
        raise DSV41LayerwiseC2Error(
            "layerwise step view and attention metadata disagree on the batch: "
            f"{len(step_view.request_ids)} bound request(s) vs "
            f"{state.num_reqs} metadata row(s)"
        )


def _request_view(
    state: _LayerRingState,
    step_view: DSV41StepRequestView,
    request_index: int,
) -> DSV41BoundaryRequestView:
    ring_metadata = state.metadata.c2_ring_metadata
    control = tuple(int(v) for v in ring_metadata[:, request_index].tolist())
    prefix_end, used, _out_base, _in_base, page = control
    slot_mapping = getattr(state.metadata, "slot_mapping", None)
    if slot_mapping is None:
        raise DSV41LayerwiseC2Error(
            f"layer {state.tensor_view.layer_name} has no state slot mapping; "
            "boundary rows cannot be proven to have been written"
        )
    rows = slot_mapping[(slot_mapping // state.tensor_view.ring_rows) == page]
    if rows.numel() == 0 and used > 0:
        raise DSV41LayerwiseC2Error(
            f"layer {state.tensor_view.layer_name} wrote {used} token(s) for "
            f"request {step_view.request_ids[request_index]!r} but no state row "
            f"belongs to page {page}"
        )
    ring_rows = state.tensor_view.ring_rows
    return DSV41BoundaryRequestView(
        layer_name=state.tensor_view.layer_name,
        execution_layer=state.tensor_view.execution_layer,
        request_id=step_view.request_ids[request_index],
        request_index=request_index,
        step_id=step_view.step_id,
        generation=step_view.generation,
        prefix_end=int(prefix_end),
        step_tokens=int(used),
        ring_rows=ring_rows,
        ring_head_size=state.tensor_view.ring_head_size,
        ring_length=ring_rows,
        ring_page=int(page),
        next_slot=int(prefix_end) % ring_rows,
        compression_ratio=state.ratio,
        compression_phase=int(prefix_end) % state.ratio,
        position_control=control,
        written_rows=tuple(int(v) for v in rows.tolist()),
    )


def _run_hook(
    layer_name: str,
    attn: Any,
    codec_call: str,
) -> None:
    if _codec is None:
        return
    if init_phase_active():
        # Capture / profile forwards carry no scheduler step; dsa_v41_forward
        # already bypassed admission for them. Never manufacture a boundary.
        return
    step_view = _require_step_view(layer_name)
    if step_view.synthetic:
        # Decoder-replay dummies replay a captured plan: they carry no
        # scheduler step and must not publish or consume a boundary.
        return
    state = _layer_ring_state(layer_name, attn)
    if state is None:
        return
    _validate_bound_requests(step_view, state)
    codec = _codec
    for request_index in range(state.num_reqs):
        request_view = _request_view(state, step_view, request_index)
        try:
            getattr(codec, codec_call)(
                request_view, state.tensor_view, stream=None
            )
        except DSV41LayerwiseC2Error:
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed, never skip bytes
            note_step_failure(f"{codec_call} failed for {request_view.request_id}")
            raise DSV41LayerwiseC2Error(
                f"{codec_call} failed for layer {request_view.layer_name} "
                f"request {request_view.request_id!r}: {exc}"
            ) from exc


def restore_layer_boundary_state(layer_name: str, attn: Any) -> None:
    """Restore this layer's C2 ring page before its first read (S1)."""
    _run_hook(layer_name, attn, "restore_layer_boundary")


def capture_layer_boundary_state(layer_name: str, attn: Any) -> None:
    """Hand this layer's freshly written ring page to the codec (S1)."""
    _run_hook(layer_name, attn, "capture_layer_boundary")


__all__ = [
    "DSV41BoundaryRequestView",
    "DSV41BoundaryStateCodec",
    "DSV41BoundaryTensorView",
    "DSV41LayerwiseC2Error",
    "DSV41StepRequestView",
    "admit_step",
    "bind_model_forward_step",
    "bind_step_request_view",
    "boundary_state_codec",
    "boundary_state_codec_is_set",
    "capture_layer_boundary_state",
    "clear_boundary_state_codec",
    "clear_step_request_view",
    "current_step_request_view",
    "NOT_BOUND",
    "next_step_id",
    "note_step_failure",
    "reset_step_state",
    "restore_layer_boundary_state",
    "set_boundary_state_codec",
    "step_failure_count",
    "step_id_counter",
]
