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

#: Suffix of the registered cache that a ratio-2 compressor ring state lives
#: in. The transport addresses those pages by absolute row while every other
#: cache is addressed by block, so the two must never be confused.
BOUNDARY_STATE_CACHE_SUFFIX = ".compressor.state_cache"


class DSV41LayerwiseC2Error(RuntimeError):
    """The S1 layerwise C2 contract was violated (fail-closed)."""


@dataclass(frozen=True)
class DSV41BoundaryRequestView:
    """One request's real forward facts for a single layer boundary."""

    layer_name: str
    execution_layer: str
    resource_id: str
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
    resource_id: str
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
    # ``(prefix_end, first_block)`` per request, in ``request_ids`` order. The
    # scheduler owns these pages, and the transport loads a boundary after the
    # forward (async KV load), when the step's ring metadata is already gone;
    # this plan keeps the page each request owns addressable for that restore.
    request_pages: tuple[tuple[int, int], ...] = ()


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

@dataclass(frozen=True)
class _BoundaryRecord:
    """Rows one layer wrote for one request, as handed to the transport."""

    resource_id: str
    request_id: str
    step_id: int
    prefix_end: int
    end_token: int
    ring_cursor: int
    compression_phase: int
    row_indices: tuple[int, ...]


_codec: DSV41BoundaryStateCodec | None = None
_provider_installed = False
# Captures keyed by ``(step_id, request_id, resource_id)``: a step may carry
# several requests, each owning its own ring page, so one request's capture
# must never overwrite another's. The transport identifies a boundary by the
# token it ends at; request ids are the stronger key when a caller has one.
_boundary_records: dict[tuple[int, str, str], _BoundaryRecord] = {}
# Bound so a long serving run cannot grow the capture map without limit.
# Only recent steps matter: a snapshot is stored right after its forward.
_BOUNDARY_RECORD_LIMIT = 4096
# Rows per page of every ratio-2 compressor ring, bound once from the model
# at install time. The transport frames whole pages by this geometry, so it
# must come from the real cache and never from a constant guess.
_ring_geometry: dict[str, int] = {}
_step_view: DSV41StepRequestView | None = None
_step_view_previous: list[DSV41StepRequestView | None] = []
_next_step_id = 0
_step_failures = 0
_validated_step_id: int | None = None
# ``request_id -> (prefix_end, first_block)`` of the last bound real step. The
# transport restores boundaries after the forward, so the live ring metadata of
# that step is gone by then; the plan still names the page each request owns.
_request_pages: dict[str, tuple[int, int]] = {}
# Installs bump the activation generation. A forward that was bound before the
# install cannot carry the step view the freshly activated path requires, so it
# degrades to a cold step (once, with a warning) instead of failing every rank.
_activation_generation = 0
_bound_generation = -1
# How many forwards were bound so far. A cold transition can only be claimed
# when a forward really was bound before the install; an armed process that
# never bound a step must still fail closed.
_bind_count = 0
_logged_transitions: set[str] = set()


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
    global _activation_generation
    _codec = codec
    _step_view = None
    _step_failures = 0
    _validated_step_id = None
    _activation_generation += 1
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
    global _step_view, _bound_generation, _bind_count
    previous = _step_view
    _step_view_previous.append(previous)
    _step_view = view
    # A binding always knows which activation it was made under, so a step that
    # started before an install is recognisable as a cold transition.
    _bound_generation = _activation_generation
    _bind_count += 1
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
    global _bound_generation, _bind_count
    _step_view = None
    _step_view_previous.clear()
    _next_step_id = 0
    _step_failures = 0
    _validated_step_id = None
    _boundary_records.clear()
    _request_pages.clear()
    _logged_transitions.clear()
    # Reset with the active generation so a test that installs a codec before
    # its first forward does not look like a mid-flight transition.
    _bound_generation = _activation_generation
    _bind_count = 0


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

    Binding is armed by either half of S1 (a codec or the boundary provider):
    the transport installs its provider at ``register_kv_caches`` time while a
    codec may arrive later, so gating on the codec alone left the first real
    forward unbound and failed every worker.
    """
    if not s1_capture_active():
        _note_unbound_step()
        return NOT_BOUND
    step_id = next_step_id()
    if init_phase_active() or decoder_replay_dummy:
        _request_pages.clear()
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
    plan = _request_page_plan(runner, req_ids)
    _request_pages.clear()
    _request_pages.update({request_id: (end, page) for request_id, end, page in plan})
    view = DSV41StepRequestView(
        step_id=step_id,
        request_ids=req_ids,
        generation=_generation_of(runner),
        request_pages=tuple((end, page) for _rid, end, page in plan),
    )
    previous = bind_step_request_view(view)
    try:
        admit_step(device)
    except Exception as exc:
        note_step_failure("TP step gate rejected the batch")
        raise DSV41LayerwiseC2Error(str(exc)) from exc
    return previous


def _note_unbound_step() -> None:
    """Record that this forward ran before S1 was armed, for the transition rule."""
    global _bound_generation, _bind_count
    _bound_generation = _activation_generation
    _bind_count += 1


def _request_page_plan(
    runner: Any, req_ids: tuple[str, ...]
) -> tuple[tuple[str, int, int], ...]:
    """``(request_id, prefix_end, first_block)`` for the step's requests.

    ``prefix_end`` is the count of tokens computed before this step and the
    first block is the request's ring page (the compressor kernel addresses a
    request's ring through ``block_table[b, 0]``). Both live in the runner's
    input batch at binding time; reading them here keeps the page addressable
    after the forward, when the transport actually loads the boundary.
    """
    if not req_ids:
        return ()
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return ()
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    block_tables = getattr(input_batch, "block_table", None)
    tables = getattr(block_tables, "block_tables", None)
    first_group = tables[0] if tables else None
    first_blocks = None
    use_cpu_tensor = getattr(first_group, "get_cpu_tensor", None)
    if callable(use_cpu_tensor):
        first_blocks = use_cpu_tensor()
    page_rows = int(getattr(first_blocks, "shape", (0, 0))[0]) if first_blocks is not None else 0
    page_cols = int(getattr(first_blocks, "shape", (0, 0))[1]) if first_blocks is not None else 0
    plan: list[tuple[str, int, int]] = []
    for index, request_id in enumerate(req_ids):
        prefix_end = 0
        if computed is not None and index < len(computed):
            prefix_end = int(computed[index])
        page = 0
        if first_blocks is not None and index < page_rows and page_cols > 0:
            # The block table lives in host memory, so this read is not a
            # device synchronisation on the forward path.
            page = int(first_blocks[index, 0].item())
        plan.append((str(request_id), prefix_end, page))
    return tuple(plan)


def _generation_of(runner: Any) -> int:
    """The connector's registration generation, when the runner knows it."""
    connector = getattr(runner, "kv_connector", None)
    generation = getattr(connector, "registration_generation", None)
    if generation is None:
        generation = getattr(connector, "server_generation", None)
    return int(generation or 0)


def _require_step_view(layer_name: str) -> DSV41StepRequestView | None:
    view = _step_view
    if view is not None:
        return view
    if _bind_count and _activation_generation != _bound_generation:
        # S1 was armed after this forward was bound (the connector installs its
        # provider from its own startup path, which can race the first real
        # forward). This step cannot carry rows; serve it cold once instead of
        # failing every worker, and say so.
        if "transition" not in _logged_transitions:
            _logged_transitions.add("transition")
            logger.warning(
                "%s: layer %s ran before S1 was armed (installed after this "
                "step was bound); serving this step cold and capturing from "
                "the next one",
                DSV41_LAYERWISE_C2_TAG,
                layer_name,
            )
        return None
    raise DSV41LayerwiseC2Error(
        f"layer {layer_name} reached the S1 C2 hook without a bound step "
        "view; a real layerwise forward must be bound to its scheduler step"
    )


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
        resource_id=str(state_prefix),
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
    ring_rows = state.tensor_view.ring_rows
    if getattr(state.metadata, "is_compressor_state", False):
        # A compressor-state buffer is addressed by its ring alone: its token
        # slot buffer intentionally stays PAD (see ``_build_metadata``), so the
        # rows this step wrote are exactly the ring slots the update kernel
        # computes: the kernel keeps the *last* ``min(used, ring_rows)`` tokens
        # of the step (``token_in_seg = used - tail_count + write_idx``) and
        # writes ``slot = (prefix_end + token_in_seg) % ring_rows`` on the
        # request's own page (``c2_ring_metadata`` row 4).
        tail_count = min(max(int(used), 0), ring_rows)
        first_token = int(prefix_end) + max(int(used), 0) - tail_count
        rows = tuple(
            int(page) * ring_rows + (first_token + index) % ring_rows
            for index in range(tail_count)
        )
    else:
        slot_mapping = getattr(state.metadata, "slot_mapping", None)
        if slot_mapping is None:
            raise DSV41LayerwiseC2Error(
                f"layer {state.tensor_view.layer_name} has no state slot mapping; "
                "boundary rows cannot be proven to have been written"
            )
        matched = slot_mapping[(slot_mapping // ring_rows) == page]
        rows = tuple(int(v) for v in matched.tolist())
        if not rows and used > 0:
            raise DSV41LayerwiseC2Error(
                f"layer {state.tensor_view.layer_name} wrote {used} token(s) for "
                f"request {step_view.request_ids[request_index]!r} but no state row "
                f"belongs to page {page}"
            )
    return DSV41BoundaryRequestView(
        layer_name=state.tensor_view.layer_name,
        execution_layer=state.tensor_view.execution_layer,
        resource_id=state.tensor_view.resource_id,
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
        written_rows=rows,
    )


def s1_capture_active() -> bool:
    """True when this process must keep real boundary rows for S1."""
    return _codec is not None or _provider_installed


def _record_boundary(view: DSV41BoundaryRequestView) -> None:
    record = _BoundaryRecord(
        resource_id=view.resource_id,
        request_id=view.request_id,
        step_id=view.step_id,
        prefix_end=view.prefix_end,
        end_token=view.prefix_end + view.step_tokens,
        ring_cursor=view.prefix_end % view.ring_rows,
        compression_phase=view.compression_phase,
        row_indices=view.written_rows,
    )
    _boundary_records[(view.step_id, view.request_id, view.resource_id)] = record
    while len(_boundary_records) > _BOUNDARY_RECORD_LIMIT:
        oldest = next(iter(_boundary_records))
        del _boundary_records[oldest]


def boundary_record(
    resource_id: str,
    request_id: str | None = None,
    step_id: int | None = None,
) -> _BoundaryRecord | None:
    """Newest capture of ``resource_id``, optionally for one request/step."""
    newest: _BoundaryRecord | None = None
    for record in _boundary_records.values():
        if record.resource_id != resource_id:
            continue
        if request_id is not None and record.request_id != request_id:
            continue
        if step_id is not None and record.step_id != step_id:
            continue
        newest = record
    return newest


def _matching_records(
    resource_id: str,
    boundary_token: int,
    request_id: str | None,
) -> list[_BoundaryRecord]:
    matched: list[_BoundaryRecord] = []
    for record in _boundary_records.values():
        if record.resource_id != resource_id:
            continue
        if request_id is not None and record.request_id != request_id:
            continue
        if record.end_token == boundary_token:
            matched.append(record)
    return matched


def provide_boundary_rows(
    resource_id: str,
    snapshot_id: str,
    boundary_token: Any,
    request_id: str | None = None,
) -> tuple[int, int, tuple[int, ...]] | None:
    """Hand the transport the real boundary rows of one request (S1).

    Installed as the layerwise worker adapter's boundary provider. Returns
    ``(ring_cursor, compression_phase, row_indices)`` for the capture that
    ends exactly at ``boundary_token``; anything else returns ``None`` so
    the snapshot stays cold-only for that request. Rows are never guessed:
    they are the state slot-map rows this forward actually wrote.

    A step may carry several requests, each with its own capture. The
    boundary token normally separates them; when two requests of the same
    step end at the same token their captures can still disagree on the
    physical page, and that ambiguity is refused (cold-only) rather than
    resolved by guessing. Passing ``request_id`` names the request outright
    and removes the ambiguity.
    """
    token = _token_int(boundary_token)
    if token is None:
        logger.warning(
            "%s: boundary token %r is not an integer; leaving %s cold-only",
            DSV41_LAYERWISE_C2_TAG,
            boundary_token,
            resource_id,
        )
        return None
    matched = _matching_records(str(resource_id), token, request_id)
    if not matched:
        known = [
            record
            for record in _boundary_records.values()
            if record.resource_id == str(resource_id)
        ]
        if known:
            logger.warning(
                "%s: boundary capture for %s ends at token %s but the "
                "transport asked for %s%s; leaving it cold-only instead of "
                "fabricating rows",
                DSV41_LAYERWISE_C2_TAG,
                resource_id,
                known[-1].end_token,
                boundary_token,
                f" (request {request_id!r})" if request_id is not None else "",
            )
        else:
            logger.warning(
                "%s: no boundary capture for resource %s ending at token %d; "
                "leaving it cold-only",
                DSV41_LAYERWISE_C2_TAG,
                resource_id,
                token,
            )
        return None
    distinct = {
        (record.ring_cursor, record.compression_phase, record.row_indices)
        for record in matched
    }
    if len(distinct) > 1:
        logger.warning(
            "%s: %d captures of %s end at token %d with different pages/rows "
            "and no request id names one; leaving it cold-only instead of "
            "guessing a page",
            DSV41_LAYERWISE_C2_TAG,
            len(matched),
            resource_id,
            token,
        )
        return None
    record = matched[-1]
    if not record.row_indices:
        logger.warning(
            "%s: boundary capture for %s wrote no rows this step; cold-only",
            DSV41_LAYERWISE_C2_TAG,
            resource_id,
        )
        return None
    return record.ring_cursor, record.compression_phase, record.row_indices


def _token_int(boundary_token: Any) -> int | None:
    try:
        return int(str(boundary_token))
    except (TypeError, ValueError):
        return None


def boundary_state_cache_name(name: Any) -> bool:
    """True when a registered cache holds a ratio-2 compressor ring state."""
    return isinstance(name, str) and name.endswith(BOUNDARY_STATE_CACHE_SUFFIX)


def bind_ring_geometry(model: Any) -> dict[str, int]:
    """Record the real rows-per-page of every compressor ring the model owns.

    The transport frames a whole ring page, so it needs the page geometry of
    the actual cache. Taking it from the model once (instead of a constant or
    a guess) keeps stores and restores on the same row space the kernel uses,
    including any layer whose ring size ever differs from the default.
    """
    geometry: dict[str, int] = {}
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return geometry
    for module in modules():
        impl = getattr(module, "v41_impl", None)
        prefix = getattr(impl, "compressor_state_prefix", None)
        if not isinstance(prefix, str) or not prefix:
            continue
        compressor = getattr(module, "compressor", None)
        state_cache = getattr(compressor, "state_cache", None)
        kv_cache = getattr(state_cache, "kv_cache", None)
        if not kv_cache:
            continue
        ring = kv_cache[0]
        if not isinstance(ring, torch.Tensor) or ring.dim() < 2:
            continue
        rows = int(ring.shape[1])
        if rows <= 0:
            continue
        geometry[prefix] = rows
    _ring_geometry.update(geometry)
    return geometry


def _ring_rows_known(resource_id: Any) -> int | None:
    rows = _ring_geometry.get(str(resource_id))
    if rows is None:
        return None
    return int(rows) if int(rows) > 0 else None


def ring_rows(resource_id: str) -> int:
    """Rows per ring page of ``resource_id``, from the real cache geometry.

    Installed as an attribute of :func:`provide_boundary_rows`; the transport
    calls it before it frames a whole page. A resource whose geometry was
    never observed raises instead of guessing 32 rows.
    """
    rows = _ring_rows_known(resource_id)
    if rows is None:
        raise DSV41LayerwiseC2Error(
            f"no ring geometry is known for {resource_id!r}; refusing to frame "
            "a boundary page whose row space was never observed"
        )
    return rows


def _live_ring_metadata(resource_id: str) -> torch.Tensor | None:
    """The live ``c2_ring_metadata`` of one boundary resource, if present."""
    from vllm.forward_context import get_forward_context

    try:
        forward_context = get_forward_context()
    except Exception:  # pragma: no cover - no context outside a forward
        return None
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        return None
    metadata = attn_metadata.get(str(resource_id))
    ring_metadata = getattr(metadata, "c2_ring_metadata", None)
    if not isinstance(ring_metadata, torch.Tensor) or ring_metadata.dim() != 2:
        return None
    if int(ring_metadata.shape[0]) < POSITION_CONTROL_WORDS:
        return None
    if int(ring_metadata.shape[1]) < 1:
        return None
    return ring_metadata


_NO_PLAN = object()


def _planned_target(
    resource_id: str, token: int, request_id: str | None
) -> tuple[int, int] | None | object:
    """Resolve a target page from the bound step's own page plan.

    The transport loads a boundary after the forward that produced the step,
    when the live ring metadata is gone but the pages the scheduler allocated
    are still exactly the ones the plan recorded. ``_NO_PLAN`` means this
    step's plan cannot answer (the caller may still try live metadata);
    ``None`` means the plan knows the request but refuses the page.
    """
    if request_id is not None:
        plan = _request_pages.get(str(request_id))
        if plan is None:
            return _NO_PLAN
        return _planned_page(resource_id, token, str(request_id), plan[0], plan[1])
    if not _request_pages:
        return _NO_PLAN
    matches = [
        (rid, end, page)
        for rid, (end, page) in _request_pages.items()
        if end == token
    ]
    pages = {page for _rid, _end, page in matches}
    if not pages:
        return _NO_PLAN
    if len(pages) > 1:
        logger.warning(
            "%s: %d requests resume at token %d with different planned pages "
            "and no request id names one; staying fail-closed for this "
            "boundary instead of guessing a page",
            DSV41_LAYERWISE_C2_TAG,
            len(matches),
            token,
        )
        return None
    rid, end, page = matches[0]
    return _planned_page(resource_id, token, rid, end, page)


def _planned_page(
    resource_id: str, token: int, request_id: str, prefix_end: int, page: int
) -> tuple[int, int] | None:
    if prefix_end != token:
        logger.warning(
            "%s: request %r resumes at token %d but the snapshot ends at %d; "
            "the planned page does not apply, staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            request_id,
            prefix_end,
            token,
        )
        return None
    if page <= 0:
        logger.warning(
            "%s: request %r has no ring page allocated for the step ending at "
            "token %d; staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            request_id,
            token,
        )
        return None
    rows = _ring_rows_known(resource_id)
    if rows is None:
        logger.warning(
            "%s: no ring geometry is known for %s; staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            resource_id,
        )
        return None
    return page, rows


def restore_target(
    resource_id: str,
    boundary_token: Any,
    request_id: str | None = None,
) -> tuple[int, int] | None:
    """Name the restoring request's own ring page for a whole-page restore (S1).

    Installed as an attribute of :func:`provide_boundary_rows`; the transport
    calls it before it rebases a captured page onto the live request. Every
    request carries its own control row in the live ``c2_ring_metadata`` and
    its own ``block_table[:, 0]`` page, so the target page is resolved per
    request rather than by requiring a one-request step:

    * with ``request_id`` the request is named outright and must resume
      exactly at ``boundary_token``;
    * without it, exactly one request of the step may resume at that token.
      Two requests resuming at the same token with different pages are
      ambiguous, and ambiguity is refused for that request only — the other
      requests of the batch keep their own restores.

    Anything unresolved returns ``None`` so the transport stays fail-closed
    instead of writing a page nobody reads.
    """
    if not _provider_installed:
        return None
    token = _token_int(boundary_token)
    if token is None:
        return None
    planned = _planned_target(resource_id, token, request_id)
    if planned is not _NO_PLAN:
        return planned
    view = _step_view
    if view is None:
        logger.warning(
            "%s: restore target asked for %s before any step view was bound; "
            "staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            resource_id,
        )
        return None
    if view.synthetic:
        return None
    ring_metadata = _live_ring_metadata(resource_id)
    if ring_metadata is None:
        logger.warning(
            "%s: no live ring metadata for %s; the target page is unknown, "
            "staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            resource_id,
        )
        return None
    rows_in_step = min(int(ring_metadata.shape[1]), len(view.request_ids))
    if rows_in_step <= 0:
        return None
    if request_id is not None:
        try:
            index = view.request_ids.index(request_id)
        except ValueError:
            logger.warning(
                "%s: restore target asked for %s of request %r, which is not "
                "in this step; staying fail-closed",
                DSV41_LAYERWISE_C2_TAG,
                resource_id,
                request_id,
            )
            return None
        if index >= rows_in_step:
            return None
        candidates = [index]
    else:
        candidates = [
            index
            for index in range(rows_in_step)
            if int(ring_metadata[0, index]) == token
        ]
        if not candidates:
            logger.warning(
                "%s: %s resumes at no request of this step at token %d; the "
                "boundary does not apply, staying fail-closed",
                DSV41_LAYERWISE_C2_TAG,
                resource_id,
                token,
            )
            return None
    # A request that carries no token of its own in this step leaves its
    # control row unset (the kernel keeps the page only when ``used > 0``),
    # so its row cannot name a page even though its prefix end matches.
    silent = [
        index for index in candidates if int(ring_metadata[1, index]) <= 0
    ]
    if silent:
        logger.warning(
            "%s: request(s) %s resume at token %d without a token of their "
            "own this step, so the kernel left their page unset; staying "
            "fail-closed for this boundary",
            DSV41_LAYERWISE_C2_TAG,
            [view.request_ids[index] for index in silent],
            token,
        )
        candidates = [index for index in candidates if index not in silent]
        if not candidates:
            return None
    pages = {int(ring_metadata[4, index]) for index in candidates}
    if len(pages) > 1:
        logger.warning(
            "%s: %d requests of this step resume at token %d with different "
            "pages and no request id names one; staying fail-closed for this "
            "boundary instead of guessing a page",
            DSV41_LAYERWISE_C2_TAG,
            len(candidates),
            token,
        )
        return None
    for index in candidates:
        prefix_end = int(ring_metadata[0, index])
        if prefix_end != token:
            logger.warning(
                "%s: request %r resumes at token %d but the snapshot ends at "
                "%d; the boundary does not apply, staying fail-closed",
                DSV41_LAYERWISE_C2_TAG,
                view.request_ids[index],
                prefix_end,
                token,
            )
            return None
    rows = _ring_rows_known(resource_id)
    if rows is None:
        logger.warning(
            "%s: no ring geometry for %s; cannot frame the restore page, "
            "staying fail-closed",
            DSV41_LAYERWISE_C2_TAG,
            resource_id,
        )
        return None
    return int(ring_metadata[4, candidates[0]]), rows


def row_space_view(value: Any) -> torch.Tensor | None:
    """Absolute-row view of a compressor ring page set, or None.

    The kernel addresses ring slots as ``page * rows + position % rows`` and
    the transport does the same, so both need one flat row axis. Only a
    contiguous ``[pages, rows, 1, 2 * head]`` (or already 3-D) page set can
    be viewed that way without moving bytes; anything else is refused.
    """
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 2:
        return value
    if value.dim() not in (3, 4):
        return None
    if value.dim() == 4 and int(value.shape[2]) != 1:
        return None
    if not value.is_contiguous():
        return None
    return value.view(-1, value.shape[-1])


def use_row_space_views(worker: Any) -> int:
    """Hand the transport absolute-row views of the boundary caches (S1).

    KV registration already hashed the real ``[pages, rows, 1, 2 * head]``
    geometry, and the engine still reads those caches by page. The transport
    indexes boundary payloads by absolute row, so after registration the
    adapter's own cache map is pointed at a view of the *same* storage. Only
    boundary-state caches are touched; a cache that cannot be flattened is
    left alone with a warning and the transport stays fail-closed for it.
    """
    caches = getattr(worker, "kv_caches", None)
    if not isinstance(caches, dict):
        return 0
    swapped = 0
    for name, value in list(caches.items()):
        if not boundary_state_cache_name(name):
            continue
        wrapped = isinstance(value, tuple) and len(value) == 1
        tensor = value[0] if wrapped else value
        view = row_space_view(tensor)
        if view is None:
            logger.warning(
                "%s: boundary cache %s cannot be viewed as absolute rows; the "
                "transport stays fail-closed for it",
                DSV41_LAYERWISE_C2_TAG,
                name,
            )
            continue
        if view is tensor:
            continue
        caches[name] = (view,) if wrapped else view
        swapped += 1
    return swapped


def install_boundary_provider(connector: Any = None, model: Any = None) -> bool:
    """Install the engine boundary-row provider on the layerwise adapter.

    Called once per worker after KV registration. Connectors that are not
    the DSV4.1 layerwise transport simply do not expose
    ``worker_adapter.set_boundary_provider``, so this is a no-op for every
    other KV connector (stock serving included).

    The provider also carries the two geometry hooks the transport asks for
    (``ring_rows`` and ``restore_target``). When the model is given, its real
    ring geometry is bound and the adapter's boundary caches are pointed at
    absolute-row views of the same storage.
    """
    global _provider_installed, _activation_generation
    if connector is None:
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return False
        connector = get_kv_transfer_group()
    worker = getattr(connector, "worker_adapter", None)
    setter = getattr(worker, "set_boundary_provider", None)
    if not callable(setter):
        return False
    setter(provide_boundary_rows)
    setattr(provide_boundary_rows, "ring_rows", ring_rows)
    setattr(provide_boundary_rows, "restore_target", restore_target)
    if model is not None:
        bind_ring_geometry(model)
    swapped = use_row_space_views(worker)
    _provider_installed = True
    _activation_generation += 1
    logger.info(
        "%s: boundary provider installed on %s: S1 snapshots take the rows "
        "this forward really wrote (%d boundary cache(s) on an absolute-row "
        "view, %d ring geometry record(s))",
        DSV41_LAYERWISE_C2_TAG,
        type(worker).__name__,
        swapped,
        len(_ring_geometry),
    )
    return True


def _run_hook(
    layer_name: str,
    attn: Any,
    codec_call: str,
) -> None:
    if codec_call == "capture_layer_boundary":
        # Capture also feeds the transport's boundary provider, so it runs
        # whenever either the push codec or the provider is installed.
        if not s1_capture_active():
            return
    elif _codec is None:
        return
    if init_phase_active():
        # Capture / profile forwards carry no scheduler step; dsa_v41_forward
        # already bypassed admission for them. Never manufacture a boundary.
        return
    step_view = _require_step_view(layer_name)
    if step_view is None:
        # This forward was bound before S1 was armed: serve the step cold.
        return
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
        if codec_call == "capture_layer_boundary":
            # The provider always sees this step's rows, whether or not a
            # push-mode codec is installed.
            _record_boundary(request_view)
        if codec is None:
            continue
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
    """Record this layer's freshly written rows for S1 (codec and/or provider)."""
    if not s1_capture_active():
        return
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
    "NOT_BOUND",
    "BOUNDARY_STATE_CACHE_SUFFIX",
    "bind_ring_geometry",
    "boundary_record",
    "boundary_state_cache_name",
    "boundary_state_codec",
    "boundary_state_codec_is_set",
    "capture_layer_boundary_state",
    "clear_boundary_state_codec",
    "clear_step_request_view",
    "current_step_request_view",
    "install_boundary_provider",
    "next_step_id",
    "note_step_failure",
    "provide_boundary_rows",
    "reset_step_state",
    "restore_target",
    "ring_rows",
    "restore_layer_boundary_state",
    "s1_capture_active",
    "set_boundary_state_codec",
    "row_space_view",
    "step_failure_count",
    "step_id_counter",
    "use_row_space_views",
]
