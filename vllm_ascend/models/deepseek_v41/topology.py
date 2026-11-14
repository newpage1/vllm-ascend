# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent DeepSeek V4.1 source and cache-consumer topology."""

from dataclasses import dataclass
from typing import Any, Literal

#: KV source layers of the fixed first-phase layerwise v2 topology. Each one
#: owns a transfer group; the engine only waits/saves at these layers.
DSV41_LW_V2_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14, 20)

#: Sources that carry a compressor boundary: every source whose compression
#: ratio is 2. The final source (20) is the ratio-1 candidate segment tail and
#: owns token-range state only.
DSV41_LW_V2_BOUNDARY_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14)

#: Backbone depth the v2 group table is defined for.
DSV41_LW_V2_BACKBONE_LAYERS = 40

#: Compression ratio whose layers own a compressor boundary (ring) state.
DSV41_BOUNDARY_COMPRESS_RATIO = 2

#: KV consumer segment per group (inclusive). Kept explicit so a topology
#: change cannot silently move a layer between groups.
DSV41_LW_V2_CONSUMER_SEGMENTS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (2, (2, 3, 4, 5, 6, 7)),
    (8, (8, 9, 10, 11, 12, 13)),
    (14, (14, 15, 16, 17, 18, 19)),
    (
        20,
        (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39),
    ),
)


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Immutable source/consumer topology of the backbone."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)

    def cache_resources(
        self, *, prefix: str = "model", include_folded_index: bool = False
    ) -> tuple["DeepseekV41CacheResource", ...]:
        """Return cache owners and consumers for MP registration planning."""
        return build_cache_resources(self, prefix=prefix, include_folded_index=include_folded_index)


@dataclass(frozen=True)
class DeepseekV41CacheResource:
    """One registered cache owner, distinct from computed top-k/candidates."""

    cache_name: str
    cache_kind: Literal["swa", "long_kv", "index_k", "folded_index_k", "compressor_state"]
    producer_layer: int
    consumer_layers: tuple[int, ...]


@dataclass(frozen=True)
class DeepseekV41TransferGroup:
    """One KV-source transfer group of the layerwise v2 protocol.

    The group head is the source layer: execution is sequential, so a single
    wait at the head covers every consumer layer of the group. The head is
    also the only layer that may save the group's freshly produced state.

    ``transfer_resources`` holds the token-range resources only (long KV and
    index K, each keeping its original registered descriptor). The compressor
    boundary is tracked separately because it is not a token-range plane and
    is absent for the final source. SWA caches never enter a transfer group;
    the engine recomputes them inside the bounded replay window.
    """

    group_id: str
    source_layer: int
    attention_prefix: str
    readiness_layer: str
    dispatch_layer: str
    consumer_layers: tuple[str, ...]
    transfer_resources: tuple[DeepseekV41CacheResource, ...]
    boundary_resource: DeepseekV41CacheResource | None = None

    @property
    def token_range_resources(self) -> tuple[DeepseekV41CacheResource, ...]:
        return self.transfer_resources

    @property
    def producer_layer(self) -> int:
        """The head layer that produces the group's state."""
        return self.source_layer

    @property
    def has_boundary(self) -> bool:
        return self.boundary_resource is not None

    @property
    def transfer_resource_count(self) -> int:
        return len(self.transfer_resources) + (0 if self.boundary_resource is None else 1)


@dataclass(frozen=True)
class DSV41LayerBinding:
    """Explicit identity binding for one DSV4.1 attention object.

    The three names come from the live ``DeepseekV41Attention`` object at
    binding time; none is derived from another by string rewriting:

    - ``attention_prefix``: the module's construction prefix
      (``model.layers.<i>.self_attn``);
    - ``dispatch_layer_name``: the custom-op dispatch key registered in the
      static forward context (``<attention_prefix>.v41_attn``);
    - ``execution_layer_name``: the layerwise plan's execution-layer ID
      (``<attention_prefix>.attn``).

    The binding builder validates these equalities on the object's own
    attributes and rejects any mismatch, instead of normalizing names.
    """

    layer_idx: int
    attention_prefix: str
    dispatch_layer_name: str
    execution_layer_name: str


def _attention_cache_prefix(prefix: str, layer_idx: int) -> str:
    """Return the registered attention prefix of one backbone layer."""
    root = f"{prefix}." if prefix else ""
    return f"{root}layers.{layer_idx}.self_attn"


def build_cache_resources(
    topology: DeepseekV41Topology,
    *,
    prefix: str = "model",
    include_folded_index: bool = False,
) -> tuple[DeepseekV41CacheResource, ...]:
    """Describe cache lifetimes without allocating tensors or loading operators.

    Args:
        topology: Backbone topology returned by ``build_layer_plan``.
        prefix: Model root before ``layers.<index>.self_attn``.
        include_folded_index: Include the packed candidate source's folded K
            cache when that cache exists on the selected hardware.

    Returns:
        Cache resources in layer order. Index K is read by index-source layers,
        while the corresponding long KV is read by every long-context consumer.
        Scratch compressor state is included explicitly for snapshot planning.
    """
    resources: list[DeepseekV41CacheResource] = []
    for role in topology.layers:
        layer_idx = role.layer_idx
        cache_prefix = _attention_cache_prefix(prefix, layer_idx)
        resources.append(DeepseekV41CacheResource(f"{cache_prefix}.swa_cache", "swa", layer_idx, (layer_idx,)))
        if not role.is_kv_source:
            continue
        resources.append(
            DeepseekV41CacheResource(
                f"{cache_prefix}.long_kv_cache", "long_kv", layer_idx, topology.kv_consumers(layer_idx)
            )
        )
        index_readers = tuple(
            consumer.layer_idx
            for consumer in topology.layers
            if consumer.kv_source_layer == layer_idx and consumer.is_index_source
        )
        resources.append(
            DeepseekV41CacheResource(f"{cache_prefix}.indexer.k_cache", "index_k", layer_idx, index_readers)
        )
        if include_folded_index and role.is_candidate_source:
            resources.append(
                DeepseekV41CacheResource(
                    f"{cache_prefix}.indexer.k_cache_folded", "folded_index_k", layer_idx, index_readers
                )
            )
        if role.compress_ratio == DSV41_BOUNDARY_COMPRESS_RATIO:
            resources.append(
                DeepseekV41CacheResource(
                    f"{cache_prefix}.compressor.state_cache", "compressor_state", layer_idx, (layer_idx,)
                )
            )
    return tuple(resources)


def build_transfer_groups(
    topology: DeepseekV41Topology, *, prefix: str = "model"
) -> tuple[DeepseekV41TransferGroup, ...]:
    """Group the KV-source caches into the layerwise v2 transfer groups.

    Pure: reads only the immutable topology and derives the same registered
    names as :func:`build_cache_resources`. No tensor is allocated, no RPC is
    issued, and no runtime state is touched, so the result can be used both
    for the engine-side layer dispatch and for plan validation.

    Args:
        topology: Backbone topology returned by ``build_layer_plan``.
        prefix: Model root before ``layers.<index>.self_attn``.

    Returns:
        One group per KV source, in source order. Each resource keeps its
        original registered descriptor; the group consumer union is the
        ordered union of its resources' consumers. SWA is never a group member.

    Raises:
        ValueError: If sources are not plain, strictly increasing integers, a
            source lacks its long KV or index K cache, or a transfer cache has
            no KV-source owner. The fixed v2 layout itself is enforced by
            :func:`validate_lw_v2_topology`.
    """
    sources = tuple(topology.kv_source_layer_ids)
    if not sources:
        raise ValueError("DSV41 layerwise v2 needs at least one KV source layer")
    if any(type(source) is not int for source in sources):
        raise ValueError(f"KV source ids must be plain ints, got {sources!r}")
    if sources != tuple(sorted(set(sources))):
        raise ValueError(f"KV source ids must be strictly increasing, got {sources!r}")

    grouped: dict[int, dict[str, DeepseekV41CacheResource]] = {source: {} for source in sources}
    for resource in topology.cache_resources(prefix=prefix, include_folded_index=False):
        if resource.cache_kind == "swa":
            # SWA is recomputed by the engine inside the bounded replay window;
            # it is registered but never transferred (contract 1.2).
            continue
        owned = grouped.get(resource.producer_layer)
        if owned is None:
            raise ValueError(
                f"{resource.cache_name} is not produced by a KV source layer: {resource.producer_layer}"
            )
        if resource.cache_kind in owned:
            raise ValueError(f"Layer {resource.producer_layer} registers two {resource.cache_kind} caches")
        owned[resource.cache_kind] = resource

    execution_positions = {role.layer_idx: index for index, role in enumerate(topology.layers)}
    groups: list[DeepseekV41TransferGroup] = []
    for source in sources:
        owned = grouped[source]
        long_kv = owned.get("long_kv")
        if long_kv is None:
            raise ValueError(f"KV source {source} has no long KV cache")
        index_k = owned.get("index_k")
        if index_k is None:
            raise ValueError(f"KV source {source} has no index K cache")
        boundary = owned.get("compressor_state")
        unsupported = sorted(set(owned) - {"long_kv", "index_k", "compressor_state"})
        if unsupported:
            raise ValueError(f"KV source {source} has unsupported transfer caches: {unsupported}")
        attention_prefix = _attention_cache_prefix(prefix, source)
        groups.append(
            DeepseekV41TransferGroup(
                group_id=f"g{source}",
                source_layer=source,
                attention_prefix=attention_prefix,
                readiness_layer=f"{attention_prefix}.attn",
                dispatch_layer=f"{attention_prefix}.v41_attn",
                consumer_layers=tuple(
                    f"{_attention_cache_prefix(prefix, layer)}.attn"
                    for layer in sorted(
                        {*long_kv.consumer_layers, *index_k.consumer_layers},
                        key=execution_positions.__getitem__,
                    )
                ),
                transfer_resources=(long_kv, index_k),
                boundary_resource=boundary,
            )
        )
    return tuple(groups)


def validate_lw_v2_topology(topology: DeepseekV41Topology) -> None:
    """Validate the fixed first-phase DSV4.1 layerwise v2 topology.

    The v2 transfer set is defined for exactly four KV source groups (layers
    2/8/14/20), boundaries on 2/8/14 and 40 backbone layers. Any other layout
    would silently change the registered transfer set, so the engine seam
    rejects it at bind time instead of degrading to a different plan.

    Args:
        topology: Backbone topology returned by ``build_layer_plan``.

    Raises:
        ValueError: If the backbone depth, KV source layers, boundary sources,
            or per-source consumer segments differ from the v2 topology.
    """
    if len(topology.layers) != DSV41_LW_V2_BACKBONE_LAYERS:
        raise ValueError(
            f"DSV41 layerwise v2 expects {DSV41_LW_V2_BACKBONE_LAYERS} backbone layers, "
            f"got {len(topology.layers)}"
        )
    sources = tuple(topology.kv_source_layer_ids)
    if any(type(source) is not int for source in sources):
        raise ValueError(f"KV source ids must be plain ints, got {sources!r}")
    if sources != DSV41_LW_V2_SOURCE_LAYERS:
        raise ValueError(
            f"DSV41 layerwise v2 expects KV source layers {DSV41_LW_V2_SOURCE_LAYERS}, got {sources}"
        )
    boundary_sources = tuple(
        role.layer_idx
        for role in topology.layers
        if role.is_kv_source and role.compress_ratio == DSV41_BOUNDARY_COMPRESS_RATIO
    )
    if boundary_sources != DSV41_LW_V2_BOUNDARY_SOURCE_LAYERS:
        raise ValueError(
            f"DSV41 layerwise v2 expects boundary sources {DSV41_LW_V2_BOUNDARY_SOURCE_LAYERS}, "
            f"got {boundary_sources}"
        )
    for source, expected_consumers in DSV41_LW_V2_CONSUMER_SEGMENTS:
        consumers = topology.kv_consumers(source)
        if consumers != expected_consumers:
            raise ValueError(
                f"DSV41 layerwise v2 expects consumers {expected_consumers} for group g{source}, got {consumers}"
            )


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the layer-sharing graph from a parsed text config.

    Extra compression ratios for speculative layers are allowed; only the first
    ``num_hidden_layers`` entries describe the backbone. Reduced SWA-only models
    may omit all source layers.

    Raises:
        ValueError: If a long-context layer lacks an earlier source, source lists
            are unordered, a KV source lacks its own indexer, or cache consumers
            disagree with the source's compression ratio.
    """
    num_layers = int(config.num_hidden_layers)
    ratios = tuple(config.compress_ratios)
    kv_sources = tuple(config.kv_source_layer_ids)
    index_sources = tuple(config.index_source_layer_ids)
    if num_layers < 1 or len(ratios) < num_layers:
        raise ValueError("compress_ratios must cover every backbone layer")
    ratios = ratios[:num_layers]
    for name, sources in (("KV", kv_sources), ("index", index_sources)):
        if sources != tuple(sorted(set(sources))):
            raise ValueError(f"{name} sources must be strictly increasing")
        if any(source < 0 or source >= num_layers or ratios[source] == 0 for source in sources):
            raise ValueError(f"{name} sources must name long-context backbone layers")
    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every KV source must also be an index source")
    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(config.engram_layer_ids)}
    candidate_source = int(config.candidate_source_layer_id)
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        if ratio not in (0, 1, 2):
            raise ValueError(f"Unsupported compress_ratio {ratio} at layer {layer_idx}")
        kv_source = next((source for source in reversed(kv_sources) if source <= layer_idx), None) if ratio else None
        index_source = (
            next((source for source in reversed(index_sources) if source <= layer_idx), None) if ratio else None
        )
        if ratio:
            if kv_source is None:
                raise ValueError(f"Layer {layer_idx} has no preceding KV source")
            if index_source is None or index_source < kv_source:
                raise ValueError(f"Layer {layer_idx} has no index source in its KV segment")
            if ratios[kv_source] != ratio:
                raise ValueError(f"Layer {layer_idx} compress_ratio differs from KV source {kv_source}")
        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )
    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=index_sources,
        candidate_source_layer_id=candidate_source,
        candidate_topk_blocks=int(config.candidate_topk_blocks),
        candidate_block_size=int(config.candidate_block_size),
        index_topk=int(config.index_topk),
    )
