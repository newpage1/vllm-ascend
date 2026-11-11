# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent DeepSeek V4.1 source and cache-consumer topology."""

from dataclasses import dataclass
from typing import Any, Literal


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
    root = f"{prefix}." if prefix else ""
    for role in topology.layers:
        layer_idx = role.layer_idx
        cache_prefix = f"{root}layers.{layer_idx}.self_attn"
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
        if role.compress_ratio == 2:
            resources.append(
                DeepseekV41CacheResource(
                    f"{cache_prefix}.compressor.state_cache", "compressor_state", layer_idx, (layer_idx,)
                )
            )
    return tuple(resources)


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
