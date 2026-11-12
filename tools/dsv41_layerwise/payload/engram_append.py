

# --- dsv41_layerwise 0.30.0rc2 backport (t71) ---
# vLLM main added engram_head_shard_rank after v0.30.0rc2; vllm-ascend's
# DSV4.1 engram embedding/parallel shards call it. Body verbatim from
# upstream; the group accessor falls back to the plain DP group when this
# vLLM release does not export the engram-specific one (with a single
# DP-shared engram table the engram DP group covers the whole DP group, so
# the computed shard rank is identical).
try:  # noqa: SIM105
    from vllm.distributed import get_engram_dp_group as _get_engram_dp_group
except ImportError:  # pragma: no cover - v0.30.0rc2 lane
    from vllm.distributed import get_dp_group as _get_engram_dp_group


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = _get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank
