# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

TOPOLOGY_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/topology.py"
SPEC = importlib.util.spec_from_file_location("dsv41_topology_cpu_test", TOPOLOGY_PATH)
assert SPEC is not None and SPEC.loader is not None
TOPOLOGY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOPOLOGY
SPEC.loader.exec_module(TOPOLOGY)
build_cache_resources = TOPOLOGY.build_cache_resources
build_layer_plan = TOPOLOGY.build_layer_plan


@pytest.fixture
def text_config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        index_topk=512,
        engram_layer_ids=[1, 14],
    )


def test_long_kv_survives_topk_source_changes(text_config):
    topology = build_layer_plan(text_config)
    resources = {resource.cache_name: resource for resource in build_cache_resources(topology)}
    source = resources["model.layers.20.self_attn.long_kv_cache"]
    assert source.producer_layer == 20
    assert source.consumer_layers == tuple(range(20, 40))
    assert topology.layer(26).kv_source_layer == 20
    assert topology.layer(26).index_source_layer == 24


def test_index_k_is_read_by_index_sources_and_not_topk_consumers(text_config):
    resources = {resource.cache_name: resource for resource in build_cache_resources(build_layer_plan(text_config))}
    index = resources["model.layers.20.self_attn.indexer.k_cache"]
    assert index.consumer_layers == (20, 24, 28, 32, 36)
    assert 26 not in index.consumer_layers
    assert all("topk" not in name and "candidate" not in name for name in resources)
    assert "model.layers.24.self_attn.indexer.k_cache" not in resources


def test_swa_and_compressor_state_have_layer_local_lifetimes(text_config):
    resources = {resource.cache_name: resource for resource in build_cache_resources(build_layer_plan(text_config))}
    assert resources["model.layers.26.self_attn.swa_cache"].consumer_layers == (26,)
    ring = resources["model.layers.2.self_attn.compressor.state_cache"]
    assert ring.cache_kind == "compressor_state"
    assert ring.consumer_layers == (2,)
    assert "model.layers.20.self_attn.compressor.state_cache" not in resources


def test_reduced_swa_only_model_and_prefix(text_config):
    text_config.num_hidden_layers = 2
    text_config.compress_ratios = [0, 0]
    text_config.kv_source_layer_ids = []
    text_config.index_source_layer_ids = []
    resources = build_cache_resources(build_layer_plan(text_config), prefix="language_model.model")
    assert [resource.cache_name for resource in resources] == [
        "language_model.model.layers.0.self_attn.swa_cache",
        "language_model.model.layers.1.self_attn.swa_cache",
    ]


def test_folded_index_lives_through_its_final_index_reader(text_config):
    topology = build_layer_plan(text_config)
    resources = topology.cache_resources(include_folded_index=True)
    folded = [resource for resource in resources if resource.cache_kind == "folded_index_k"]
    assert len(folded) == 1
    assert folded[0].producer_layer == 20
    assert folded[0].consumer_layers[-1] == 36


def test_nonuniform_source_groups_are_derived_from_configuration(text_config):
    text_config.num_hidden_layers = 7
    text_config.compress_ratios = [0, 2, 2, 2, 1, 1, 1]
    text_config.kv_source_layer_ids = [1, 4]
    text_config.index_source_layer_ids = [1, 3, 4, 6]
    resources = {resource.cache_name: resource for resource in build_layer_plan(text_config).cache_resources()}
    assert resources["model.layers.1.self_attn.long_kv_cache"].consumer_layers == (1, 2, 3)
    assert resources["model.layers.4.self_attn.indexer.k_cache"].consumer_layers == (4, 6)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("compress_ratios", [0], "compress_ratios"),
        ("compress_ratios", [0, 0] + [3] * 38, "compress_ratio"),
        ("kv_source_layer_ids", [8, 14, 20], "KV source"),
        ("kv_source_layer_ids", [8, 2, 14, 20], "strictly increasing"),
        ("index_source_layer_ids", [2, 14, 20], "index source"),
    ],
)
def test_invalid_source_graph_fails_before_cache_registration(text_config, field, value, message):
    setattr(text_config, field, value)
    with pytest.raises(ValueError, match=message):
        build_layer_plan(text_config)


build_transfer_groups = TOPOLOGY.build_transfer_groups
validate_lw_v2_topology = TOPOLOGY.validate_lw_v2_topology


def _execution_layer(index: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{index}.self_attn.attn"


def _dispatch_layer(index: int, prefix: str = "model") -> str:
    return f"{prefix}.layers.{index}.self_attn.v41_attn"


def test_transfer_groups_cover_the_four_kv_sources(text_config):
    topology = build_layer_plan(text_config)
    validate_lw_v2_topology(topology)
    groups = build_transfer_groups(topology)
    assert [group.group_id for group in groups] == ["g2", "g8", "g14", "g20"]
    assert [group.source_layer for group in groups] == [2, 8, 14, 20]
    assert [group.producer_layer for group in groups] == [2, 8, 14, 20]
    assert [group.readiness_layer for group in groups] == [_execution_layer(i) for i in (2, 8, 14, 20)]
    assert [group.dispatch_layer for group in groups] == [_dispatch_layer(i) for i in (2, 8, 14, 20)]
    assert [group.consumer_layers for group in groups] == [
        tuple(_execution_layer(i) for i in range(2, 8)),
        tuple(_execution_layer(i) for i in range(8, 14)),
        tuple(_execution_layer(i) for i in range(14, 20)),
        tuple(_execution_layer(i) for i in range(20, 40)),
    ]
    assert [len(group.transfer_resources) for group in groups] == [2, 2, 2, 2]
    assert sum(group.transfer_resource_count for group in groups) == 11
    assert [group.source_layer for group in groups if group.has_boundary] == [2, 8, 14]
    assert groups[-1].boundary_resource is None
    swa = [resource for resource in topology.cache_resources() if resource.cache_kind == "swa"]
    assert len(swa) == 40
    assert not any(resource.cache_kind == "swa" for group in groups for resource in group.transfer_resources)


def test_group_resources_keep_registered_names_and_true_consumer_subsets(text_config):
    groups = {group.group_id: group for group in build_transfer_groups(build_layer_plan(text_config))}
    long_kv, index_k = groups["g20"].transfer_resources
    assert long_kv.cache_name == "model.layers.20.self_attn.long_kv_cache"
    assert long_kv.consumer_layers == tuple(range(20, 40))
    assert index_k.cache_name == "model.layers.20.self_attn.indexer.k_cache"
    assert index_k.consumer_layers == (20, 24, 28, 32, 36)
    assert groups["g20"].has_boundary is False
    assert groups["g2"].boundary_resource.cache_name == "model.layers.2.self_attn.compressor.state_cache"


def test_group_builder_rejects_a_source_without_a_source_role(text_config):
    inconsistent = replace(build_layer_plan(text_config), kv_source_layer_ids=(2, 8, 14, 20, 26))
    with pytest.raises(ValueError, match="has no long KV cache"):
        build_transfer_groups(inconsistent)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kv_source_layer_ids", (2, 8, 14, 20, 24), "expects KV source layers"),
        ("kv_source_layer_ids", (8, 2, 14, 20), "strictly increasing"),
        ("compress_ratios", [0, 0] + [2] * 38, "boundary sources"),
        ("num_hidden_layers", 41, "backbone layers"),
        ("num_hidden_layers", 2, "backbone layers"),
    ],
)
def test_lw_v2_topology_rejects_any_other_layout(text_config, field, value, message):
    setattr(text_config, field, value)
    with pytest.raises(ValueError, match=message):
        validate_lw_v2_topology(build_layer_plan(text_config))


def test_lw_v2_topology_rejects_a_moved_consumer_segment(text_config):
    topology = build_layer_plan(text_config)
    moved = SimpleNamespace(
        layers=topology.layers,
        kv_source_layer_ids=topology.kv_source_layer_ids,
        kv_consumers=lambda source: (2,) if source == 2 else topology.kv_consumers(source),
    )
    with pytest.raises(ValueError, match="consumers"):
        validate_lw_v2_topology(moved)
