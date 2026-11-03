# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from pathlib import Path
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
