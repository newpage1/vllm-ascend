# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""vLLM main ``kv_cache_planning`` module shim for the v0.30.0rc2 image.

vLLM #61183 moved the planning helpers out of
``vllm.v1.core.kv_cache_utils`` into ``vllm.v1.core.kv_cache_planning``.
``vllm-ascend`` main imports the new home, but the v0.30.0rc2 image still
ships the helpers in ``kv_cache_utils`` only.  This overlay installs this
module at ``vllm/v1/core/kv_cache_planning.py`` so the branch import
surface works unmodified.

``DefaultKVCacheConfigBuilder`` does not exist anywhere in v0.30.0rc2; the
v0.30 planner never resolves it (``platform.get_kv_cache_config_builder_cls``
is dead code on this vLLM), so the fallback class below only has to be
importable.

As an import side effect this module also injects the
``VllmConfig.aux_output_config`` namespace expected by vllm-ascend main
(``enabled`` / ``enable_return_routed_experts`` both default to ``False``);
v0.30 ``VllmConfig`` has no such field and stock reads stay on the
``getattr(..., None)`` early-exit path.

Installed by ``tools/dsv41_layerwise/overlay-0.30.0rc2.sh`` (t71).
"""

from vllm.v1.core import kv_cache_utils as _kv_cache_utils

get_kv_cache_groups = _kv_cache_utils.get_kv_cache_groups
unify_hybrid_kv_cache_specs = _kv_cache_utils.unify_hybrid_kv_cache_specs
may_override_num_blocks = _kv_cache_utils.may_override_num_blocks
create_kv_cache_group_specs = _kv_cache_utils.create_kv_cache_group_specs


def __getattr__(name: str):
    # Forward any other planning symbol that still lives in kv_cache_utils on
    # this vLLM release, so branch imports keep working without enumerating
    # every helper here.
    try:
        return getattr(_kv_cache_utils, name)
    except AttributeError:
        raise AttributeError(name) from None


class DefaultKVCacheConfigBuilder:
    """Placeholder for the vLLM-main builder absent from v0.30.0rc2.

    The v0.30 planner resolves cache sizes through
    ``initialize_kv_cache``/``may_override_num_blocks`` and never asks the
    platform for a builder class, so nothing instantiates this fallback.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "DefaultKVCacheConfigBuilder is a vLLM-main API and is not "
            "available in the v0.30.0rc2 planner."
        )

    @staticmethod
    def _get_pool_bytes_per_block(_vllm_config, _kv_cache_groups):
        # vllm-ascend main monkeypatches and super()-chains into this hook
        # (patch_kv_cache_utils / AscendKVCacheConfigBuilder). The v0.30
        # planner never runs the builder path, so the hook only has to
        # resolve; failing loudly beats guessing a pool size if it is ever
        # reached on this lane.
        raise NotImplementedError(
            "DefaultKVCacheConfigBuilder._get_pool_bytes_per_block is a "
            "vLLM-main API and is not available in the v0.30.0rc2 planner."
        )


def _inject_aux_output_config() -> None:
    """Give v0.30 ``VllmConfig`` the ``aux_output_config`` vllm-ascend main reads."""
    from types import SimpleNamespace

    from vllm.config import VllmConfig

    if hasattr(VllmConfig, "aux_output_config"):
        return
    VllmConfig.aux_output_config = SimpleNamespace(
        enabled=False,
        enable_return_routed_experts=False,
    )


_inject_aux_output_config()


# ── B12 (t71): 镜像补丁透传 ─────────────────────────────────────────────
# vllm-ascend main 在本模块上 monkeypatch 规划钩子（patch_kv_cache_utils
# 尾部对 ``_kv_cache_planning.<name>`` 赋值，共 7 个符号）。vLLM main 的
# 消费方从 planning 命名空间解析这些钩子；vLLM 0.30 的消费方全部从
# ``vllm.v1.core.kv_cache_utils`` 解析（get_kv_cache_groups 内部对
# _get_packed_kv_cache_groups 等做模块全局查找）。本模块在 0.30 lane 上
# 只是别名，因此把之后的属性写入镜像回 kv_cache_utils，让分支的 DSV4.1
# 分组（_ascend_get_packed_kv_cache_groups → group_cache_specs/
# make_cache_groups）真正接入 0.30 的规划路径——与镜像树 stock 0.30 对
# 同名符号直接打在 kv_cache_utils 上的做法一致。
import sys as _sys
import types as _types


class _MirroringModule(_types.ModuleType):

    # vLLM 0.30 的 vllm 内部调用点以单参调用 _pool_bytes_per_block(groups)，
    # 而 vllm-ascend main 版本是 (vllm_config, groups) 双参。该符号不直接
    # 镜像，改由下方 _pool_bytes_per_block_030_adapter 适配。
    _SKIP_MIRROR = frozenset({"_backing", "_pool_bytes_per_block"})

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name.startswith("__") or name in _MirroringModule._SKIP_MIRROR:
            return
        backing = self.__dict__.get("_backing")
        if backing is not None:
            setattr(backing, name, value)


def _pool_bytes_per_block_030_adapter(kv_cache_groups):
    # 单参（vLLM 0.30 调用形态）→ 分支双参实现。DSV4.1/GLM/TQ 路径不读
    # vllm_config；仅当分支实现本身不接受该形态时回退 stock 0.30 记账。
    branch_fn = globals().get("_pool_bytes_per_block")
    if branch_fn is not None and branch_fn is not _pool_bytes_per_block_030_adapter:
        try:
            return branch_fn(None, kv_cache_groups)
        except TypeError:
            pass
    return _kv_cache_utils.__dict__.get(
        "_stock_pool_bytes_per_block_030", lambda groups: 0
    )(kv_cache_groups)


_self = _sys.modules.get(__name__)
if _self is not None:
    _self.__class__ = _MirroringModule
    _self._backing = _kv_cache_utils
    # 保存 stock 实现并挂上适配器（分支补丁加载前执行一次，幂等）
    if not hasattr(_kv_cache_utils, "_stock_pool_bytes_per_block_030"):
        _kv_cache_utils._stock_pool_bytes_per_block_030 = _kv_cache_utils._pool_bytes_per_block
    _kv_cache_utils._pool_bytes_per_block = _pool_bytes_per_block_030_adapter
