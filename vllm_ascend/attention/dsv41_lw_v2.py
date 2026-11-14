# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side seam for the DeepSeek V4.1 layerwise v2 group protocol.

The v2 design transfers four KV-source groups (source layers 2, 8, 14 and 20).
Only a group head touches the transport: it waits for the group's load,
restores the group's compressor boundary, runs attention, then captures the
boundary and submits the group's store. Every other layer is a pure no-op: it
performs the group-table lookup and nothing else (no RPC, future, ring, codec
or counter access).

This module owns that dispatch. It is inert until a v2 connector explicitly
binds its group facade, so stock serving and the legacy ``dsv41_layerwise``
path in :func:`dsa_v41_forward` keep their own behaviour when v2 is disabled.

Deliberate non-reuse (contract 1.7/6.3): this seam imports no admission
registry, lease, snapshot or boundary-codec state from the legacy layerwise
modules. Exceptions from the group facade are never caught here: a failed group
load must fail the current step instead of resuming computation on an
incomplete prefix (contract 4.2).

The binding is one process-wide, write-once value. The connector installs it
during worker KV registration -- before the first real forward -- and only
worker teardown or tests remove it. A single immutable binding is the only
global state in this module, mirroring the module-level registry pattern the
stock layerwise hooks already use; every other value is passed explicitly.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from vllm_ascend.models.deepseek_v41.topology import (
    DSV41_LW_V2_SOURCE_LAYERS,
    DeepseekV41Topology,
    DeepseekV41TransferGroup,
    build_transfer_groups,
    validate_lw_v2_topology,
)


class DSV41LwV2GroupHooks(Protocol):
    """Group-level transport facade implemented by the v2 connector.

    The engine calls exactly these methods (contract 4.1). ``g20`` has no
    boundary, so its restore/capture methods are never called by the engine.
    """

    def wait_group_load(self, group_id: str) -> None: ...

    def restore_group_boundary(self, group_id: str) -> None: ...

    def capture_group_boundary(self, group_id: str) -> None: ...

    def submit_group_store(self, group_id: str) -> None: ...


_GROUP_HOOK_METHODS: tuple[str, ...] = (
    "wait_group_load",
    "restore_group_boundary",
    "capture_group_boundary",
    "submit_group_store",
)


class DSV41LwV2BindingError(RuntimeError):
    """The v2 engine seam cannot be bound or was bound inconsistently."""


@dataclass(frozen=True)
class DSV41LwV2EngineBinding:
    """Immutable dispatch table of one bound v2 connector.

    ``groups`` is the topology-derived group table; ``hooks`` is the
    connector's public facade. Only group heads are addressable by dispatch
    layer name, so a non-head lookup can never reach the transport.
    """

    hooks: DSV41LwV2GroupHooks
    groups: tuple[DeepseekV41TransferGroup, ...]
    prefix: str
    _group_by_dispatch_layer: Mapping[str, DeepseekV41TransferGroup] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_dispatch_layer: dict[str, DeepseekV41TransferGroup] = {}
        for group in self.groups:
            if group.dispatch_layer in by_dispatch_layer:
                raise DSV41LwV2BindingError(f"dispatch layer {group.dispatch_layer} is listed by two groups")
            by_dispatch_layer[group.dispatch_layer] = group
        object.__setattr__(self, "_group_by_dispatch_layer", MappingProxyType(by_dispatch_layer))

    def group_for_layer(self, dispatch_layer_name: str) -> DeepseekV41TransferGroup | None:
        """Return the group headed by ``dispatch_layer_name``; non-heads return ``None``."""
        return self._group_by_dispatch_layer.get(dispatch_layer_name)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups)

    @property
    def head_layers(self) -> tuple[str, ...]:
        return tuple(self._group_by_dispatch_layer)


_engine_binding: DSV41LwV2EngineBinding | None = None


def lw_v2_engine() -> DSV41LwV2EngineBinding | None:
    """Return the bound v2 engine seam, or ``None`` when v2 is disabled."""
    return _engine_binding


def lw_v2_active() -> bool:
    """Whether a v2 connector bound its group facade in this process."""
    return _engine_binding is not None


def bind_lw_v2_engine(
    hooks: DSV41LwV2GroupHooks,
    *,
    topology: DeepseekV41Topology,
    prefix: str = "model",
) -> DSV41LwV2EngineBinding:
    """Bind the connector's group facade and derive the group-head dispatch table.

    Must run before the first real forward -- the connector does it during
    worker KV registration -- because a real step that reaches attention
    without a bound facade would take the stock path.

    Args:
        hooks: Connector implementing the four group facade methods.
        topology: Backbone topology returned by ``build_layer_plan``. Only the
            fixed first-phase v2 topology is accepted.
        prefix: Model root before ``layers.<index>.self_attn``.

    Returns:
        The installed binding (also stored process-wide).

    Raises:
        DSV41LwV2BindingError: A facade method is missing, or another connector
            instance is already bound.
        ValueError: The topology is not the fixed v2 topology.
    """
    global _engine_binding

    missing = [name for name in _GROUP_HOOK_METHODS if not callable(getattr(hooks, name, None))]
    if missing:
        raise DSV41LwV2BindingError(f"layerwise v2 connector is missing group facade methods: {missing}")
    validate_lw_v2_topology(topology)
    groups = build_transfer_groups(topology, prefix=prefix)
    if len(groups) != len(DSV41_LW_V2_SOURCE_LAYERS):
        raise DSV41LwV2BindingError(
            f"layerwise v2 expects {len(DSV41_LW_V2_SOURCE_LAYERS)} groups, built {len(groups)}"
        )
    existing = _engine_binding
    if existing is not None:
        if existing.hooks is hooks and existing.groups == groups and existing.prefix == prefix:
            return existing
        raise DSV41LwV2BindingError("layerwise v2 engine is already bound to a different connector instance")
    binding = DSV41LwV2EngineBinding(hooks=hooks, groups=groups, prefix=prefix)
    _engine_binding = binding
    return binding


def unbind_lw_v2_engine() -> None:
    """Remove the process-wide binding (worker teardown and tests)."""
    global _engine_binding
    _engine_binding = None


def run_group_forward(
    binding: DSV41LwV2EngineBinding,
    dispatch_layer_name: str,
    forward: Callable[[], None],
) -> None:
    """Run one V4.1 layer forward under the v2 group-head protocol.

    Order (contract 1.4): ``wait_group_load -> restore_group_boundary ->
    forward -> capture_group_boundary -> submit_group_store``. Only the four
    group heads touch the transport; any other layer performs the table lookup
    and runs ``forward`` unchanged.

    Exceptions from the facade propagate on purpose: once the scheduler skipped
    a prefix, a failed group load must fail this step instead of continuing.

    Args:
        binding: Installed engine binding.
        dispatch_layer_name: Custom-op key passed to :func:`dsa_v41_forward`.
        forward: The attention call for this layer.
    """
    group = binding.group_for_layer(dispatch_layer_name)
    if group is None:
        forward()
        return
    hooks = binding.hooks
    hooks.wait_group_load(group.group_id)
    if group.has_boundary:
        hooks.restore_group_boundary(group.group_id)
    forward()
    if group.has_boundary:
        hooks.capture_group_boundary(group.group_id)
    hooks.submit_group_store(group.group_id)


__all__ = [
    "DSV41LwV2BindingError",
    "DSV41LwV2EngineBinding",
    "DSV41LwV2GroupHooks",
    "bind_lw_v2_engine",
    "lw_v2_active",
    "lw_v2_engine",
    "run_group_forward",
    "unbind_lw_v2_engine",
]
