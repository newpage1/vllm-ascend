# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-lane symbol shims for the DeepSeek V4.1 model closure.

The ``dsv41_layerwise`` branch tracks vLLM main, while the validated serving
image (``quay.io/ascend/vllm-ascend:v0.30.0rc2-a3``) ships vLLM 0.30.0rc2.
This module keeps the DSV4.1 closure importable and construction-compatible
on both lanes without version-string guessing: each helper probes the actual
runtime symbols and falls through only when a probed module or attribute is
missing, so a lane that moves or drops a symbol is resolved from evidence,
not from a version table that can drift.
"""

import importlib
import inspect
from collections.abc import Callable
from typing import Any

# vLLM #61183 moved ``may_override_num_blocks`` to ``kv_cache_planning`` on
# main; the release lanes (0.30.0rc2, 0.32.0) still ship it in
# ``kv_cache_utils``. Probe the release location first so both lanes resolve.
_MAY_OVERRIDE_NUM_BLOCKS_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("vllm.v1.core.kv_cache_utils", "may_override_num_blocks"),
    ("vllm.v1.core.kv_cache_planning", "may_override_num_blocks"),
)


def resolve_may_override_num_blocks() -> Callable[..., Any]:
    """Return this runtime's ``may_override_num_blocks`` implementation.

    Probes the candidate ``(module, symbol)`` pairs in order and returns the
    first callable match. Only ``ImportError`` (candidate module absent) and
    ``AttributeError`` (symbol absent from an imported module) fall through
    to the next candidate; errors raised while importing a present module
    propagate, so a broken dependency is never mistaken for a moved symbol.

    Returns:
        The resolved callable.

    Raises:
        ImportError: No candidate module exposes the symbol.
        TypeError: A candidate resolved to a non-callable attribute.
    """
    failures: list[str] = []
    for module_name, symbol in _MAY_OVERRIDE_NUM_BLOCKS_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            resolved = getattr(module, symbol)
        except (ImportError, AttributeError) as exc:
            failures.append(f"{module_name}.{symbol}: {exc!r}")
            continue
        if not callable(resolved):
            raise TypeError(f"{module_name}.{symbol} resolved to a non-callable {resolved!r}")
        return resolved
    raise ImportError(
        "may_override_num_blocks was not found in any candidate vLLM module: " + "; ".join(failures)
    )


def declared_kwarg(owner: type, name: str, value: Any) -> dict[str, Any]:
    """Spell one optional field out for a constructor that declares it.

    Used for spec fields upstream adds or removes across lanes --
    ``storage_block_size`` and ``bounded_replay`` on the MLA/SWA cache specs.
    Naming such a keyword unconditionally is a ``TypeError`` on the lane that
    does not declare it, and omitting it silently changes allocation sizes on
    the lane that requires it; probing the constructor keeps that decision in
    one place. This mirrors ``vllm_ascend.core.kv_cache_interface``'s helper
    so the DSV4.1 overlay stays self-contained on the 0.30.0rc2 image, whose
    ``core.kv_cache_interface`` does not export it.
    """
    return {name: value} if name in inspect.signature(owner).parameters else {}
