# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401
"""Re-export the real v0.30.0rc2 speculator graph manager.

``vllm_ascend/worker/v2/spec_decode/target_dependent_ar/aclgraph.py`` (and
the patch module) import ``SpeculatorCudaGraphManager`` from this module;
v0.30.0rc2 ships the class in ``...spec_decode.autoregressive``, so re-export
that class instead of stubbing it.
"""

from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (  # noqa: F401
    SpeculatorCudaGraphManager,
)
