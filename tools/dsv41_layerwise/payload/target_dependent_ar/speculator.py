# SPDX-License-Identifier: Apache-2.0
"""Stub ``TargetDependentARSpeculator`` for the v0.30.0rc2 overlay.

The v1-worker lane of the dsv41_layerwise overlay never constructs this
speculator; ``vllm_ascend.patch.worker.patch_eagle_speculator`` imports this
module only to rebind ``SpeculatorCudaGraphManager`` to the Ascend manager.
The real upstream implementation arrived after v0.30.0rc2.
"""

from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)


class TargetDependentARSpeculator:
    """Import-time placeholder; never instantiated on this lane."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "TargetDependentARSpeculator is a vLLM-main API and is not "
            "available in the v0.30.0rc2 image."
        )
