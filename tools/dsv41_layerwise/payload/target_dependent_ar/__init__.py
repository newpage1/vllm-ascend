# SPDX-License-Identifier: Apache-2.0
"""vLLM main ``target_dependent_ar`` shim for the v0.30.0rc2 image.

vllm-ascend main imports ``vllm.v1.worker.gpu.spec_decode.target_dependent_ar``
unconditionally (``vllm_ascend/patch/worker/__init__.py`` loads
``patch_eagle_speculator``, which immediately rebinds
``SpeculatorCudaGraphManager`` to the Ascend manager), so this shim only has
to be importable and to expose the manager symbols. The real graph-manager
class is re-exported from the ``autoregressive`` package that v0.30.0rc2
ships; ``TargetDependentARSpeculator`` is a stub because the v1 worker lane
of this overlay never executes target-dependent speculation.

Installed by ``tools/dsv41_layerwise/overlay-0.30.0rc2.sh`` (t71).
"""

from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import (  # noqa: F401
    cudagraph_utils,
    speculator,
)
