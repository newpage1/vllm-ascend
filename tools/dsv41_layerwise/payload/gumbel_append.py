

# --- dsv41_layerwise 0.30.0rc2 backport (t71) ---
# vLLM main added _log1p_neg_stable after v0.30.0rc2; vllm-ascend's dflash2
# speculator consumes it. Verbatim from upstream (vllm-v1 v0.32.0 line-identical).
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

@triton.jit
def _log1p_neg_stable(value):
    # Preserve precision for the positive Gumbel tail without relying on a
    # backend-specific libdevice log1p. The degree-8 series has absolute error
    # below 6e-7 on [0, 0.25]; subtraction is well-conditioned elsewhere for
    # the part of the distribution that can win the argmax.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial

    direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)

