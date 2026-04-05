"""GPU pricing kernels for batch option pricing.

Supports multiple backends via dtype management module:
- Metal (Apple Silicon): Uses Float32
- CUDA/HIP: Uses Float64
- CPU fallback: Always available

Kernels use comptime layouts for GPU compatibility.
"""

from std.gpu import global_idx
from layout import LayoutTensor
from gpu_utils.dtype import (
    METAL_DTYPE, METAL_VEC_LAYOUT, METAL_MAX_N,
    CUDA_DTYPE, CUDA_VEC_LAYOUT, CUDA_MAX_N,
)
from std.sys import has_apple_gpu_accelerator


# Backend-specific types - automatically selected via ternary expressions
comptime PRICER_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime PRICER_MAX_OPTIONS = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime PRICER_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT


def payoff_integration_kernel(
    pdf: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    s_points: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    v_points: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    ds_weights: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    dv_weights: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    strikes: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    barriers: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    prices: LayoutTensor[PRICER_DTYPE, PRICER_VEC_LAYOUT, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
    n_options: Int,
):
    """GPU kernel: one thread per option. Integrate payoff over PDF grid.

    Uses backend-appropriate dtype from dtype management module.
    Each thread computes:
    price = Σ_i Σ_j payoff(S_i, K, barrier) × pdf(S_i, V_j) × dS_i × dV_j
    
    Payoff: European call = max(S - K, 0) with barrier knock-out.
    """
    var option_idx = global_idx.x
    
    if option_idx >= n_options:
        return
    
    var K = rebind[Scalar[PRICER_DTYPE]](strikes[option_idx])
    var barrier = rebind[Scalar[PRICER_DTYPE]](barriers[option_idx])

    var price: Scalar[PRICER_DTYPE] = 0.0
    var i = 0
    while i < n_s:
        var S = rebind[Scalar[PRICER_DTYPE]](s_points[i])
        var ds_w = rebind[Scalar[PRICER_DTYPE]](ds_weights[i])
        
        # European call payoff with barrier knock-out
        var payoff: Scalar[PRICER_DTYPE] = S - K
        if payoff < 0.0:
            payoff = 0.0
        if S >= barrier:
            payoff = 0.0
        
        if payoff > 0.0:
            var j = 0
            while j < n_v:
                var pdf_val = rebind[Scalar[PRICER_DTYPE]](pdf[i * PRICER_MAX_OPTIONS + j])
                var dv_w = rebind[Scalar[PRICER_DTYPE]](dv_weights[j])
                price = price + payoff * pdf_val * ds_w * dv_w
                j += 1
        i += 1

    prices[option_idx] = rebind[prices.element_type](price)
