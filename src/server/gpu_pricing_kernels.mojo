"""GPU pricing kernels for batch option pricing.

Supports multiple backends:
- Metal (Apple Silicon): Uses Float32
- CUDA/HIP: Uses Float64
- CPU fallback: Always available

Kernels use comptime layouts for GPU compatibility.
"""

from std.gpu import global_idx
from layout import Layout, LayoutTensor


# Concrete comptime layouts for GPU pricing kernels
comptime PRICER_MAX_OPTIONS = 1024
comptime PRICER_MAX_S = 128
comptime PRICER_MAX_V = 128
comptime PRICER_VEC_LAYOUT = Layout.row_major(PRICER_MAX_OPTIONS)
comptime PRICER_MAT_LAYOUT = Layout.row_major(PRICER_MAX_S, PRICER_MAX_V)


def payoff_integration_kernel(
    pdf: LayoutTensor[DType.float32, PRICER_MAT_LAYOUT, MutAnyOrigin],
    s_points: LayoutTensor[DType.float32, PRICER_VEC_LAYOUT, MutAnyOrigin],
    v_points: LayoutTensor[DType.float32, PRICER_VEC_LAYOUT, MutAnyOrigin],
    strikes: LayoutTensor[DType.float32, PRICER_VEC_LAYOUT, MutAnyOrigin],
    barriers: LayoutTensor[DType.float32, PRICER_VEC_LAYOUT, MutAnyOrigin],
    prices: LayoutTensor[DType.float32, PRICER_VEC_LAYOUT, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
):
    """GPU kernel: one thread per option. Integrate payoff over PDF grid.

    Uses Float32 for Metal compatibility. Each thread computes:
    price = Σ_i Σ_j payoff(S_i, K, barrier) × pdf(S_i, V_j) × dS_i × dV_j
    """
    var option_idx = global_idx.x
    
    var K = rebind[Scalar[DType.float32]](strikes[option_idx])
    var barrier = rebind[Scalar[DType.float32]](barriers[option_idx])

    var price: Scalar[DType.float32] = 0.0
    var i = 0
    while i < n_s:
        var S = rebind[Scalar[DType.float32]](s_points[i])
        var payoff: Scalar[DType.float32] = S - K
        if payoff < 0.0:
            payoff = 0.0
        if S >= barrier:
            payoff = 0.0

        var j = 0
        while j < n_v:
            var pdf_val = rebind[Scalar[DType.float32]](pdf[i, j])
            price = price + payoff * pdf_val
            j += 1
        i += 1

    if option_idx < n_s:  # Use n_s as proxy for option count bound
        prices[option_idx] = rebind[prices.element_type](price)
