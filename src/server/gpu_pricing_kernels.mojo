"""GPU pricing kernels for batch option pricing.

Supports multiple backends via dtype management module:
- Metal (Apple Silicon): Uses Float32
- CUDA/HIP): Uses Float64
- CPU fallback: Always available

Kernels use comptime layouts for GPU compatibility.

TODO: Update kernel to support BarrierPayoff option_type dispatch (10 types)
instead of hard-coded EuropeanCall + barrier knock-out. See server.payoffs.BarrierPayoff.
"""

from std.gpu import block_idx, thread_idx, block_dim
from layout import LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT, GPU_MAX_N


comptime PRICER_DTYPE = GPU_DTYPE
comptime PRICER_MAX_OPTIONS = GPU_MAX_N
comptime PRICER_VEC_LAYOUT = GPU_VEC_LAYOUT


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
    """GPU kernel: Block to option mapping instance processing.

    Architecture:
    grid_dim.x binds directly to the specific independent option logic chain.
    thread_idx.x delegates workload locally (sequentially here on master thread 0 or mapped loops).
    """
    var b = block_idx.x
    if Int(b) >= n_options:
        return

    var K = rebind[Scalar[PRICER_DTYPE]](strikes[Int(b)])
    var barrier = rebind[Scalar[PRICER_DTYPE]](barriers[Int(b)])

    # Isolate integration compute entirely to one thread proxy per block
    # to avoid synchronization overhead lacking shared reductions.
    if Int(thread_idx.x) == 0:
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
                    var pdf_val = rebind[Scalar[PRICER_DTYPE]](
                        pdf[i * PRICER_MAX_OPTIONS + j]
                    )
                    var dv_w = rebind[Scalar[PRICER_DTYPE]](dv_weights[j])
                    price = price + payoff * pdf_val * ds_w * dv_w
                    j += 1
            i += 1

        prices[Int(b)] = rebind[prices.element_type](price)
