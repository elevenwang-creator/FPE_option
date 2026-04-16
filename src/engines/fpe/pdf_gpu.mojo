"""GPU kernels for FPE PDF computation and option pricing integration."""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, METAL_MAX_N, CUDA_DTYPE, CUDA_VEC_LAYOUT, CUDA_MAX_N
from std.sys import has_apple_gpu_accelerator

comptime GPU_PDF_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_PDF_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_PDF_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_PDF_SCALAR = Scalar[GPU_PDF_DTYPE]


def integrate_gpu_kernel(
    pdf_out: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    q_in: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    phi_in: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    n_basis: Int,
    n_points: Int,
    n_s: Int,
    n_v: Int,
):
    """Compute PDF from ODE solution: pdf(t) = Phi * q(T)."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base_phi = b * GPU_PDF_MAX_N * GPU_PDF_MAX_N
    var base_q = b * n_basis
    var base_pdf = b * n_points

    var i = Int(tid)
    while i < n_points:
        var sum: Float64 = 0.0
        for j in range(n_basis):
            var phi_val = Float64(rebind[GPU_PDF_SCALAR](phi_in[base_phi + i * GPU_PDF_MAX_N + j]))
            var q_val = Float64(rebind[GPU_PDF_SCALAR](q_in[base_q + j]))
            sum += phi_val * q_val
        pdf_out[base_pdf + i] = GPU_PDF_SCALAR(sum)
        i += Int(threads)


def price_integration_kernel(
    pdf: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    s_points: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    v_points: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    ds_weights: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    dv_weights: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    strikes: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    barriers: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    prices_out: LayoutTensor[GPU_PDF_DTYPE, GPU_PDF_VEC, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
    n_options: Int,
):
    """GPU kernel: integrate payoff over PDF grid for batch option pricing."""
    var b = Int(block_idx.x)
    if Int(b) >= n_options:
        return

    var K = Float64(rebind[GPU_PDF_SCALAR](strikes[Int(b)]))
    var barrier = Float64(rebind[GPU_PDF_SCALAR](barriers[Int(b)]))

    if Int(thread_idx.x) == 0:
        var price: Float64 = 0.0
        for i in range(n_s):
            var S = Float64(rebind[GPU_PDF_SCALAR](s_points[i]))
            var ds_w = Float64(rebind[GPU_PDF_SCALAR](ds_weights[i]))
            var payoff = S - K
            if payoff < 0.0:
                payoff = 0.0
            if barrier > 0.0 and S >= barrier:
                payoff = 0.0
            if payoff > 0.0:
                for j in range(n_v):
                    var pdf_val = Float64(rebind[GPU_PDF_SCALAR](pdf[i * GPU_PDF_MAX_N + j]))
                    var dv_w = Float64(rebind[GPU_PDF_SCALAR](dv_weights[j]))
                    price = price + payoff * pdf_val * ds_w * dv_w
        prices_out[Int(b)] = GPU_PDF_SCALAR(price)