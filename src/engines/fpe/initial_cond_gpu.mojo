"""GPU kernels for FPE initial condition: delta function and NNLS."""

from std.gpu import block_idx, thread_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT, GPU_MAX_N
from std.math import exp, pi

comptime GPU_IC_DTYPE = GPU_DTYPE
comptime GPU_IC_VEC = GPU_VEC_LAYOUT
comptime GPU_IC_MAX_N = GPU_MAX_N
comptime GPU_IC_SCALAR = Scalar[GPU_IC_DTYPE]


def delta_gpu_kernel(
    delta_out: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    grid_in: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    params: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
    n_s_ext: Int,
):
    """Bivariate Gaussian delta function for initial condition."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var s0 = Float64(rebind[GPU_IC_SCALAR](params[b * 12 + 6]))
    var v0 = Float64(rebind[GPU_IC_SCALAR](params[b * 12 + 7]))
    var sigma = Float64(rebind[GPU_IC_SCALAR](params[b * 12 + 2]))
    var s_sigma = sigma
    if s_sigma < 0.01:
        s_sigma = 0.01
    var v_sigma = 0.1 * sigma
    if v_sigma < 0.001:
        v_sigma = 0.001
    var norm = 1.0 / (2.0 * 3.14159265358979323846 * s_sigma * v_sigma)
    var base_grid = b * (n_s_ext + n_s_ext)
    var base_out = b * n_s * n_v

    var idx = Int(tid)
    while idx < n_s * n_v:
        var i_s = idx / n_v
        var j_v = idx % n_v
        var s = Float64(rebind[GPU_IC_SCALAR](grid_in[base_grid + i_s]))
        var v = Float64(
            rebind[GPU_IC_SCALAR](grid_in[base_grid + n_s_ext + j_v])
        )
        var ds = (s - s0) / s_sigma
        var dv = (v - v0) / v_sigma
        var val = norm * exp(-0.5 * (ds * ds + dv * dv))
        delta_out[base_out + idx] = GPU_IC_SCALAR(val)
        idx += Int(threads)

    barrier()
    # Thread-0 normalizes sequentially for correctness
    if tid == 0:
        var sum_val: Float64 = 0.0
        for j in range(n_s * n_v):
            sum_val = sum_val + Float64(
                rebind[GPU_IC_SCALAR](delta_out[base_out + j])
            )
        if sum_val > 0.0:
            var inv_sum = 1.0 / sum_val
            for j in range(n_s * n_v):
                delta_out[base_out + j] = GPU_IC_SCALAR(
                    Float64(rebind[GPU_IC_SCALAR](delta_out[base_out + j]))
                    * inv_sum
                )


def initial_gpu_kernel(
    initial_out: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    delta_in: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    phi_in: LayoutTensor[GPU_IC_DTYPE, GPU_IC_VEC, MutAnyOrigin],
    n_basis: Int,
    n_points: Int,
    max_iter: Int,
):
    """Solve initial condition q0 via projected gradient NNLS."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base_q = b * n_basis
    var base_delta = b * n_points
    var base_phi = b * GPU_IC_MAX_N * GPU_IC_MAX_N

    var i = Int(tid)
    while i < n_basis:
        initial_out[base_q + i] = GPU_IC_SCALAR(0.0)
        i += Int(threads)
    barrier()

    # Compute max diagonal element for step size (thread-0 sequential)
    var step: Float64 = 0.001
    if tid == 0:
        var max_diag: Float64 = 0.0
        for i_diag in range(n_basis):
            var diag_val: Float64 = 0.0
            for k in range(n_points):
                var phi_ki = Float64(
                    rebind[GPU_IC_SCALAR](
                        phi_in[base_phi + k * GPU_IC_MAX_N + i_diag]
                    )
                )
                diag_val = diag_val + phi_ki * phi_ki
            if diag_val > max_diag:
                max_diag = diag_val
        if max_diag > 0.0:
            step = 1.0 / (max_diag + 1e-10)

    barrier()

    var step_size = GPU_IC_SCALAR(step)

    for _iter in range(max_iter):
        # Compute residual: r = delta - Phi^T * q (store in q[n_basis..2n_basis])
        var i_res = Int(tid)
        while i_res < n_points:
            var r_val: Float64 = 0.0 - Float64(
                rebind[GPU_IC_SCALAR](delta_in[base_delta + i_res])
            )
            for j in range(n_basis):
                var phi_val = Float64(
                    rebind[GPU_IC_SCALAR](
                        phi_in[base_phi + i_res * GPU_IC_MAX_N + j]
                    )
                )
                var q_val = Float64(
                    rebind[GPU_IC_SCALAR](initial_out[base_q + j])
                )
                r_val = r_val + phi_val * q_val
            initial_out[base_q + n_basis + i_res] = GPU_IC_SCALAR(r_val)
            i_res += Int(threads)
        barrier()

        # Gradient step: q = max(0, q - step * Phi * r)
        var i_q = Int(tid)
        while i_q < n_basis:
            var g_val: Float64 = 0.0
            for k in range(n_points):
                var phi_val = Float64(
                    rebind[GPU_IC_SCALAR](
                        phi_in[base_phi + k * GPU_IC_MAX_N + i_q]
                    )
                )
                var r_val = Float64(
                    rebind[GPU_IC_SCALAR](initial_out[base_q + n_basis + k])
                )
                g_val = g_val + phi_val * r_val
            var current_q = Float64(
                rebind[GPU_IC_SCALAR](initial_out[base_q + i_q])
            )
            var new_q = current_q - step * g_val
            if new_q < 0.0:
                new_q = 0.0
            initial_out[base_q + i_q] = GPU_IC_SCALAR(new_q)
            i_q += Int(threads)
        barrier()

    # Project to non-negative and normalize (thread-0 sequential)
    i = Int(tid)
    while i < n_basis:
        var val = Float64(rebind[GPU_IC_SCALAR](initial_out[base_q + i]))
        if val < 0.0:
            initial_out[base_q + i] = GPU_IC_SCALAR(0.0)
        i += Int(threads)
    barrier()
    if tid == 0:
        var sum_q: Float64 = 0.0
        for j in range(n_basis):
            sum_q = sum_q + Float64(
                rebind[GPU_IC_SCALAR](initial_out[base_q + j])
            )
        if sum_q > 0.0:
            var inv_sum = 1.0 / sum_q
            for j in range(n_basis):
                initial_out[base_q + j] = GPU_IC_SCALAR(
                    Float64(rebind[GPU_IC_SCALAR](initial_out[base_q + j]))
                    * inv_sum
                )
