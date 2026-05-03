"""GPU kernels for FPE initial condition: delta function and NNLS."""

from std.gpu import block_idx, thread_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import (
    METAL_DTYPE,
    METAL_VEC_LAYOUT,
    METAL_MAX_N,
    CUDA_DTYPE,
    CUDA_VEC_LAYOUT,
    CUDA_MAX_N,
)
from std.sys import has_apple_gpu_accelerator
from std.math import exp

comptime GPU_IC_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_IC_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_IC_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
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
    var s0 = rebind[GPU_IC_SCALAR](params[b * 12 + 6])
    var v0 = rebind[GPU_IC_SCALAR](params[b * 12 + 7])
    var sigma = rebind[GPU_IC_SCALAR](params[b * 12 + 2])
    var s_sigma: GPU_IC_SCALAR = sigma
    if s_sigma < 0.01:
        s_sigma = 0.01
    var v_sigma: GPU_IC_SCALAR = 0.1 * sigma
    if v_sigma < 0.001:
        v_sigma = 0.001
    var norm: GPU_IC_SCALAR = 1.0 / (
        2.0 * GPU_IC_SCALAR(3.14159265358979) * s_sigma * v_sigma
    )
    var base_grid = b * (n_s_ext + n_s_ext)
    var base_out = b * n_s * n_v

    var idx = Int(tid)
    while idx < n_s * n_v:
        var i_s = idx / n_v
        var j_v = idx % n_v
        var s = rebind[GPU_IC_SCALAR](grid_in[base_grid + i_s])
        var v = rebind[GPU_IC_SCALAR](grid_in[base_grid + n_s_ext + j_v])
        var ds_f = Float64(s - s0) / Float64(s_sigma)
        var dv_f = Float64(v - v0) / Float64(v_sigma)
        var arg = -0.5 * (ds_f * ds_f + dv_f * dv_f)
        var val = GPU_IC_SCALAR(Float64(norm) * exp(arg))
        delta_out[base_out + idx] = rebind[delta_out.element_type](val)
        idx += Int(threads)

    barrier()
    if tid == 0:
        var sum_val: GPU_IC_SCALAR = 0.0
        for j in range(n_s * n_v):
            sum_val = sum_val + rebind[GPU_IC_SCALAR](delta_out[base_out + j])
        if sum_val > 0.0:
            var inv_sum: GPU_IC_SCALAR = 1.0 / sum_val
            for j in range(n_s * n_v):
                delta_out[base_out + j] = rebind[delta_out.element_type](
                    rebind[GPU_IC_SCALAR](delta_out[base_out + j]) * inv_sum
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
        initial_out[base_q + i] = rebind[initial_out.element_type](
            GPU_IC_SCALAR(0.0)
        )
        i += Int(threads)
    barrier()

    var step: GPU_IC_SCALAR = 0.001
    if tid == 0:
        var max_diag: GPU_IC_SCALAR = 0.0
        for i_diag in range(n_basis):
            var diag_val: GPU_IC_SCALAR = 0.0
            for k in range(n_points):
                var phi_ki = rebind[GPU_IC_SCALAR](
                    phi_in[base_phi + k * GPU_IC_MAX_N + i_diag]
                )
                diag_val = diag_val + phi_ki * phi_ki
            if diag_val > max_diag:
                max_diag = diag_val
        if max_diag > 0.0:
            step = 1.0 / (max_diag + GPU_IC_SCALAR(1e-10))

    barrier()

    for _iter in range(max_iter):
        var i_res = Int(tid)
        while i_res < n_points:
            var r_val: GPU_IC_SCALAR = 0.0 - rebind[GPU_IC_SCALAR](
                delta_in[base_delta + i_res]
            )
            for j in range(n_basis):
                var phi_val = rebind[GPU_IC_SCALAR](
                    phi_in[base_phi + i_res * GPU_IC_MAX_N + j]
                )
                var q_val = rebind[GPU_IC_SCALAR](initial_out[base_q + j])
                r_val = r_val + phi_val * q_val
            initial_out[base_q + n_basis + i_res] = rebind[
                initial_out.element_type
            ](r_val)
            i_res += Int(threads)
        barrier()

        var i_q = Int(tid)
        while i_q < n_basis:
            var g_val: GPU_IC_SCALAR = 0.0
            for k in range(n_points):
                var phi_val = rebind[GPU_IC_SCALAR](
                    phi_in[base_phi + k * GPU_IC_MAX_N + i_q]
                )
                var r_val = rebind[GPU_IC_SCALAR](
                    initial_out[base_q + n_basis + k]
                )
                g_val = g_val + phi_val * r_val
            var current_q = rebind[GPU_IC_SCALAR](initial_out[base_q + i_q])
            var new_q: GPU_IC_SCALAR = current_q - step * g_val
            if new_q < 0.0:
                new_q = 0.0
            initial_out[base_q + i_q] = rebind[initial_out.element_type](new_q)
            i_q += Int(threads)
        barrier()

        i = Int(tid)
        while i < n_basis:
            var val = rebind[GPU_IC_SCALAR](initial_out[base_q + i])
            if val < 0.0:
                initial_out[base_q + i] = rebind[initial_out.element_type](
                    GPU_IC_SCALAR(0.0)
                )
            i += Int(threads)
        barrier()
        if tid == 0:
            var sum_q: GPU_IC_SCALAR = 0.0
            for j in range(n_basis):
                sum_q = sum_q + rebind[GPU_IC_SCALAR](initial_out[base_q + j])
            if sum_q > 0.0:
                var inv_sum: GPU_IC_SCALAR = 1.0 / sum_q
                for j in range(n_basis):
                    initial_out[base_q + j] = rebind[initial_out.element_type](
                        rebind[GPU_IC_SCALAR](initial_out[base_q + j]) * inv_sum
                    )
