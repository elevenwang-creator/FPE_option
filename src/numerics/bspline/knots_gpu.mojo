"""GPU kernel for B-spline knot generation."""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_KNOT_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_KNOT_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_KNOT_SCALAR = Scalar[GPU_KNOT_DTYPE]


def generate_knots_gpu_kernel(
    knots_out: LayoutTensor[GPU_KNOT_DTYPE, GPU_KNOT_VEC, MutAnyOrigin],
    weights_out: LayoutTensor[GPU_KNOT_DTYPE, GPU_KNOT_VEC, MutAnyOrigin],
    params: LayoutTensor[GPU_KNOT_DTYPE, GPU_KNOT_VEC, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
    degree: Int,
):
    """Generate non-uniform B-spline knots and quadrature weights on GPU.

    Matches CPU GenerateKnots + FPEDomain:
    - Chebyshev-clustered internal knots centered at S0/V0
    - Degree-fold boundary padding
    - Trapezoidal quadrature weights

    params layout per batch: [kappa, theta, sigma, rho, r, T, S0, V0, S_min, S_max, V_min, V_max] (12 fields)
    """
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)

    var s0 = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 6]))
    var v0 = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 7]))
    var s_min = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 8]))
    var s_max = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 9]))
    var v_min = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 10]))
    var v_max = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 11]))
    var sigma = Float64(rebind[GPU_KNOT_SCALAR](params[b * 12 + 2]))

    var internal_num_s = n_s - 2 * degree
    if internal_num_s < 2:
        internal_num_s = 2
    var internal_num_v = n_v - 2 * degree
    if internal_num_v < 2:
        internal_num_v = 2
    var n_s_ext = n_s + 2 * degree
    var n_v_ext = n_v + 2 * degree
    var base_k = b * (n_s_ext + n_v_ext)
    var base_w = b * (n_s_ext + n_v_ext)

    var i = tid
    while i < n_s_ext:
        var val: Float64 = 0.0
        var w_val: Float64 = 0.0
        if i < degree:
            val = s_min
            w_val = 0.5 * (s_max - s_min) / Float64(n_s - 1)
        elif i >= n_s_ext - degree:
            val = s_max
            w_val = 0.5 * (s_max - s_min) / Float64(n_s - 1)
        else:
            var idx = i - degree
            var t = Float64(idx) / Float64(internal_num_s + 1)
            var s_std = sigma * 0.1
            if s_std < 0.01:
                s_std = 0.01
            var x_min = s0 - 4.5 * s_std
            var x_max = s0 + 4.5 * s_std
            var lb = (x_min - s_min) / (s_max - s_min)
            var ub = (x_max - s_min) / (s_max - s_min)
            if lb < 0.0:
                lb = 0.0
            if ub > 1.0:
                ub = 1.0
            var t_mapped = lb + t * (ub - lb)
            if t_mapped < 0.0:
                t_mapped = 0.0
            if t_mapped > 1.0:
                t_mapped = 1.0
            val = s_min + t_mapped * (s_max - s_min)
            if idx == 0 or idx == internal_num_s - 1:
                w_val = 0.5 * (s_max - s_min) / Float64(internal_num_s + 1)
            else:
                w_val = (s_max - s_min) / Float64(internal_num_s + 1)
        knots_out[base_k + i] = GPU_KNOT_SCALAR(val)
        weights_out[base_w + i] = GPU_KNOT_SCALAR(w_val)
        i += Int(threads)

    var j = tid
    while j < n_v_ext:
        var val_v: Float64 = 0.0
        var w_val_v: Float64 = 0.0
        if j < degree:
            val_v = v_min
            w_val_v = 0.5 * (v_max - v_min) / Float64(n_v - 1)
        elif j >= n_v_ext - degree:
            val_v = v_max
            w_val_v = 0.5 * (v_max - v_min) / Float64(n_v - 1)
        else:
            var idx = j - degree
            var t = Float64(idx) / Float64(internal_num_v + 1)
            var v_std = 0.1 * sigma
            if v_std < 0.001:
                v_std = 0.001
            var x_min = v0 - 4.5 * v_std
            var x_max = v0 + 4.5 * v_std
            var lb = (x_min - v_min) / (v_max - v_min)
            var ub = (x_max - v_min) / (v_max - v_min)
            if lb < 0.0:
                lb = 0.0
            if ub > 1.0:
                ub = 1.0
            var t_mapped = lb + t * (ub - lb)
            if t_mapped < 0.0:
                t_mapped = 0.0
            if t_mapped > 1.0:
                t_mapped = 1.0
            val_v = v_min + t_mapped * (v_max - v_min)
            if idx == 0 or idx == internal_num_v - 1:
                w_val_v = 0.5 * (v_max - v_min) / Float64(internal_num_v + 1)
            else:
                w_val_v = (v_max - v_min) / Float64(internal_num_v + 1)
        knots_out[base_k + n_s_ext + j] = GPU_KNOT_SCALAR(val_v)
        weights_out[base_w + n_s_ext + j] = GPU_KNOT_SCALAR(w_val_v)
        j += Int(threads)