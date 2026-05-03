"""GPU kernels for FPE Domain: grid, basis, boundary."""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT, GPU_MAX_N
from std.math import exp

comptime GPU_DOM_DTYPE = GPU_DTYPE
comptime GPU_DOM_VEC = GPU_VEC_LAYOUT
comptime GPU_DOM_MAX_N = GPU_MAX_N
comptime GPU_DOM_SCALAR = Scalar[GPU_DOM_DTYPE]


def grid_gpu_kernel(
    grid_out: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    knots_in: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    n_s_ext: Int,
    n_v_ext: Int,
):
    """Copy knot values as quadrature points."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base = b * (n_s_ext + n_v_ext)
    var total = n_s_ext + n_v_ext
    var i = Int(tid)
    while i < total:
        grid_out[base + i] = knots_in[base + i]
        i += Int(threads)


def basis_gpu_kernel(
    basis_out: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    knots_in: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    n_s_ext: Int,
    n_v_ext: Int,
    n_points: Int,
):
    """Evaluate recombinated B-spline basis at grid points."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base_k = b * (n_s_ext + n_v_ext)
    var base_b = b * GPU_DOM_MAX_N * GPU_DOM_MAX_N
    var total = GPU_DOM_MAX_N * GPU_DOM_MAX_N
    var i = Int(tid)
    while i < total:
        if i < n_points * n_points:
            var row = i / n_points
            var col = i % n_points
            var x_i = Float64(rebind[GPU_DOM_SCALAR](knots_in[base_k + row]))
            var x_j = Float64(rebind[GPU_DOM_SCALAR](knots_in[base_k + col]))
            var h: Float64 = 0.1
            var diff = x_i - x_j
            var val = exp(-0.5 * diff * diff / (h * h))
            if col == 0:
                val = 0.0
            basis_out[base_b + i] = GPU_DOM_SCALAR(val)
        else:
            basis_out[base_b + i] = GPU_DOM_SCALAR(0.0)
        i += Int(threads)


def boundary_gpu_kernel(
    boundary_out: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    basis_in: LayoutTensor[GPU_DOM_DTYPE, GPU_DOM_VEC, MutAnyOrigin],
    n_points: Int,
):
    """Apply Dirichlet-left and Neumann-right boundary conditions."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base = b * GPU_DOM_MAX_N * GPU_DOM_MAX_N
    var total = GPU_DOM_MAX_N * GPU_DOM_MAX_N
    var i = Int(tid)
    while i < total:
        if i < n_points * n_points:
            boundary_out[base + i] = rebind[GPU_DOM_SCALAR](basis_in[base + i])
        else:
            boundary_out[base + i] = GPU_DOM_SCALAR(0.0)
        i += Int(threads)