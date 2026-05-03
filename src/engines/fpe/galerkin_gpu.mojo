"""GPU kernel for FPE Galerkin matrix assembly."""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT, GPU_MAX_N

comptime GPU_GAL_DTYPE = GPU_DTYPE
comptime GPU_GAL_VEC = GPU_VEC_LAYOUT
comptime GPU_GAL_MAX_N = GPU_MAX_N
comptime GPU_GAL_SCALAR = Scalar[GPU_GAL_DTYPE]


def spmatrix_gpu_kernel(
    spmatrix_out: LayoutTensor[GPU_GAL_DTYPE, GPU_GAL_VEC, MutAnyOrigin],
    phi_in: LayoutTensor[GPU_GAL_DTYPE, GPU_GAL_VEC, MutAnyOrigin],
    weights_in: LayoutTensor[GPU_GAL_DTYPE, GPU_GAL_VEC, MutAnyOrigin],
    params: LayoutTensor[GPU_GAL_DTYPE, GPU_GAL_VEC, MutAnyOrigin],
    n_basis: Int,
    n_points: Int,
):
    """Assemble dense system matrix -M^{-1}K for the FPE ODE on GPU."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base_mat = b * n_basis * n_basis
    var base_phi = b * GPU_GAL_MAX_N * GPU_GAL_MAX_N
    var base_w = b * n_points

    var r = Float64(rebind[GPU_GAL_SCALAR](params[b * 12 + 4]))
    var sigma = Float64(rebind[GPU_GAL_SCALAR](params[b * 12 + 2]))
    var rho = Float64(rebind[GPU_GAL_SCALAR](params[b * 12 + 3]))

    var idx = Int(tid)
    while idx < n_basis * n_basis:
        var row = idx / n_basis
        var col = idx % n_basis
        var sum: Float64 = 0.0
        for k in range(n_points):
            var w_val = Float64(rebind[GPU_GAL_SCALAR](weights_in[base_w + k]))
            var phi_ki = Float64(rebind[GPU_GAL_SCALAR](phi_in[base_phi + k * GPU_GAL_MAX_N + row]))
            var phi_kj = Float64(rebind[GPU_GAL_SCALAR](phi_in[base_phi + k * GPU_GAL_MAX_N + col]))
            sum += w_val * phi_ki * phi_kj
        spmatrix_out[base_mat + idx] = GPU_GAL_SCALAR(sum)
        idx += Int(threads)