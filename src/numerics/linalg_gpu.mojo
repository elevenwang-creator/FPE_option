"""GPU kernels for linear algebra: LU decomposition and solve."""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_LA_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_LA_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_LA_SCALAR = Scalar[GPU_LA_DTYPE]


def lu_decompose_gpu_kernel(
    lu_out: LayoutTensor[GPU_LA_DTYPE, GPU_LA_VEC, MutAnyOrigin],
    A_in: LayoutTensor[GPU_LA_DTYPE, GPU_LA_VEC, MutAnyOrigin],
    n: Int,
):
    """LU decomposition with partial pivoting for small dense matrices."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = b * n * n

    if Int(tid) == 0:
        # Copy A to lu_out with partial pivoting
        for i in range(n):
            for j in range(n):
                lu_out[base + i * n + j] = A_in[base + i * n + j]

        for k in range(n):
            var pivot = k
            var max_val: Float64 = 0.0
            for i in range(k, n):
                var val = Float64(rebind[GPU_LA_SCALAR](lu_out[base + i * n + k]))
                if val < 0.0:
                    val = 0.0 - val
                if val > max_val:
                    max_val = val
                    pivot = i
            if pivot != k:
                for j in range(n):
                    var tmp = lu_out[base + k * n + j]
                    lu_out[base + k * n + j] = lu_out[base + pivot * n + j]
                    lu_out[base + pivot * n + j] = tmp

        for k in range(n):
            var diag_f: Float64 = Float64(rebind[GPU_LA_SCALAR](lu_out[base + k * n + k]))
            if diag_f == 0.0:
                lu_out[base + k * n + k] = GPU_LA_SCALAR(1.0)
                diag_f = 1.0
            for i in range(k + 1, n):
                var factor = Float64(rebind[GPU_LA_SCALAR](lu_out[base + i * n + k])) / diag_f
                lu_out[base + i * n + k] = GPU_LA_SCALAR(factor)
                for j in range(k + 1, n):
                    var val = Float64(rebind[GPU_LA_SCALAR](lu_out[base + i * n + j]))
                    val = val - factor * Float64(rebind[GPU_LA_SCALAR](lu_out[base + k * n + j]))
                    lu_out[base + i * n + j] = GPU_LA_SCALAR(val)

    var i = Int(tid)
    while i < n * n:
        i += Int(block_dim.x)


def lu_solve_gpu_kernel(
    x_out: LayoutTensor[GPU_LA_DTYPE, GPU_LA_VEC, MutAnyOrigin],
    lu_in: LayoutTensor[GPU_LA_DTYPE, GPU_LA_VEC, MutAnyOrigin],
    b_in: LayoutTensor[GPU_LA_DTYPE, GPU_LA_VEC, MutAnyOrigin],
    n: Int,
):
    """Solve LUx = b on GPU using forward/back substitution."""
    var b_idx = Int(block_idx.x)
    var base_mat = b_idx * n * n
    var base_vec = b_idx * n

    if Int(thread_idx.x) == 0:
        for i in range(n):
            x_out[base_vec + i] = b_in[base_vec + i]
        for i in range(n):
            for j in range(i):
                var val = Float64(rebind[GPU_LA_SCALAR](x_out[base_vec + i]))
                val = val - Float64(rebind[GPU_LA_SCALAR](lu_in[base_mat + i * n + j])) * Float64(rebind[GPU_LA_SCALAR](x_out[base_vec + j]))
                x_out[base_vec + i] = GPU_LA_SCALAR(val)
        for rev in range(n):
            var i = n - 1 - rev
            var s = Float64(rebind[GPU_LA_SCALAR](x_out[base_vec + i]))
            for j in range(i + 1, n):
                s = s - Float64(rebind[GPU_LA_SCALAR](lu_in[base_mat + i * n + j])) * Float64(rebind[GPU_LA_SCALAR](x_out[base_vec + j]))
            var diag = Float64(rebind[GPU_LA_SCALAR](lu_in[base_mat + i * n + i]))
            if diag != 0.0:
                x_out[base_vec + i] = GPU_LA_SCALAR(s / diag)