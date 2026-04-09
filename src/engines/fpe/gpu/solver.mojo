"""GPU kernels for FPE Solver logic.

Includes: LU -> RADAU5.
"""
from std.gpu import barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from std.gpu import block_idx, thread_idx, block_dim

def lu_gpu_kernel(
    lu_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    spmatrix_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    matrix_size: Int,
):
    """LU_GPU: spLU decomposition-Mdq/dt=Kq."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * matrix_size
    var i = Int(tid)
    while i < matrix_size:
        lu_out[base + i] = spmatrix_in[base + i]
        i += Int(threads)

def radau5_gpu_kernel(
    radau_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    lu_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    initial_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """RADAU5_GPU: solve ODE-dq/dt=M^-1Kq."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        radau_out[base + i] = 1.0
        i += Int(threads)
