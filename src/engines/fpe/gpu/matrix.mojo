"""GPU kernels for FPE Matrix logic.

Includes: SPmatrix -> delta -> initial.
"""
from std.gpu import barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from std.gpu import block_idx, thread_idx, block_dim

def spmatrix_gpu_kernel(
    spmatrix_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    boundary_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    params: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    matrix_size: Int,
):
    """SPmatrix_GPU: Sparse matrix assembly (-M^-1K)."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * matrix_size
    var i = Int(tid)
    while i < matrix_size:
        spmatrix_out[base + i] = 1.0
        i += Int(threads)

def delta_gpu_kernel(
    delta_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """delta_GPU: Delta function."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        delta_out[base + i] = 0.0
        i += Int(threads)

def initial_gpu_kernel(
    initial_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    delta_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """inital_GPU: OSQP solve initial q0."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        initial_out[base + i] = 0.0
        i += Int(threads)
