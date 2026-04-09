"""GPU kernels for FPE Integration logic.

Includes: integrate.
"""
from std.gpu import barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from std.gpu import block_idx, thread_idx, block_dim

def integrate_gpu_kernel(
    price_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    radau_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """Integrate_GPU: option pricing integration over PDF."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        price_out[base + i] = radau_in[base + i]
        i += Int(threads)
