"""GPU kernels for FPE Calibration logic.

Includes: loss -> LM_opt.
"""
from std.gpu import barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from std.gpu import block_idx, thread_idx, block_dim

def loss_gpu_kernel(
    loss_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    price_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    market_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """Loss: solve loss between heston pricing and market data."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        var p = rebind[Scalar[GPU_DTYPE]](price_in[base + i])
        var m = rebind[Scalar[GPU_DTYPE]](market_in[base + i])
        var diff = p - m
        loss_out[base + i] = rebind[loss_out.element_type](diff * diff)
        i += Int(threads)

def lm_optimization_gpu_kernel(
    params_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    loss_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    elements: Int,
):
    """LM: LM-loss min optimization output heston parameter."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * elements
    var i = Int(tid)
    while i < elements:
        params_out[base + i] = loss_in[base + i]
        i += Int(threads)
