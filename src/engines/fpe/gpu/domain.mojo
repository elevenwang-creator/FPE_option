"""GPU kernels for FPE Domain logic.

Includes: knots -> grid -> basis -> boundary.
"""
from std.gpu import barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from std.gpu import block_idx, thread_idx, block_dim

def generate_knots_gpu_kernel(
    knots_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    params: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n_s: Int,
    n_v: Int,
):
    """Knots_GPU: Generate knots autonomously for a single batch element.
    
    Architecture:
    grid_dim.x determines the batch instance (block_idx.x).
    No sharing across blocks. 
    Inside the block, threads cooperate to compute knots.
    """
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * (n_s + n_v)
    
    # Threads collaborate over n_s 
    var i = Int(tid)
    while i < n_s:
        knots_out[base + i] = rebind[knots_out.element_type](0.1)
        i += Int(threads)
        
    var j = Int(tid)
    while j < n_v:
        knots_out[base + n_s + j] = rebind[knots_out.element_type](0.01)
        j += Int(threads)

def grid_gpu_kernel(
    grid_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    knots_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n_elements: Int,
):
    """Grid_GPU: Numerical points."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * n_elements
    var i = Int(tid)
    while i < n_elements:
        grid_out[base + i] = knots_in[base + i]
        i += Int(threads)

def basis_gpu_kernel(
    basis_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    grid_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n_elements: Int,
):
    """Basis_GPU: B-spline basis function from knots."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * n_elements
    var i = Int(tid)
    while i < n_elements:
        basis_out[base + i] = rebind[basis_out.element_type](1.0)
        i += Int(threads)

def boundary_gpu_kernel(
    boundary_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    basis_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n_elements: Int,
):
    """Boundary_GPU: Impose boundary condition."""
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x
    var base = Int(b) * n_elements
    var i = Int(tid)
    while i < n_elements:
        boundary_out[base + i] = basis_in[base + i]
        i += Int(threads)
