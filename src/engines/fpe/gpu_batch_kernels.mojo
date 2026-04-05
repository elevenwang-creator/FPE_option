"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

Cross-platform design:
- Uses dtype management module for automatic backend selection
- Metal: Float32, CUDA/HIP: Float64
- Layouts sized appropriately per backend
- Single source, compiled differently per target

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters with concrete comptime layouts
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from layout import LayoutTensor
from gpu_utils.dtype import (
    METAL_DTYPE, METAL_MAT_LAYOUT, METAL_VEC_LAYOUT, METAL_MAX_N,
    CUDA_DTYPE, CUDA_MAT_LAYOUT, CUDA_VEC_LAYOUT, CUDA_MAX_N,
)
from std.sys import has_apple_gpu_accelerator


# Backend-specific types - automatically selected via ternary expressions
# This is the key to "write once, deploy anywhere" - same code, different types per backend
comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAT_LAYOUT = METAL_MAT_LAYOUT if has_apple_gpu_accelerator() else CUDA_MAT_LAYOUT
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N


def batch_euler_step(
    mat: LayoutTensor[GPU_DTYPE, GPU_MAT_LAYOUT, MutAnyOrigin],
    q_in: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    q_out: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n: Int,
    dt: Scalar[GPU_DTYPE],
):
    """GPU explicit Euler ODE step kernel with double-buffering.

    Computes dq/dt = mat @ q_in, writes q_out = q_in + dt * dq.
    One thread per row. No race conditions since read/write are separate buffers.

    Cross-platform: dtype and layouts selected automatically per backend via
    ternary expressions using constants from gpu_utils.dtype module.

    Args:
        mat: n×n matrix in row-major order (LayoutTensor, backend dtype).
        q_in: Input state vector (read-only, LayoutTensor, backend dtype).
        q_out: Output state vector (write-only, LayoutTensor, backend dtype).
        n: Matrix/vector dimension.
        dt: Time step (backend dtype).
    """
    var row = global_idx.x
    if row < n:
        var r = Int(row)

        # Compute dq[r] = mat[r,:] @ q_in (read from input buffer)
        var dq_r: Scalar[GPU_DTYPE] = 0.0
        var j = 0
        while j < n:
            var mat_val = rebind[Scalar[GPU_DTYPE]](mat[r, j])
            var q_val = rebind[Scalar[GPU_DTYPE]](q_in[j])
            dq_r = dq_r + mat_val * q_val
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        var dt_scalar = rebind[Scalar[GPU_DTYPE]](dt)
        var q_in_val = rebind[Scalar[GPU_DTYPE]](q_in[r])
        q_out[r] = rebind[q_out.element_type](q_in_val + dt_scalar * dq_r)
