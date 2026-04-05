"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

Cross-platform design:
- Uses dtype management module for backend configuration constants
- Metal: Float32 with 64 max size
- CUDA/HIP: Float64 with 256 max size
- Single source, compiled differently per target via comptime constants

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters with concrete comptime layouts
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_MAX_N, METAL_VEC_LAYOUT, CUDA_MAX_N, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator


# Backend-specific types - selected at compile time
# Uses constants from dtype management module for consistency
comptime if has_apple_gpu_accelerator():
    # Metal backend: Float32, smaller max size
    comptime KERNEL_DTYPE = DType.float32
    comptime KERNEL_MAT_LAYOUT = Layout.row_major(METAL_MAX_N, METAL_MAX_N)
    comptime KERNEL_VEC_LAYOUT = METAL_VEC_LAYOUT
    comptime KERNEL_MAX_N = METAL_MAX_N
else:
    # CUDA/HIP/CPU backend: Float64, larger max size
    comptime KERNEL_DTYPE = DType.float64
    comptime KERNEL_MAT_LAYOUT = Layout.row_major(CUDA_MAX_N, CUDA_MAX_N)
    comptime KERNEL_VEC_LAYOUT = CUDA_VEC_LAYOUT
    comptime KERNEL_MAX_N = CUDA_MAX_N


def batch_euler_step(
    mat: LayoutTensor[KERNEL_DTYPE, KERNEL_MAT_LAYOUT, MutAnyOrigin],
    q_in: LayoutTensor[KERNEL_DTYPE, KERNEL_VEC_LAYOUT, MutAnyOrigin],
    q_out: LayoutTensor[KERNEL_DTYPE, KERNEL_VEC_LAYOUT, MutAnyOrigin],
    n: Int,
    dt: Scalar[KERNEL_DTYPE],
):
    """GPU explicit Euler ODE step kernel with double-buffering.

    Computes dq/dt = mat @ q_in, writes q_out = q_in + dt * dq.
    One thread per row. No race conditions since read/write are separate buffers.

    Cross-platform: dtype and layouts selected automatically per backend via
    comptime if using constants from gpu_utils.dtype module.

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
        var dq_r: Scalar[KERNEL_DTYPE] = 0.0
        var j = 0
        while j < n:
            var mat_val = rebind[Scalar[KERNEL_DTYPE]](mat[r, j])
            var q_val = rebind[Scalar[KERNEL_DTYPE]](q_in[j])
            dq_r = dq_r + mat_val * q_val
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        var dt_scalar = rebind[Scalar[KERNEL_DTYPE]](dt)
        var q_in_val = rebind[Scalar[KERNEL_DTYPE]](q_in[r])
        q_out[r] = rebind[q_out.element_type](q_in_val + dt_scalar * dq_r)
