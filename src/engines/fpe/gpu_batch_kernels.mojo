"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

Metal backend (Apple Silicon): Uses Float32 — Metal doesn't support Float64 kernels.
CUDA/HIP backend: Can use Float64.

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters with concrete comptime layouts
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from layout import Layout, LayoutTensor


# Concrete comptime layouts for GPU kernels
# These must match the layouts used in gpu_batch_executor.mojo
comptime KERNEL_MAX_N = 64
comptime KERNEL_MAT_LAYOUT = Layout.row_major(KERNEL_MAX_N, KERNEL_MAX_N)
comptime KERNEL_VEC_LAYOUT = Layout.row_major(KERNEL_MAX_N)


def batch_euler_step(
    mat: LayoutTensor[DType.float32, KERNEL_MAT_LAYOUT, MutAnyOrigin],
    q_in: LayoutTensor[DType.float32, KERNEL_VEC_LAYOUT, MutAnyOrigin],
    q_out: LayoutTensor[DType.float32, KERNEL_VEC_LAYOUT, MutAnyOrigin],
    n: Int,
    dt: Float32,
):
    """GPU explicit Euler ODE step kernel with double-buffering.

    Computes dq/dt = mat @ q_in, writes q_out = q_in + dt * dq.
    One thread per row. No race conditions since read/write are separate buffers.

    Uses Float32 for Metal compatibility.

    Args:
        mat: n×n matrix in row-major order (LayoutTensor, Float32).
        q_in: Input state vector (read-only, LayoutTensor, Float32).
        q_out: Output state vector (write-only, LayoutTensor, Float32).
        n: Matrix/vector dimension.
        dt: Time step (Float32).
    """
    var row = global_idx.x
    if row < n:
        var r = Int(row)

        # Compute dq[r] = mat[r,:] @ q_in (read from input buffer)
        var dq_r: Scalar[DType.float32] = 0.0
        var j = 0
        while j < n:
            var mat_val = rebind[Scalar[DType.float32]](mat[r, j])
            var q_val = rebind[Scalar[DType.float32]](q_in[j])
            dq_r = dq_r + mat_val * q_val
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        var dt_scalar = rebind[Scalar[DType.float32]](dt)
        var q_in_val = rebind[Scalar[DType.float32]](q_in[r])
        q_out[r] = rebind[q_out.element_type](q_in_val + dt_scalar * dq_r)
