"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

Metal backend (Apple Silicon): Uses Float32 — Metal doesn't support Float64 kernels.
CUDA/HIP backend: Can use Float64.

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from layout import Layout, LayoutTensor


def batch_euler_step[
    mat_layout: Layout,
    vec_layout: Layout,
](
    mat: LayoutTensor[DType.float32, mat_layout, MutAnyOrigin],
    q_in: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
    q_out: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
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
        var dq_r: Float32 = 0.0
        var j = 0
        while j < n:
            dq_r += mat[r * n + j] * q_in[j]
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        q_out[r] = q_in[r] + dt * dq_r
