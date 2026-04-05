"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

NOTE: Requires working Metal/CUDA/HIP compiler. On M1 Pro with Mojo v0.26.2,
the Metal compiler may fail with "Metal Compiler failed to compile metallib".
This is a known Mojo toolchain issue, not a code bug.

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters (Mojo v0.26.2 requirement)
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from layout import Layout, LayoutTensor


def batch_euler_step[
    mat_layout: Layout,
    vec_layout: Layout,
](
    mat: LayoutTensor[DType.float64, mat_layout, MutAnyOrigin],
    q_in: LayoutTensor[DType.float64, vec_layout, MutAnyOrigin],
    q_out: LayoutTensor[DType.float64, vec_layout, MutAnyOrigin],
    n: Int,
    dt: Float64,
):
    """GPU explicit Euler ODE step kernel with double-buffering.

    Computes dq/dt = mat @ q_in, writes q_out = q_in + dt * dq.
    One thread per row. No race conditions since read/write are separate buffers.

    Args:
        mat: n×n matrix in row-major order (LayoutTensor).
        q_in: Input state vector (read-only, LayoutTensor).
        q_out: Output state vector (write-only, LayoutTensor).
        n: Matrix/vector dimension.
        dt: Time step.
    """
    var row = global_idx.x
    if row < UInt(n):
        var r = Int(row)

        # Compute dq[r] = mat[r,:] @ q_in (read from input buffer)
        var dq_r: Float64 = 0.0
        var j = 0
        while j < n:
            dq_r += Float64(mat[r * n + j]) * Float64(q_in[j])
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        q_out[r] = rebind[q_out.element_type](Scalar[DType.float64](Float64(q_in[r]) + dt * dq_r))
