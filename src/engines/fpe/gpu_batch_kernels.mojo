"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Double-buffered: reads from q_in, writes to q_out — no race conditions.

Kernel constraints:
- nonraising (no exceptions on GPU)
- UnsafePointer for all parameters (required by compile_function/enqueue_function)
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx


def batch_euler_step(
    mat_ptr: UnsafePointer[Float64, MutAnyOrigin],
    q_in: UnsafePointer[Float64, MutAnyOrigin],
    q_out: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
    dt: Float64,
):
    """GPU explicit Euler ODE step kernel with double-buffering.

    Computes dq/dt = mat @ q_in, writes q_out = q_in + dt * dq.
    One thread per row. No race conditions since read/write are separate buffers.

    Args:
        mat_ptr: Pointer to n×n matrix in row-major order.
        q_in: Pointer to input state vector (read-only).
        q_out: Pointer to output state vector (write-only).
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
            dq_r += mat_ptr[r * n + j] * q_in[j]
            j += 1

        # Euler update: q_out[r] = q_in[r] + dt * dq[r] (write to output buffer)
        q_out[r] = q_in[r] + dt * dq_r
