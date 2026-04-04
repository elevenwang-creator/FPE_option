"""GPU batch ODE integration kernel for FPE solver.

Uses explicit Euler: q_{n+1} = q_n + dt * (-M⁻¹K @ q_n)
Each thread handles one row of the matrix-vector multiply for one batch element.

Kernel constraints:
- nonraising (no exceptions on GPU)
- UnsafePointer for all parameters (required by compile_function/enqueue_function)
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx


def batch_euler_step(
    mat_ptr: UnsafePointer[Float64, MutAnyOrigin],
    q_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n: Int,
    dt: Float64,
):
    """GPU explicit Euler ODE step kernel.

    Computes dq/dt = mat @ q, updates q in-place: q += dt * mat @ q.
    One thread per row.

    Args:
        mat_ptr: Pointer to n×n matrix in row-major order.
        q_ptr: Pointer to n-element state vector (modified in-place).
        n: Matrix/vector dimension.
        dt: Time step.
    """
    var row = global_idx.x
    if row < UInt(n):
        var r = Int(row)

        # Compute dq[r] = mat[r,:] @ q
        var dq_r: Float64 = 0.0
        var j = 0
        while j < n:
            dq_r += mat_ptr[r * n + j] * q_ptr[j]
            j += 1

        # Euler update: q[r] += dt * dq[r]
        q_ptr[r] = q_ptr[r] + dt * dq_r
