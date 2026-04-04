"""Host-side GPU batch execution for FPE ODE integration.

Manages GPU device context, buffer transfers, kernel compilation/enqueue,
and result retrieval for batch parallel ODE solving.

Backend detection:
- Apple Silicon → Metal via DeviceContext(api="metal")
- Other GPU → Generic DeviceContext()
- No GPU → Falls back to CPU explicit Euler

Usage:
    from engines.fpe.gpu_batch_executor import gpu_batch_solve
    results = gpu_batch_solve(
        neg_M_inv_K=matrix, q0=initial_states, t_end=0.1,
        batch_size=2, num_steps=1000,
    )
"""

from engines.fpe.gpu_batch_kernels import batch_euler_step
from gpu_utils.detect import is_gpu_available, get_device_api_name
from gpu_utils.host_utils import create_device_context
from sparse.csr import CSRMatrix
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import global_idx
from std.memory import UnsafePointer, MutAnyOrigin


def _csr_to_dense_float(A: CSRMatrix[DType.float64]) -> List[List[Float64]]:
    """Convert CSR to dense List[List[Float64]]."""
    var d = A.to_dense()
    var out: List[List[Float64]] = []
    for i in range(A.nrows):
        var row: List[Float64] = []
        for j in range(A.ncols):
            row.append(d[i][j])
        out.append(row^)
    return out^


def _project_nonnegative(mut states: List[List[Float64]]):
    """Project ODE solution states to non-negative values and normalize."""
    for i in range(len(states)):
        var row_sum = 0.0
        for j in range(len(states[i])):
            if states[i][j] < 0.0:
                states[i][j] = 0.0
            row_sum += states[i][j]
        if row_sum > 0.0:
            for j in range(len(states[i])):
                states[i][j] = states[i][j] / row_sum


def _cpu_euler_solve(
    neg_M_inv_K: List[List[Float64]],
    q0: List[Float64],
    t_end: Float64,
    num_steps: Int,
) -> List[Float64]:
    """CPU explicit Euler ODE solver."""
    var n = len(q0)
    var dt = t_end / Float64(num_steps)
    var q = q0.copy()

    var step = 0
    while step < num_steps:
        var dq: List[Float64] = []
        for i in range(n):
            var acc = 0.0
            for j in range(n):
                acc += neg_M_inv_K[i][j] * q[j]
            dq.append(acc)
        for i in range(n):
            q[i] = q[i] + dt * dq[i]
        step += 1

    return q^


def gpu_batch_solve(
    neg_M_inv_K: CSRMatrix[DType.float64],
    q0: List[Float64],
    t_end: Float64,
    batch_size: Int,
    num_steps: Int = 1000,
) raises -> List[List[Float64]]:
    """Solve batch of ODE systems on GPU using explicit Euler.

    Args:
        neg_M_inv_K: Pre-computed -M⁻¹K matrix (shared across batch)
        q0: Initial state vector (replicated for each batch element)
        t_end: Final integration time
        batch_size: Number of parallel ODE systems
        num_steps: Number of Euler time steps

    Returns:
        State vectors at final time, shape [batch_size, num_states]
    """
    var n = len(q0)

    # Convert CSR to dense for GPU transfer
    var mat_dense = _csr_to_dense_float(neg_M_inv_K)
    var mat_flat: List[Float64] = []
    for i in range(n):
        for j in range(n):
            mat_flat.append(mat_dense[i][j])

    # GPU execution if available
    comptime if is_gpu_available():
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var q = _gpu_single_solve(mat_flat, q0, t_end, num_steps, n)
            results.append(q^)
        _project_nonnegative(results)
        return results^
    else:
        # CPU fallback
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var q = _cpu_euler_solve(mat_dense, q0, t_end, num_steps)
            results.append(q^)
        _project_nonnegative(results)
        return results^


def _gpu_single_solve(
    mat_flat: List[Float64],
    q0: List[Float64],
    t_end: Float64,
    num_steps: Int,
    n: Int,
) raises -> List[Float64]:
    """Solve a single ODE system on GPU."""
    var dt = t_end / Float64(num_steps)

    var ctx = create_device_context()

    # Create host buffers
    var mat_host = ctx.enqueue_create_host_buffer[DType.float64](n * n)
    var q_host = ctx.enqueue_create_host_buffer[DType.float64](n)
    ctx.synchronize()

    # Copy data to host buffers
    for i in range(n * n):
        mat_host[i] = Float64(mat_flat[i])
    for i in range(n):
        q_host[i] = Float64(q0[i])

    # Create device buffers
    var mat_dev = ctx.enqueue_create_buffer[DType.float64](n * n)
    var q_dev = ctx.enqueue_create_buffer[DType.float64](n)

    # Copy to device
    ctx.enqueue_copy(dst_buf=mat_dev, src_buf=mat_host)
    ctx.enqueue_copy(dst_buf=q_dev, src_buf=q_host)
    ctx.synchronize()

    # Launch GPU kernel
    var step = 0
    while step < num_steps:
        ctx.enqueue_function[batch_euler_step, batch_euler_step](
            mat_dev, q_dev, n, dt,
            grid_dim=n, block_dim=256,
        )
        step += 1

    # Copy results back
    ctx.enqueue_copy(dst_buf=q_host, src_buf=q_dev)
    ctx.synchronize()

    # Read results
    var q_out: List[Float64] = []
    for i in range(n):
        q_out.append(Float64(q_host[i]))
    return q_out^
