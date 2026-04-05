"""Host-side GPU batch execution for FPE ODE integration.

Manages GPU device context, buffer transfers, kernel launch,
and result retrieval for batch parallel ODE solving.

Backend detection:
- Apple Silicon → Metal via DeviceContext(api="metal")
- Other GPU → Generic DeviceContext()
- No GPU → Falls back to CPU explicit Euler

Metal uses Float32 (Metal doesn't support Float64 kernels).
Requires: --target-accelerator=metal:1 (or equivalent for CUDA/HIP)

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
from layout import Layout, LayoutTensor
from numerics.utils import zeros


# Comptime layout sizes for GPU kernels (Metal requires comptime layouts)
comptime GPU_MAX_N = 64
comptime GPU_MAT_LAYOUT = Layout.row_major(GPU_MAX_N, GPU_MAX_N)
comptime GPU_VEC_LAYOUT = Layout.row_major(GPU_MAX_N)


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
    """CPU explicit Euler ODE solver with double-buffering."""
    var n = len(q0)
    var dt = t_end / Float64(num_steps)
    var q = q0.copy()
    var q_next = zeros(n)

    var step = 0
    while step < num_steps:
        for i in range(n):
            var acc = 0.0
            for j in range(n):
                acc += neg_M_inv_K[i][j] * q[j]
            q_next[i] = q[i] + dt * acc
        for i in range(n):
            q[i] = q_next[i]
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

    Uses runtime dispatch for GPU detection.
    Each batch element gets its own double-buffered device buffers.
    Metal backend uses Float32 (Metal doesn't support Float64 kernels).

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
    var mat_dense = _csr_to_dense_float(neg_M_inv_K)
    var dt = t_end / Float64(num_steps)

    if is_gpu_available():
        return _gpu_batch_solve_impl(mat_dense, q0, t_end, batch_size, num_steps, n, dt)
    else:
        # CPU fallback
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var q = _cpu_euler_solve(mat_dense, q0, t_end, num_steps)
            results.append(q^)
        _project_nonnegative(results)
        return results^


def _gpu_batch_solve_impl(
    mat_dense: List[List[Float64]],
    q0: List[Float64],
    t_end: Float64,
    batch_size: Int,
    num_steps: Int,
    n: Int,
    dt: Float64,
) raises -> List[List[Float64]]:
    """Actual GPU batch solve implementation with double-buffering.

    Uses Float32 for Metal compatibility. Each batch element gets its own
    pair of device buffers (q_in, q_out) with no race conditions.
    """
    var ctx = create_device_context()

    # Convert matrix to Float32 for Metal
    var mat_flat: List[Float32] = []
    for i in range(n):
        for j in range(n):
            mat_flat.append(Float32(mat_dense[i][j]))

    # Convert q0 to Float32
    var q0_flat: List[Float32] = []
    for i in range(n):
        q0_flat.append(Float32(q0[i]))

    # Create host buffers (padded to comptime size)
    var mat_host = ctx.enqueue_create_host_buffer[DType.float32](GPU_MAX_N * GPU_MAX_N)
    var q_in_host = ctx.enqueue_create_host_buffer[DType.float32](GPU_MAX_N)
    var q_out_host = ctx.enqueue_create_host_buffer[DType.float32](GPU_MAX_N)
    ctx.synchronize()

    # Copy matrix to host buffer
    for i in range(n * n):
        mat_host[i] = mat_flat[i]

    # Create device buffers
    var mat_dev = ctx.enqueue_create_buffer[DType.float32](GPU_MAX_N * GPU_MAX_N)
    var q_in_dev = ctx.enqueue_create_buffer[DType.float32](GPU_MAX_N)
    var q_out_dev = ctx.enqueue_create_buffer[DType.float32](GPU_MAX_N)

    # Copy to device
    ctx.enqueue_copy(dst_buf=mat_dev, src_buf=mat_host)
    ctx.synchronize()

    # Create LayoutTensor views with comptime layouts
    var mat_tensor = LayoutTensor[DType.float32, GPU_MAT_LAYOUT](mat_dev)
    var q_in_tensor = LayoutTensor[DType.float32, GPU_VEC_LAYOUT](q_in_dev)
    var q_out_tensor = LayoutTensor[DType.float32, GPU_VEC_LAYOUT](q_out_dev)

    # Process each batch element
    var results: List[List[Float64]] = []
    for b in range(batch_size):
        # Copy initial condition to host buffer
        for i in range(n):
            q_in_host[i] = q0_flat[i]

        # Copy to device
        ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_in_host)
        ctx.synchronize()

        # Launch GPU kernel for all time steps
        var step = 0
        while step < num_steps:
            ctx.enqueue_function[batch_euler_step, batch_euler_step](
                mat_tensor,
                q_in_tensor,
                q_out_tensor,
                n,
                Float32(dt),
                grid_dim=n,
                block_dim=256,
            )
            # Swap: copy q_out to q_in for next step
            ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_out_dev)
            step += 1

        # Copy result back
        ctx.enqueue_copy(dst_buf=q_out_host, src_buf=q_out_dev)
        ctx.synchronize()

        # Read result (convert Float32 → Float64)
        var row: List[Float64] = []
        for i in range(n):
            row.append(Float64(q_out_host[i]))
        results.append(row^)

    _project_nonnegative(results)
    return results^
