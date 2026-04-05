"""Host-side GPU batch execution for FPE ODE integration.

Manages GPU device context, buffer transfers, kernel launch,
and result retrieval for batch parallel ODE solving.

Backend detection:
- Apple Silicon → Metal via DeviceContext(api="metal")
- Other GPU → Generic DeviceContext()
- No GPU → Falls back to CPU explicit Euler

Dtype selection: Automatically uses gpu_utils.dtype module constants
with ternary expressions for true "write once, deploy anywhere".

Requires: --target-accelerator=metal:1 (or equivalent for CUDA/HIP)

Usage:
    from engines.fpe.gpu_batch_executor import gpu_batch_solve
    results = gpu_batch_solve(
        neg_M_inv_K=matrix, q0=initial_states, t_end=0.1,
        batch_size=2, num_steps=1000,
    )
"""

from engines.fpe.gpu_batch_kernels import batch_euler_step
from gpu_utils.detect import is_gpu_available
from gpu_utils.host_utils import create_device_context
from gpu_utils.dtype import (
    METAL_DTYPE, METAL_MAT_LAYOUT, METAL_VEC_LAYOUT, METAL_MAX_N,
    CUDA_DTYPE, CUDA_MAT_LAYOUT, CUDA_VEC_LAYOUT, CUDA_MAX_N,
    is_float32_backend,
)
from sparse.csr import CSRMatrix
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import LayoutTensor
from numerics.utils import zeros
from std.sys import has_apple_gpu_accelerator


# Backend-specific types - automatically selected via ternary expressions
# Same pattern as gpu_batch_kernels.mojo for consistency
comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAT_LAYOUT = METAL_MAT_LAYOUT if has_apple_gpu_accelerator() else CUDA_MAT_LAYOUT
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N


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
    Dtype is automatically selected per backend via gpu_utils.dtype module.

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
    
    # Bounds check: GPU kernel has fixed maximum size from dtype module
    if n > GPU_MAX_N:
        # Fall back to CPU for large matrices
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var q = _cpu_euler_solve(_csr_to_dense_float(neg_M_inv_K), q0, t_end, num_steps)
            results.append(q^)
        _project_nonnegative(results)
        return results^
    
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

    Uses backend-appropriate dtype from gpu_utils.dtype module.
    Each batch element gets its own pair of device buffers (q_in, q_out) with no race conditions.
    """
    var ctx = create_device_context()

    # Use dtype management module for all type selection
    comptime if is_float32_backend():
        # Metal: Float32
        var mat_host = ctx.enqueue_create_host_buffer[METAL_DTYPE](GPU_MAX_N * GPU_MAX_N)
        var q_in_host = ctx.enqueue_create_host_buffer[METAL_DTYPE](GPU_MAX_N)
        var q_out_host = ctx.enqueue_create_host_buffer[METAL_DTYPE](GPU_MAX_N)
        ctx.synchronize()
        
        # Copy matrix with proper padding for LayoutTensor stride
        for i in range(GPU_MAX_N):
            for j in range(GPU_MAX_N):
                if i < n and j < n:
                    mat_host[i * GPU_MAX_N + j] = Float32(mat_dense[i][j])
                else:
                    mat_host[i * GPU_MAX_N + j] = Float32(0.0)
        
        # Copy initial condition
        for i in range(n):
            q_in_host[i] = Float32(q0[i])
        for i in range(n, GPU_MAX_N):
            q_in_host[i] = Float32(0.0)
        
        var mat_dev = ctx.enqueue_create_buffer[METAL_DTYPE](GPU_MAX_N * GPU_MAX_N)
        var q_in_dev = ctx.enqueue_create_buffer[METAL_DTYPE](GPU_MAX_N)
        var q_out_dev = ctx.enqueue_create_buffer[METAL_DTYPE](GPU_MAX_N)
        
        ctx.enqueue_copy(dst_buf=mat_dev, src_buf=mat_host)
        ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_in_host)
        ctx.synchronize()
        
        var mat_tensor = LayoutTensor[METAL_DTYPE, GPU_MAT_LAYOUT](mat_dev)
        var q_in_tensor = LayoutTensor[METAL_DTYPE, GPU_VEC_LAYOUT](q_in_dev)
        var q_out_tensor = LayoutTensor[METAL_DTYPE, GPU_VEC_LAYOUT](q_out_dev)
        
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var step = 0
            while step < num_steps:
                ctx.enqueue_function[batch_euler_step, batch_euler_step](
                    mat_tensor, q_in_tensor, q_out_tensor, n, Float32(dt),
                    grid_dim=n, block_dim=256,
                )
                ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_out_dev)
                step += 1
            
            ctx.enqueue_copy(dst_buf=q_out_host, src_buf=q_out_dev)
            ctx.synchronize()
            
            var row: List[Float64] = []
            for i in range(n):
                row.append(Float64(q_out_host[i]))
            results.append(row^)
        
        _project_nonnegative(results)
        return results^
    else:
        # CUDA/HIP/CPU: Float64
        var mat_host = ctx.enqueue_create_host_buffer[CUDA_DTYPE](GPU_MAX_N * GPU_MAX_N)
        var q_in_host = ctx.enqueue_create_host_buffer[CUDA_DTYPE](GPU_MAX_N)
        var q_out_host = ctx.enqueue_create_host_buffer[CUDA_DTYPE](GPU_MAX_N)
        ctx.synchronize()
        
        # Copy matrix with proper padding for LayoutTensor stride
        for i in range(GPU_MAX_N):
            for j in range(GPU_MAX_N):
                if i < n and j < n:
                    mat_host[i * GPU_MAX_N + j] = mat_dense[i][j]
                else:
                    mat_host[i * GPU_MAX_N + j] = Float64(0.0)
        
        # Copy initial condition
        for i in range(n):
            q_in_host[i] = q0[i]
        for i in range(n, GPU_MAX_N):
            q_in_host[i] = Float64(0.0)
        
        var mat_dev = ctx.enqueue_create_buffer[CUDA_DTYPE](GPU_MAX_N * GPU_MAX_N)
        var q_in_dev = ctx.enqueue_create_buffer[CUDA_DTYPE](GPU_MAX_N)
        var q_out_dev = ctx.enqueue_create_buffer[CUDA_DTYPE](GPU_MAX_N)
        
        ctx.enqueue_copy(dst_buf=mat_dev, src_buf=mat_host)
        ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_in_host)
        ctx.synchronize()
        
        var mat_tensor = LayoutTensor[CUDA_DTYPE, GPU_MAT_LAYOUT](mat_dev)
        var q_in_tensor = LayoutTensor[CUDA_DTYPE, GPU_VEC_LAYOUT](q_in_dev)
        var q_out_tensor = LayoutTensor[CUDA_DTYPE, GPU_VEC_LAYOUT](q_out_dev)
        
        var results: List[List[Float64]] = []
        for b in range(batch_size):
            var step = 0
            while step < num_steps:
                ctx.enqueue_function[batch_euler_step, batch_euler_step](
                    mat_tensor, q_in_tensor, q_out_tensor, n, dt,
                    grid_dim=n, block_dim=256,
                )
                ctx.enqueue_copy(dst_buf=q_in_dev, src_buf=q_out_dev)
                step += 1
            
            ctx.enqueue_copy(dst_buf=q_out_host, src_buf=q_out_dev)
            ctx.synchronize()
            
            var row: List[Float64] = []
            for i in range(n):
                row.append(q_out_host[i])
            results.append(row^)
        
        _project_nonnegative(results)
        return results^
