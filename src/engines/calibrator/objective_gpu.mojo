"""GPU kernels for loss computation and LM optimization."""

from std.gpu import block_idx, thread_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import METAL_DTYPE, METAL_VEC_LAYOUT, CUDA_DTYPE, CUDA_VEC_LAYOUT
from std.sys import has_apple_gpu_accelerator

comptime GPU_OBJ_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_OBJ_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_OBJ_SCALAR = Scalar[GPU_OBJ_DTYPE]
comptime N_PARAMS: Int = 5


def loss_gpu_kernel(
    loss_out: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    price_in: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    market_in: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    n_options: Int,
):
    """Compute squared pricing error: loss[i] = (model_price[i] - market_price[i])^2."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base = Int(b) * n_options
    var i = Int(tid)
    while i < n_options:
        var p = Float64(rebind[GPU_OBJ_SCALAR](price_in[base + i]))
        var m = Float64(rebind[GPU_OBJ_SCALAR](market_in[base + i]))
        var diff = p - m
        loss_out[base + i] = GPU_OBJ_SCALAR(diff * diff)
        i += Int(threads)


def loss_sum_gpu_kernel(
    total_loss: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    loss_in: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    n_options: Int,
):
    """Compute total loss as sum of squared residuals (thread-0 sequential for correctness)."""
    var b = Int(block_idx.x)
    if b >= n_options:
        return
    if Int(thread_idx.x) == 0:
        var sum_val: Float64 = 0.0
        for i in range(n_options):
            sum_val = sum_val + Float64(rebind[GPU_OBJ_SCALAR](loss_in[b * n_options + i]))
        total_loss[b] = GPU_OBJ_SCALAR(sum_val)


def lm_step_gpu_kernel(
    params_out: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    params_in: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    jacobian: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    residuals: LayoutTensor[GPU_OBJ_DTYPE, GPU_OBJ_VEC, MutAnyOrigin],
    lambda_val: Scalar[GPU_OBJ_DTYPE],
    n_options: Int,
):
    """One Levenberg-Marquardt optimization step on GPU."""
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)
    var base_j = b * n_options * N_PARAMS
    var base_r = b * n_options
    var base_p = b * 5

    if Int(tid) == 0:
        # JtJ accumulation and lambda addition
        for i_row in range(N_PARAMS):
            for j_col in range(N_PARAMS):
                var total: Float64 = 0.0
                for k in range(n_options):
                    var ji = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + k * N_PARAMS + i_row]))
                    var jj = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + k * N_PARAMS + j_col]))
                    total += ji * jj
                if i_row == j_col:
                    total = total + Float64(rebind[GPU_OBJ_SCALAR](lambda_val))
                jacobian[base_j + i_row * N_PARAMS + j_col] = GPU_OBJ_SCALAR(total)

        # JtR accumulation
        for i_row in range(N_PARAMS):
            var sum_val: Float64 = 0.0
            for k in range(n_options):
                var ji = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + k * N_PARAMS + i_row]))
                var rk = Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + k]))
                sum_val = sum_val + ji * rk
            residuals[base_r + i_row] = GPU_OBJ_SCALAR(0.0 - sum_val)

        # LU solve 5x5
        for k in range(N_PARAMS):
            var pivot = k
            var max_val: Float64 = 0.0
            for i in range(k, N_PARAMS):
                var val = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + i * N_PARAMS + k]))
                if val < 0.0:
                    val = 0.0 - val
                if val > max_val:
                    max_val = val
                    pivot = i
            if pivot != k:
                for jj in range(N_PARAMS):
                    var tmp = jacobian[base_j + k * N_PARAMS + jj]
                    jacobian[base_j + k * N_PARAMS + jj] = jacobian[base_j + pivot * N_PARAMS + jj]
                    jacobian[base_j + pivot * N_PARAMS + jj] = tmp
                var tmp_r = residuals[base_r + k]
                residuals[base_r + k] = residuals[base_r + pivot]
                residuals[base_r + pivot] = tmp_r
            var diag = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + k * N_PARAMS + k]))
            if diag != 0.0:
                for i in range(k + 1, N_PARAMS):
                    var factor = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + i * N_PARAMS + k])) / diag
                    for jj in range(k + 1, N_PARAMS):
                        var val = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + i * N_PARAMS + jj]))
                        val = val - factor * Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + k * N_PARAMS + jj]))
                        jacobian[base_j + i * N_PARAMS + jj] = GPU_OBJ_SCALAR(val)
                    var r_val = Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + i]))
                    r_val = r_val - factor * Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + k]))
                    residuals[base_r + i] = GPU_OBJ_SCALAR(r_val)
                    jacobian[base_j + i * N_PARAMS + k] = GPU_OBJ_SCALAR(0.0)

        for rev in range(N_PARAMS):
            var i = N_PARAMS - 1 - rev
            var s: Float64 = Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + i]))
            for j in range(i + 1, N_PARAMS):
                s = s - Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + i * N_PARAMS + j])) * Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + j]))
            var d = Float64(rebind[GPU_OBJ_SCALAR](jacobian[base_j + i * N_PARAMS + i]))
            if d != 0.0:
                residuals[base_r + i] = GPU_OBJ_SCALAR(s / d)

        for i in range(N_PARAMS):
            var delta = Float64(rebind[GPU_OBJ_SCALAR](residuals[base_r + i]))
            var new_val = Float64(rebind[GPU_OBJ_SCALAR](params_in[base_p + i])) + delta
            params_out[base_p + i] = GPU_OBJ_SCALAR(new_val)
    barrier()