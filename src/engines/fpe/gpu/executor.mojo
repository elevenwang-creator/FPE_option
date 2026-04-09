"""GPU Executor for the FULL Heston batch pricing and calibration logic chain.

Orchestrates the kernels defined in separated modular files.
Fulfills the strict requirement: all modules MUST be on GPU.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from std.sys import has_accelerator
from std.math import ceildiv
from gpu_utils.dtype import (
    METAL_DTYPE,
    METAL_VEC_LAYOUT,
    CUDA_DTYPE,
    CUDA_VEC_LAYOUT,
)
from std.sys import has_apple_gpu_accelerator

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT

from engines.fpe.gpu.domain import (
    generate_knots_gpu_kernel,
    grid_gpu_kernel,
    basis_gpu_kernel,
    boundary_gpu_kernel,
)
from engines.fpe.gpu.matrix import (
    spmatrix_gpu_kernel,
    delta_gpu_kernel,
    initial_gpu_kernel,
)
from engines.fpe.gpu.solver import (
    lu_gpu_kernel,
    radau5_gpu_kernel,
)
from engines.fpe.gpu.integration import integrate_gpu_kernel
from engines.fpe.gpu.calibration import (
    loss_gpu_kernel,
    lm_optimization_gpu_kernel,
)


@fieldwise_init
struct GPUFullChainExecutor[B: Int]:
    var n_s: Int
    var n_v: Int

    def execute_batch_pricing(self) raises:
        """Executes the full Heston batch pricing logic chain entirely on GPU.
        """
        comptime assert (
            has_accelerator()
        ), "GPU is required for batch pricing logic chain!"
        var ctx = DeviceContext()

        var batch_size = Self.B
        var bs = 256
        var grid = batch_size  # Launch exactly 1 block per batch element
        var elements = self.n_s + self.n_v
        var matrix_size = 1
        var total_size = batch_size * elements

        var knots_buf = ctx.enqueue_create_buffer[GPU_DTYPE](total_size)
        var params_buf = ctx.enqueue_create_buffer[GPU_DTYPE](batch_size * 5)
        var grid_buf = ctx.enqueue_create_buffer[GPU_DTYPE](total_size)
        var basis_buf = ctx.enqueue_create_buffer[GPU_DTYPE](total_size)
        var boundary_buf = ctx.enqueue_create_buffer[GPU_DTYPE](total_size)

        var spmatrix_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * matrix_size
        )
        var delta_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var initial_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var lu_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * matrix_size
        )
        var radau5_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var price_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )

        var knots = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](knots_buf)
        var params = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](params_buf)
        var grid_d = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](grid_buf)
        var basis = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](basis_buf)
        var boundary = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](boundary_buf)
        var spmatrix = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](spmatrix_buf)
        var delta = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](delta_buf)
        var initial = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](initial_buf)
        var lu = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](lu_buf)
        var radau5 = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](radau5_buf)
        var price = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](price_buf)

        ctx.enqueue_function[
            generate_knots_gpu_kernel, generate_knots_gpu_kernel
        ](knots, params, self.n_s, self.n_v, grid_dim=grid, block_dim=bs)
        ctx.enqueue_function[grid_gpu_kernel, grid_gpu_kernel](
            grid_d, knots, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[basis_gpu_kernel, basis_gpu_kernel](
            basis, grid_d, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[boundary_gpu_kernel, boundary_gpu_kernel](
            boundary, basis, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[spmatrix_gpu_kernel, spmatrix_gpu_kernel](
            spmatrix, boundary, params, matrix_size, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[delta_gpu_kernel, delta_gpu_kernel](
            delta, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[initial_gpu_kernel, initial_gpu_kernel](
            initial, delta, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[lu_gpu_kernel, lu_gpu_kernel](
            lu, spmatrix, matrix_size, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[radau5_gpu_kernel, radau5_gpu_kernel](
            radau5, lu, initial, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[integrate_gpu_kernel, integrate_gpu_kernel](
            price, radau5, elements, grid_dim=grid, block_dim=bs
        )

        ctx.synchronize()

    def execute_calibration_logic(self) raises:
        """Executes the full Calibration logic chain entirely on GPU."""
        comptime assert (
            has_accelerator()
        ), "GPU is required for calibration logic chain!"
        var ctx = DeviceContext()

        var batch_size = Self.B
        var bs = 256
        var grid = batch_size
        var elements = 1

        # Execute pricing chain first
        self.execute_batch_pricing()

        var market_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var price_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var loss_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
            batch_size * elements
        )
        var params_buf = ctx.enqueue_create_buffer[GPU_DTYPE](batch_size * 5)

        var market = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](market_buf)
        var price = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](price_buf)
        var loss_t = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](loss_buf)
        var out_params = LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT](params_buf)

        ctx.enqueue_function[loss_gpu_kernel, loss_gpu_kernel](
            loss_t, price, market, elements, grid_dim=grid, block_dim=bs
        )
        ctx.enqueue_function[
            lm_optimization_gpu_kernel, lm_optimization_gpu_kernel
        ](out_params, loss_t, elements, grid_dim=grid, block_dim=bs)

        ctx.synchronize()
