"""GPU kernels for NAIS training.

NOTE: GPU-accelerated NAIS training is not yet implemented.
The CPU-based Trainer (in trainer.mojo) is the fully functional training path.

This module serves as a placeholder for future GPU acceleration work.
"""

from std.gpu import block_idx, thread_idx, block_dim
from layout import LayoutTensor
from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT


@fieldwise_init
struct NAISGPUTrainExecutor:
    var learning_rate: Float64
    var n_iter: Int
    var n_params: Int

    def execute_training_on_gpu(self) raises -> List[Float64]:
        raise Error(
            "GPU-accelerated NAIS training is not yet implemented. "
            "Use the CPU-based Trainer in engines.nais.trainer instead."
        )
