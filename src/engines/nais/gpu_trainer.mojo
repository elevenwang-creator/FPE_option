"""CPU-parallel NAIS training loop with GPU forward kernel scaffolding.

Accelerates the O(n_params) forward passes needed for finite-difference
gradient computation by using CPU parallelism via parallelize[].

The GPU forward kernel (nais_forward_kernel) is defined in
gpu_forward_kernels.mojo and can be dispatched when GPU is available,
but CPU parallelism provides the bulk of the acceleration for training.

NAIS inference stays on CPU (ultra-low latency requirement for trading).
"""

from engines.nais.fbsde import FBSDEParams, FBSDELoss
from engines.nais.nais_net import NaisNet
from engines.nais.utils import _flatten_net_params
from engines.nais.variance import VarianceProcess
from numerics.utils import linspace
from std.math import abs
from std.algorithm import parallelize


@fieldwise_init
struct GPUTrainer[B: Int]:
    """CPU-parallel training loop for NAIS-Net.

    Uses parallelize[] for CPU batch parallelism in finite-difference
    gradient computation. GPU forward kernel dispatch is available via
    nais_forward_kernel for future GPU acceleration.
    """

    var learning_rate: Float64
    var n_iter: Int

    def train(
        mut self, mut net: NaisNet, params: FBSDEParams
    ) raises -> List[Float64]:
        """Training loop entirely on GPU using NAISGPUTrainExecutor."""
        from engines.nais.gpu_train_kernels import NAISGPUTrainExecutor

        var net_params = _flatten_net_params(net)
        var n_params = len(net_params)

        # Dispatch training to GPU executor
        var executor = NAISGPUTrainExecutor(
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            n_params=n_params,
        )
        var losses = executor.execute_training_on_gpu()
        return losses^
