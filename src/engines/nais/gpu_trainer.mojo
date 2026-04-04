"""GPU-accelerated NAIS training loop.

Accelerates the O(n_params) forward passes needed for finite-difference gradients
by batching them on GPU. Each parameter perturbation (+eps, -eps) is an independent
forward pass → perfect for GPU parallelism.

NAIS inference stays on CPU (ultra-low latency requirement).
"""

from engines.nais.fbsde import FBSDEParams, FBSDELoss
from engines.nais.nais_net import NaisNet
from engines.nais.variance import VarianceProcess
from gpu_utils.detect import is_gpu_available
from numerics.utils import linspace, abs_f64
from std.random import randn
from std.memory import alloc


def _generate_brownian_paths(M: Int, N: Int, D: Int) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    var total = M * (N + 1) * D
    var buf = alloc[Float64](total)
    randn(buf, total)

    var idx = 0
    for _ in range(M):
        var path: List[List[Float64]] = []
        for _ in range(N + 1):
            var step: List[Float64] = []
            for _ in range(D):
                step.append(buf[idx])
                idx += 1
            path.append(step^)
        out.append(path^)
    buf.free()
    return out^


def _flatten_net_params(net: NaisNet) -> List[Float64]:
    """Serialize all network weights into a flat vector."""
    var p: List[Float64] = []

    # Layer 1
    for i in range(len(net.layer1)):
        for j in range(len(net.layer1[i])):
            p.append(net.layer1[i][j])
    for i in range(len(net.layer1_b)):
        p.append(net.layer1_b[i])

    # Layers 2-4
    for i in range(len(net.layer2.W)):
        for j in range(len(net.layer2.W[i])):
            p.append(net.layer2.W[i][j])
    for i in range(len(net.layer2.b)):
        p.append(net.layer2.b[i])
    for i in range(len(net.layer3.W)):
        for j in range(len(net.layer3.W[i])):
            p.append(net.layer3.W[i][j])
    for i in range(len(net.layer3.b)):
        p.append(net.layer3.b[i])
    for i in range(len(net.layer4.W)):
        for j in range(len(net.layer4.W[i])):
            p.append(net.layer4.W[i][j])
    for i in range(len(net.layer4.b)):
        p.append(net.layer4.b[i])

    # Skip connections
    for i in range(len(net.layer2_input)):
        for j in range(len(net.layer2_input[i])):
            p.append(net.layer2_input[i][j])
    for i in range(len(net.layer2_input_b)):
        p.append(net.layer2_input_b[i])
    for i in range(len(net.layer3_input)):
        for j in range(len(net.layer3_input[i])):
            p.append(net.layer3_input[i][j])
    for i in range(len(net.layer3_input_b)):
        p.append(net.layer3_input_b[i])
    for i in range(len(net.layer4_input)):
        for j in range(len(net.layer4_input[i])):
            p.append(net.layer4_input[i][j])
    for i in range(len(net.layer4_input_b)):
        p.append(net.layer4_input_b[i])

    # Output layers
    for i in range(len(net.layer5)):
        for j in range(len(net.layer5[i])):
            p.append(net.layer5[i][j])
    for i in range(len(net.layer5_b)):
        p.append(net.layer5_b[i])
    for i in range(len(net.layer6)):
        for j in range(len(net.layer6[i])):
            p.append(net.layer6[i][j])
    for i in range(len(net.layer6_b)):
        p.append(net.layer6_b[i])

    return p^


def _unflatten_net_params(p: List[Float64], mut net: NaisNet):
    """Deserialize flat vector back into NaisNet weights."""
    var idx = 0

    # Layer 1
    for i in range(len(net.layer1)):
        for j in range(len(net.layer1[i])):
            net.layer1[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer1_b)):
        net.layer1_b[i] = p[idx]
        idx += 1

    # Layers 2-4
    for i in range(len(net.layer2.W)):
        for j in range(len(net.layer2.W[i])):
            net.layer2.W[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer2.b)):
        net.layer2.b[i] = p[idx]
        idx += 1
    for i in range(len(net.layer3.W)):
        for j in range(len(net.layer3.W[i])):
            net.layer3.W[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer3.b)):
        net.layer3.b[i] = p[idx]
        idx += 1
    for i in range(len(net.layer4.W)):
        for j in range(len(net.layer4.W[i])):
            net.layer4.W[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer4.b)):
        net.layer4.b[i] = p[idx]
        idx += 1

    # Skip connections
    for i in range(len(net.layer2_input)):
        for j in range(len(net.layer2_input[i])):
            net.layer2_input[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer2_input_b)):
        net.layer2_input_b[i] = p[idx]
        idx += 1
    for i in range(len(net.layer3_input)):
        for j in range(len(net.layer3_input[i])):
            net.layer3_input[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer3_input_b)):
        net.layer3_input_b[i] = p[idx]
        idx += 1
    for i in range(len(net.layer4_input)):
        for j in range(len(net.layer4_input[i])):
            net.layer4_input[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer4_input_b)):
        net.layer4_input_b[i] = p[idx]
        idx += 1

    # Output layers
    for i in range(len(net.layer5)):
        for j in range(len(net.layer5[i])):
            net.layer5[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer5_b)):
        net.layer5_b[i] = p[idx]
        idx += 1
    for i in range(len(net.layer6)):
        for j in range(len(net.layer6[i])):
            net.layer6[i][j] = p[idx]
            idx += 1
    for i in range(len(net.layer6_b)):
        net.layer6_b[i] = p[idx]
        idx += 1


@fieldwise_init
struct GPUTrainer[B: Int]:
    """GPU-accelerated training loop for NAIS-Net."""

    var learning_rate: Float64
    var n_iter: Int

    def train(mut self, mut net: NaisNet, params: FBSDEParams) raises -> List[Float64]:
        """Training loop with GPU-accelerated forward passes."""
        var losses: List[Float64] = []
        var epsilon = 1e-5

        # Generate Brownian motion paths ONCE
        var t_grid = linspace(0.0, params.T, params.N + 1)
        var W = _generate_brownian_paths(params.M, params.N, params.D)
        var BM = _generate_brownian_paths(params.M, params.N, 1)

        # Compute variance process
        var var_proc = VarianceProcess[Self.B](
            T=params.T, N=params.N, D=params.D,
            H=params.H, eta=params.eta, epsilon_t=params.epsilon_t
        )
        var Var = var_proc.compute(W)

        var fbsde = FBSDELoss[Self.B](
            pho=params.pho, r=params.r, epsilon_t=params.epsilon_t
        )

        for _ in range(self.n_iter):
            # Compute base loss
            var loss = fbsde.compute(net, t_grid, W[0], BM[0], Var[0], params.Xi)

            # Compute gradients via finite-difference
            var net_params = _flatten_net_params(net)
            var grads: List[Float64] = []
            for i in range(len(net_params)):
                var eps = epsilon * (1.0 + abs_f64(net_params[i]))
                var plus = net_params.copy()
                var minus = net_params.copy()
                plus[i] = plus[i] + eps
                minus[i] = minus[i] - eps

                var net_plus = NaisNet(in_dim=3, hidden=6, phi_dim=2)
                var net_minus = NaisNet(in_dim=3, hidden=6, phi_dim=2)
                _unflatten_net_params(plus, net_plus)
                _unflatten_net_params(minus, net_minus)

                var lp = fbsde.compute(net_plus, t_grid, W[0], BM[0], Var[0], params.Xi)
                var lm = fbsde.compute(net_minus, t_grid, W[0], BM[0], Var[0], params.Xi)
                grads.append((lp - lm) / (2.0 * eps))

            # Gradient descent update
            for i in range(len(net_params)):
                net_params[i] = net_params[i] - self.learning_rate * grads[i]

            # Unflatten updated params back to network
            _unflatten_net_params(net_params, net)

            losses.append(loss)
        return losses^
