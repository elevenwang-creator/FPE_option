from numerics.utils import linspace, abs_f64
from std.random import randn
from std.memory import alloc

from engines.nais.fbsde import FBSDEParams, FBSDELoss
from engines.nais.nais_net import NaisNet
from engines.nais.variance import VarianceProcess
from numerics.nn.adam import Adam
from numerics.nn.autograd import GradientTape, Tape


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

def _flatten_mat(mut p: List[Float64], W: List[List[Float64]]):
    """Append all elements of a 2D weight matrix to the flat vector."""
    for i in range(len(W)):
        for j in range(len(W[i])):
            p.append(W[i][j])


def _flatten_vec(mut p: List[Float64], b: List[Float64]):
    """Append all elements of a bias vector to the flat vector."""
    for j in range(len(b)):
        p.append(b[j])


def _flatten_net_params(net: NaisNet) -> List[Float64]:
    """Serialize all network weights into a flat vector for optimization."""
    var p: List[Float64] = []

    # Layer 1
    _flatten_mat(p, net.layer1)
    _flatten_vec(p, net.layer1_b)

    # Layers 2-4
    _flatten_mat(p, net.layer2.W)
    _flatten_vec(p, net.layer2.b)
    _flatten_mat(p, net.layer3.W)
    _flatten_vec(p, net.layer3.b)
    _flatten_mat(p, net.layer4.W)
    _flatten_vec(p, net.layer4.b)

    # Skip connections
    _flatten_mat(p, net.layer2_input)
    _flatten_vec(p, net.layer2_input_b)
    _flatten_mat(p, net.layer3_input)
    _flatten_vec(p, net.layer3_input_b)
    _flatten_mat(p, net.layer4_input)
    _flatten_vec(p, net.layer4_input_b)

    # Output layers
    _flatten_mat(p, net.layer5)
    _flatten_vec(p, net.layer5_b)
    _flatten_mat(p, net.layer6)
    _flatten_vec(p, net.layer6_b)

    return p^


def _unflatten_mat(p: List[Float64], mut idx: Int, mut W: List[List[Float64]]):
    """Read elements from flat vector into a 2D weight matrix."""
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = p[idx]
            idx += 1


def _unflatten_vec(p: List[Float64], mut idx: Int, mut b: List[Float64]):
    """Read elements from flat vector into a bias vector."""
    for j in range(len(b)):
        b[j] = p[idx]
        idx += 1


def _unflatten_net_params(p: List[Float64], mut net: NaisNet):
    """Deserialize flat vector back into NaisNet weights."""
    var idx = 0

    _unflatten_mat(p, idx, net.layer1)
    _unflatten_vec(p, idx, net.layer1_b)

    _unflatten_mat(p, idx, net.layer2.W)
    _unflatten_vec(p, idx, net.layer2.b)
    _unflatten_mat(p, idx, net.layer3.W)
    _unflatten_vec(p, idx, net.layer3.b)
    _unflatten_mat(p, idx, net.layer4.W)
    _unflatten_vec(p, idx, net.layer4.b)

    _unflatten_mat(p, idx, net.layer2_input)
    _unflatten_vec(p, idx, net.layer2_input_b)
    _unflatten_mat(p, idx, net.layer3_input)
    _unflatten_vec(p, idx, net.layer3_input_b)
    _unflatten_mat(p, idx, net.layer4_input)
    _unflatten_vec(p, idx, net.layer4_input_b)

    _unflatten_mat(p, idx, net.layer5)
    _unflatten_vec(p, idx, net.layer5_b)
    _unflatten_mat(p, idx, net.layer6)
    _unflatten_vec(p, idx, net.layer6_b)


def _collect_param_indices(net: NaisNet, mut tape: Tape) -> List[Int]:
    """Record all network parameters on tape and return their indices."""
    var indices: List[Int] = []

    # Layer 1
    for i in range(len(net.layer1)):
        for j in range(len(net.layer1[i])):
            indices.append(tape.record_value(net.layer1[i][j]))
    for i in range(len(net.layer1_b)):
        indices.append(tape.record_value(net.layer1_b[i]))

    # Layers 2-4
    for i in range(len(net.layer2.W)):
        for j in range(len(net.layer2.W[i])):
            indices.append(tape.record_value(net.layer2.W[i][j]))
    for i in range(len(net.layer2.b)):
        indices.append(tape.record_value(net.layer2.b[i]))
    for i in range(len(net.layer3.W)):
        for j in range(len(net.layer3.W[i])):
            indices.append(tape.record_value(net.layer3.W[i][j]))
    for i in range(len(net.layer3.b)):
        indices.append(tape.record_value(net.layer3.b[i]))
    for i in range(len(net.layer4.W)):
        for j in range(len(net.layer4.W[i])):
            indices.append(tape.record_value(net.layer4.W[i][j]))
    for i in range(len(net.layer4.b)):
        indices.append(tape.record_value(net.layer4.b[i]))

    # Skip connections
    for i in range(len(net.layer2_input)):
        for j in range(len(net.layer2_input[i])):
            indices.append(tape.record_value(net.layer2_input[i][j]))
    for i in range(len(net.layer2_input_b)):
        indices.append(tape.record_value(net.layer2_input_b[i]))
    for i in range(len(net.layer3_input)):
        for j in range(len(net.layer3_input[i])):
            indices.append(tape.record_value(net.layer3_input[i][j]))
    for i in range(len(net.layer3_input_b)):
        indices.append(tape.record_value(net.layer3_input_b[i]))
    for i in range(len(net.layer4_input)):
        for j in range(len(net.layer4_input[i])):
            indices.append(tape.record_value(net.layer4_input[i][j]))
    for i in range(len(net.layer4_input_b)):
        indices.append(tape.record_value(net.layer4_input_b[i]))

    # Output layers
    for i in range(len(net.layer5)):
        for j in range(len(net.layer5[i])):
            indices.append(tape.record_value(net.layer5[i][j]))
    for i in range(len(net.layer5_b)):
        indices.append(tape.record_value(net.layer5_b[i]))
    for i in range(len(net.layer6)):
        for j in range(len(net.layer6[i])):
            indices.append(tape.record_value(net.layer6[i][j]))
    for i in range(len(net.layer6_b)):
        indices.append(tape.record_value(net.layer6_b[i]))

    return indices^


def _apply_gradients(mut net: NaisNet, grads: List[Float64], lr: Float64):
    """Apply gradients to network weights using gradient descent."""
    var idx = 0

    # Layer 1
    for i in range(len(net.layer1)):
        for j in range(len(net.layer1[i])):
            net.layer1[i][j] = net.layer1[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer1_b)):
        net.layer1_b[i] = net.layer1_b[i] - lr * grads[idx]
        idx += 1

    # Layers 2-4
    for i in range(len(net.layer2.W)):
        for j in range(len(net.layer2.W[i])):
            net.layer2.W[i][j] = net.layer2.W[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer2.b)):
        net.layer2.b[i] = net.layer2.b[i] - lr * grads[idx]
        idx += 1
    for i in range(len(net.layer3.W)):
        for j in range(len(net.layer3.W[i])):
            net.layer3.W[i][j] = net.layer3.W[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer3.b)):
        net.layer3.b[i] = net.layer3.b[i] - lr * grads[idx]
        idx += 1
    for i in range(len(net.layer4.W)):
        for j in range(len(net.layer4.W[i])):
            net.layer4.W[i][j] = net.layer4.W[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer4.b)):
        net.layer4.b[i] = net.layer4.b[i] - lr * grads[idx]
        idx += 1

    # Skip connections
    for i in range(len(net.layer2_input)):
        for j in range(len(net.layer2_input[i])):
            net.layer2_input[i][j] = net.layer2_input[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer2_input_b)):
        net.layer2_input_b[i] = net.layer2_input_b[i] - lr * grads[idx]
        idx += 1
    for i in range(len(net.layer3_input)):
        for j in range(len(net.layer3_input[i])):
            net.layer3_input[i][j] = net.layer3_input[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer3_input_b)):
        net.layer3_input_b[i] = net.layer3_input_b[i] - lr * grads[idx]
        idx += 1
    for i in range(len(net.layer4_input)):
        for j in range(len(net.layer4_input[i])):
            net.layer4_input[i][j] = net.layer4_input[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer4_input_b)):
        net.layer4_input_b[i] = net.layer4_input_b[i] - lr * grads[idx]
        idx += 1

    # Output layers
    for i in range(len(net.layer5)):
        for j in range(len(net.layer5[i])):
            net.layer5[i][j] = net.layer5[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer5_b)):
        net.layer5_b[i] = net.layer5_b[i] - lr * grads[idx]
        idx += 1
    for i in range(len(net.layer6)):
        for j in range(len(net.layer6[i])):
            net.layer6[i][j] = net.layer6[i][j] - lr * grads[idx]
            idx += 1
    for i in range(len(net.layer6_b)):
        net.layer6_b[i] = net.layer6_b[i] - lr * grads[idx]
        idx += 1

@fieldwise_init
struct Trainer[B: Int]:
    """Training loop for NAIS-Net."""

    var learning_rate: Float64
    var n_iter: Int

    def train(mut self, mut net: NaisNet, params: FBSDEParams) raises -> List[Float64]:
        """Training loop: forward → loss → gradient → update."""
        var losses: List[Float64] = []
        var epsilon = 1e-5

        # Generate Brownian motion paths ONCE for consistent training
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
            # Compute FBSDE loss
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

                # Create perturbed networks with SAME dimensions as input net
                var net_plus = net.copy()
                var net_minus = net.copy()
                _unflatten_net_params(plus, net_plus)
                _unflatten_net_params(minus, net_minus)

                var lp = fbsde.compute(net_plus, t_grid, W[0], BM[0], Var[0], params.Xi)
                var lm = fbsde.compute(net_minus, t_grid, W[0], BM[0], Var[0], params.Xi)
                grads.append((lp - lm) / (2.0 * eps))

            # Apply gradient descent
            _apply_gradients(net, grads, self.learning_rate)

            losses.append(loss)
        return losses^
