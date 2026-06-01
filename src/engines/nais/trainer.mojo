from numerics.utils import linspace
from std.math import abs

from engines.nais.fbsde import FBSDEParams, FBSDELoss
from engines.nais.nais_net import NaisNet
from engines.nais.utils import _generate_brownian_paths, _flatten_net_params
from engines.nais.variance import VarianceProcess
from numerics.nn.adam import Adam
from numerics.nn.autograd import GradientTape, Tape


def _unflatten_vec(p: List[Float64], idx: Int, mut b: List[Float64]) -> Int:
    var pos = idx
    for j in range(len(b)):
        b[j] = p[pos]
        pos += 1
    return pos


def _unflatten_net_params(p: List[Float64], mut net: NaisNet):
    var idx = 0

    for i in range(len(net.layer1_T_flat)):
        net.layer1_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer1_b)

    for i in range(len(net.layer2.W_T_flat)):
        net.layer2.W_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer2.b)
    for i in range(len(net.layer3.W_T_flat)):
        net.layer3.W_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer3.b)
    for i in range(len(net.layer4.W_T_flat)):
        net.layer4.W_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer4.b)

    for i in range(len(net.layer2_input_T_flat)):
        net.layer2_input_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer2_input_b)
    for i in range(len(net.layer3_input_T_flat)):
        net.layer3_input_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer3_input_b)
    for i in range(len(net.layer4_input_T_flat)):
        net.layer4_input_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer4_input_b)

    for i in range(len(net.layer5_T_flat)):
        net.layer5_T_flat[i] = p[idx]; idx += 1
    idx = _unflatten_vec(p, idx, net.layer5_b)
    for i in range(len(net.layer6_T_flat)):
        net.layer6_T_flat[i] = p[idx]; idx += 1
    _ = _unflatten_vec(p, idx, net.layer6_b)


def _collect_param_indices(net: NaisNet, mut tape: Tape) -> List[Int]:
    var indices: List[Int] = []

    for i in range(len(net.layer1_T_flat)):
        indices.append(tape.record_value(net.layer1_T_flat[i]))
    for i in range(len(net.layer1_b)):
        indices.append(tape.record_value(net.layer1_b[i]))

    for i in range(len(net.layer2.W_T_flat)):
        indices.append(tape.record_value(net.layer2.W_T_flat[i]))
    for i in range(len(net.layer2.b)):
        indices.append(tape.record_value(net.layer2.b[i]))
    for i in range(len(net.layer3.W_T_flat)):
        indices.append(tape.record_value(net.layer3.W_T_flat[i]))
    for i in range(len(net.layer3.b)):
        indices.append(tape.record_value(net.layer3.b[i]))
    for i in range(len(net.layer4.W_T_flat)):
        indices.append(tape.record_value(net.layer4.W_T_flat[i]))
    for i in range(len(net.layer4.b)):
        indices.append(tape.record_value(net.layer4.b[i]))

    for i in range(len(net.layer2_input_T_flat)):
        indices.append(tape.record_value(net.layer2_input_T_flat[i]))
    for i in range(len(net.layer2_input_b)):
        indices.append(tape.record_value(net.layer2_input_b[i]))
    for i in range(len(net.layer3_input_T_flat)):
        indices.append(tape.record_value(net.layer3_input_T_flat[i]))
    for i in range(len(net.layer3_input_b)):
        indices.append(tape.record_value(net.layer3_input_b[i]))
    for i in range(len(net.layer4_input_T_flat)):
        indices.append(tape.record_value(net.layer4_input_T_flat[i]))
    for i in range(len(net.layer4_input_b)):
        indices.append(tape.record_value(net.layer4_input_b[i]))

    for i in range(len(net.layer5_T_flat)):
        indices.append(tape.record_value(net.layer5_T_flat[i]))
    for i in range(len(net.layer5_b)):
        indices.append(tape.record_value(net.layer5_b[i]))
    for i in range(len(net.layer6_T_flat)):
        indices.append(tape.record_value(net.layer6_T_flat[i]))
    for i in range(len(net.layer6_b)):
        indices.append(tape.record_value(net.layer6_b[i]))

    return indices^


def _apply_gradients(mut net: NaisNet, grads: List[Float64], lr: Float64):
    var idx = 0

    for i in range(len(net.layer1_T_flat)):
        net.layer1_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer1_b)):
        net.layer1_b[i] -= lr * grads[idx]; idx += 1

    for i in range(len(net.layer2.W_T_flat)):
        net.layer2.W_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer2.b)):
        net.layer2.b[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer3.W_T_flat)):
        net.layer3.W_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer3.b)):
        net.layer3.b[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer4.W_T_flat)):
        net.layer4.W_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer4.b)):
        net.layer4.b[i] -= lr * grads[idx]; idx += 1

    for i in range(len(net.layer2_input_T_flat)):
        net.layer2_input_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer2_input_b)):
        net.layer2_input_b[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer3_input_T_flat)):
        net.layer3_input_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer3_input_b)):
        net.layer3_input_b[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer4_input_T_flat)):
        net.layer4_input_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer4_input_b)):
        net.layer4_input_b[i] -= lr * grads[idx]; idx += 1

    for i in range(len(net.layer5_T_flat)):
        net.layer5_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer5_b)):
        net.layer5_b[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer6_T_flat)):
        net.layer6_T_flat[i] -= lr * grads[idx]; idx += 1
    for i in range(len(net.layer6_b)):
        net.layer6_b[i] -= lr * grads[idx]; idx += 1


@fieldwise_init
struct Trainer[B: Int]:
    var learning_rate: Float64
    var n_iter: Int

    def train(
        mut self, mut net: NaisNet, params: FBSDEParams
    ) raises -> List[Float64]:
        var losses: List[Float64] = []
        var epsilon = 1e-5

        var t_grid = linspace(0.0, params.T, params.N + 1)
        var W = _generate_brownian_paths(params.M, params.N, params.D)
        var BM = _generate_brownian_paths(params.M, params.N, 1)

        var var_proc = VarianceProcess[Self.B](
            T=params.T,
            N=params.N,
            D=params.D,
            H=params.H,
            eta=params.eta,
            epsilon_t=params.epsilon_t,
        )
        var Var = var_proc.compute(W)

        var fbsde = FBSDELoss[Self.B](
            pho=params.pho, r=params.r, epsilon_t=params.epsilon_t
        )

        for _ in range(self.n_iter):
            var loss = 0.0
            for m in range(params.M):
                loss += fbsde.compute(
                    net, t_grid, W[m], BM[m], Var[m], params.Xi
                )
            loss /= Float64(params.M)

            var net_params = _flatten_net_params(net)
            var grads: List[Float64] = []
            for i in range(len(net_params)):
                var eps = epsilon * (1.0 + abs(net_params[i]))
                var plus = net_params.copy()
                var minus = net_params.copy()
                plus[i] = plus[i] + eps
                minus[i] = minus[i] - eps

                var net_plus = net.copy()
                var net_minus = net.copy()
                _unflatten_net_params(plus, net_plus)
                _unflatten_net_params(minus, net_minus)

                var lp = 0.0
                var lm = 0.0
                for m in range(params.M):
                    lp += fbsde.compute(
                        net_plus, t_grid, W[m], BM[m], Var[m], params.Xi
                    )
                    lm += fbsde.compute(
                        net_minus, t_grid, W[m], BM[m], Var[m], params.Xi
                    )
                lp /= Float64(params.M)
                lm /= Float64(params.M)
                grads.append((lp - lm) / (2.0 * eps))

            _apply_gradients(net, grads, self.learning_rate)

            losses.append(loss)
        return losses^
