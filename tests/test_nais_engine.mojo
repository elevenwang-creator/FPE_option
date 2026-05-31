from engines.nais.fbsde import FBSDEParams
from engines.nais.nais_net import NaisNet
from engines.nais.trainer import Trainer
from engines.nais.volterra import VolterraProcess
from numerics.nn.stable_linear import StableLinear
from std.testing import assert_true, TestSuite


def _zeros(n: Int) -> List[Float64]:
    var out = List[Float64](length=n, fill=0.0)
    return out^


def _zeros_2d(n0: Int, n1: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for _ in range(n0):
        out.append(_zeros(n1))
    return out^


def _zeros_3d(n0: Int, n1: Int, n2: Int) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    for _ in range(n0):
        out.append(_zeros_2d(n1, n2))
    return out^


def test_stable_linear_forward_shape() raises:
    var W: List[List[Float64]] = []
    W.append([0.1, -0.2, 0.3])
    W.append([0.0, 0.4, -0.1])
    W.append([0.2, 0.1, 0.2])
    var b: List[Float64] = [0.0, 0.1, -0.1]
    var layer = StableLinear(W=W^, b=b^, epsilon=0.01)

    var y = layer.forward([1.0, 2.0, -1.0])
    assert_true(len(y) == 3)


def test_nais_net_forward_output_shapes() raises:
    var net = NaisNet(in_dim=3, hidden=8, phi_dim=2)
    var out = net.forward(0.1, [100.0, 0.05])
    var u = out[0]
    var phi = out[1].copy()
    _ = u
    assert_true(len(phi) == 2)


def test_volterra_generate_shape() raises:
    var M = 3
    var N = 6
    var D = 1
    var W = _zeros_3d(M, N + 1, D)
    for m in range(M):
        for n in range(1, N + 1):
            W[m][n][0] = W[m][n - 1][0] + 0.01 * Float64(m + n)

    var proc = VolterraProcess[1](T=0.2, N=N, D=D, H=0.1)
    var X = proc.generate(W)
    assert_true(len(X) == M)
    assert_true(len(X[0]) == N + 1)
    assert_true(len(X[0][0]) == D)


def test_trainer_loss_decreases() raises:
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var trainer = Trainer[1](learning_rate=1e-2, n_iter=5)
    var params = FBSDEParams(
        Xi=[100.0, 0.04],
        T=0.2,
        M=4,
        N=6,
        D=1,
        H=0.1,
        eta=1.2,
        pho=-0.7,
        r=0.02,
        epsilon_t=0.09,
    )
    var losses = trainer.train(net, params)
    assert_true(len(losses) == 5)
    assert_true(losses[4] < losses[0], "loss should decrease after training")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
