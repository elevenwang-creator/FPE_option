from engines.nais.fbsde import FBSDELoss, FBSDEParams
from engines.nais.nais_net import NaisNet
from numerics.nn.autograd import Tape
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, atol: Float64 = 1e-4) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_compute_tracked_matches_compute() raises:
    """Tracked loss should produce same value as regular loss."""
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var loss_fn = FBSDELoss(pho=-0.7, r=0.02, epsilon_t=0.09)

    var t: List[Float64] = [0.0, 0.1, 0.2]
    var W: List[List[Float64]] = [[100.0, 101.0, 102.0]]
    var BM: List[List[Float64]] = [[0.0, 0.1, 0.2]]
    var Var: List[List[Float64]] = [[0.04, 0.041, 0.042]]
    var Xi: List[Float64] = [100.0, 0.04]

    var loss = loss_fn.compute(net, t, W, BM, Var, Xi)

    var tape = Tape()
    var loss_idx = loss_fn.compute_tracked(net, t, W, BM, Var, Xi, tape)

    assert_close(loss, tape.values[loss_idx])


def test_compute_tracked_records_operations() raises:
    """Tracked loss should record many operations."""
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var loss_fn = FBSDELoss(pho=-0.7, r=0.02, epsilon_t=0.09)

    var t: List[Float64] = [0.0, 0.1, 0.2]
    var W: List[List[Float64]] = [[100.0, 101.0, 102.0]]
    var BM: List[List[Float64]] = [[0.0, 0.1, 0.2]]
    var Var: List[List[Float64]] = [[0.04, 0.041, 0.042]]
    var Xi: List[Float64] = [100.0, 0.04]

    var tape = Tape()
    var _ = loss_fn.compute_tracked(net, t, W, BM, Var, Xi, tape)

    assert_true(len(tape.entries) > 20, "should record many operations")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
