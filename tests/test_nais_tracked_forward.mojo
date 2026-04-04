from engines.nais.nais_net import NaisNet
from numerics.nn.autograd import Tape
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, atol: Float64 = 1e-6) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_forward_tracked_matches_forward() raises:
    """Tracked forward pass should produce same values as regular forward."""
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var t = 0.1
    var x: List[Float64] = [100.0, 0.04, 0.0]

    var result_forward = net.forward(t, x)
    var u = result_forward[0]
    var phi = result_forward[1].copy()

    var tape = Tape()
    var result = net.forward_tracked(t, x, tape)
    var u_tracked = result[0]
    var phi_tracked = result[1].copy()

    assert_close(u, tape.values[u_tracked])
    for i in range(len(phi)):
        assert_close(phi[i], tape.values[phi_tracked[i]])


def test_forward_tracked_records_operations() raises:
    """Tracked forward should record many operations on the tape."""
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var t = 0.1
    var x: List[Float64] = [100.0, 0.04, 0.0]

    var tape = Tape()
    var _ = net.forward_tracked(t, x, tape)

    assert_true(len(tape.entries) > 10, "should record many operations")
    assert_true(len(tape.values) > 10, "should have many values")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
