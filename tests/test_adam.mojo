from numerics.nn.adam import Adam
from numerics.utils import abs_f64
from std.testing import assert_true, TestSuite


def _close(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    return abs_f64(a - b) < tol


def test_adam_converges_to_minimum() raises:
    """Adam should converge params toward minimum of f(x) = x^2."""
    var adam = Adam(lr=0.1)
    var params: List[Float64] = [5.0]

    var i = 0
    while i < 100:
        var grads: List[Float64] = [2.0 * params[0]]
        params = adam.step(params, grads)
        i += 1

    # Adam with lr=0.1 should get close to 0 but may not reach exactly 0
    assert_true(abs_f64(params[0]) < 0.5, "should converge toward 0, got " + String(params[0]))


def test_adam_multi_dim() raises:
    """Adam should converge in multi-dimensional case."""
    var adam = Adam(lr=0.1)
    var params: List[Float64] = [3.0, -4.0]

    var i = 0
    while i < 200:
        var grads: List[Float64] = [2.0 * params[0], 2.0 * params[1]]
        params = adam.step(params, grads)
        i += 1

    assert_true(_close(params[0], 0.0, 0.01))
    assert_true(_close(params[1], 0.0, 0.01))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
