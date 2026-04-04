from numerics.nn.autograd import Tape
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, atol: Float64 = 1e-6) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_tape_linear_2x1() raises:
    """Test y = W @ x + b where W is 2x1, x is scalar, b is 2-vector.
    W = [[1], [2]], x = [3], b = [4, 5]
    y = [1*3+4, 2*3+5] = [7, 11]
    dy/dW[0][0] = x = 3, dy/db[0] = 1
    dy/dW[1][0] = x = 3, dy/db[1] = 1
    dy/dx = W[0][0] + W[1][0] = 1 + 2 = 3
    """
    var tape = Tape()
    var w00 = tape.record_value(1.0)
    var w10 = tape.record_value(2.0)
    var x0 = tape.record_value(3.0)
    var b0 = tape.record_value(4.0)
    var b1 = tape.record_value(5.0)

    var W_idx = [w00, w10]
    var b_idx = [b0, b1]
    var x_idx = [x0]
    var y_idx = tape.record_linear(W_idx, b_idx, x_idx)

    assert_close(tape.values[y_idx[0]], 7.0)
    assert_close(tape.values[y_idx[1]], 11.0)

    tape.backward(y_idx[0])
    var grads = tape.gradients_for([w00, x0, b0])
    assert_close(grads[0], 3.0)
    assert_close(grads[1], 1.0)
    assert_close(grads[2], 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
