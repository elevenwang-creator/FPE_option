from numerics.nn.autograd import Tape
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, atol: Float64 = 1e-9) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_tape_record_value() raises:
    var tape = Tape()
    var idx = tape.record_value(3.0)
    assert_true(idx == 0, "first value should be at index 0")
    assert_close(tape.values[idx], 3.0)


def test_tape_add() raises:
    var tape = Tape()
    var a = tape.record_value(3.0)
    var b = tape.record_value(2.0)
    var c = tape.record_add(a, b)
    assert_close(tape.values[c], 5.0)


def test_tape_mul() raises:
    var tape = Tape()
    var a = tape.record_value(3.0)
    var b = tape.record_value(4.0)
    var c = tape.record_mul(a, b)
    assert_close(tape.values[c], 12.0)


def test_tape_chain_rule() raises:
    """Test: c = (a + b) * a, where a=3, b=2
    dc/da = (a+b) + a = 5 + 3 = 8
    dc/db = a = 3
    """
    var tape = Tape()
    var a = tape.record_value(3.0)
    var b = tape.record_value(2.0)
    var sum = tape.record_add(a, b)
    var c = tape.record_mul(sum, a)

    tape.backward(c)

    assert_close(tape.gradients_for([a])[0], 8.0, 1e-6)
    assert_close(tape.gradients_for([b])[0], 3.0, 1e-6)


def test_tape_sin_derivative() raises:
    """Test: y = sin(x), x=0, dy/dx = cos(0) = 1"""
    var tape = Tape()
    var x = tape.record_value(0.0)
    var y = tape.record_sin(x)

    tape.backward(y)
    var grads = tape.gradients_for([x])
    assert_close(grads[0], 1.0, 1e-6)


def test_tape_sin_at_pi() raises:
    """Test: y = sin(x), x=pi, dy/dx = cos(pi) = -1"""
    from std.math import pi
    var tape = Tape()
    var x = tape.record_value(Float64(pi))
    var y = tape.record_sin(x)

    tape.backward(y)
    var grads = tape.gradients_for([x])
    assert_close(grads[0], -1.0, 1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
