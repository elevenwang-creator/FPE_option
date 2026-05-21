from server.payoffs import BarrierPayoff
from std.testing import assert_true, assert_equal, TestSuite


def assert_close(a: Float64, b: Float64, tol: Float64 = 1e-12) raises:
    var diff = a - b
    if diff < 0.0:
        diff = -diff
    assert_true(diff < tol, "assert_close failed: " + String(a) + " vs " + String(b))


def assert_list_close(actual: List[Float64], expected: List[Float64], tol: Float64 = 1e-12) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_close(actual[i], expected[i], tol)


def test_down_and_in_call_in_otm() raises:
    var p = BarrierPayoff(option_type=0, strikes=[100.0], barrier=80.0)
    var r = p.evaluate(S=70.0)
    assert_list_close(r, [0.0])


def test_down_and_in_call_not_in() raises:
    var p = BarrierPayoff(option_type=0, strikes=[100.0], barrier=80.0)
    var r = p.evaluate(S=90.0)
    assert_list_close(r, [0.0])


def test_down_and_in_call_in_deep_otm() raises:
    var p = BarrierPayoff(option_type=0, strikes=[100.0], barrier=80.0)
    var r = p.evaluate(S=60.0)
    assert_list_close(r, [0.0])


def test_down_and_out_call_active() raises:
    var p = BarrierPayoff(option_type=2, strikes=[100.0], barrier=80.0)
    var r = p.evaluate(S=120.0)
    assert_list_close(r, [20.0])


def test_down_and_out_call_knocked_out() raises:
    var p = BarrierPayoff(option_type=2, strikes=[100.0], barrier=80.0)
    var r = p.evaluate(S=70.0)
    assert_list_close(r, [0.0])


def test_up_and_out_call_active() raises:
    var p = BarrierPayoff(option_type=6, strikes=[100.0], barrier=120.0)
    var r = p.evaluate(S=110.0)
    assert_list_close(r, [10.0])


def test_up_and_out_call_knocked_out() raises:
    var p = BarrierPayoff(option_type=6, strikes=[100.0], barrier=120.0)
    var r = p.evaluate(S=130.0)
    assert_list_close(r, [0.0])


def test_up_and_in_put_in_otm() raises:
    var p = BarrierPayoff(option_type=5, strikes=[100.0], barrier=120.0)
    var r = p.evaluate(S=130.0)
    assert_list_close(r, [0.0])


def test_up_and_in_put_not_in() raises:
    var p = BarrierPayoff(option_type=5, strikes=[100.0], barrier=120.0)
    var r = p.evaluate(S=90.0)
    assert_list_close(r, [0.0])


def test_european_call_itm() raises:
    var p = BarrierPayoff(option_type=8, strikes=[100.0], barrier=0.0)
    var r = p.evaluate(S=120.0)
    assert_list_close(r, [20.0])


def test_european_call_otm() raises:
    var p = BarrierPayoff(option_type=8, strikes=[100.0], barrier=0.0)
    var r = p.evaluate(S=80.0)
    assert_list_close(r, [0.0])


def test_european_put_itm() raises:
    var p = BarrierPayoff(option_type=9, strikes=[100.0], barrier=0.0)
    var r = p.evaluate(S=80.0)
    assert_list_close(r, [20.0])


def test_european_put_otm() raises:
    var p = BarrierPayoff(option_type=9, strikes=[100.0], barrier=0.0)
    var r = p.evaluate(S=120.0)
    assert_list_close(r, [0.0])


def test_multi_strike_european_call() raises:
    var p = BarrierPayoff(option_type=8, strikes=[95.0, 100.0, 105.0], barrier=0.0)
    var r = p.evaluate(S=110.0)
    assert_list_close(r, [15.0, 10.0, 5.0])


def test_multi_strike_barrier_active() raises:
    var p = BarrierPayoff(option_type=6, strikes=[95.0, 100.0, 105.0], barrier=120.0)
    var r = p.evaluate(S=110.0)
    assert_list_close(r, [15.0, 10.0, 5.0])


def test_multi_strike_barrier_knocked_out() raises:
    var p = BarrierPayoff(option_type=6, strikes=[95.0, 100.0, 105.0], barrier=120.0)
    var r = p.evaluate(S=130.0)
    assert_list_close(r, [0.0, 0.0, 0.0])


def test_name() raises:
    var p = BarrierPayoff(option_type=8, strikes=[100.0], barrier=0.0)
    assert_equal(p.name(), "BarrierPayoff")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
