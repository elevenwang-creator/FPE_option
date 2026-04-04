from server.pdf_cache import PDFGrid
from server.interpolator import Interpolator
from server.payoffs import BarrierUpAndOut, EuropeanCall
from server.pricer import PricingRequest, Pricer
from server.greeks import Greeks
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_float_close(a: Float64, b: Float64, atol: Float64 = 1e-9) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def make_test_grid() -> PDFGrid:
    var pdf: List[List[Float64]] = []
    pdf.append([1.0, 2.0])
    pdf.append([3.0, 4.0])
    var s_points: List[Float64] = [10.0, 20.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_points^, v_points=v_points^, T=1.0, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    return grid^


def test_interpolator_corners_and_midpoint() raises:
    var grid = make_test_grid()
    var interp = Interpolator()

    var f00 = interp.interpolate(grid, 10.0, 0.1)
    var f10 = interp.interpolate(grid, 20.0, 0.1)
    var f01 = interp.interpolate(grid, 10.0, 0.2)
    var f11 = interp.interpolate(grid, 20.0, 0.2)
    var fmid = interp.interpolate(grid, 15.0, 0.15)

    assert_float_close(f00, 1.0)
    assert_float_close(f10, 3.0)
    assert_float_close(f01, 2.0)
    assert_float_close(f11, 4.0)
    assert_float_close(fmid, 2.5)


def test_barrier_up_and_out_payoff() raises:
    var payoff = BarrierUpAndOut()
    assert_float_close(payoff.evaluate(120.0, 100.0, 150.0), 20.0)
    assert_float_close(payoff.evaluate(160.0, 100.0, 150.0), 0.0)


def test_european_call_payoff() raises:
    var payoff = EuropeanCall()
    assert_float_close(payoff.evaluate(120.0, 100.0, 999.0), 20.0)
    assert_float_close(payoff.evaluate(80.0, 100.0, 999.0), 0.0)


def test_integrate_payoff_uniform_pdf() raises:
    var pdf: List[List[Float64]] = []
    pdf.append([1.0, 1.0])
    pdf.append([1.0, 1.0])
    var s_points: List[Float64] = [90.0, 110.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_points^, v_points=v_points^, T=1.0, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()

    var req = PricingRequest(
        S=100.0,
        K=100.0,
        V=0.15,
        barrier=200.0,
        payoff_type=1,
        param_hash=1,
    )

    var pricer = Pricer[1](interpolator=Interpolator(), greeks_computer=Greeks[1](h_s=1e-2, h_v=1e-3))
    var value = pricer._integrate_payoff(grid, req)

    # With trapezoidal spacing in pricer:
    # S=90 payoff=0, S=110 payoff=10; sum over 2 V points with dv=1 each.
    # price = (0 + 10) * 2 = 20.
    assert_float_close(value, 20.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
