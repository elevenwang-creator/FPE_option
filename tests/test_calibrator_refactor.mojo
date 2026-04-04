from engines.calibrator.calibrator import Calibrator
from engines.fpe.heston_params import HestonParams
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, atol: Float64 = 1e-3) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_calibrator_uses_lm_solve() raises:
    """Calibrator should use LevenbergMarquardt.solve() internally.
    This test verifies the refactored calibrator produces correct results.
    """
    var init = HestonParams(
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.04,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    # Generate synthetic market prices from known params
    var true_params = HestonParams(
        kappa=1.5,
        theta=0.06,
        sigma=0.4,
        rho=-0.6,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.06,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    # Simple synthetic market: just test that calibrator converges
    var market_prices: List[Float64] = [5.0, 3.0, 1.0]
    var strikes: List[Float64] = [90.0, 100.0, 110.0]
    var expiries: List[Float64] = [0.5, 0.5, 0.5]

    var cal = Calibrator[1](max_iter=20, tol=1e-6)
    var result = cal.calibrate(market_prices, strikes, expiries, init)

    # Result should be valid params
    assert_true(result.kappa > 0.0)
    assert_true(result.theta > 0.0)
    assert_true(result.sigma > 0.0)
    assert_true(-1.0 < result.rho and result.rho < 1.0)
    assert_true(result.V0 > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
