from engines.calibrator.calibrator import Calibrator
from engines.calibrator.objective import ObjectiveFunction
from engines.fpe.heston_params import HestonParams
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _rel_err(a: Float64, b: Float64) -> Float64:
    var denom = _abs(b)
    if denom < 1e-8:
        denom = 1.0
    return _abs(a - b) / denom


def test_calibrator_converges_on_synthetic_market() raises:
    var true_params = HestonParams(
        kappa=1.1,
        theta=0.06,
        sigma=0.3,
        rho=-0.35,
        r=0.02,
        T=0.2,
        S0=100.0,
        V0=0.05,
        S_min=40.0,
        S_max=180.0,
        V_min=0.0,
        V_max=1.0,
    )

    var strikes: List[Float64] = [90.0, 100.0, 110.0]
    var expiries: List[Float64] = [0.1, 0.15, 0.2]
    var zero_market: List[Float64] = [0.0, 0.0, 0.0]

    var obj_zero = ObjectiveFunction[1](zero_market.copy(), strikes.copy(), expiries.copy())
    var market_prices = obj_zero.compute(true_params)

    var init_params = HestonParams(
        kappa=true_params.kappa * 1.01,
        theta=true_params.theta * 0.99,
        sigma=true_params.sigma * 1.01,
        rho=true_params.rho * 0.99,
        r=true_params.r,
        T=true_params.T,
        S0=true_params.S0,
        V0=true_params.V0 * 1.01,
        S_min=true_params.S_min,
        S_max=true_params.S_max,
        V_min=true_params.V_min,
        V_max=true_params.V_max,
    )

    var calibrator = Calibrator[1](max_iter=20, tol=1e-6)
    var fitted = calibrator.calibrate(market_prices, strikes, expiries, init_params)

    assert_true(_rel_err(fitted.kappa, true_params.kappa) < 0.10)
    assert_true(_rel_err(fitted.theta, true_params.theta) < 0.10)
    assert_true(_rel_err(fitted.sigma, true_params.sigma) < 0.10)
    assert_true(_rel_err(fitted.rho, true_params.rho) < 0.10)
    assert_true(_rel_err(fitted.V0, true_params.V0) < 0.10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
