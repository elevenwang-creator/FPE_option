"""End-to-end GPU pipeline test: FPE solve + batch pricing.

Tests the complete logic chain:
1. FPE Solver (ODE integration)
2. PricingEngine (batch pricing)
3. Dtype management for cross-platform compatibility
"""
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams
from server.pricing_engine import PricingEngine
from engines.calibrator.calibrator import Calibrator

from gpu_utils.dtype import GPU_DTYPE


def test_gpu_fpe_solver_produces_valid_pricing() raises:
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var fp = FpeParams(
        heston=params^, n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[60.0]
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_true(len(results) > 0)
    assert_true(results[0].success)


def test_gpu_batch_pricing_produces_results() raises:
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var fp = FpeParams(
        heston=params^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[55.0, 60.0, 65.0, 70.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_true(len(results) == 4)
    for i in range(len(results)):
        assert_true(results[i].success)
        assert_true(results[i].price >= 0.0)


def test_gpu_calibration_converges() raises:
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

    var market_prices: List[Float64] = [5.0, 3.0, 1.0]
    var strikes: List[Float64] = [90.0, 100.0, 110.0]
    var expiries: List[Float64] = [0.5, 0.5, 0.5]

    var cal = Calibrator[1](max_iter=5, tol=1e-4)
    var result = cal.calibrate(market_prices, strikes, expiries, init)

    assert_true(result.kappa > 0.0)
    assert_true(result.theta > 0.0)
    assert_true(result.sigma > 0.0)
    assert_true(result.rho > -1.0 and result.rho < 1.0)
    assert_true(result.V0 > 0.0)


def test_gpu_dtype_is_float() raises:
    assert_true(GPU_DTYPE == DType.float32 or GPU_DTYPE == DType.float64)


def main() raises:
    print("=" * 60)
    print("End-to-End GPU Pipeline Test")
    print("GPU_DTYPE:", GPU_DTYPE)
    print("GPU available:", has_accelerator())
    print("=" * 60)

    TestSuite.discover_tests[__functions_in_module()]().run()
