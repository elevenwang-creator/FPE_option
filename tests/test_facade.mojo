from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _make_heston() -> HestonParams:
    return HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=1e-4,
        V_max=1.0,
    )


def test_fpe_params_validation() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[100.0]
    )
    assert_true(fp.is_valid(), "Valid params should pass")


def test_fpe_params_invalid_barrier() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=16, n_v=16, barrier=120.0, option_type=8, strikes=[100.0]
    )
    assert_false(fp.is_valid(), "Vanilla with barrier should fail")


def test_pricing_engine_call_option() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[100.0]
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Call pricing should succeed")
    assert_true(results[0].price > 0.0, "Call price should be positive")


def test_pricing_engine_barrier_up_and_out() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=16, n_v=16, barrier=120.0, option_type=6, strikes=[100.0]
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Barrier pricing should succeed")
    assert_true(results[0].price >= 0.0, "Barrier price should be non-negative")


def test_pricing_engine_put_option() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=16, n_v=16, barrier=0.0, option_type=9, strikes=[110.0]
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Put pricing should succeed")
    assert_true(results[0].price > 0.0, "Put price should be positive")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
