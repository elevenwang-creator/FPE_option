from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams
from server.pricing_engine import PricingEngine
from std.testing import TestSuite, assert_true, assert_false


def test_pricing_engine_invalid_params() raises:
    var h_invalid = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.1,
        S0=100.0,
        V0=0.1,
        S_min=0.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )
    var fp = FpeParams(
        heston=h_invalid^, n_s=8, n_v=8, barrier=90.0, option_type=6, strikes=[60.0]
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_true(len(results) == 1)
    assert_false(results[0].success)


def test_pricing_engine_with_valid_params() raises:
    var h = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=20.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )
    var fp = FpeParams(
        heston=h^, n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[60.0]
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_true(len(results) == 1)
    assert_true(results[0].success)
    assert_true(results[0].price >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
