from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from std.math import exp
from std.testing import assert_true, assert_equal, TestSuite


def _heston() -> HestonParams:
    return HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=0.0,
        S_max=180.0,
        V_min=0.0,
        V_max=1.0,
    )


def test_european_call_positive() raises:
    var h = _heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[60.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_equal(len(results), 1)
    assert_true(results[0].success)
    assert_true(results[0].price > 0.0)


def test_european_put_positive() raises:
    var h = _heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=9,
        strikes=[60.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_equal(len(results), 1)
    assert_true(results[0].success)
    assert_true(results[0].price > 0.0)


def test_up_and_out_call() raises:
    var h = _heston()
    var fp_vanilla = FpeParams(
        heston=h.copy(),
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[60.0],
    )
    var fp_uoc = FpeParams(
        heston=h.copy(),
        n_s=16,
        n_v=16,
        barrier=80.0,
        option_type=6,
        strikes=[60.0],
    )
    var engine = PricingEngine(num_insert=30)
    var vanilla = engine.price(fp_vanilla)
    var uoc = engine.price(fp_uoc)
    assert_true(vanilla[0].success)
    assert_true(uoc[0].success)
    assert_true(uoc[0].price > 0.0)
    assert_true(uoc[0].price < vanilla[0].price)


def test_down_and_out_call() raises:
    var h = _heston()
    var fp_vanilla = FpeParams(
        heston=h.copy(),
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[60.0],
    )
    var fp_doc = FpeParams(
        heston=h.copy(),
        n_s=16,
        n_v=16,
        barrier=40.0,
        option_type=2,
        strikes=[60.0],
    )
    var engine = PricingEngine(num_insert=30)
    var vanilla = engine.price(fp_vanilla)
    var doc = engine.price(fp_doc)
    assert_true(vanilla[0].success)
    assert_true(doc[0].success)
    assert_true(doc[0].price > 0.0)
    assert_true(doc[0].price < vanilla[0].price)


def test_multi_strike_all_positive() raises:
    var h = _heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[50.0, 55.0, 60.0, 65.0, 70.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_equal(len(results), 5)
    for i in range(len(results)):
        assert_true(results[i].success)
        assert_true(results[i].price >= 0.0)


def test_call_monotonicity() raises:
    var h = _heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[50.0, 55.0, 60.0, 65.0, 70.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    for i in range(len(results) - 1):
        assert_true(results[i].price >= results[i + 1].price)


def test_invalid_params_returns_failure() raises:
    var h = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=0.0,
        S_max=180.0,
        V_min=0.0,
        V_max=1.0,
    )
    var fp = FpeParams(
        heston=h^,
        n_s=8,
        n_v=8,
        barrier=50.0,
        option_type=6,
        strikes=[60.0],
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_true(len(results) >= 1)
    assert_true(not results[0].success)


def test_facade_single_price() raises:
    from fpe_option import price
    var h = _heston()
    var result = price(h.copy(), K=60.0, barrier=0.0, payoff_type=8, n_s=16, n_v=16)
    assert_true(result.success)
    assert_true(result.price > 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
