from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine, PDFGrid
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _make_heston() -> HestonParams:
    return HestonParams(
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


def test_european_call() raises:
    var h = _make_heston()
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
    assert_true(results[0].price >= 0.0)


def test_up_and_out_call() raises:
    var h = _make_heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=80.0,
        option_type=6,
        strikes=[60.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_true(results[0].success)
    assert_true(results[0].price >= 0.0)


def test_multi_strike_european() raises:
    var h = _make_heston()
    var fp = FpeParams(
        heston=h^,
        n_s=16,
        n_v=16,
        barrier=0.0,
        option_type=8,
        strikes=[55.0, 60.0, 65.0],
    )
    var engine = PricingEngine(num_insert=30)
    var results = engine.price(fp)
    assert_equal(len(results), 3)
    for k in range(len(results)):
        assert_true(results[k].success)
        assert_true(results[k].price >= 0.0)


def test_invalid_params() raises:
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
        heston=h_invalid^,
        n_s=8,
        n_v=8,
        barrier=90.0,
        option_type=6,
        strikes=[60.0],
    )
    var engine = PricingEngine()
    var results = engine.price(fp)
    assert_false(results[0].success)


def main() raises:
    test_european_call()
