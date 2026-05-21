from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams, PricingResult
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def _make_heston() -> HestonParams:
    return HestonParams(
        kappa=1.5,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        r=0.03,
        T=1.0,
        S0=100.0,
        V0=0.04,
        S_min=10.0,
        S_max=500.0,
        V_min=0.0,
        V_max=1.0,
    )


def test_vanilla_valid() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[100.0])
    assert_true(fp.is_valid())


def test_up_barrier_valid() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=120.0, option_type=6, strikes=[100.0])
    assert_true(fp.is_valid())


def test_down_barrier_valid() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=80.0, option_type=0, strikes=[100.0])
    assert_true(fp.is_valid())


def test_up_barrier_invalid_low() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=90.0, option_type=6, strikes=[100.0])
    assert_false(fp.is_valid())


def test_vanilla_with_barrier_invalid() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=120.0, option_type=8, strikes=[100.0])
    assert_false(fp.is_valid())


def test_revised_heston_up() raises:
    var h = _make_heston()
    var fp = FpeParams(heston=h.copy(), n_s=64, n_v=32, barrier=120.0, option_type=6, strikes=[100.0])
    var rev = fp.revised_heston()
    assert_equal(rev.S_max, 120.0)
    assert_equal(rev.S_min, h.S_min)
    assert_equal(rev.S0, h.S0)


def test_revised_heston_down() raises:
    var h = _make_heston()
    var fp = FpeParams(heston=h.copy(), n_s=64, n_v=32, barrier=80.0, option_type=0, strikes=[100.0])
    var rev = fp.revised_heston()
    assert_equal(rev.S_min, 80.0)
    assert_equal(rev.S_max, h.S_max)
    assert_equal(rev.S0, h.S0)


def test_revised_heston_vanilla() raises:
    var h = _make_heston()
    var fp = FpeParams(heston=h.copy(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[100.0])
    var rev = fp.revised_heston()
    assert_equal(rev.S_min, h.S_min)
    assert_equal(rev.S_max, h.S_max)


def test_s_left_cond_down() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=80.0, option_type=0, strikes=[100.0])
    assert_equal(fp.s_left_cond(), "dirichlet")


def test_s_left_cond_up() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=120.0, option_type=6, strikes=[100.0])
    assert_equal(fp.s_left_cond(), "neumann")


def test_s_left_cond_vanilla() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[100.0])
    assert_equal(fp.s_left_cond(), "dirichlet")


def test_s_right_cond_down() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=80.0, option_type=0, strikes=[100.0])
    assert_equal(fp.s_right_cond(), "neumann")


def test_s_right_cond_up() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=120.0, option_type=6, strikes=[100.0])
    assert_equal(fp.s_right_cond(), "dirichlet")


def test_s_right_cond_vanilla() raises:
    var fp = FpeParams(heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[100.0])
    assert_equal(fp.s_right_cond(), "neumann")


def test_multi_strikes() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[95.0, 100.0, 105.0]
    )
    assert_true(fp.is_valid())
    assert_equal(len(fp.strikes), 3)


def test_empty_strikes_invalid() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[]
    )
    assert_false(fp.is_valid())


def test_negative_strike_invalid() raises:
    var fp = FpeParams(
        heston=_make_heston(), n_s=64, n_v=32, barrier=0.0, option_type=8, strikes=[-1.0]
    )
    assert_false(fp.is_valid())


def test_pricing_result() raises:
    var pr = PricingResult(price=1.0, delta=0.5, gamma=0.1, vega=0.2, success=True)
    assert_true(pr.success)
    assert_equal(pr.price, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
