from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams
from server.compute_pipeline import ComputePipeline
from std.testing import TestSuite, assert_true, assert_false, assert_equal


def _make_fp() -> FpeParams:
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
    return FpeParams(
        heston=h^, n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[60.0]
    )


def test_knots() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var tup = pipe.knots()
    assert_true(len(tup[0]) > 0)
    assert_true(len(tup[1]) > 0)


def test_grid_points() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var tup = pipe.grid_points()
    assert_true(len(tup[0]) > 0)
    assert_true(len(tup[1]) > 0)
    assert_equal(len(tup[0]), len(tup[2]))
    assert_equal(len(tup[1]), len(tup[3]))


def test_basis_1d() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var tup = pipe.basis_1d()
    assert_true(tup[0].nrows > 0)


def test_initial_condition() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var q0 = pipe.initial_condition()
    assert_true(len(q0) > 0)
    var q0b = pipe.initial_condition()
    assert_equal(len(q0b), len(q0))


def test_solve() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var sol = pipe.solve()
    assert_true(len(sol) > 0)
    assert_true(len(sol[0]) > 0)


def test_pdf() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var pdf = pipe.pdf()
    assert_true(len(pdf) > 0)


def test_price_at() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var prices = pipe.price_at([60.0])
    assert_equal(len(prices), 1)
    assert_true(prices[0] >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
