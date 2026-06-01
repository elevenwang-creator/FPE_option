"""Integration tests for Compute (stepwise API) and price (one-shot API)."""

import numpy as np
import pytest
from scipy import sparse

from fpe_engine import Compute, price, KnotsResult, GridPointsResult, Basis1DResult, GreeksResult, PriceResult


@pytest.fixture
def ctx():
    return Compute(S0=60.0, V0=0.1, T=0.6, r=0.1, n_s=8, n_v=8)


class TestInvalidParams:
    def test_invalid_n_s(self):
        with pytest.raises(Exception):
            Compute(n_s=1)

    def test_invalid_option_type(self):
        with pytest.raises(Exception):
            Compute(option_type="invalid")


class TestKnots:
    def test_returns_knots_result(self, ctx):
        k = ctx.knots
        assert isinstance(k, KnotsResult)
        assert isinstance(k.s, np.ndarray)
        assert isinstance(k.v, np.ndarray)
        assert k.s.dtype == np.float64
        assert len(k.s) > 0
        assert len(k.v) > 0

    def test_cached(self, ctx):
        k1 = ctx.knots
        k2 = ctx.knots
        assert k1 is k2


class TestGridPoints:
    def test_returns_grid_points_result(self, ctx):
        gp = ctx.grid_points
        assert isinstance(gp, GridPointsResult)
        assert isinstance(gp.s, np.ndarray)
        assert gp.s.dtype == np.float64
        assert len(gp.s) > 0

    def test_s_weights(self, ctx):
        gp = ctx.grid_points
        assert isinstance(gp.s_weights, np.ndarray)
        assert gp.s_weights.dtype == np.float64
        assert len(gp.s_weights) == len(gp.s)
        assert np.all(gp.s_weights >= 0.0)

    def test_cached(self, ctx):
        gp1 = ctx.grid_points
        gp2 = ctx.grid_points
        assert gp1 is gp2


class TestBasis1D:
    def test_returns_basis_1d_result(self, ctx):
        b = ctx.basis_1d
        assert isinstance(b, Basis1DResult)
        assert isinstance(b.Bs, sparse.csr_matrix)
        assert b.Bs.shape[0] > 0

    def test_cached(self, ctx):
        b1 = ctx.basis_1d
        b2 = ctx.basis_1d
        assert b1 is b2


class TestBasis2D:
    def test_returns_csr_matrix(self, ctx):
        b = ctx.basis_2d
        assert isinstance(b, sparse.csr_matrix)
        assert b.shape[0] > 0


class TestInitialCondition:
    def test_returns_ndarray(self, ctx):
        q0 = ctx.initial_condition
        assert isinstance(q0, np.ndarray)
        assert q0.dtype == np.float64


class TestSolve:
    def test_returns_list_of_ndarray(self, ctx):
        sol = ctx.solve
        assert isinstance(sol, list)
        assert len(sol) > 0
        for t in sol:
            assert isinstance(t, np.ndarray)
            assert t.dtype == np.float64


class TestPDF:
    def test_returns_2d_array(self, ctx):
        pdf = ctx.pdf
        assert isinstance(pdf, np.ndarray)
        assert pdf.ndim == 2
        assert pdf.dtype == np.float64


class TestPayoffPrice:
    def test_single_strike(self, ctx):
        p = ctx.payoff_price([100.0])
        assert isinstance(p, np.ndarray)
        assert len(p) == 1

    def test_multiple_strikes(self, ctx):
        p = ctx.payoff_price([80.0, 100.0, 120.0])
        assert isinstance(p, np.ndarray)
        assert len(p) == 3

    def test_k_variation(self, ctx):
        p1 = ctx.payoff_price([80.0])
        p2 = ctx.payoff_price([120.0])
        assert p1[0] != p2[0]


class TestPriceOneShot:
    def test_returns_price_result(self):
        pr = price(S0=60.0, K=100.0, T=0.6, n_s=8, n_v=8)
        assert isinstance(pr, PriceResult)
        assert isinstance(pr.prices, np.ndarray)
        assert isinstance(pr.deltas, np.ndarray)

    def test_multiple_strikes(self):
        pr = price(S0=60.0, K=[80.0, 100.0, 120.0], T=0.6, n_s=8, n_v=8)
        assert len(pr.prices) == 3


class TestDomainBounds:
    def test_default_s_range(self):
        c = Compute(S0=60.0, V0=0.1, T=0.6, r=0.1, n_s=8, n_v=8)
        gp = c.grid_points
        assert gp.s[0] >= 0.0
        assert gp.s[-1] <= 60.0 * 3.0

    def test_custom_s_range(self):
        c = Compute(
            S0=60.0, V0=0.1, T=0.6, r=0.1,
            n_s=8, n_v=8,
            s_min=10.0, s_max=200.0,
        )
        gp = c.grid_points
        assert gp.s[0] >= 10.0
        assert gp.s[-1] <= 200.0

    def test_custom_s_range_one_shot(self):
        pr = price(
            S0=60.0, K=50.0, T=0.6, n_s=16, n_v=16,
            s_min=10.0, s_max=200.0,
        )
        assert len(pr.prices) == 1
        assert pr.prices[0] > 0


class TestGreeks:
    def test_single_strike(self, ctx):
        g = ctx.greeks([100.0])
        assert isinstance(g, GreeksResult)
        assert isinstance(g.delta, np.ndarray)
        assert g.delta.shape == (1,)
        assert g.gamma.shape == (1,)
        assert g.vega.shape == (1,)

    def test_multiple_strikes(self, ctx):
        g = ctx.greeks([65.0, 70.0, 75.0, 80.0])
        assert g.delta.shape == (4,)
        assert g.gamma.shape == (4,)
        assert g.vega.shape == (4,)

    def test_delta_negative(self, ctx):
        g = ctx.greeks([65.0, 100.0])
        assert np.all(g.delta < 0)
