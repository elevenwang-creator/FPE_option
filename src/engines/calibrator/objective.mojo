from numerics.utils import max_f64

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.pdf import PDFComputer
from engines.fpe.solver import FPESolver


def _with_maturity(base: HestonParams, maturity: Float64) -> HestonParams:
    return HestonParams(
        kappa=base.kappa,
        theta=base.theta,
        sigma=base.sigma,
        rho=base.rho,
        r=base.r,
        T=maturity,
        S0=base.S0,
        V0=base.V0,
        S_min=base.S_min,
        S_max=base.S_max,
        V_min=base.V_min,
        V_max=base.V_max,
    )


def _integrate_call_price(
    domain: FPEDomain, pdf: List[List[Float64]], strike: Float64
) -> Float64:
    var price = 0.0
    var n_s = len(domain.s_points)
    var n_v = len(domain.v_points)

    for i in range(n_s):
        var s_phys = domain.s_points_phys[i]
        var ds = 1.0
        if i > 0 and i < n_s - 1:
            var s_prev = domain.s_points_phys[i - 1]
            var s_next = domain.s_points_phys[i + 1]
            ds = 0.5 * (s_next - s_prev)

        var payoff = max_f64(s_phys - strike, 0.0)
        for j in range(n_v):
            var dv = 1.0
            if j > 0 and j < n_v - 1:
                var v_prev = domain.v_points_phys[j - 1]
                var v_next = domain.v_points_phys[j + 1]
                dv = 0.5 * (v_next - v_prev)
            price += payoff * pdf[i][j] * ds * dv

    return price


struct ObjectiveFunction[B: Int](Copyable, Movable):
    """Calibration objective: sum of squared pricing errors."""

    var market_prices: List[Float64]
    var strikes: List[Float64]
    var expiries: List[Float64]

    def __init__(
        out self,
        var market_prices: List[Float64],
        var strikes: List[Float64],
        var expiries: List[Float64],
    ):
        self.market_prices = market_prices^
        self.strikes = strikes^
        self.expiries = expiries^

    def __init__(out self, *, copy: Self):
        self.market_prices = copy.market_prices.copy()
        self.strikes = copy.strikes.copy()
        self.expiries = copy.expiries.copy()

    def compute(self, params: HestonParams) raises -> List[Float64]:
        """Returns residuals: model_price[i] - market_price[i] for each option.
        """
        if len(self.market_prices) != len(self.strikes) or len(
            self.market_prices
        ) != len(self.expiries):
            raise Error(
                "ObjectiveFunction: market/strike/expiry lengths must match"
            )

        var residuals: List[Float64] = []
        var solver = FPESolver[Self.B](
            rtol=1e-5, atol=1e-7, max_step=0.02, first_step=0.0
        )
        var pdf_comp = PDFComputer[Self.B]()

        for i in range(len(self.market_prices)):
            var maturity = self.expiries[i]
            var local_params = _with_maturity(params, maturity)
            var domain = FPEDomain[3, 3](local_params, n_s=8, n_v=8)
            var t_eval: List[Float64] = [0.0, maturity]
    var q_path = solver.solve(domain, local_params, t_eval^)
            var q_terminal = q_path[len(q_path) - 1].copy()
            var pdf = pdf_comp.compute(domain, q_terminal)
            var model_price = _integrate_call_price(
                domain, pdf, self.strikes[i]
            )
            residuals.append(model_price - self.market_prices[i])

        return residuals^
