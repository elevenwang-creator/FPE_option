from std.math import exp

from engines.fpe.domain import FPEDomain
from engines.fpe.solver import FPESolver
from engines.fpe.pdf import pdf_from_cached
from server.option_types import FpeParams, PricingResult
from server.payoffs import BarrierPayoff


@fieldwise_init
struct PDFGrid(Copyable, Movable):
    var pdf: List[List[Float64]]
    var s_points: List[Float64]
    var v_points: List[Float64]
    var T: Float64
    var ds_weights: List[Float64]
    var dv_weights: List[Float64]


def _compute_trap_weights(points: List[Float64]) -> List[Float64]:
    var n = len(points)
    var w: List[Float64] = []
    for i in range(n):
        if i == 0 or i == n - 1:
            w.append(1.0)
        else:
            w.append((points[i + 1] - points[i - 1]) * 0.5)
    return w^


struct PricingEngine:
    var rtol: Float64
    var atol: Float64
    var num_insert: Int

    def __init__(out self, rtol: Float64 = 1e-4, atol: Float64 = 1e-6, num_insert: Int = 50):
        self.rtol = rtol
        self.atol = atol
        self.num_insert = num_insert

    def price(self, fpe_params: FpeParams) raises -> List[PricingResult]:
        if not fpe_params.is_valid():
            var err: List[PricingResult] = []
            err.append(
                PricingResult(
                    price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False
                )
            )
            return err^

        var revised = fpe_params.revised_heston()

        var domain = FPEDomain[3, 3](
            revised,
            n_s=fpe_params.n_s,
            n_v=fpe_params.n_v,
            num_insert=self.num_insert,
            s_left_cond=fpe_params.s_left_cond(),
            s_right_cond=fpe_params.s_right_cond(),
        )

        var solver = FPESolver[1](
            rtol=self.rtol,
            atol=self.atol,
            max_step=revised.T / 5.0,
            first_step=1e-6,
        )
        var sol = solver.solve(domain, revised)

        var cached = domain.cached_basis()
        var pdf_grid = pdf_from_cached[3, 3](cached, sol[len(sol) - 1])

        var payoff = BarrierPayoff(
            option_type=fpe_params.option_type,
            strikes=fpe_params.strikes.copy(),
            barrier=fpe_params.barrier,
        )

        var n_strikes = len(fpe_params.strikes)
        var prices: List[Float64] = []
        for _ in range(n_strikes):
            prices.append(0.0)

        var n_s = len(domain.s_points_phys)
        var n_v = len(domain.v_points_phys)
        var discount = exp(-revised.r * revised.T)

        for i in range(n_s):
            var S = domain.s_points_phys[i]
            var payoff_vals = payoff.evaluate(S)
            var payoff_nonzero = False
            for k in range(n_strikes):
                if payoff_vals[k] != 0.0:
                    payoff_nonzero = True
                    break
            if not payoff_nonzero:
                continue
            var ds_w = domain.s_weights[i]
            for j in range(n_v):
                var pdf_dv = pdf_grid[i][j] * domain.v_weights[j]
                for k in range(n_strikes):
                    prices[k] += payoff_vals[k] * pdf_dv * ds_w

        var results: List[PricingResult] = []
        for k in range(n_strikes):
            results.append(
                PricingResult(
                    price=prices[k] * discount,
                    delta=0.0,
                    gamma=0.0,
                    vega=0.0,
                    success=True,
                )
            )
        return results^
