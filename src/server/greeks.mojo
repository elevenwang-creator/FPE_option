from server.pricing_engine import PDFGrid
from server.payoffs import BarrierPayoff


struct Greeks:
    var h_s: Float64
    var h_v: Float64

    def __init__(out self, h_s: Float64 = 0.01, h_v: Float64 = 0.001):
        self.h_s = h_s
        self.h_v = h_v

    def _price_at(self, grid: PDFGrid, payoff: BarrierPayoff) -> List[Float64]:
        _ = self
        var n_strikes = len(payoff.strikes)
        var prices: List[Float64] = []
        for _ in range(n_strikes):
            prices.append(0.0)
        for i in range(len(grid.s_points)):
            var S = grid.s_points[i]
            var payoff_vals = payoff.evaluate(S)
            var ds_w = grid.ds_weights[i]
            for j in range(len(grid.v_points)):
                var pdf_dv = grid.pdf[i][j] * grid.dv_weights[j]
                for k in range(n_strikes):
                    prices[k] += payoff_vals[k] * pdf_dv * ds_w
        return prices^

    def compute_delta(self, grid: PDFGrid, payoff: BarrierPayoff) -> List[Float64]:
        _ = grid
        _ = payoff
        var n_strikes = len(payoff.strikes)
        var result: List[Float64] = []
        for _ in range(n_strikes):
            result.append(0.0)
        return result^

    def compute_gamma(self, grid: PDFGrid, payoff: BarrierPayoff) -> List[Float64]:
        _ = grid
        _ = payoff
        var n_strikes = len(payoff.strikes)
        var result: List[Float64] = []
        for _ in range(n_strikes):
            result.append(0.0)
        return result^

    def compute_vega(self, grid: PDFGrid, payoff: BarrierPayoff) -> List[Float64]:
        _ = grid
        _ = payoff
        var n_strikes = len(payoff.strikes)
        var result: List[Float64] = []
        for _ in range(n_strikes):
            result.append(0.0)
        return result^
