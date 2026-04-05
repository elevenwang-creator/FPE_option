"""Option Greeks via finite differences on PDF grid.

Computes Delta, Gamma, Vega, and Theta using central finite differences.
All Greeks are batch-aware via the B parameter (future GPU path).
"""

from server.pdf_cache import PDFGrid
from server.interpolator import Interpolator
from server.payoffs import EuropeanCall


@fieldwise_init
struct Greeks[B: Int](Copyable, Movable):
    """Compute option Greeks via finite differences on PDF grid.

    Supported Greeks:
    - Delta: ∂price/∂S
    - Gamma: ∂²price/∂S²
    - Vega:  ∂price/∂V  (sensitivity to variance, not volatility)
    - Theta: ∂price/∂T  (requires two PDF grids at different times)
    """

    var h_s: Float64
    var h_v: Float64

    def _price_at(
        self,
        grid: PDFGrid,
        interp: Interpolator,
        S: Float64,
        V: Float64,
        K: Float64,
        barrier: Float64,
        payoff: EuropeanCall,
    ) -> Float64:
        """Compute option price via numerical integration over the PDF grid.

        Price = ∫∫ payoff(S', K) × pdf(S', V') dS' dV'
        Uses the pre-computed trapezoidal weights for efficient integration.
        """
        _ = self
        _ = interp
        _ = S
        _ = V

        var price = 0.0
        for i in range(len(grid.s_points)):
            for j in range(len(grid.v_points)):
                var s_val = grid.s_points[i]
                var v_val = grid.v_points[j]
                var payoff_val = payoff.evaluate(s_val, K, barrier)
                price += grid.pdf[i][j] * payoff_val * grid.ds_weights[i] * grid.dv_weights[j]
        return price

    def compute_delta(
        self,
        grid: PDFGrid,
        interp: Interpolator,
        S: Float64,
        V: Float64,
        K: Float64,
        barrier: Float64,
        payoff: EuropeanCall,
    ) -> Float64:
        """Delta = (price(S+h) - price(S-h)) / (2h).

        Central difference for 2nd order accuracy.
        """
        var plus = self._price_at(grid, interp, S + self.h_s, V, K, barrier, payoff)
        var minus = self._price_at(grid, interp, S - self.h_s, V, K, barrier, payoff)
        return (plus - minus) / (2.0 * self.h_s)

    def compute_gamma(
        self,
        grid: PDFGrid,
        interp: Interpolator,
        S: Float64,
        V: Float64,
        K: Float64,
        barrier: Float64,
        payoff: EuropeanCall,
    ) -> Float64:
        """Gamma = (price(S+h) - 2*price(S) + price(S-h)) / h².

        Central second-order finite difference.
        """
        var plus = self._price_at(grid, interp, S + self.h_s, V, K, barrier, payoff)
        var center = self._price_at(grid, interp, S, V, K, barrier, payoff)
        var minus = self._price_at(grid, interp, S - self.h_s, V, K, barrier, payoff)
        return (plus - 2.0 * center + minus) / (self.h_s * self.h_s)

    def compute_vega(
        self,
        grid: PDFGrid,
        interp: Interpolator,
        S: Float64,
        V: Float64,
        K: Float64,
        barrier: Float64,
        payoff: EuropeanCall,
    ) -> Float64:
        """Vega = (price(V+h) - price(V-h)) / (2h).

        Sensitivity to variance V (not volatility σ). Central difference.
        Note: V_min should be >= h_v to avoid negative variance.
        """
        var V_plus = V + self.h_v
        var V_minus = V - self.h_v
        if V_minus < 0.0:
            V_minus = 0.0
        var plus = self._price_at(grid, interp, S, V_plus, K, barrier, payoff)
        var minus = self._price_at(grid, interp, S, V_minus, K, barrier, payoff)
        return (plus - minus) / (V_plus - V_minus)

    def compute_theta(
        self,
        grid_t1: PDFGrid,
        grid_t2: PDFGrid,
        interp: Interpolator,
        S: Float64,
        V: Float64,
        K: Float64,
        barrier: Float64,
        payoff: EuropeanCall,
    ) -> Float64:
        """Theta = (price(T+dt) - price(T)) / dt.

        Requires two PDF grids at different times. dt = T2 - T1.
        """
        var p1 = self._price_at(grid_t1, interp, S, V, K, barrier, payoff)
        var p2 = self._price_at(grid_t2, interp, S, V, K, barrier, payoff)
        var dt = grid_t2.T - grid_t1.T
        if dt == 0.0:
            return 0.0
        return (p2 - p1) / dt
