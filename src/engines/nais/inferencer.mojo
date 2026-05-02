from numerics.utils import pow_pos, max_f64, abs_f64

from engines.nais.nais_net import NaisNet
from std.math import exp, log, sqrt


def _norm_cdf(x: Float64) -> Float64:
    """Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
    var a1 = 0.254829592
    var a2 = -0.284496736
    var a3 = 1.421413741
    var a4 = -1.453152027
    var a5 = 1.061405429
    var p = 0.3275911
    var sign = 1.0
    if x < 0.0:
        sign = -1.0
    var ax = x * sign
    var t = 1.0 / (1.0 + p * ax)
    var y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    return 0.5 * (1.0 + sign * y)


def _bs_call_price(S: Float64, K: Float64, T: Float64, r: Float64, sigma: Float64) -> Float64:
    """Black-Scholes call option price."""
    if sigma <= 0.0 or T <= 0.0:
        return max_f64(S - K, 0.0)
    var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    var d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)


def _bs_vega(S: Float64, K: Float64, T: Float64, r: Float64, sigma: Float64) -> Float64:
    """Black-Scholes Vega = dC/d(sigma)."""
    if sigma <= 0.0 or T <= 0.0:
        return 0.0
    var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return S * _norm_cdf(d1) * sqrt(T)


def _implied_vol_newton(
    price: Float64, S: Float64, K: Float64, T: Float64, r: Float64
) -> Float64:
    """Implied volatility via Newton's method on Black-Scholes call price."""
    if T <= 1e-10:
        return 0.0
    var sigma = 0.3
    for _ in range(50):
        var bs_price = _bs_call_price(S, K, T, r, sigma)
        var diff = bs_price - price
        if abs_f64(diff) < 1e-8:
            return sigma
        var vega = _bs_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        sigma = sigma - diff / vega
        if sigma < 0.001:
            sigma = 0.001
        if sigma > 5.0:
            sigma = 5.0
    return sigma


@fieldwise_init
struct Inferencer[B: Int]:
    """Online inference: (t, S, V) -> (price, delta, vol_surface)."""

    var net: NaisNet
    var risk_free_rate: Float64

    def infer(
        self, t: Float64, S: Float64, V: Float64
    ) -> Tuple[Float64, Float64]:
        """Returns (price, delta)."""
        var out = self.net.forward(t, [S, V])
        var u = out[0]
        var phi = out[1].copy()
        var delta = 0.0
        if len(phi) > 0:
            delta = phi[0]
        return (u, delta)

    def vol_surface(
        self, strikes: List[Float64], expiries: List[Float64]
    ) -> List[List[Float64]]:
        """Generate implied vol surface by inverting Black-Scholes on the KxT grid."""
        var r = self.risk_free_rate
        var surface: List[List[Float64]] = []
        for i in range(len(expiries)):
            var row: List[Float64] = []
            var t = expiries[i]
            for j in range(len(strikes)):
                var K = strikes[j]
                var price, _ = self.infer(t, K, 0.04)
                if price <= 0.0 or K <= 0.0 or t <= 1e-10:
                    row.append(0.0)
                else:
                    var iv = _implied_vol_newton(price, K, K, t, r)
                    if iv < 0.0:
                        iv = 0.0
                    row.append(iv)
            surface.append(row^)
        return surface^