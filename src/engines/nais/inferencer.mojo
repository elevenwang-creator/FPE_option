from numerics.utils import pow_pos

from engines.nais.nais_net import NaisNet
from std.math import exp, log




@fieldwise_init
struct Inferencer[B: Int]:
    """Online inference: (t, S, V) → (price, delta, vol_surface)."""

    var net: NaisNet

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
        """Generate implied vol surface by sweeping K×T grid."""
        var surface: List[List[Float64]] = []
        for i in range(len(expiries)):
            var row: List[Float64] = []
            var t = expiries[i]
            for j in range(len(strikes)):
                var K = strikes[j]
                var price, _ = self.infer(t, K, 0.04)
                var denom = K * pow_pos(t + 1e-6, 0.5)
                if denom <= 1e-8:
                    row.append(0.0)
                else:
                    var iv = price / denom
                    if iv < 0.0:
                        iv = -iv
                    row.append(iv)
            surface.append(row^)
        return surface^
