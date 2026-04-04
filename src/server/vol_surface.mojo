from engines.nais.inferencer import Inferencer
from engines.nais.nais_net import NaisNet

@fieldwise_init
struct VolSurfaceGenerator:
    """Generate implied volatility surface from NAIS-Net inference."""
    var net: NaisNet

    def generate(
        self, strikes: List[Float64], expiries: List[Float64]
    ) -> List[List[Float64]]:
        """Compute implied vol at each (K, T) grid point."""
        var surface: List[List[Float64]] = []
        for t_idx in range(len(expiries)):
            var row: List[Float64] = []
            for k_idx in range(len(strikes)):
                var result = self.net.forward(expiries[t_idx], [strikes[k_idx], 0.04])
                row.append(result[0])
            surface.append(row^)
        return surface^
