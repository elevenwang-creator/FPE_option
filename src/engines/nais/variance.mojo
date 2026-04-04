from numerics.utils import pow_pos

from engines.nais.volterra import VolterraProcess
from std.math import exp, log




@fieldwise_init
struct VarianceProcess[B: Int]:
    """Rough Bergomi variance process: ε(t)·exp(η·X̃ - 0.5η²t^{2H})."""

    var T: Float64
    var N: Int
    var D: Int
    var H: Float64
    var eta: Float64
    var epsilon_t: Float64

    def compute(self, W: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var volterra = VolterraProcess[Self.B](T=self.T, N=self.N, D=self.D, H=self.H)
        var X_tilde = volterra.generate(W)
        var M = len(X_tilde)
        var var_process = X_tilde.copy()

        for m in range(M):
            for n in range(self.N + 1):
                var t = self.T * Float64(n) / Float64(self.N)
                var correction = 0.5 * self.eta * self.eta * pow_pos(t, 2.0 * self.H)
                for d in range(self.D):
                    var_process[m][n][d] = self.epsilon_t * exp(self.eta * X_tilde[m][n][d] - correction)

        return var_process^
