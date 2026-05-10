from numerics.utils import min_f64, zeros, zeros_mat

from std.math import sqrt


def _matmul_vec(
    W: List[List[Float64]], x: List[Float64], b: List[Float64]
) -> List[Float64]:
    var out_dim = len(b)
    var y = zeros(out_dim)
    for j in range(out_dim):
        y[j] = b[j]

    for i in range(len(W)):
        var xi = 0.0
        if i < len(x):
            xi = x[i]
        for j in range(out_dim):
            y[j] = y[j] + W[i][j] * xi

    return y^


@fieldwise_init
struct StableLinear(Copyable, Movable):
    """Weight-constrained linear layer from NAIS-Net.
    Constraint: ||W^T W||_2 <= 1 - 2*epsilon
    """

    var W: List[List[Float64]]
    var b: List[Float64]
    var epsilon: Float64

    def forward(self, x: List[Float64]) -> List[Float64]:
        """Y = constrained_W @ x + b."""
        var W_c = self._constrain_weight()
        if len(W_c) == 0:
            var b_copy = List[Float64]()
            for i in range(len(self.b)):
                b_copy.append(self.b[i])
            return b_copy^
        var in_dim = len(W_c)
        var out_dim = len(self.b)

        var result = List[Float64]()
        for j in range(out_dim):
            var sum = 0.0
            for i in range(in_dim):
                if i < len(x):
                    sum += W_c[i][j] * x[i]
            result.append(sum + self.b[j])

        return result^

    def _constrain_weight(self) -> List[List[Float64]]:
        """Apply weight constraint: W_c = -A where A = RtR_update + eps*I."""
        var in_features = len(self.W)
        if in_features == 0:
            var empty: List[List[Float64]] = []
            return empty^

        var out_features = len(self.W[0])
        var delta = 1.0 - 2.0 * self.epsilon
        if delta < 1e-12:
            delta = 1e-12

        var RtR = zeros_mat(out_features, out_features)
        for i in range(out_features):
            for j in range(out_features):
                var s = 0.0
                for k in range(in_features):
                    s = s + self.W[k][i] * self.W[k][j]
                RtR[i][j] = s

        # Estimate spectral norm via power iteration (20 iterations)
        # instead of Frobenius norm which doesn't bound the operator norm
        var v: List[Float64] = []
        for i in range(out_features):
            v.append(1.0)

        var spectral_norm: Float64 = 1.0
        for _ in range(20):
            var Rv: List[Float64] = zeros(out_features)
            for i in range(out_features):
                for j in range(out_features):
                    Rv[i] += RtR[i][j] * v[j]
            spectral_norm = 0.0
            for i in range(out_features):
                spectral_norm += Rv[i] * Rv[i]
            spectral_norm = sqrt(spectral_norm)
            if spectral_norm > 1e-12:
                for i in range(out_features):
                    v[i] = Rv[i] / spectral_norm

        if spectral_norm < 1e-12:
            spectral_norm = 1e-12

        var scale = sqrt(delta) / sqrt(spectral_norm)
        var RtR_update = zeros_mat(out_features, out_features)
        for i in range(out_features):
            for j in range(out_features):
                RtR_update[i][j] = scale * RtR[i][j]

        var A = zeros_mat(out_features, out_features)
        for i in range(out_features):
            for j in range(out_features):
                A[i][j] = RtR_update[i][j]
            A[i][i] = A[i][i] + self.epsilon

        var constrained = zeros_mat(out_features, out_features)
        for i in range(out_features):
            for j in range(out_features):
                constrained[i][j] = -A[i][j]
        return constrained^


def make_stable_linear(
    in_features: Int, out_features: Int, epsilon: Float64 = 0.01
) -> StableLinear:
    var W = zeros_mat(in_features, out_features)
    var b = zeros(out_features)
    for i in range(in_features):
        for j in range(out_features):
            var seed_i = ((i + 1) * 17 + (j + 1) * 13) % 11
            W[i][j] = 0.02 * (Float64(seed_i) - 5.0) / 5.0
    return StableLinear(W=W^, b=b^, epsilon=epsilon)
