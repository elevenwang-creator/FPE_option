from numerics.utils import zeros_mat, mat_vec_mul, mat_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major

from std.math import sqrt
from std.memory import Span


@always_inline
def _tile_to_list(t: TileTensor[DType.float64, ...], rows: Int, cols: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for i in range(rows):
        var row: List[Float64] = []
        for j in range(cols):
            row.append(t.raw_load[width=1](i * cols + j)[0])
        out.append(row^)
    return out^


def _matmul_vec(
    W: List[List[Float64]], x: List[Float64], b: List[Float64]
) -> List[Float64]:
    var in_dim = len(W)
    var out_dim = len(b)
    if in_dim == 0:
        return b.copy()
    var n = min(in_dim, len(x))

    var flat = List[Float64](length=n * out_dim, fill=0.0)
    for i in range(n):
        var wi = W[i].copy()
        for j in range(out_dim):
            flat[j * n + i] = wi[j]
    var A = TileTensor(flat, row_major(coord[DType.int64]((out_dim, n))))
    var y = List[Float64](length=out_dim, fill=0.0)
    var y_span = Span[mut=True, Float64](y)
    mat_vec_mul(A, Span[Float64](x), y_span)
    for j in range(out_dim):
        y[j] += b[j]
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
            return self.b.copy()
        var in_dim = len(W_c)
        var out_dim = len(self.b)
        var n = min(in_dim, len(x))

        var flat = List[Float64](length=n * out_dim, fill=0.0)
        for i in range(n):
            var wi = W_c[i].copy()
            for j in range(out_dim):
                flat[j * n + i] = wi[j]
        var A = TileTensor(flat, row_major(coord[DType.int64]((out_dim, n))))
        var result = List[Float64](length=out_dim, fill=0.0)
        var r_span = Span[mut=True, Float64](result)
        mat_vec_mul(A, Span[Float64](x), r_span)
        for j in range(out_dim):
            result[j] += self.b[j]
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

        # Build Gram matrix: RtR = W^T @ W using SIMD mat_mul
        var W_flat = List[Float64](length=in_features * out_features, fill=0.0)
        var WT_flat = List[Float64](length=in_features * out_features, fill=0.0)
        for k in range(in_features):
            var wk = self.W[k].copy()
            for i in range(out_features):
                var val = wk[i]
                W_flat[k * out_features + i] = val
                WT_flat[i * in_features + k] = val

        var W_t = TileTensor(W_flat, row_major(coord[DType.int64]((in_features, out_features))))
        var WT_t = TileTensor(WT_flat, row_major(coord[DType.int64]((out_features, in_features))))
        var RtR_flat = List[Float64](length=out_features * out_features, fill=0.0)
        var RtR_t = TileTensor(RtR_flat, row_major(coord[DType.int64]((out_features, out_features))))
        mat_mul(WT_t, W_t, RtR_t)

        # Convert RtR back to List[List[Float64]] for downstream use
        var RtR = _tile_to_list(RtR_t, out_features, out_features)

        # Estimate spectral norm via power iteration (20 iterations)
        var v = List[Float64](length=out_features, fill=1.0)
        var spectral_norm: Float64 = 1.0
        for _ in range(20):
            var Rv: List[Float64] = List[Float64](length=out_features, fill=0.0)
            var Rv_span = Span[mut=True, Float64](Rv)
            mat_vec_mul(RtR_t, Span[Float64](v), Rv_span)
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
    var b = List[Float64](length=out_features, fill=0.0)
    for i in range(in_features):
        for j in range(out_features):
            var seed_i = ((i + 1) * 17 + (j + 1) * 13) % 11
            W[i][j] = 0.02 * (Float64(seed_i) - 5.0) / 5.0
    return StableLinear(W=W^, b=b^, epsilon=epsilon)
