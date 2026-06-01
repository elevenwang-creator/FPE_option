from numerics.utils import mat_vec_mul, mat_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major

from std.math import sqrt
from std.memory import Span


@fieldwise_init
struct StableLinear(Copyable, Movable):
    var W_T_flat: List[Float64]
    var b: List[Float64]
    var epsilon: Float64
    var in_features: Int
    var out_features: Int

    def forward(self, x: List[Float64]) -> List[Float64]:
        var W_c = self._constrain_weight()
        if len(W_c) == 0:
            return self.b.copy()
        var A = TileTensor(
            W_c, row_major(coord[DType.int64]((self.out_features, self.out_features)))
        )
        var result = List[Float64](length=self.out_features, fill=0.0)
        var r_span = Span[mut=True, Float64](result)
        mat_vec_mul(A, Span[Float64](x), r_span)
        for j in range(self.out_features):
            result[j] += self.b[j]
        return result^

    def _constrain_weight(self) -> List[Float64]:
        """W_c = -A where A = scale * (W^T @ W) + eps*I."""
        var nf = self.in_features
        var mf = self.out_features
        if nf == 0 or mf == 0:
            return List[Float64]()

        var delta = 1.0 - 2.0 * self.epsilon
        if delta < 1e-12:
            delta = 1e-12

        var W_flat = List[Float64](length=nf * mf, fill=0.0)
        for i in range(nf):
            for j in range(mf):
                W_flat[i * mf + j] = self.W_T_flat[j * nf + i]

        var WT_t = TileTensor(
            self.W_T_flat, row_major(coord[DType.int64]((mf, nf)))
        )
        var W_t = TileTensor(
            W_flat, row_major(coord[DType.int64]((nf, mf)))
        )

        var RtR_flat = List[Float64](length=mf * mf, fill=0.0)
        var RtR_t = TileTensor(
            RtR_flat, row_major(coord[DType.int64]((mf, mf)))
        )
        mat_mul(WT_t, W_t, RtR_t)

        var v = List[Float64](length=mf, fill=1.0)
        var spectral_norm: Float64 = 1.0
        for _ in range(20):
            var Rv = List[Float64](length=mf, fill=0.0)
            var Rv_span = Span[mut=True, Float64](Rv)
            mat_vec_mul(RtR_t, Span[Float64](v), Rv_span)
            spectral_norm = 0.0
            for i in range(mf):
                spectral_norm += Rv[i] * Rv[i]
                v[i] = Rv[i]
            spectral_norm = sqrt(spectral_norm)
            if spectral_norm > 1e-12:
                for i in range(mf):
                    v[i] /= spectral_norm

        if spectral_norm < 1e-12:
            spectral_norm = 1e-12

        var scale = sqrt(delta) / sqrt(spectral_norm)
        var res = List[Float64](length=mf * mf, fill=0.0)
        for i in range(mf):
            for j in range(mf):
                res[i * mf + j] = -scale * RtR_flat[i * mf + j]
            res[i * mf + i] -= self.epsilon
        return res^


def make_stable_linear(
    in_features: Int, out_features: Int, epsilon: Float64 = 0.01
) -> StableLinear:
    var W_flat = List[Float64](length=out_features * in_features, fill=0.0)
    for i in range(in_features):
        for j in range(out_features):
            var seed_i = ((i + 1) * 17 + (j + 1) * 13) % 11
            W_flat[j * in_features + i] = 0.02 * (Float64(seed_i) - 5.0) / 5.0
    var b = List[Float64](length=out_features, fill=0.0)
    return StableLinear(
        W_T_flat=W_flat^,
        b=b^,
        epsilon=epsilon,
        in_features=in_features,
        out_features=out_features,
    )
