"""B-spline knot generation with Gauss-Legendre quadrature.

Optimized with vectorize + Span + unsafe_ptr:
- chebyshev_knots: vectorize angle/cos/transform
- func_parabolic: sort(Span) + vectorize eval
- internal knot merge: extend + sort(Span) + inline round + unique scan
- generate_knots: sequential boundary fill + copy internal
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.math import pi, cos, sqrt, abs, round
from std.sys import simd_width_of
from numerics.utils.helpers import linspace, normalize

comptime SIMD_W: Int = simd_width_of[DType.float64]()


struct GaussLegendre:
    var order: Int
    var nodes: List[Float64]
    var weights: List[Float64]

    def __init__(out self, order: Int):
        self.order = order
        if order == 2:
            var s = sqrt(1.0 / 3.0)
            self.nodes = [-s, s]
            self.weights = [0.5, 0.5]
        elif order == 3:
            self.nodes = [-sqrt(3.0 / 5.0), 0.0, sqrt(3.0 / 5.0)]
            self.weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif order == 4:
            var s1 = sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(5.0 / 6.0))
            var s2 = sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(5.0 / 6.0))
            var w1 = (18.0 - sqrt(30.0)) / 36.0
            var w2 = (18.0 + sqrt(30.0)) / 36.0
            self.nodes = [-s1, -s2, s2, s1]
            self.weights = [w1, w2, w2, w1]
        elif order == 5:
            self.nodes = [
                -0.9061798459386639927976,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386639927976,
            ]
            self.weights = [
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ]
        elif order == 6:
            var _s1 = sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0
            var s2 = sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)) / 3.0
            var s3 = sqrt(5.0 / 3.0 - 2.0 / 21.0 * sqrt(5.0 / 2.0))
            var s4 = sqrt(5.0 / 3.0 + 2.0 / 21.0 * sqrt(5.0 / 2.0))
            var w1 = (322.0 + 13.0 * sqrt(70.0)) / 1800.0
            var w2 = (322.0 - 13.0 * sqrt(70.0)) / 1800.0
            var w3 = 128.0 / 225.0
            self.nodes = [-s2, -s4, -s3, s3, s4, s2]
            self.weights = [w1, w3, w2, w2, w3, w1]
        else:
            self.nodes = [0.0]
            self.weights = [1.0]


@align(64)
struct GenerateKnots(Copyable, Movable):
    var n: Int
    var degree: Int
    var method: String
    var center: Float64
    var boundary: Tuple[Float64, Float64]
    var mean: Float64
    var std: Float64
    var cheby_knots: Int

    def __init__(
        out self,
        n: Int,
        degree: Int,
        method: String = "uniform",
        center: Float64 = 0.2,
        boundary: Tuple[Float64, Float64] = (0.0, 1.0),
        mean: Float64 = 50.0,
        std: Float64 = 0.1,
        cheby_num: Int = 13,
    ):
        self.n = n
        self.degree = degree
        self.method = method
        self.center = center
        self.boundary = boundary
        self.mean = mean
        self.std = std
        self.cheby_knots = cheby_num

    def func_parabolic(
        self, n: Int, boundary: Tuple[Float64, Float64]
    ) -> Tuple[Int, List[Float64]]:
        var low_y = boundary[0]
        var high_y = boundary[1]
        var centor = self.center

        var divide = sqrt((high_y - centor) / (centor - low_y)) + 1.0
        var n_adj: Int
        if Float64(n) % divide == 0.0:
            n_adj = n
        else:
            n_adj = Int(Float64(n) / divide) * Int(divide)
        if n_adj < 1:
            n_adj = n
        var factor = 2.0

        var a = abs(2.0 * (low_y - centor) / (factor * factor))

        var upward = sqrt(2.0 * (high_y - centor) / a) + factor

        var x = linspace(0.0, upward, n_adj - 1)
        x.append(factor)

        var x_len = len(x)
        sort(Span(x))

        var y = List[Float64](length=x_len, fill=0.0)

        var y_ptr = y.unsafe_ptr()
        var x_span = Span(x)

        def vparab[
            width: Int
        ](p_off: Int) {
            read a,
            read factor,
            read centor,
            read x_span,
            read y_ptr,
        }:
            for w in range(width):
                var xi = x_span[p_off + w]
                y_ptr[p_off + w] = (
                    -0.5 * a * (xi - factor) * (xi - factor) + centor
                )

        vectorize[SIMD_W](x_len, vparab)

        var min_dist = abs(y[0] - centor)
        var center_idx = 0
        for i in range(1, x_len):
            var d = abs(y[i] - centor)
            if d < min_dist:
                min_dist = d
                center_idx = i

        for i in range(center_idx + 1, x_len):
            y[i] = 0.5 * a * (x[i] - factor) * (x[i] - factor) + centor

        sort(Span(y))

        return (n_adj, y^)

    def chebyshev_knots(self, n: Int, a: Float64, b: Float64) -> List[Float64]:
        var total = n + 2
        var out = List[Float64](length=total, fill=0.0)

        var r_ptr = out.unsafe_ptr()

        r_ptr[0] = a
        r_ptr[total - 1] = b

        var scale = (b - a) / 2.0
        var shift = (a + b) / 2.0
        var inv_2n = pi / (2.0 * Float64(n))

        def vcheby[
            width: Int
        ](p_off: Int) {read inv_2n, read scale, read shift, read r_ptr}:
            for w in range(width):
                var idx = p_off + w
                var angle = (2.0 * Float64(idx) + 1.0) * inv_2n
                r_ptr[1 + idx] = scale * cos(angle) + shift

        vectorize[SIMD_W](n, vcheby)

        return out^



    def generate_knots(self) -> List[Float64]:
        var p = self.degree
        var boundary = self.boundary
        var boundary_normal = normalize([boundary[0], boundary[1]])
        var mean = self.mean
        var std = self.std
        var x_min = mean - 4.5 * std
        var x_min_normal = (x_min - boundary[0]) / (boundary[1] - boundary[0])
        var x_max = mean + 4.5 * std
        var x_max_normal = (x_max - boundary[0]) / (boundary[1] - boundary[0])
        var n_knots = self.cheby_knots

        var internal_num = self.n - 2 * p

        var internal_knots: List[Float64] = []

        if self.method == "uniform":
            internal_knots = linspace(0.0, 1.0, internal_num)
        elif self.method == "non-uniform":
            var x_knots = self.chebyshev_knots(n_knots, x_min, x_max)
            var x_normal: List[Float64] = []
            for i in range(len(x_knots)):
                x_normal.append(
                    (x_knots[i] - boundary[0]) / (boundary[1] - boundary[0])
                )
            sort(Span(x_normal))

            var parabolic_result = self.func_parabolic(
                internal_num, (boundary_normal[0], boundary_normal[1])
            )
            var parabolic_knots = parabolic_result[1].copy()

            # Filter parabolic knots into left/right by threshold
            var left_knots: List[Float64] = []
            var right_knots: List[Float64] = []
            for i in range(len(parabolic_knots)):
                var v = parabolic_knots[i]
                if abs(v - x_min_normal) < 1e-10 or v < x_min_normal:
                    left_knots.append(v)
                if abs(v - x_max_normal) < 1e-10 or v > x_max_normal:
                    right_knots.append(v)

            # Merge, sort, round, dedup
            var merged = left_knots^
            merged += x_normal^
            merged += right_knots^
            sort(Span(merged))

            var m_total = len(merged)
            for i in range(m_total):
                merged[i] = round(merged[i] * 1e8) / 1e8

            var unique: List[Float64] = []
            if m_total > 0:
                unique.append(merged[0])
                for i in range(1, m_total):
                    if merged[i] > unique[len(unique) - 1] + 1e-9:
                        unique.append(merged[i])

            internal_knots = unique^

        var i_len = len(internal_knots)
        var total = 2 * p + i_len
        var final_knots = List[Float64](length=total, fill=0.0)

        var fk_ptr = final_knots.unsafe_ptr()
        var ik_span = Span(internal_knots)
        var bv_lo = boundary_normal[0]
        var bv_hi = boundary_normal[1]

        for i in range(total):
            if i < p:
                fk_ptr[i] = bv_lo
            elif i < p + i_len:
                fk_ptr[i] = ik_span[i - p]
            else:
                fk_ptr[i] = bv_hi

        return final_knots^



