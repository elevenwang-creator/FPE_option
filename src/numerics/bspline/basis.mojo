"""B-spline basis evaluation with direct CSR construction.

Optimized with InlineArray + unsafe_ptr:
- de_boor_cox: InlineArray stack alloc (no heap), knots_span for direct access
- eval_all: sequential count + fill rows, vectorize zero-init, inline prefix-sum
- first_derivative_all: same pattern, InlineArray work arrays
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


struct BSplineBasis[degree: Int](Copyable, Movable):
    var knots: List[Float64]
    var num_basis: Int

    def __init__(out self, var knots: List[Float64]):
        self.knots = knots^
        self.num_basis = len(self.knots) - Self.degree - 1

    def base_basis(self, x: Float64, i: Int) -> Float64:
        if i < 0 or i + 1 >= len(self.knots):
            return 0.0

        var left = self.knots[i]
        var right = self.knots[i + 1]
        var last_knot = self.knots[len(self.knots) - 1]
        if left <= x and x < right:
            return 1.0
        if x == last_knot and x == right:
            return 1.0
        return 0.0

    def de_boor_cox(self, x: Float64, i: Int) -> Float64:
        if i < 0 or i >= self.num_basis:
            return 0.0

        var k_span = Span(self.knots)
        var d = Self.degree + 1

        var work = InlineArray[Float64, Self.degree + 1](fill=0.0)
        for j in range(d):
            work[j] = self.base_basis(x, i + j)

        comptime for k in range(1, Self.degree + 1):
            var nxt = InlineArray[Float64, Self.degree + 1](fill=0.0)
            var upper = Self.degree - k + 1
            for j in range(upper):
                var idx = i + j

                var denom1 = k_span[idx + k] - k_span[idx]
                var alpha = Float64(0.0)
                if denom1 != 0.0:
                    alpha = (x - k_span[idx]) / denom1

                var denom2 = k_span[idx + k + 1] - k_span[idx + 1]
                var beta = Float64(0.0)
                if denom2 != 0.0:
                    beta = (x - k_span[idx + 1]) / denom2

                nxt[j] = alpha * work[j] + (1.0 - beta) * work[j + 1]

            work = nxt

        return work[0]

    def eval_all(self, points: List[Float64]) -> CSRMatrix:
        var n_pts = len(points)
        var n_basis = self.num_basis

        var row_counts = List[Int](length=n_pts, fill=0)
        var rc_ptr = row_counts.unsafe_ptr()

        var pts_span = Span(points)

        for row in range(n_pts):
            var x = pts_span[row]
            var cnt = 0
            for col in range(n_basis):
                var value = self.de_boor_cox(x, col)
                if value > 1e-12 or value < -1e-12:
                    cnt += 1
            rc_ptr[row] = cnt

        var total_nnz = 0
        for i in range(n_pts):
            total_nnz += rc_ptr[i]

        var result = CSRMatrix(n_pts, n_basis, total_nnz)

        var d_ptr = result.data.unsafe_ptr()
        var idx_ptr = result.indices.unsafe_ptr()
        var ip_ptr = result.indptr.unsafe_ptr()

        ip_ptr[0] = 0
        for i in range(n_pts):
            ip_ptr[i + 1] = ip_ptr[i] + rc_ptr[i]

        var pos = List[Int](length=n_pts, fill=0)
        var pos_ptr = pos.unsafe_ptr()

        for i in range(n_pts):
            pos_ptr[i] = ip_ptr[i]

        for row in range(n_pts):
            var x = pts_span[row]
            var dest = pos_ptr[row]
            for col in range(n_basis):
                var value = self.de_boor_cox(x, col)
                if value > 1e-12 or value < -1e-12:
                    d_ptr[dest] = value
                    idx_ptr[dest] = col
                    dest += 1
            pos_ptr[row] = dest

        return result^

    def first_derivative_all(self, points: List[Float64]) -> CSRMatrix:
        var n_pts = len(points)
        var n_basis = self.num_basis

        comptime if Self.degree == 0:
            return CSRMatrix(n_pts, n_basis)

        var lower = BSplineBasis[Self.degree - 1](self.knots.copy())

        var row_counts = List[Int](length=n_pts, fill=0)
        var rc_ptr = row_counts.unsafe_ptr()

        var pts_span = Span(points)
        var k_span = Span(self.knots)

        for row in range(n_pts):
            var x = pts_span[row]
            var cnt = 0
            for col in range(n_basis):
                var left = Float64(0.0)
                var denom1 = k_span[col + Self.degree] - k_span[col]
                if denom1 != 0.0:
                    left = (
                        Float64(Self.degree)
                        / denom1
                        * lower.de_boor_cox(x, col)
                    )

                var right = Float64(0.0)
                var denom2 = k_span[col + Self.degree + 1] - k_span[col + 1]
                if denom2 != 0.0:
                    right = (
                        Float64(Self.degree)
                        / denom2
                        * lower.de_boor_cox(x, col + 1)
                    )

                var value = left - right
                if value > 1e-12 or value < -1e-12:
                    cnt += 1
            rc_ptr[row] = cnt

        var total_nnz = 0
        for i in range(n_pts):
            total_nnz += rc_ptr[i]

        var result = CSRMatrix(n_pts, n_basis, total_nnz)

        var d_ptr = result.data.unsafe_ptr()
        var idx_ptr = result.indices.unsafe_ptr()
        var ip_ptr = result.indptr.unsafe_ptr()

        ip_ptr[0] = 0
        for i in range(n_pts):
            ip_ptr[i + 1] = ip_ptr[i] + rc_ptr[i]

        var pos = List[Int](length=n_pts, fill=0)
        var pos_ptr = pos.unsafe_ptr()

        for i in range(n_pts):
            pos_ptr[i] = ip_ptr[i]

        for row in range(n_pts):
            var x = pts_span[row]
            var dest = pos_ptr[row]
            for col in range(n_basis):
                var left = Float64(0.0)
                var denom1 = k_span[col + Self.degree] - k_span[col]
                if denom1 != 0.0:
                    left = (
                        Float64(Self.degree)
                        / denom1
                        * lower.de_boor_cox(x, col)
                    )

                var right = Float64(0.0)
                var denom2 = k_span[col + Self.degree + 1] - k_span[col + 1]
                if denom2 != 0.0:
                    right = (
                        Float64(Self.degree)
                        / denom2
                        * lower.de_boor_cox(x, col + 1)
                    )

                var value = left - right
                if value > 1e-12 or value < -1e-12:
                    d_ptr[dest] = value
                    idx_ptr[dest] = col
                    dest += 1
            pos_ptr[row] = dest

        return result^
