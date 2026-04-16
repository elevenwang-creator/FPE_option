"""B-spline basis evaluation with direct CSR construction.

Two-pass algorithm: count nnz per row, allocate, fill.
No COO intermediate. Direct CSR output.
"""

from sparse.csr import CSRMatrix


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

        var work: List[Float64] = []
        for _ in range(Self.degree + 1):
            work.append(0.0)

        for j in range(Self.degree + 1):
            work[j] = self.base_basis(x, i + j)

        comptime for k in range(1, Self.degree + 1):
            var next: List[Float64] = []
            for _ in range(Self.degree + 1):
                next.append(0.0)

            var upper = Self.degree - k + 1
            for j in range(upper):
                var idx = i + j

                var denom1 = self.knots[idx + k] - self.knots[idx]
                var alpha = Float64(0.0)
                if denom1 != 0.0:
                    alpha = (x - self.knots[idx]) / denom1

                var denom2 = self.knots[idx + k + 1] - self.knots[idx + 1]
                var beta = Float64(0.0)
                if denom2 != 0.0:
                    beta = (x - self.knots[idx + 1]) / denom2

                next[j] = alpha * work[j] + (1.0 - beta) * work[j + 1]

            work = next^

        return work[0]

    def eval_all(self, points: List[Float64]) -> CSRMatrix:
        var n_pts = len(points)
        var n_basis = self.num_basis

        var row_counts = alloc[Int](n_pts)
        for i in range(n_pts):
            row_counts[i] = 0

        for row in range(n_pts):
            var x = points[row]
            for col in range(n_basis):
                var value = self.de_boor_cox(x, col)
                if value > 1e-12 or value < -1e-12:
                    row_counts[row] += 1

        var total_nnz = 0
        for i in range(n_pts):
            total_nnz += row_counts[i]

        var result = CSRMatrix(n_pts, n_basis, total_nnz)
        result.indptr[0] = 0
        for i in range(n_pts):
            result.indptr[i + 1] = result.indptr[i] + row_counts[i]

        var pos = alloc[Int](n_pts)
        for i in range(n_pts):
            pos[i] = result.indptr[i]

        for row in range(n_pts):
            var x = points[row]
            for col in range(n_basis):
                var value = self.de_boor_cox(x, col)
                if value > 1e-12 or value < -1e-12:
                    var dest = pos[row]
                    result.data[dest] = value
                    result.indices[dest] = col
                    pos[row] = dest + 1

        row_counts.free()
        pos.free()
        return result^

    def first_derivative_all(self, points: List[Float64]) -> CSRMatrix:
        var n_pts = len(points)
        var n_basis = self.num_basis

        comptime if Self.degree == 0:
            return CSRMatrix(n_pts, n_basis)

        var lower = BSplineBasis[Self.degree - 1](self.knots.copy())

        var row_counts = alloc[Int](n_pts)
        for i in range(n_pts):
            row_counts[i] = 0

        for row in range(n_pts):
            var x = points[row]
            for col in range(n_basis):
                var left = Float64(0.0)
                var denom1 = self.knots[col + Self.degree] - self.knots[col]
                if denom1 != 0.0:
                    left = Float64(Self.degree) / denom1 * lower.de_boor_cox(x, col)

                var right = Float64(0.0)
                var denom2 = self.knots[col + Self.degree + 1] - self.knots[col + 1]
                if denom2 != 0.0:
                    right = Float64(Self.degree) / denom2 * lower.de_boor_cox(x, col + 1)

                var value = left - right
                if value > 1e-12 or value < -1e-12:
                    row_counts[row] += 1

        var total_nnz = 0
        for i in range(n_pts):
            total_nnz += row_counts[i]

        var result = CSRMatrix(n_pts, n_basis, total_nnz)
        result.indptr[0] = 0
        for i in range(n_pts):
            result.indptr[i + 1] = result.indptr[i] + row_counts[i]

        var pos = alloc[Int](n_pts)
        for i in range(n_pts):
            pos[i] = result.indptr[i]

        for row in range(n_pts):
            var x = points[row]
            for col in range(n_basis):
                var left = Float64(0.0)
                var denom1 = self.knots[col + Self.degree] - self.knots[col]
                if denom1 != 0.0:
                    left = Float64(Self.degree) / denom1 * lower.de_boor_cox(x, col)

                var right = Float64(0.0)
                var denom2 = self.knots[col + Self.degree + 1] - self.knots[col + 1]
                if denom2 != 0.0:
                    right = Float64(Self.degree) / denom2 * lower.de_boor_cox(x, col + 1)

                var value = left - right
                if value > 1e-12 or value < -1e-12:
                    var dest = pos[row]
                    result.data[dest] = value
                    result.indices[dest] = col
                    pos[row] = dest + 1

        row_counts.free()
        pos.free()
        return result^
