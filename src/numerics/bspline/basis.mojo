from sparse.coo import COOMatrix
from sparse.csr import CSRMatrix


@align(64)
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

    def eval_all(self, points: List[Float64]) -> CSRMatrix[DType.float64]:
        var out = COOMatrix[DType.float64](len(points), self.num_basis)

        for row in range(len(points)):
            var x = points[row]
            for col in range(self.num_basis):
                var value = self.de_boor_cox(x, col)
                var abs_value = value
                if abs_value < 0.0:
                    abs_value = -abs_value
                if abs_value > 1e-12:
                    out.append(row, col, value)
        return out.to_csr()

    def _base_basis_simd[width: Int](self, x: SIMD[DType.float64, width], i: Int) -> SIMD[DType.float64, width]:
        var result = SIMD[DType.float64, width]()
        if i < 0 or i + 1 >= len(self.knots):
            return result
        var sleft = SIMD[DType.float64, width](self.knots[i])
        var sright = SIMD[DType.float64, width](self.knots[i + 1])
        var slast = SIMD[DType.float64, width](self.knots[len(self.knots) - 1])
        
        var cond1 = x.ge(sleft) & x.lt(sright)
        var cond2 = x.eq(slast) & x.eq(sright)
        # Use select to turn bool SIMD into float SIMD
        return (cond1 | cond2).cast[DType.float64]()

    def _de_boor_cox_simd[width: Int](self, x: SIMD[DType.float64, width], i: Int) -> SIMD[DType.float64, width]:
        if i < 0 or i >= self.num_basis:
            return SIMD[DType.float64, width]()

        var work = List[SIMD[DType.float64, width]]()
        for _ in range(Self.degree + 1):
            work.append(SIMD[DType.float64, width]())

        for j in range(Self.degree + 1):
            work[j] = self._base_basis_simd[width](x, i + j)

        comptime for k in range(1, Self.degree + 1):
            var next_work = List[SIMD[DType.float64, width]]()
            for _ in range(Self.degree + 1):
                next_work.append(SIMD[DType.float64, width]())
            
            var upper = Self.degree - k + 1
            for j in range(upper):
                var idx = i + j
                var denom1 = self.knots[idx + k] - self.knots[idx]
                var denom2 = self.knots[idx + k + 1] - self.knots[idx + 1]
                
                var alpha = SIMD[DType.float64, width]()
                if denom1 != 0.0:
                    alpha = (x - self.knots[idx]) / denom1
                    
                var beta = SIMD[DType.float64, width]()
                if denom2 != 0.0:
                    beta = (x - self.knots[idx + 1]) / denom2
                    
                next_work[j] = alpha * work[j] + (1.0 - beta) * work[j + 1]
            work = next_work^

        return work[0]

    def evaluate_batch_simd(self, points: List[Float64]) -> CSRMatrix[DType.float64]:
        """SIMD-vectorized B-Spline evaluation over the batch dim."""
        from std.sys import simd_width_of
        var out = COOMatrix[DType.float64](len(points), self.num_basis)
        comptime width = simd_width_of[DType.float64]()
        var n = len(points)
        var row = 0
        
        while row + width <= n:
            var points_vec = SIMD[DType.float64, width]()
            for k in range(width):
                points_vec[k] = points[row + k]
                
            for col in range(self.num_basis):
                var vals = self._de_boor_cox_simd[width](points_vec, col)
                for k in range(width):
                    var val = vals[k]
                    var abs_val = val
                    if abs_val < 0.0:
                        abs_val = -abs_val
                    if abs_val > 1e-12:
                        out.append(row + k, col, val)
            row += width
            
        while row < n:
            for col in range(self.num_basis):
                var val = self.de_boor_cox(points[row], col)
                var abs_val = val
                if abs_val < 0.0:
                    abs_val = -abs_val
                if abs_val > 1e-12:
                    out.append(row, col, val)
            row += 1
            
        return out.to_csr()

    def first_derivative_all(self, points: List[Float64]) -> CSRMatrix[DType.float64]:
        var out = COOMatrix[DType.float64](len(points), self.num_basis)

        comptime if Self.degree == 0:
            return out.to_csr()
        else:
            var lower = BSplineBasis[Self.degree - 1](self.knots.copy())

            for row in range(len(points)):
                var x = points[row]
                for col in range(self.num_basis):
                    var left = Float64(0.0)
                    var denom1 = self.knots[col + Self.degree] - self.knots[col]
                    if denom1 != 0.0:
                        left = Float64(Self.degree) / denom1 * lower.de_boor_cox(x, col)

                    var right = Float64(0.0)
                    var denom2 = self.knots[col + Self.degree + 1] - self.knots[col + 1]
                    if denom2 != 0.0:
                        right = Float64(Self.degree) / denom2 * lower.de_boor_cox(x, col + 1)

                    var value = left - right
                    var abs_value = value
                    if abs_value < 0.0:
                        abs_value = -abs_value
                    if abs_value > 1e-12:
                        out.append(row, col, value)
            return out.to_csr()
