from numerics.utils.linalg import lu_solve
from numerics.utils import mat_vec_mul, mat_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major
from std.memory import Span
from std.math import abs


trait ResidualCallable:
    def __call__(self, x: List[Float64]) raises -> List[Float64]:
        ...


trait JacobianCallable:
    def __call__(self, x: List[Float64]) raises -> List[List[Float64]]:
        ...


@fieldwise_init
struct LevenbergMarquardt:
    """Nonlinear least squares: min ||f(x)||²."""

    var max_iter: Int
    var tol: Float64
    var lambda_init: Float64
    var lambda_up: Float64
    var lambda_down: Float64

    def solve[
        ResidualFn: ResidualCallable,
        JacobianFn: JacobianCallable,
    ](
        self,
        residual_fn: ResidualFn,
        jacobian_fn: JacobianFn,
        x0: List[Float64],
    ) raises -> List[Float64]:
        var x = x0.copy()
        var lam = self.lambda_init
        var n = len(x)

        for _ in range(self.max_iter):
            var r = residual_fn(x)
            var J = jacobian_fn(x)
            var m = len(r)

            if len(J) != m:
                raise Error(
                    "LevenbergMarquardt: Jacobian rows must match residual size"
                )
            for k in range(m):
                if len(J[k]) != n:
                    raise Error(
                        "LevenbergMarquardt: Jacobian cols must match parameter"
                        " size"
                    )

            var J_flat = List[Float64](length=m * n, fill=0.0)
            var JT_flat = List[Float64](length=m * n, fill=0.0)
            for k in range(m):
                var jk = J[k].copy()
                for i in range(n):
                    var val = jk[i]
                    J_flat[k * n + i] = val
                    JT_flat[i * m + k] = val

            var J_t = TileTensor(J_flat, row_major(coord[DType.int64]((m, n))))
            var JT_t = TileTensor(JT_flat, row_major(coord[DType.int64]((n, m))))

            var JtJ_flat = List[Float64](length=n * n, fill=0.0)
            var JtJ_t = TileTensor(JtJ_flat, row_major(coord[DType.int64]((n, n))))
            mat_mul(JT_t, J_t, JtJ_t)

            var JtJ: List[List[Float64]] = []
            for i in range(n):
                var row: List[Float64] = []
                for j in range(n):
                    row.append(JtJ_flat[i * n + j])
                JtJ.append(row^)

            var Jtr: List[Float64] = List[Float64](length=n, fill=0.0)
            var Jtr_span = Span[mut=True, Float64](Jtr)
            mat_vec_mul(JT_t, Span[Float64](r), Jtr_span)
            var grad_inf = 0.0
            for i in range(n):
                var abs_s = abs(Jtr[i])
                if abs_s > grad_inf:
                    grad_inf = abs_s

            if grad_inf < self.tol:
                break

            for i in range(n):
                JtJ[i][i] += lam

            var neg_Jtr: List[Float64] = []
            for i in range(n):
                neg_Jtr.append(-Jtr[i])

            var delta = lu_solve(JtJ, neg_Jtr)

            var x_new: List[Float64] = []
            for i in range(n):
                x_new.append(x[i] + delta[i])

            var r_new = residual_fn(x_new)
            var cost_old = 0.0
            var cost_new = 0.0
            for k in range(m):
                cost_old += r[k] * r[k]
                cost_new += r_new[k] * r_new[k]

            if cost_new < cost_old:
                x = x_new^
                lam = lam * self.lambda_down
                if lam < 1e-10:
                    lam = 1e-10
            else:
                lam = lam * self.lambda_up

        return x^
