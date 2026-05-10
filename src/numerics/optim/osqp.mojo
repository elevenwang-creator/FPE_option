"""ADMM-based OSQP solver using sparse LU factorization.

Solves: min 0.5 ||M @ c - b||^2 + (lambda/2) * ||c||^2  s.t. c >= 0

Equivalent QP form: min 0.5 c^T P c + q^T c  s.t. c >= 0
  where P = M^T M + lambda * I,  q = -M^T b

ADMM with pre-factorized KKT:
  KKT = P + (sigma + rho) * I
  Each ADMM iteration is ONE sparse LU solve (zero heap allocation).

Performance optimizations:
  - Pre-allocated FixedSizeVector buffers (zero heap allocation in ADMM loop)
  - In-place LU solve via SparseLU.solve_inplace
  - Fused convergence norm computation (single loop for all 5 norms)
  - In-place diagonal shift (no CSR copy)
  - SIMD vectorization via FixedSizeVector methods

Correctness fixes:
  - LU factorization failure returns zero vector instead of silent garbage
  - Dual residual properly scaled by rho
"""

from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix
from numerics.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from std.math import sqrt, abs, max
from std.sys import simd_width_of

comptime SIMD_W = simd_width_of[DType.float64]()


def _add_alpha_to_diagonal_inplace(mut A: CSRMatrix, alpha: Float64):
    """Add alpha to diagonal entries of A in-place (no copy)."""
    for i in range(A.nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[p] == i:
                A.data[p] = A.data[p] + alpha
                break


@fieldwise_init
struct OSQPSolver(Copyable, Movable):
    var max_iter: Int
    var eps_abs: Float64
    var eps_rel: Float64
    var rho: Float64
    var sigma: Float64
    var lambda_reg: Float64

    def solve_nnls_sparse(
        self,
        M: CSRMatrix,
        b: List[Float64],
    ) -> List[Float64]:
        var n = M.ncols
        if n == 0:
            return []

        # Build P = M^T M + lambda * I, then KKT = P + (sigma + rho) * I
        var Mt = M.transpose()
        var MtM = Mt @ M

        # In-place diagonal shift: MtM += (lambda + sigma + rho) * I
        var alpha = self.lambda_reg + self.sigma + self.rho
        _add_alpha_to_diagonal_inplace(MtM, alpha)

        # Factorize KKT (= MtM after shift)
        var KKT_csc = MtM.to_csc()
        var lu = SparseLU(n)
        var factorize_ok = True
        try:
            lu.factorize(KKT_csc)
        except:
            factorize_ok = False

        # C2 fix: if factorization failed, return zero vector
        if not factorize_ok:
            var fallback: List[Float64] = []
            for _ in range(n):
                fallback.append(0.0)
            return fallback^

        # Linear term: q = M^T b (positive sign — used as +q in RHS)
        var q_list = Mt.spmv_new(b)

        # Pre-allocate all FixedSizeVector buffers (zero heap allocation in loop)
        var q = FixedSizeVector(n)
        q.copy_from(q_list)

        var x = FixedSizeVector(n)
        var z = FixedSizeVector(n)
        var z_old = FixedSizeVector(n)
        var u = FixedSizeVector(n)
        var rhs = FixedSizeVector(n)
        var work = FixedSizeVector(n)

        var rho_val = self.rho
        var sigma_val = self.sigma
        var eps_abs_val = self.eps_abs
        var eps_rel_val = self.eps_rel

        for _ in range(self.max_iter):
            # x-update: solve KKT * x_new = q + rho*(z - u) + sigma*x
            comptime width = SIMD_W
            var i = 0
            while i + width <= n:
                var sq = (q.ptr() + i).load[width=width]()
                var sz = (z.ptr() + i).load[width=width]()
                var su = (u.ptr() + i).load[width=width]()
                var sx = (x.ptr() + i).load[width=width]()
                var sr = sq + rho_val * (sz - su) + sigma_val * sx
                (rhs.ptr() + i).store[width=width](sr)
                i += width
            while i < n:
                rhs[i] = q[i] + rho_val * (z[i] - u[i]) + sigma_val * x[i]
                i += 1

            # In-place LU solve: rhs is overwritten with solution
            lu.solve_inplace(rhs, work)

            # Copy solution to x
            x.copy_from_fixed(rhs)

            # Save z_old for dual residual
            z_old.copy_from_fixed(z)

            # z-update: z = max(x + u, 0)  (projection onto non-negative orthant)
            # u-update: u = u + x - z
            # Fused convergence norms: primal, dual, z_norm, x_norm, u_norm
            var primal_res_sq = 0.0
            var dual_res_sq = 0.0
            var z_norm_sq = 0.0
            var x_norm_sq = 0.0
            var u_norm_sq = 0.0

            for j in range(n):
                # z-update with projection
                var z_new = x[j] + u[j]
                if z_new < 0.0:
                    z_new = 0.0
                z[j] = z_new

                # u-update
                u[j] = u[j] + x[j] - z_new

                # Primal residual: ||x - z||
                var pv = x[j] - z_new
                primal_res_sq += pv * pv

                # C4 fix: Dual residual with rho scaling: ||rho*(z - z_old)||
                var dv = rho_val * (z_new - z_old[j])
                dual_res_sq += dv * dv

                # Norms for tolerance computation
                z_norm_sq += z_new * z_new
                x_norm_sq += x[j] * x[j]
                u_norm_sq += u[j] * u[j]

            # Convergence check (standard OSQP criteria)
            var eps_prim = eps_abs_val + eps_rel_val * sqrt(
                max(z_norm_sq, x_norm_sq)
            )
            var eps_dual = eps_abs_val + eps_rel_val * rho_val * sqrt(u_norm_sq)

            if sqrt(primal_res_sq) < eps_prim and sqrt(dual_res_sq) < eps_dual:
                break

        return z.to_list()

    def solve_nnls_dense(
        self,
        A: List[List[Float64]],
        b: List[Float64],
    ) -> List[Float64]:
        var nrows = len(A)
        if nrows == 0:
            return []
        var M = CSRMatrix.from_dense(A)
        return self.solve_nnls_sparse(M, b)


# --- Backward-compatible aliases for existing tests ---


struct OSQP(Copyable, Movable):
    """Legacy alias wrapping OSQPSolver with simplified (max_iter, tol) API."""

    var _solver: OSQPSolver

    def __init__(out self, max_iter: Int = 5000, tol: Float64 = 1e-8):
        self._solver = OSQPSolver(
            max_iter=max_iter,
            eps_abs=tol,
            eps_rel=tol,
            rho=0.1,
            sigma=1e-6,
            lambda_reg=1e-6,
        )

    def solve_nnls(
        self, A: List[List[Float64]], b: List[Float64]
    ) -> List[Float64]:
        return self._solver.solve_nnls_dense(A, b)


struct ProjectedGradient(Copyable, Movable):
    """Simple projected gradient descent for NNLS: min ||Ac - b||^2  s.t. c >= 0.
    """

    var max_iter: Int
    var tol: Float64
    var step_size: Float64

    def __init__(
        out self,
        max_iter: Int = 1000,
        tol: Float64 = 1e-8,
        step_size: Float64 = -1.0,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size

    def solve(self, A: List[List[Float64]], b: List[Float64]) -> List[Float64]:
        var nrows = len(A)
        if nrows == 0:
            return []
        var ncols = len(A[0])
        if ncols == 0:
            return []

        # Compute A^T A and A^T b
        var AtA: List[List[Float64]] = []
        var Atb: List[Float64] = []
        for i in range(ncols):
            var row: List[Float64] = []
            for j in range(ncols):
                var s = 0.0
                for k in range(nrows):
                    s += A[k][i] * A[k][j]
                row.append(s)
            AtA.append(row^)
            var sb = 0.0
            for k in range(nrows):
                sb += A[k][i] * b[k]
            Atb.append(sb)

        # Auto step size: 1 / ||A^T A||_F
        var alpha = self.step_size
        if alpha <= 0.0:
            var norm_sq = 0.0
            for i in range(ncols):
                for j in range(ncols):
                    norm_sq += AtA[i][j] * AtA[i][j]
            if norm_sq > 0.0:
                alpha = 1.0 / sqrt(norm_sq)
            else:
                alpha = 1e-3

        var c: List[Float64] = []
        for _ in range(ncols):
            c.append(0.0)

        for _ in range(self.max_iter):
            # gradient = A^T A c - A^T b
            var grad: List[Float64] = []
            var grad_norm = 0.0
            for i in range(ncols):
                var g = -Atb[i]
                for j in range(ncols):
                    g += AtA[i][j] * c[j]
                grad.append(g)
                grad_norm += g * g

            if sqrt(grad_norm) < self.tol:
                break

            # Projected step: c = max(c - alpha * grad, 0)
            for i in range(ncols):
                c[i] = c[i] - alpha * grad[i]
                if c[i] < 0.0:
                    c[i] = 0.0

        return c^
