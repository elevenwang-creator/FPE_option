"""ADMM-based OSQP solver using sparse LU factorization.

Solves: min 0.5 ||M @ c - b||^2 + lambda * ||c||^2  s.t. c >= 0

ADMM with pre-factorized KKT:
  KKT = M^T M + (lambda + sigma + rho) * I
  Each ADMM iteration is ONE sparse LU solve.
"""

from sparse.csr import CSRMatrix
from sparse.csc import csr_to_csc
from numerics.sparse_lu import SparseLU


def _add_alpha_to_diagonal(
    A: CSRMatrix, alpha: Float64
) -> CSRMatrix:
    var nnz_val = A.nnz()
    var result = CSRMatrix(A.nrows, A.ncols, nnz_val)
    for i in range(A.nrows + 1):
        result.indptr[i] = A.indptr[i]
    for p in range(nnz_val):
        result.data[p] = A.data[p]
        result.indices[p] = A.indices[p]
    for i in range(result.nrows):
        for p in range(result.indptr[i], result.indptr[i + 1]):
            if result.indices[p] == i:
                result.data[p] = result.data[p] + alpha
                break
    return result^


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

        var Mt = M.transpose()
        var MtM = _spgemm(Mt, M)

        var alpha = self.lambda_reg + self.sigma + self.rho
        var KKT = _add_alpha_to_diagonal(MtM, alpha)

        var KKT_csc = csr_to_csc(KKT)
        var lu = SparseLU(n)
        try:
            lu.factorize(KKT_csc)
        except:
            pass

        var q = Mt.spmv(b)
        for i in range(len(q)):
            q[i] = -q[i]

        var x: List[Float64] = []
        var z: List[Float64] = []
        var u: List[Float64] = []
        for _ in range(n):
            x.append(0.0)
            z.append(0.0)
            u.append(0.0)

        for _ in range(self.max_iter):
            var rhs: List[Float64] = []
            for i in range(n):
                rhs.append(-q[i] + self.rho * (z[i] - u[i]) + self.sigma * x[i])

            x = lu.solve(rhs)

            var z_old = z.copy()
            for i in range(n):
                z[i] = x[i] + u[i]
                if z[i] < 0.0:
                    z[i] = 0.0

            for i in range(n):
                u[i] = u[i] + x[i] - z[i]

            var primal_res_norm = 0.0
            var dual_res_norm = 0.0
            for i in range(n):
                var pv = x[i] - z[i]
                var dv = z[i] - z_old[i]
                primal_res_norm += pv * pv
                dual_res_norm += dv * dv

            var z_norm = 0.0
            var x_norm = 0.0
            var u_norm = 0.0
            for i in range(n):
                z_norm += z[i] * z[i]
                x_norm += x[i] * x[i]
                u_norm += u[i] * u[i]

            var eps_prim = self.eps_abs + self.eps_rel * max(z_norm, x_norm) ** 0.5
            var eps_dual = self.eps_abs + self.eps_rel * abs(self.rho) * u_norm ** 0.5

            if primal_res_norm ** 0.5 < eps_prim and dual_res_norm ** 0.5 < eps_dual:
                break

        return z^

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


def _spgemm(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    from sparse.ops import spgemm as sparse_spgemm
    return sparse_spgemm(A, B)
