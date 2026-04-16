"""Initial condition computation using 1D basis evaluations.

Key optimization: exploit Kronecker structure to avoid forming
328K×600 intermediate Phi matrix.

Galerkin projection: c = Bs^T @ W_delta @ Bv (flattened)
Where W_delta[s, v] = sw[s] * vw[v] * delta(s, v)

Mass matrix for OSQP: M = kron(Ms, Mv) from 1D Galerkin matrices.
Basis integral for normalization: m[i_s*n_v+i_v] = Ms_rowsum[i_s] * Mv_rowsum[i_v]
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from numerics.optim.osqp import OSQPSolver
from sparse.csr import CSRMatrix
from std.math import exp, pi, sqrt, abs


def _normalize_nonnegative(mut x: List[Float64]):
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = 0.0


def _delta_approx_2d(
    domain: FPEDomain, params: HestonParams, sigma0: Float64
) -> List[List[Float64]]:
    var s0 = params.S0
    var v0 = params.V0
    #var rho = params.rho
    var rho = 0.0
    var s_range_span = domain.s_max - domain.s_min
    var v_stdev = sigma0 / s_range_span if s_range_span > 0.0 else sigma0
    var s_sigma = sigma0
    var v_sigma = v_stdev
    if s_sigma <= 0.0:
        s_sigma = 1e-3
    if v_sigma <= 0.0:
        v_sigma = 1e-3

    var rho2 = rho * rho
    var one_minus_rho2 = 1.0 - rho2
    if one_minus_rho2 < 1e-10:
        one_minus_rho2 = 1e-10
    var norm = 1.0 / (2.0 * pi * s_sigma * v_sigma * sqrt(one_minus_rho2))
    var inv_2_omr2 = 0.5 / one_minus_rho2

    var n_s = len(domain.s_points_phys)
    var n_v = len(domain.v_points_phys)

    var result: List[List[Float64]] = []
    for i in range(n_s):
        var row: List[Float64] = []
        var s = domain.s_points_phys[i]
        var ds = (s - s0) / s_sigma
        var ds2 = ds * ds
        for j in range(n_v):
            if ds2 > 50.0:
                row.append(0.0)
                continue
            var v = domain.v_points_phys[j]
            var dv = (v - v0) / v_sigma
            var dv2 = dv * dv
            var z = ds2 - 2.0 * rho * ds * dv + dv2
            if z * inv_2_omr2 > 50.0:
                row.append(0.0)
            else:
                var val = norm * exp(-z * inv_2_omr2)
                if val < 1e-6:
                    val = 0.0
                row.append(val)
        result.append(row^)
    return result^


def _galerkin_1d(B: CSRMatrix, weights: List[Float64]) -> List[List[Float64]]:
    """Compute B^T @ diag(weights) @ B as dense matrix."""
    var n = B.ncols
    var result: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        for _ in range(n):
            row.append(0.0)
        result.append(row^)

    for row_idx in range(B.nrows):
        var w = weights[row_idx]
        if w == 0.0:
            continue
        for p1 in range(B.indptr[row_idx], B.indptr[row_idx + 1]):
            var j1 = B.indices[p1]
            var v1 = B.data[p1]
            var wv1 = w * v1
            for p2 in range(p1, B.indptr[row_idx + 1]):
                var j2 = B.indices[p2]
                var v2 = B.data[p2]
                result[j1][j2] += wv1 * v2
                if j1 != j2:
                    result[j2][j1] += wv1 * v2

    return result^


def _dense_kron_to_csr(
    A: List[List[Float64]], B: List[List[Float64]]
) -> CSRMatrix:
    """Kronecker product of two dense matrices as CSR."""
    var m = len(A)
    if m == 0:
        return CSRMatrix(0, 0)
    var n = len(A[0])
    var p = len(B)
    if p == 0:
        return CSRMatrix(0, 0)
    var q = len(B[0])

    var out_rows = m * p
    var out_cols = n * q

    var row_counts = alloc[Int](out_rows)
    for i in range(out_rows):
        row_counts[i] = 0

    for i in range(m):
        for k in range(p):
            var count = 0
            for j in range(n):
                if abs(A[i][j]) < 1e-15:
                    continue
                for l in range(q):
                    if abs(B[k][l]) < 1e-15:
                        continue
                    count += 1
            row_counts[i * p + k] = count

    var total_nnz = 0
    for i in range(out_rows):
        total_nnz += row_counts[i]

    var result = CSRMatrix(out_rows, out_cols, total_nnz)
    var dest = 0
    result.indptr[0] = 0

    for i in range(m):
        for k in range(p):
            for j in range(n):
                if abs(A[i][j]) < 1e-15:
                    continue
                for l in range(q):
                    if abs(B[k][l]) < 1e-15:
                        continue
                    result.data[dest] = A[i][j] * B[k][l]
                    result.indices[dest] = j * q + l
                    dest += 1
            result.indptr[i * p + k + 1] = dest

    row_counts.free()
    return result^


def _dense_matmul(
    A: List[List[Float64]], B: List[List[Float64]]
) -> List[List[Float64]]:
    var m = len(A)
    if m == 0:
        return A.copy()
    var k = len(A[0])
    var n = len(B[0])
    var result: List[List[Float64]] = []
    for i in range(m):
        var row: List[Float64] = []
        for j in range(n):
            var sum = 0.0
            for p in range(k):
                sum += A[i][p] * B[p][j]
            row.append(sum)
        result.append(row^)
    return result^


def _dense_transpose(A: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(A)
    if m == 0:
        return A.copy()
    var n = len(A[0])
    var result: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        for j in range(m):
            row.append(A[j][i])
        result.append(row^)
    return result^


struct InitialCondition[B: Int]:
    def __init__(out self):
        pass

    def compute(
        self,
        domain: FPEDomain,
        params: HestonParams,
        sigma0: Float64 = 0.1,
    ) raises -> List[Float64]:
        var basis = domain.build_basis()

        var Bs = basis.basis_s.eval_all(domain.s_points)
        var Bv = basis.basis_v.eval_all(domain.v_points)

        var Ms = _galerkin_1d(Bs, domain.s_weights)
        var Mv = _galerkin_1d(Bv, domain.v_weights)

        var delta_2d = _delta_approx_2d(domain, params, sigma0)

        var n_s = len(domain.s_weights)
        var n_v = len(domain.v_weights)
        var W_delta: List[List[Float64]] = []
        for i in range(n_s):
            var row: List[Float64] = []
            for j in range(n_v):
                row.append(domain.s_weights[i] * domain.v_weights[j] * delta_2d[i][j])
            W_delta.append(row^)

        var BsT = _dense_transpose(_csr_to_dense(Bs))
        var Bv_dense = _csr_to_dense(Bv)

        var C_2d = _dense_matmul(_dense_matmul(BsT, W_delta), Bv_dense)

        var n_s_col = Bs.ncols
        var n_v_col = Bv.ncols
        var galerkin_proj: List[Float64] = []
        for i in range(n_s_col):
            for j in range(n_v_col):
                galerkin_proj.append(C_2d[i][j])

        var M = _dense_kron_to_csr(Ms, Mv)

        var osqp = OSQPSolver(
            max_iter=50000,
            eps_abs=1e-8,
            eps_rel=1e-8,
            rho=0.1,
            sigma=1e-6,
            lambda_reg=1e-6,
        )
        var result = osqp.solve_nnls_sparse(M, galerkin_proj)

        _normalize_nonnegative(result)

        var Ms_rowsum: List[Float64] = []
        for i in range(n_s_col):
            var s = 0.0
            for j in range(n_s_col):
                s += Ms[i][j]
            Ms_rowsum.append(s)

        var Mv_rowsum: List[Float64] = []
        for j in range(n_v_col):
            var s = 0.0
            for k in range(n_v_col):
                s += Mv[j][k]
            Mv_rowsum.append(s)

        var m_dot_r = 0.0
        for i in range(n_s_col):
            for j in range(n_v_col):
                m_dot_r += Ms_rowsum[i] * Mv_rowsum[j] * result[i * n_v_col + j]

        if m_dot_r > 0.0:
            for i in range(len(result)):
                result[i] = result[i] / m_dot_r

        return result^


def _csr_to_dense(A: CSRMatrix) -> List[List[Float64]]:
    var result: List[List[Float64]] = []
    for i in range(A.nrows):
        var row: List[Float64] = []
        for _ in range(A.ncols):
            row.append(0.0)
        result.append(row^)
    for i in range(A.nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            result[i][A.indices[p]] = A.data[p]
    return result^


def _scale_csr(A: CSRMatrix, d_inv: List[Float64]) -> CSRMatrix:
    var nnz_val = A.nnz()
    var result = CSRMatrix(A.nrows, A.ncols, nnz_val)
    for i in range(A.nrows + 1):
        result.indptr[i] = A.indptr[i]
    var dest = 0
    for i in range(A.nrows):
        var li = d_inv[i]
        for p in range(A.indptr[i], A.indptr[i + 1]):
            var j = A.indices[p]
            result.data[dest] = li * A.data[p] * d_inv[j]
            result.indices[dest] = j
            dest += 1
    return result^
