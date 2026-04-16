"""Galerkin assembler exploiting Kronecker structure.

Key insight: FPE mass and stiffness matrices have Kronecker structure
because weights are separable (sw ⊗ vw) and basis functions are tensor products.

Instead of forming 328K×600 intermediate matrices, we compute:
  M = Ms ⊗ Mv  (mass: two small dense 1D matrices, then kron)
  K = sum of kron(As_i, Av_i)  (stiffness: sum of 1D kron products)

1D dense Galerkin matrices are computed directly from CSR basis evaluations:
  Ms = Bs^T @ diag(sw) @ Bs  (≈24×24 dense)
  Mv = Bv^T @ diag(vw) @ Bv  (≈25×25 dense)

This reduces memory from O(n_s * n_v * Rcol) to O(Rcol_s^2 + Rcol_v^2).
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from sparse.csr import CSRMatrix
from sparse.ops import add, kron, scale


def _zeros_2d(rows: Int, cols: Int) -> List[List[Float64]]:
    var result: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        for _ in range(cols):
            row.append(0.0)
        result.append(row^)
    return result^


def _galerkin_1d(B: CSRMatrix, weights: List[Float64]) -> List[List[Float64]]:
    var n = B.ncols
    var result = _zeros_2d(n, n)

    for row_idx in range(B.nrows):
        var w = weights[row_idx]
        if w == 0.0:
            continue
        var row_start = B.indptr[row_idx]
        var row_end = B.indptr[row_idx + 1]
        for p1 in range(row_start, row_end):
            var j1 = B.indices[p1]
            var v1 = B.data[p1]
            var wv1 = w * v1
            for p2 in range(p1, row_end):
                var j2 = B.indices[p2]
                var v2 = B.data[p2]
                result[j1][j2] += wv1 * v2
                if j1 != j2:
                    result[j2][j1] += wv1 * v2

    return result^


def _galerkin_mixed_1d(
    B1: CSRMatrix, B2: CSRMatrix, weights: List[Float64]
) -> List[List[Float64]]:
    """Compute B1^T @ diag(weights) @ B2 as dense matrix from CSR bases.

    Used for mixed terms like (dBs/ds)^T @ diag(sw*s) @ Bs.
    """
    var n1 = B1.ncols
    var n2 = B2.ncols
    var result = _zeros_2d(n1, n2)

    for row_idx in range(B1.nrows):
        var w = weights[row_idx]
        if w == 0.0:
            continue
        for p1 in range(B1.indptr[row_idx], B1.indptr[row_idx + 1]):
            var j1 = B1.indices[p1]
            var v1 = B1.data[p1]
            var wv1 = w * v1
            for p2 in range(B2.indptr[row_idx], B2.indptr[row_idx + 1]):
                var j2 = B2.indices[p2]
                var v2 = B2.data[p2]
                result[j1][j2] += wv1 * v2

    return result^


def _dense_kron_to_csr(
    A: List[List[Float64]], B: List[List[Float64]]
) -> CSRMatrix:
    """Compute Kronecker product of two dense matrices as CSR.

    A is m×n, B is p×q, result is (m*p)×(n*q) CSR.
    Only stores non-zero entries (threshold 1e-15).
    """
    var m = len(A)
    if m == 0:
        return CSRMatrix(0, 0)
    var n = len(A[0])
    var p = len(B)
    if p == 0:
        return CSRMatrix(m * 0, n * 0)
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


def _dense_add(
    mut A: List[List[Float64]], B: List[List[Float64]], alpha: Float64 = 1.0
):
    """A += alpha * B in-place for dense matrices."""
    for i in range(len(A)):
        for j in range(len(A[i])):
            A[i][j] += alpha * B[i][j]


def _dense_scale(A: List[List[Float64]], alpha: Float64) -> List[List[Float64]]:
    var m = len(A)
    if m == 0:
        return A.copy()
    var n = len(A[0])
    var result = _zeros_2d(m, n)
    for i in range(m):
        for j in range(n):
            result[i][j] = alpha * A[i][j]
    return result^


def _dense_transpose(A: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(A)
    if m == 0:
        return A.copy()
    var n = len(A[0])
    var result = _zeros_2d(n, m)
    for i in range(m):
        for j in range(n):
            result[j][i] = A[i][j]
    return result^


struct GalerkinAssembler[B: Int]:
    def __init__(out self):
        pass

    def mass_matrix(self, domain: FPEDomain) -> CSRMatrix:
        var basis = domain.build_basis()

        var Bs = basis.basis_s.eval_all(domain.s_points)
        var Bv = basis.basis_v.eval_all(domain.v_points)

        var Ms = _galerkin_1d(Bs, domain.s_weights)
        var Mv = _galerkin_1d(Bv, domain.v_weights)

        return _dense_kron_to_csr(Ms, Mv)

    def stiffness_matrix(
        self, domain: FPEDomain, params: HestonParams
    ) -> CSRMatrix:
        var basis = domain.build_basis()

        var Bs = basis.basis_s.eval_all(domain.s_points)
        var dBs = basis.basis_s.first_derivative_all(domain.s_points)
        var Bv = basis.basis_v.eval_all(domain.v_points)
        var dBv = basis.basis_v.first_derivative_all(domain.v_points)

        var sw = domain.s_weights.copy()
        var vw = domain.v_weights.copy()
        var s_phys = domain.s_points_phys.copy()
        var v_phys = domain.v_points_phys.copy()
        var j = domain.jacobian_factor()

        var sw_s: List[Float64] = []
        var sw_s2: List[Float64] = []
        for i in range(len(sw)):
            sw_s.append(sw[i] * s_phys[i])
            sw_s2.append(sw[i] * s_phys[i] * s_phys[i])

        var vw_v: List[Float64] = []
        for i in range(len(vw)):
            vw_v.append(vw[i] * v_phys[i])

        var Ms = _galerkin_1d(Bs, sw)
        var Mv = _galerkin_1d(Bv, vw)

        var Ks_s = _galerkin_mixed_1d(dBs, Bs, sw_s)
        var Ks_ss = _galerkin_1d(dBs, sw_s2)
        var Ks_sT = _dense_transpose(Ks_s)

        var Kv_v = _galerkin_1d(Bv, vw_v)
        var Kv_cross = _galerkin_mixed_1d(dBv, Bv, vw)
        var Kv_v_cross = _galerkin_mixed_1d(dBv, Bv, vw_v)
        var Kv_vv = _galerkin_1d(dBv, vw_v)
        var Kv_cross_T = _dense_transpose(Kv_cross)
        var Kv_v_cross_T = _dense_transpose(Kv_v_cross)

        var k1 = (-params.r + 0.5 * params.rho * params.sigma) / j
        var k2 = 1.0 / j
        var k3 = 0.5 / (j * j)
        var k4 = 0.5 * params.rho * params.sigma / j
        var k5 = 0.5 * params.sigma * params.sigma - params.kappa * params.theta
        var k6 = params.kappa + 0.5 * params.rho * params.sigma
        var k8 = 0.5 * params.sigma * params.sigma

        var Mv_k1k2 = _dense_scale(Mv, k1)
        _dense_add(Mv_k1k2, Kv_v, k2)

        var K_sb = _dense_kron_to_csr(Ks_s, Mv_k1k2)
        var K_ssv = _dense_kron_to_csr(Ks_ss, _dense_scale(Kv_v, k3))
        var K_sv = _dense_kron_to_csr(Ks_s, _dense_scale(Kv_v_cross_T, k4))

        var Kv_k5k6 = _dense_scale(Kv_cross, k5)
        _dense_add(Kv_k5k6, Kv_v_cross, k6)

        var K_vb = _dense_kron_to_csr(Ms, Kv_k5k6)
        var K_vv = _dense_kron_to_csr(Ms, _dense_scale(Kv_vv, k8))
        var K_vs = _dense_kron_to_csr(Ks_sT, _dense_scale(Kv_v_cross, k4))

        var K = add(K_sb, K_ssv)
        var K2 = add(K, K_sv)
        var K3 = add(K2, K_vb)
        var K4 = add(K3, K_vv)
        var K5 = add(K4, K_vs)

        return K5^
