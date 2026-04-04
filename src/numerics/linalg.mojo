"""Shared linear algebra operations.

Consolidates the 4× duplicated LU solve into a single implementation.
Provides the shared dense linear solve used by radau, lm, solver, calibrator.
"""

from numerics.utils import abs_f64, zeros, copy_vec, copy_mat, swap_rows


def lu_solve(A: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
    """Solve Ax = b via LU factorization with partial pivoting.

    Single implementation replacing 4 copies across:
    - numerics/ode/radau.mojo
    - numerics/optim/lm.mojo
    - engines/fpe/solver.mojo
    - engines/calibrator/calibrator.mojo
    """
    var n = len(b)
    var M = copy_mat(A)
    var rhs = copy_vec(b)

    # Forward elimination with partial pivoting
    for k in range(n):
        var pivot = k
        var pivot_value = abs_f64(M[k][k])
        for i in range(k + 1, n):
            var cand = abs_f64(M[i][k])
            if cand > pivot_value:
                pivot_value = cand
                pivot = i

        if pivot_value == 0.0:
            raise Error("Singular linear system in lu_solve")

        if pivot != k:
            swap_rows(M, k, pivot)
            var tmp_rhs = rhs[k]
            rhs[k] = rhs[pivot]
            rhs[pivot] = tmp_rhs

        for i in range(k + 1, n):
            var factor = M[i][k] / M[k][k]
            M[i][k] = 0.0
            for j in range(k + 1, n):
                M[i][j] = M[i][j] - factor * M[k][j]
            rhs[i] = rhs[i] - factor * rhs[k]

    # Back substitution
    var x = zeros(n)
    for rev in range(n):
        var i = n - 1 - rev
        var s = rhs[i]
        for j in range(i + 1, n):
            s = s - M[i][j] * x[j]
        x[i] = s / M[i][i]
    return x^


def dense_matvec(
    A: List[List[Float64]], x: List[Float64]
) -> List[Float64]:
    """Dense matrix-vector multiply: y = A @ x."""
    var n = len(A)
    var y = zeros(n)
    for i in range(n):
        var acc = 0.0
        for j in range(len(A[i])):
            acc += A[i][j] * x[j]
        y[i] = acc
    return y^


def csr_to_dense_float(
    A_data: List[Scalar[DType.float64]],
    A_indices: List[Int],
    A_indptr: List[Int],
    nrows: Int,
    ncols: Int,
) -> List[List[Float64]]:
    """Convert CSR components to dense List[List[Float64]]."""
    var out: List[List[Float64]] = []
    for _ in range(nrows):
        var row: List[Float64] = []
        for _ in range(ncols):
            row.append(0.0)
        out.append(row^)

    for i in range(nrows):
        var row_start = A_indptr[i]
        var row_end = A_indptr[i + 1]
        for p in range(row_start, row_end):
            out[i][A_indices[p]] = A_data[p]

    return out^
