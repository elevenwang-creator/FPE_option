"""Sparse linear algebra operations for FPE systems.

Imports modular LU from sparse_lu.mojo for sparse factorization.
"""

from numerics.utils import abs_f64, zeros
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix
from numerics.sparse_lu import SparseLU
from std.sys import simd_width_of


comptime SIMD_WIDTH = simd_width_of[DType.float64]()


def lu_solve(A: List[List[Float64]], b: List[Float64]) -> List[Float64]:
    """Dense LU with partial pivoting."""
    var n = len(b)
    var LU = copy_mat(A)
    var x = copy_vec(b)
    var perm = List[Int]()
    for i in range(n):
        perm.append(i)

    for k in range(n):
        var pivot = k
        var max_val = abs_f64(LU[k][k])
        for i in range(k + 1, n):
            if abs_f64(LU[i][k]) > max_val:
                max_val = abs_f64(LU[i][k])
                pivot = i

        if max_val < 1e-14:
            continue

        if pivot != k:
            for j in range(n):
                var tmp = LU[k][j]
                LU[k][j] = LU[pivot][j]
                LU[pivot][j] = tmp
            var tmp_perm = x[k]
            x[k] = x[pivot]
            x[pivot] = tmp_perm
            var tmp_i = perm[k]
            perm[k] = perm[pivot]
            perm[pivot] = tmp_i

        var piv = LU[k][k]
        for i in range(k + 1, n):
            LU[i][k] = LU[i][k] / piv
            for j in range(k + 1, n):
                LU[i][j] = LU[i][j] - LU[i][k] * LU[k][j]

    for i in range(n):
        for j in range(i):
            x[i] = x[i] - LU[i][j] * x[j]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] = x[i] - LU[i][j] * x[j]
        x[i] = x[i] / LU[i][i]

    return x^


def compute_jacobian(
    M: CSRMatrix,
    K: CSRMatrix,
) raises -> List[List[Float64]]:
    """Compute J = -M^(-1) @ K using sparse LU."""
    var n = M.nrows
    var lu = SparseLU(n)
    try:
        lu.factorize(M.to_csc())
    except:
        pass

    var neg_K = List[List[Float64]]()
    for _ in range(n):
        var row = zeros(n)
        neg_K.append(row^)

    for i in range(n):
        for p in range(K.indptr[i], K.indptr[i + 1]):
            var col = K.indices[p]
            neg_K[i][col] = -K.data[p]

    var J = List[List[Float64]]()
    for col in range(n):
        var rhs = zeros(n)
        for i in range(n):
            rhs[i] = neg_K[i][col]
        var x = lu.solve(rhs)
        J.append(x^)

    return J^


def dense_matvec(A: List[List[Float64]], x: List[Float64]) -> List[Float64]:
    """Dense matrix-vector multiply."""
    var n = len(A)
    var y = zeros(n)
    for i in range(n):
        var acc: Float64 = 0.0
        for j in range(n):
            acc += A[i][j] * x[j]
        y[i] = acc
    return y^


def sparse_matvec(A: CSRMatrix, x: List[Float64]) -> List[Float64]:
    """Sparse matrix-vector multiply with SIMD."""
    comptime width = SIMD_WIDTH
    var n = A.nrows
    var y = zeros(n)

    for i in range(n):
        var row_start = A.indptr[i]
        var row_end = A.indptr[i + 1]
        var acc: Float64 = 0.0
        var p = row_start

        while p + width <= row_end:
            var vals = SIMD[DType.float64, width]()
            var x_vals = SIMD[DType.float64, width]()
            for k in range(width):
                vals[k] = A.data[p + k]
                x_vals[k] = x[A.indices[p + k]]
            acc += (vals * x_vals).reduce_add()
            p += width

        while p < row_end:
            acc += A.data[p] * x[A.indices[p]]
            p += 1

        y[i] = acc

    return y^


def copy_mat(A: List[List[Float64]]) -> List[List[Float64]]:
    var result: List[List[Float64]] = []
    for i in range(len(A)):
        var row: List[Float64] = []
        for j in range(len(A[i])):
            row.append(A[i][j])
        result.append(row^)
    return result^


def copy_vec(v: List[Float64]) -> List[Float64]:
    var result: List[Float64] = []
    for i in range(len(v)):
        result.append(v[i])
    return result^
