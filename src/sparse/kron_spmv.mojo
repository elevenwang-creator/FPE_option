"""Kronecker-structured sparse matrix-vector products.

For kron(A,B).spmv(x) = vec(A @ X @ B^T) where X = reshape(x, p, q),
this does 1D SpMVs instead of building the 2.1M-row kron matrix.

For kron(sw_diag, vw_diag).spmv(v) = elementwise sw[i]*vw[j]*v[k].
"""

from sparse.csr import CSRMatrix
from sparse.scratch import ScratchBuffer


def kron_spmv(
    A: CSRMatrix, B: CSRMatrix, x: List[Float64]
) -> List[Float64]:
    """- kron(A, B).spmv(x) = vec(A @ X @ B^T) where X = reshape(x, p, q).

    A: m x p, B: n x q, x: p*q
    result: m*n

    Step 1: W = X @ B^T (p x n). W[k,j] = sum_l X[k,l] * B[j,l]
      Iterate B's rows (CSR-friendly).
    Step 2: Y = A @ W (m x n). Y[i,j] = sum_k A[i,k] * W[k,j]
    """
    var m = A.nrows
    var p = A.ncols
    var n = B.nrows
    var q = B.ncols

    var X = ScratchBuffer[Float64](p * q)
    for k in range(p * q):
        X[k] = x[k]

    var W = ScratchBuffer[Float64](p * n)
    for k in range(p * n):
        W[k] = 0.0

    for k in range(p):
        for j in range(n):
            var w_val = 0.0
            var b_start = B.indptr[j]
            var b_end = B.indptr[j + 1]
            for bp in range(b_start, b_end):
                var l = B.indices[bp]
                if l < 0 or l >= q:
                    continue
                w_val += B.data[bp] * X[k * q + l]
            W[k * n + j] = w_val

    var Y = ScratchBuffer[Float64](m * n)
    for k in range(m * n):
        Y[k] = 0.0

    for i in range(m):
        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        for ap in range(a_start, a_end):
            var a_val = A.data[ap]
            var k = A.indices[ap]
            if k < 0 or k >= p:
                continue
            for j in range(n):
                Y[i * n + j] += a_val * W[k * n + j]

    var result: List[Float64] = []
    for k in range(m * n):
        result.append(Y[k])

    return result^


def kron_T_spmv(
    A_T: CSRMatrix, B_T: CSRMatrix, v: List[Float64]
) -> List[Float64]:
    """- kron(A, B)^T.spmv(v) = kron(A^T, B^T).spmv(v) = vec(A^T @ V @ B).

    A^T: p x m, B^T: q x n (i.e. B: n x q), v: m*n
    result: p*q

    Step 1: W = V @ B (m x q). W[i,l] = sum_j V[i,j] * B[j,l]
      Iterate B_T rows: B_T[l,:] gives B[:,l].
    Step 2: Y = A^T @ W (p x q). Y[k,l] = sum_i A_T[k,i] * W[i,l]
    """
    var p = A_T.nrows
    var m = A_T.ncols
    var q = B_T.nrows
    var n = B_T.ncols

    var V = ScratchBuffer[Float64](m * n)
    for k in range(m * n):
        V[k] = v[k]

    var W = ScratchBuffer[Float64](m * q)
    for k in range(m * q):
        W[k] = 0.0

    for l in range(q):
        var bt_start = B_T.indptr[l]
        var bt_end = B_T.indptr[l + 1]
        for bp in range(bt_start, bt_end):
            var j = B_T.indices[bp]
            var b_val = B_T.data[bp]
            if j < 0 or j >= n:
                continue
            for i in range(m):
                W[i * q + l] += b_val * V[i * n + j]

    var Y = ScratchBuffer[Float64](p * q)
    for k in range(p * q):
        Y[k] = 0.0

    for k in range(p):
        var at_start = A_T.indptr[k]
        var at_end = A_T.indptr[k + 1]
        for atp in range(at_start, at_end):
            var at_val = A_T.data[atp]
            var i = A_T.indices[atp]
            if i < 0 or i >= m:
                continue
            for l in range(q):
                Y[k * q + l] += at_val * W[i * q + l]

    var result: List[Float64] = []
    for k in range(p * q):
        result.append(Y[k])

    return result^


def weights_spmv(
    sw: List[Float64], vw: List[Float64], v: List[Float64]
) -> List[Float64]:
    """- kron(sw_diag, vw_diag).spmv(v) = elementwise sw[i]*vw[j]*v[k].

    v: n_s * n_v, result: n_s * n_v
    result[i*n_v + j] = sw[i] * vw[j] * v[i*n_v + j]
    """
    var n_s = len(sw)
    var n_v = len(vw)
    var result: List[Float64] = []
    for i in range(n_s):
        var sw_i = sw[i]
        for j in range(n_v):
            var idx = i * n_v + j
            result.append(sw_i * vw[j] * v[idx])
    return result^
