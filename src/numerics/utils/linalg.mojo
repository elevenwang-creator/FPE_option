"""Dense linear algebra operations for FPE systems."""

from numerics.utils.helpers import abs_f64, zeros, copy_mat, copy_vec


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
