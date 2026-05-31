"""Dense linear algebra operations for FPE systems."""

from numerics.utils.helpers import copy_mat


from std.math import abs

def lu_solve(A: List[List[Float64]], b: List[Float64]) -> List[Float64]:
    """Dense LU with partial pivoting."""
    var n = len(b)
    var LU = copy_mat(A)
    var x = b.copy()
    var perm = List[Int]()
    for i in range(n):
        perm.append(i)

    for k in range(n):
        var pivot = k
        var max_val = abs(LU[k][k])
        for i in range(k + 1, n):
            if abs(LU[i][k]) > max_val:
                max_val = abs(LU[i][k])
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



