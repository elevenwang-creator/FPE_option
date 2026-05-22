"""Test SparseLU symbolic/numeric factorization split."""

from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.math import abs


def main() raises:
    print("=== SparseLU Symbolic/Numeric Split Test ===")
    print()

    print("[Test 1] Sym+Num matches factorize on 5x5 tridiagonal")
    var n = 5
    var A_csr = CSRMatrix(n, n, 0)
    var indptr: List[Int] = [0]
    var indices: List[Int] = []
    var data: List[Float64] = []
    for i in range(n):
        if i > 0:
            indices.append(i - 1)
            data.append(-1.0)
        indices.append(i)
        data.append(4.0)
        if i < n - 1:
            indices.append(i + 1)
            data.append(-1.0)
        indptr.append(len(indices))
    A_csr.indptr = indptr^
    A_csr.indices = indices^
    A_csr.data = data^
    A_csr._nnz = len(A_csr.data)

    var A_csc = A_csr.to_csc()

    var lu_full = SparseLU(n)
    lu_full.factorize(A_csc)

    var lu_split = SparseLU(n)
    lu_split.factorize_symbolic(A_csc)
    lu_split.factorize_numeric(A_csc)

    var b: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    var x_full = lu_full.solve(b)
    var x_split = lu_split.solve(b)

    var diff = 0.0
    for i in range(n):
        diff = diff + abs(x_full[i] - x_split[i])
    print("  ||x_full - x_split|| = ", diff)
    var ok1 = diff < 1e-12
    print("  PASS" if ok1 else "  FAIL")
    print()

    print("[Test 2] Re-numeric with different diagonal values")
    var A2_csr = CSRMatrix(n, n, 0)
    var indptr2: List[Int] = [0]
    var indices2: List[Int] = []
    var data2: List[Float64] = []
    for i in range(n):
        if i > 0:
            indices2.append(i - 1)
            data2.append(-2.0)
        indices2.append(i)
        data2.append(8.0)
        if i < n - 1:
            indices2.append(i + 1)
            data2.append(-2.0)
        indptr2.append(len(indices2))
    A2_csr.indptr = indptr2^
    A2_csr.indices = indices2^
    A2_csr.data = data2^
    A2_csr._nnz = len(A2_csr.data)

    var A2_csc = A2_csr.to_csc()

    lu_split.factorize_numeric(A2_csc)

    var b2: List[Float64] = [4.0, 4.0, 4.0, 4.0, 4.0]
    var x2 = lu_split.solve(b2)

    var lu2_full = SparseLU(n)
    lu2_full.factorize(A2_csc)
    var x2_full = lu2_full.solve(b2)

    var diff2 = 0.0
    for i in range(n):
        diff2 = diff2 + abs(x2[i] - x2_full[i])
    print("  ||x2_renum - x2_full|| = ", diff2)
    var ok2 = diff2 < 1e-10
    print("  PASS" if ok2 else "  FAIL")
    print()

    print("[Test 3] solve_inplace matches solve after sym+num")
    var b3_vec = FixedSizeVector(n)
    var work3 = FixedSizeVector(n)
    var b3_data: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    for i in range(n):
        b3_vec.ptr()[i] = b3_data[i]

    lu_split.factorize_numeric(A_csc)
    lu_split.solve_inplace(b3_vec, work3)
    var x3_inplace = b3_vec.to_list()
    var x3_solve = lu_split.solve(b3_data)

    var diff3 = 0.0
    for i in range(n):
        diff3 = diff3 + abs(x3_inplace[i] - x3_solve[i])
    print("  ||solve - solve_inplace|| = ", diff3)
    var ok3 = diff3 < 1e-12
    print("  PASS" if ok3 else "  FAIL")
    print()


    print("[Test 4] Sym+Num with pivoting (zero diagonal 4x4)")
    var A4_csr = CSRMatrix(4, 4, 0)
    var indptr4: List[Int] = [0]
    var indices4: List[Int] = []
    var data4: List[Float64] = []
    indices4.append(1); data4.append(2.0)
    indices4.append(2); data4.append(1.0)
    indptr4.append(2)
    indices4.append(0); data4.append(1.0)
    indices4.append(3); data4.append(3.0)
    indptr4.append(4)
    indices4.append(1); data4.append(1.0)
    indices4.append(2); data4.append(4.0)
    indices4.append(3); data4.append(1.0)
    indptr4.append(7)
    indices4.append(2); data4.append(1.0)
    indices4.append(3); data4.append(5.0)
    indptr4.append(9)
    A4_csr.indptr = indptr4^
    A4_csr.indices = indices4^
    A4_csr.data = data4^
    A4_csr._nnz = 9
    var A4_csc = A4_csr.to_csc()

    var lu4_full = SparseLU(4)
    lu4_full.factorize(A4_csc)
    var lu4_split = SparseLU(4)
    lu4_split.factorize_symbolic(A4_csc)
    lu4_split.factorize_numeric(A4_csc)

    var b4: List[Float64] = [4.0, 10.0, 17.0, 18.0]
    var x4_full = lu4_full.solve(b4)
    var x4_split = lu4_split.solve(b4)

    var diff4 = 0.0
    for i in range(4):
        diff4 = diff4 + abs(x4_full[i] - x4_split[i])
    print(" ||x4_full - x4_split|| = ", diff4)
    var ok4 = diff4 < 1e-12
    print(" PASS" if ok4 else " FAIL")
    print()

    print("[Test 5] Re-numeric with pivoting (modified 4x4)")
    var A5_csr = CSRMatrix(4, 4, 0)
    var indptr5: List[Int] = [0]
    var indices5: List[Int] = []
    var data5: List[Float64] = []
    indices5.append(1); data5.append(3.0)
    indices5.append(2); data5.append(2.0)
    indptr5.append(2)
    indices5.append(0); data5.append(2.0)
    indices5.append(3); data5.append(5.0)
    indptr5.append(4)
    indices5.append(1); data5.append(2.0)
    indices5.append(2); data5.append(6.0)
    indices5.append(3); data5.append(2.0)
    indptr5.append(7)
    indices5.append(2); data5.append(2.0)
    indices5.append(3); data5.append(8.0)
    indptr5.append(9)
    A5_csr.indptr = indptr5^
    A5_csr.indices = indices5^
    A5_csr.data = data5^
    A5_csr._nnz = 9
    var A5_csc = A5_csr.to_csc()

    lu4_split.factorize_numeric(A5_csc)
    var lu5_full = SparseLU(4)
    lu5_full.factorize(A5_csc)

    var b5: List[Float64] = [7.0, 12.0, 28.0, 30.0]
    var x5_renum = lu4_split.solve(b5)
    var x5_full = lu5_full.solve(b5)

    var diff5 = 0.0
    for i in range(4):
        diff5 = diff5 + abs(x5_renum[i] - x5_full[i])
    print(" ||x5_renum - x5_full|| = ", diff5)
    var ok5 = diff5 < 1e-10
    print(" PASS" if ok5 else " FAIL")
    print()

    var all_pass = ok1 and ok2 and ok3 and ok4 and ok5
    if all_pass:
        print("=== ALL SYM/NUM TESTS PASS ===")
    else:
        print("=== SOME TESTS FAILED ===")
