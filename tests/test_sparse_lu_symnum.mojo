"""Test SparseLU factorize + solve + solve_inplace on various matrices."""

from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.math import abs


def main() raises:
    print("=== SparseLU Factor/Solve Test ===")
    print()

    var all_pass = True

    # Test 1: 5x5 tridiagonal
    print("[Test 1] 5x5 tridiagonal solve")
    var n1 = 5
    var csr1 = CSRMatrix(n1, n1, 0)
    var iptr1: List[Int] = [0]
    var idxs1: List[Int] = []
    var dat1: List[Float64] = []
    for i in range(n1):
        if i > 0:
            idxs1.append(i - 1)
            dat1.append(-1.0)
        idxs1.append(i)
        dat1.append(4.0)
        if i < n1 - 1:
            idxs1.append(i + 1)
            dat1.append(-1.0)
        iptr1.append(len(idxs1))
    csr1.indptr = iptr1^
    csr1.indices = idxs1^
    csr1.data = dat1^
    csr1._nnz = len(csr1.data)
    var A1 = csr1.to_csc()
    var b1: List[Float64] = [3.0, 2.0, 2.0, 2.0, 3.0]
    var lu1 = SparseLU(5)
    lu1.factorize(A1)
    var x1 = lu1.solve(b1)
    var err1 = 0.0
    for i in range(5):
        err1 += abs(x1[i] - 1.0)
    var ok1 = err1 < 1e-10
    print("  ||x - 1|| =", err1, "PASS" if ok1 else "FAIL")
    all_pass = all_pass and ok1
    print()

    # Test 2: Different tridiagonal values
    print("[Test 2] 5x5 tridiagonal (diag=8, off=-2)")
    var n2 = 5
    var csr2 = CSRMatrix(n2, n2, 0)
    var iptr2: List[Int] = [0]
    var idxs2: List[Int] = []
    var dat2: List[Float64] = []
    for i in range(n2):
        if i > 0:
            idxs2.append(i - 1)
            dat2.append(-2.0)
        idxs2.append(i)
        dat2.append(8.0)
        if i < n2 - 1:
            idxs2.append(i + 1)
            dat2.append(-2.0)
        iptr2.append(len(idxs2))
    csr2.indptr = iptr2^
    csr2.indices = idxs2^
    csr2.data = dat2^
    csr2._nnz = len(csr2.data)
    var A2 = csr2.to_csc()
    var b2: List[Float64] = [6.0, 4.0, 4.0, 4.0, 6.0]
    var lu2 = SparseLU(5)
    lu2.factorize(A2)
    var x2 = lu2.solve(b2)
    var err2 = 0.0
    for i in range(5):
        err2 += abs(x2[i] - 1.0)
    var ok2 = err2 < 1e-10
    print("  ||x - 1|| =", err2, "PASS" if ok2 else "FAIL")
    all_pass = all_pass and ok2
    print()

    # Test 3: solve_inplace matches solve
    print("[Test 3] solve_inplace matches solve")
    var n3 = 5
    var csr3 = CSRMatrix(n3, n3, 0)
    var iptr3: List[Int] = [0]
    var idxs3: List[Int] = []
    var dat3: List[Float64] = []
    for i in range(n3):
        if i > 0:
            idxs3.append(i - 1)
            dat3.append(-1.0)
        idxs3.append(i)
        dat3.append(4.0)
        if i < n3 - 1:
            idxs3.append(i + 1)
            dat3.append(-1.0)
        iptr3.append(len(idxs3))
    csr3.indptr = iptr3^
    csr3.indices = idxs3^
    csr3.data = dat3^
    csr3._nnz = len(csr3.data)
    var A3 = csr3.to_csc()
    var b3: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    var lu3 = SparseLU(5)
    lu3.factorize(A3)
    var x3_ref = lu3.solve(b3)
    var b3_vec = FixedSizeVector(5)
    var work3 = FixedSizeVector(5)
    for i in range(5):
        b3_vec.ptr()[i] = b3[i]
    lu3.solve_inplace(b3_vec, work3)
    var x3_inplace = b3_vec.to_list()
    var err3 = 0.0
    for i in range(5):
        err3 += abs(x3_ref[i] - x3_inplace[i])
    var ok3 = err3 < 1e-12
    print("  ||solve - solve_inplace|| =", err3, "PASS" if ok3 else "FAIL")
    all_pass = all_pass and ok3
    print()

    # Test 4: 4x4 with pivoting (zero diagonal)
    print("[Test 4] 4x4 pivoting (zero diagonal)")
    var csr4 = CSRMatrix(4, 4, 9)
    var iptr4: List[Int] = [0]
    var idxs4: List[Int] = []
    var dat4: List[Float64] = []
    idxs4.append(1); dat4.append(2.0)
    idxs4.append(2); dat4.append(1.0)
    iptr4.append(2)
    idxs4.append(0); dat4.append(1.0)
    idxs4.append(3); dat4.append(3.0)
    iptr4.append(4)
    idxs4.append(1); dat4.append(1.0)
    idxs4.append(2); dat4.append(4.0)
    idxs4.append(3); dat4.append(1.0)
    iptr4.append(7)
    idxs4.append(2); dat4.append(1.0)
    idxs4.append(3); dat4.append(5.0)
    iptr4.append(9)
    csr4.indptr = iptr4^
    csr4.indices = idxs4^
    csr4.data = dat4^
    csr4._nnz = 9
    var A4 = csr4.to_csc()
    var b4: List[Float64] = [4.0, 10.0, 17.0, 18.0]
    var lu4 = SparseLU(4)
    lu4.factorize(A4)
    var x4 = lu4.solve(b4)
    var Ax4 = csr4.spmv_new(x4)
    var resid4 = 0.0
    for i in range(4):
        resid4 += abs(Ax4[i] - b4[i])
    var ok4 = resid4 < 1e-10
    print("  ||Ax - b|| =", resid4, "PASS" if ok4 else "FAIL")
    all_pass = all_pass and ok4
    print()

    if all_pass:
        print("=== ALL TESTS PASS ===")
    else:
        print("=== SOME TESTS FAILED ===")
