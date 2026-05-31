"""Unit test for SparseLU correctness and performance."""

from numerics.utils.sparse_lu import SparseLU
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.math import abs


def main() raises:
    print("=== SparseLU Unit Test ===")
    print()

    print("[Test 1] 3x3 dense matrix")
    var A3_csr = CSRMatrix(3, 3, 0)
    var indptr3: List[Int] = [0]
    var indices3: List[Int] = []
    var data3: List[Float64] = []
    indices3.append(0); data3.append(2.0)
    indices3.append(1); data3.append(1.0)
    indptr3.append(2)
    indices3.append(0); data3.append(1.0)
    indices3.append(1); data3.append(3.0)
    indices3.append(2); data3.append(1.0)
    indptr3.append(5)
    indices3.append(1); data3.append(1.0)
    indices3.append(2); data3.append(2.0)
    indptr3.append(7)
    A3_csr.indptr = indptr3^
    A3_csr.indices = indices3^
    A3_csr.data = data3^
    A3_csr._nnz = 7
    var A3 = A3_csr.to_csc()

    var lu3 = SparseLU(3)
    lu3.factorize(A3)

    var b3: List[Float64] = [5.0, 10.0, 8.0]
    var x3 = lu3.solve(b3)

    print(" A = [[2,1,0],[1,3,1],[0,1,2]]")
    print(" b = [5, 10, 8]")
    print(" x = ", end="")
    for i in range(3):
        print(String(x3[i]), end=" ")
    print()

    var residual3 = 0.0
    var Ax3 = List[Float64](length=3, fill=0.0)
    for col in range(3):
        for p in range(A3.colptr[col], A3.colptr[col + 1]):
            var row = A3.indices[p]
            Ax3[row] = Ax3[row] + A3.data[p] * x3[col]
    for i in range(3):
        residual3 = residual3 + abs(Ax3[i] - b3[i])
    print(" ||Ax - b|| = ", residual3)
    print()

    print("[Test 2] 5x5 tridiagonal matrix")
    var n5 = 5
    var A5_csr = CSRMatrix(n5, n5, 0)
    var indptr5: List[Int] = [0]
    var indices5: List[Int] = []
    var data5: List[Float64] = []
    for i in range(n5):
        if i > 0:
            indices5.append(i - 1)
            data5.append(-1.0)
        indices5.append(i)
        data5.append(4.0)
        if i < n5 - 1:
            indices5.append(i + 1)
            data5.append(-1.0)
        indptr5.append(len(indices5))
    A5_csr.indptr = indptr5^
    A5_csr.indices = indices5^
    A5_csr.data = data5^
    A5_csr._nnz = len(A5_csr.data)

    var A5 = A5_csr.to_csc()

    var lu5 = SparseLU(n5)
    lu5.factorize(A5)

    var b5: List[Float64] = []
    b5.append(3.0)
    b5.append(6.0)
    b5.append(3.0)
    b5.append(6.0)
    b5.append(4.0)
    var x5 = lu5.solve(b5)

    print(" x = ", end="")
    for i in range(n5):
        print(String(x5[i]), end=" ")
    print()

    var Ax5 = List[Float64](length=n5, fill=0.0)
    for col in range(n5):
        for p in range(A5.colptr[col], A5.colptr[col + 1]):
            var row = A5.indices[p]
            Ax5[row] = Ax5[row] + A5.data[p] * x5[col]

    var residual5 = 0.0
    for i in range(n5):
        residual5 = residual5 + abs(Ax5[i] - b5[i])
    print(" ||Ax - b|| = ", residual5)
    print()

    print("[Test 3] Non-symmetric 4x4 matrix (built via CSR)")
    var A4_csr = CSRMatrix(4, 4, 9)
    A4_csr.indptr = [0, 2, 5, 7, 9]
    A4_csr.indices = [0, 2, 1, 2, 3, 0, 2, 1, 3]
    A4_csr.data = [2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 5.0]
    var A4 = A4_csr.to_csc()

    var lu4 = SparseLU(4)
    lu4.factorize(A4)

    var b4: List[Float64] = [4.0, 8.0, 6.0, 12.0]
    var x4 = lu4.solve(b4)

    print(" x = ", end="")
    for i in range(4):
        print(String(x4[i]), end=" ")
    print()

    var Ax4 = List[Float64](length=4, fill=0.0)
    for col in range(4):
        for p in range(A4.colptr[col], A4.colptr[col + 1]):
            var row = A4.indices[p]
            Ax4[row] = Ax4[row] + A4.data[p] * x4[col]

    var residual4 = 0.0
    for i in range(4):
        residual4 = residual4 + abs(Ax4[i] - b4[i])
    print(" ||Ax - b|| = ", residual4)
    print()

    print("[Test 4] 32x32 pentadiagonal matrix (bandwidth 2)")
    var n32 = 32
    var A32_csr = CSRMatrix(n32, n32, 0)
    var indptr32: List[Int] = [0]
    var indices32: List[Int] = []
    var data32: List[Float64] = []
    for i in range(n32):
        if i > 1:
            indices32.append(i - 2)
            data32.append(-1.0)
        if i > 0:
            indices32.append(i - 1)
            data32.append(-1.0)
        indices32.append(i)
        data32.append(6.0)
        if i < n32 - 1:
            indices32.append(i + 1)
            data32.append(-1.0)
        if i < n32 - 2:
            indices32.append(i + 2)
            data32.append(-1.0)
        indptr32.append(len(indices32))
    A32_csr.indptr = indptr32^
    A32_csr.indices = indices32^
    A32_csr.data = data32^
    A32_csr._nnz = len(A32_csr.data)

    var A32 = A32_csr.to_csc()

    var lu32 = SparseLU(n32)
    lu32.factorize(A32)

    var b32: List[Float64] = []
    b32.append(4.0)
    b32.append(4.0)
    for _ in range(2, n32 - 2):
        b32.append(2.0)
    b32.append(4.0)
    b32.append(4.0)
    var x32 = lu32.solve(b32)

    var Ax32 = List[Float64](length=n32, fill=0.0)
    for col in range(n32):
        for p in range(A32.colptr[col], A32.colptr[col + 1]):
            var row = A32.indices[p]
            Ax32[row] = Ax32[row] + A32.data[p] * x32[col]

    var residual32 = 0.0
    for i in range(n32):
        residual32 = residual32 + abs(Ax32[i] - b32[i])
    var ok4 = residual32 < 1e-10
    print(" ||Ax - b|| = ", residual32)
    print(" PASS" if ok4 else " FAIL")
    print()

    print("=== All tests complete ===")
