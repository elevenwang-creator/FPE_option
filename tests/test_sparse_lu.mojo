"""Unit test for SparseLU correctness and performance."""

from numerics.sparse_lu import SparseLU
from sparse.csc import CSCMatrix
from numerics.utils import zeros, abs_f64


def main() raises:
    print("=== SparseLU Unit Test ===")
    print()

    print("[Test 1] 3x3 dense matrix")
    var A3 = CSCMatrix[DType.float64](3, 3)
    A3.indptr = [0, 3, 6, 9]
    A3.row = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    A3.data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]
    A3.nrows = 3
    A3.ncols = 3

    var lu3 = SparseLU(3)
    lu3.factorize(A3)

    var b3: List[Float64] = [5.0, 10.0, 8.0]
    var x3 = lu3.solve(b3)

    print("  A = [[2,1,0],[1,3,1],[0,1,2]]")
    print("  b = [5, 10, 8]")
    print("  x = ", end="")
    for i in range(3):
        print(String(x3[i]), end=" ")
    print()

    var residual3 = 0.0
    var expected3: List[Float64] = [1.0, 2.0, 3.0]
    for i in range(3):
        residual3 = residual3 + abs_f64(x3[i] - expected3[i])
    print("  residual = ", residual3)
    print()

    print("[Test 2] 5x5 sparse matrix")
    var n5 = 5
    var A5 = CSCMatrix[DType.float64](n5, n5)
    A5.indptr = [0, 2, 4, 6, 8, 10]
    A5.row = [0, 1, 0, 1, 2, 3, 2, 3, 3, 4]
    A5.data = [4.0, -1.0, -1.0, 4.0, 4.0, -1.0, -1.0, 4.0, 4.0, 4.0]
    A5.nrows = n5
    A5.ncols = n5

    var lu5 = SparseLU(n5)
    lu5.factorize(A5)

    var b5: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    var x5 = lu5.solve(b5)

    print("  x = ", end="")
    for i in range(n5):
        print(String(x5[i]), end=" ")
    print()

    var Ax5: List[Float64] = []
    for _ in range(n5):
        Ax5.append(0.0)
    for col in range(n5):
        for p in range(A5.indptr[col], A5.indptr[col + 1]):
            var row = A5.row[p]
            Ax5[row] = Ax5[row] + A5.data[p] * x5[col]

    var residual5 = 0.0
    for i in range(n5):
        residual5 = residual5 + abs_f64(Ax5[i] - b5[i])
    print("  ||Ax - b|| = ", residual5)
    print()

    print("[Test 3] Non-symmetric 4x4 matrix")
    var A4 = CSCMatrix[DType.float64](4, 4)
    A4.indptr = [0, 2, 5, 7, 9]
    A4.row = [0, 2, 1, 2, 3, 0, 2, 1, 3]
    A4.data = [2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 5.0]
    A4.nrows = 4
    A4.ncols = 4

    var lu4 = SparseLU(4)
    lu4.factorize(A4)

    var b4: List[Float64] = [4.0, 8.0, 6.0, 12.0]
    var x4 = lu4.solve(b4)

    print("  x = ", end="")
    for i in range(4):
        print(String(x4[i]), end=" ")
    print()

    var Ax4: List[Float64] = []
    for _ in range(4):
        Ax4.append(0.0)
    for col in range(4):
        for p in range(A4.indptr[col], A4.indptr[col + 1]):
            var row = A4.row[p]
            Ax4[row] = Ax4[row] + A4.data[p] * x4[col]

    var residual4 = 0.0
    for i in range(4):
        residual4 = residual4 + abs_f64(Ax4[i] - b4[i])
    print("  ||Ax - b|| = ", residual4)
    print()

    print("=== All tests complete ===")
