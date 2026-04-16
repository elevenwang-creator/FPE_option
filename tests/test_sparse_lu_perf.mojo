"""Performance test for SparseLU with larger matrices."""

from numerics.sparse_lu import SparseLU
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.csr import CSRMatrix
from numerics.utils import zeros, abs_f64


def main() raises:
    print("=== SparseLU Performance Test ===")
    print()

    var n = 196
    print("[Test] Building 196x196 tridiagonal matrix...")

    var A_csr = CSRMatrix[DType.float64](n, n)
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
    A_csr.nrows = n
    A_csr.ncols = n

    print("  nnz = ", len(A_csr.data))

    print("  Converting CSR to CSC...")
    var A_csc = csr_to_csc(A_csr)

    print("  Factorizing...")
    var lu = SparseLU(n)
    lu.factorize(A_csc)
    print("  Factorization done!")

    var b = zeros(n)
    b[0] = 4.0
    b[n - 1] = 4.0
    for i in range(1, n - 1):
        b[i] = 2.0

    print("  Solving...")
    var x = lu.solve(b)
    print("  Solve done!")

    var Ax = A_csr.spmv(x)
    var residual = 0.0
    for i in range(n):
        residual = residual + abs_f64(Ax[i] - b[i])
    print("  ||Ax - b|| = ", residual)
    print()

    print("[Test 2] 392x392 block matrix (2n system)...")
    var n2 = 2 * n
    var B_csr = CSRMatrix[DType.float64](n2, n2)
    var indptr2: List[Int] = [0]
    var indices2: List[Int] = []
    var data2: List[Float64] = []

    for i in range(n2):
        if i > 0:
            indices2.append(i - 1)
            data2.append(-0.5)
        indices2.append(i)
        data2.append(2.0)
        if i < n2 - 1:
            indices2.append(i + 1)
            data2.append(-0.5)
        indptr2.append(len(indices2))

    B_csr.indptr = indptr2^
    B_csr.indices = indices2^
    B_csr.data = data2^
    B_csr.nrows = n2
    B_csr.ncols = n2

    print("  nnz = ", len(B_csr.data))

    print("  Converting CSR to CSC...")
    var B_csc = csr_to_csc(B_csr)

    print("  Factorizing...")
    var lu2 = SparseLU(n2)
    lu2.factorize(B_csc)
    print("  Factorization done!")

    print("  Solving...")
    var b2 = zeros(n2)
    for i in range(n2):
        b2[i] = 1.0
    var x2 = lu2.solve(b2)
    print("  Solve done!")
    print()

    print("=== Performance test complete ===")
