"""Performance test for SparseLU with larger matrices."""

from numerics.sparse_lu import SparseLU
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from numerics.utils import abs_f64


def main() raises:
    print("=== SparseLU Performance Test ===")
    print()

    var n = 196
    print("[Test] Building 196x196 tridiagonal matrix...")

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

    print(" nnz = ", len(A_csr.data))

    print(" Converting CSR to CSC...")
    var A_csc = A_csr.to_csc()

    print(" Factorizing...")
    var lu = SparseLU(n)
    lu.factorize(A_csc)
    print(" Factorization done!")

    var b: List[Float64] = []
    for _ in range(n):
        b.append(0.0)
    b[0] = 4.0
    b[n - 1] = 4.0
    for i in range(1, n - 1):
        b[i] = 2.0

    print(" Solving...")
    var x = lu.solve(b)
    print(" Solve done!")

    var Ax = A_csr.spmv_new(x)
    var residual = 0.0
    for i in range(n):
        residual = residual + abs_f64(Ax[i] - b[i])
    print(" ||Ax - b|| = ", residual)
    print()

    print("[Test 2] 392x392 block matrix (2n system)...")
    var n2 = 2 * n
    var B_csr = CSRMatrix(n2, n2, 0)
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
    B_csr._nnz = len(B_csr.data)

    print(" nnz = ", len(B_csr.data))

    print(" Converting CSR to CSC...")
    var B_csc = B_csr.to_csc()

    print(" Factorizing...")
    var lu2 = SparseLU(n2)
    lu2.factorize(B_csc)
    print(" Factorization done!")

    print(" Solving...")
    var b2: List[Float64] = []
    for _ in range(n2):
        b2.append(1.0)
    _ = lu2.solve(b2)
    print(" Solve done!")
    print()

    print("=== Performance test complete ===")
