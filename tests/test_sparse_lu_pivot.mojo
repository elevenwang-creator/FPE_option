"""Test SparseLU with matrices that require pivot swaps."""

from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.math import abs


def main() raises:
    print("=== SparseLU Pivot Swap Test ===")
    print()

    print("[Test 1] 4x4 matrix requiring row swap (zero diagonal)")
    var A1_csr = CSRMatrix(4, 4, 0)
    A1_csr.indptr = [0, 2, 5, 7, 9]
    A1_csr.indices = [1, 2, 0, 2, 3, 1, 3, 0, 2]
    A1_csr.data = [1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0]
    A1_csr._nnz = 9
    var A1 = A1_csr.to_csc()

    var lu1 = SparseLU(4)
    lu1.factorize(A1)

    var b1: List[Float64] = [3.0, 8.0, 11.0, 8.0]
    var x1 = lu1.solve(b1)

    print(" x = ", end="")
    for i in range(4):
        print(String(x1[i]), end=" ")
    print()

    var Ax1: List[Float64] = []
    for _ in range(4):
        Ax1.append(0.0)
    for col in range(4):
        for p in range(A1.colptr[col], A1.colptr[col + 1]):
            var row = A1.indices[p]
            Ax1[row] = Ax1[row] + A1.data[p] * x1[col]

    var res1 = 0.0
    for i in range(4):
        res1 = res1 + abs(Ax1[i] - b1[i])
    print(" ||Ax - b|| = ", res1)
    var ok1 = res1 < 1e-10
    print(" PASS" if ok1 else " FAIL")
    print()

    print("[Test 2] 6x6 non-symmetric with multiple swaps")
    var A2_csr = CSRMatrix(6, 6, 0)
    var indptr2: List[Int] = [0]
    var indices2: List[Int] = []
    var data2: List[Float64] = []
    var row_data: List[List[Tuple[Int, Float64]]] = []
    for _ in range(6):
        row_data.append(List[Tuple[Int, Float64]]())

    row_data[0].append((2, 1.0))
    row_data[0].append((3, 2.0))
    row_data[1].append((0, 3.0))
    row_data[1].append((1, 1.0))
    row_data[1].append((4, 1.0))
    row_data[2].append((1, 2.0))
    row_data[2].append((2, 4.0))
    row_data[2].append((5, 1.0))
    row_data[3].append((0, 1.0))
    row_data[3].append((3, 3.0))
    row_data[3].append((5, 2.0))
    row_data[4].append((2, 1.0))
    row_data[4].append((4, 5.0))
    row_data[5].append((1, 1.0))
    row_data[5].append((3, 2.0))
    row_data[5].append((5, 3.0))

    for i in range(6):
        for j in range(len(row_data[i])):
            indices2.append(row_data[i][j][0])
            data2.append(row_data[i][j][1])
        indptr2.append(len(indices2))
    A2_csr.indptr = indptr2^
    A2_csr.indices = indices2^
    A2_csr.data = data2^
    A2_csr._nnz = len(A2_csr.data)
    var A2 = A2_csr.to_csc()

    var lu2 = SparseLU(6)
    lu2.factorize(A2)

    var b2: List[Float64] = [3.0, 5.0, 7.0, 6.0, 6.0, 6.0]
    var x2 = lu2.solve(b2)

    print(" x = ", end="")
    for i in range(6):
        print(String(x2[i]), end=" ")
    print()

    var Ax2: List[Float64] = []
    for _ in range(6):
        Ax2.append(0.0)
    for col in range(6):
        for p in range(A2.colptr[col], A2.colptr[col + 1]):
            var row = A2.indices[p]
            Ax2[row] = Ax2[row] + A2.data[p] * x2[col]

    var res2 = 0.0
    for i in range(6):
        res2 = res2 + abs(Ax2[i] - b2[i])
    print(" ||Ax - b|| = ", res2)
    var ok2 = res2 < 1e-10
    print(" PASS" if ok2 else " FAIL")
    print()

    print("[Test 3] solve_inplace with pivoting")
    var b3_vec = FixedSizeVector(4)
    var work3 = FixedSizeVector(4)
    var b3_ptr_init = b3_vec.ptr()
    b3_ptr_init[0] = 3.0
    b3_ptr_init[1] = 8.0
    b3_ptr_init[2] = 11.0
    b3_ptr_init[3] = 8.0

    lu1.solve_inplace(b3_vec, work3)

    var b3_result = b3_vec.to_list()

    var res3 = 0.0
    for i in range(4):
        var diff = abs(b3_result[i] - x1[i])
        res3 = res3 + diff
    print(" ||solve - solve_inplace|| = ", res3)
    var ok3 = res3 < 1e-10
    print(" PASS" if ok3 else " FAIL")
    print()

    var all_pass = ok1 and ok2 and ok3
    if all_pass:
        print("=== ALL PIVOT SWAP TESTS PASS ===")
    else:
        print("=== SOME TESTS FAILED ===")

    print()
    print("[Test 4] 64x64 block-pentadiagonal with forced pivots")
    var n_bp = 64
    var A_bp_csr = CSRMatrix(n_bp, n_bp, 0)
    var indptr_bp: List[Int] = [0]
    var indices_bp: List[Int] = []
    var data_bp: List[Float64] = []
    for i in range(n_bp):
        if i > 1:
            indices_bp.append(i - 2)
            data_bp.append(-0.25)
        if i > 0:
            indices_bp.append(i - 1)
            data_bp.append(-0.5)
        indices_bp.append(i)
        data_bp.append(4.0 + Float64(i % 7))
        if i < n_bp - 1:
            indices_bp.append(i + 1)
            data_bp.append(-0.5)
        if i < n_bp - 2:
            indices_bp.append(i + 2)
            data_bp.append(-0.25)
        indptr_bp.append(len(indices_bp))
    A_bp_csr.indptr = indptr_bp^
    A_bp_csr.indices = indices_bp^
    A_bp_csr.data = data_bp^
    A_bp_csr._nnz = len(A_bp_csr.data)
    var A_bp = A_bp_csr.to_csc()
    var lu_bp = SparseLU(n_bp)
    lu_bp.factorize(A_bp)
    var b_bp: List[Float64] = []
    for i in range(n_bp):
        b_bp.append(Float64(i + 1))
    var x_bp = lu_bp.solve(b_bp)
    var Ax_bp = A_bp_csr.spmv_new(x_bp)
    var res_bp = 0.0
    for i in range(n_bp):
        res_bp = res_bp + abs(Ax_bp[i] - b_bp[i])
    print(" ||Ax - b|| = ", res_bp)
    var ok4 = res_bp < 1e-8
    print(" PASS" if ok4 else " FAIL")

    all_pass = all_pass and ok4
    if all_pass:
        print("=== ALL PIVOT SWAP TESTS PASS ===")
    else:
        print("=== SOME TESTS FAILED ===")
