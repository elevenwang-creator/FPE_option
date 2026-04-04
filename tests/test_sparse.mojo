from sparse.csr import CSRMatrix
from sparse.coo import COOMatrix
from sparse.diag import DiagMatrix
from sparse.ops import kron, spgemm, spmm
from std.testing import assert_true, assert_equal, TestSuite


def assert_float_close(a: Float64, b: Float64, atol: Float64 = 1e-10) raises:
    var diff = a - b
    if diff < 0:
        diff = -diff
    assert_true(diff < atol, "Expected " + String(b) + " got " + String(a))


# ---------------------------------------------------------------------------
# CSRMatrix.spmv
# A = [[1, 0, 2],
#      [0, 3, 0],
#      [4, 0, 5]]
# x = [1, 2, 3]
# expected y = [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
# ---------------------------------------------------------------------------
def test_csr_spmv() raises:
    var dense: List[List[Scalar[DType.float64]]] = []
    var row0: List[Scalar[DType.float64]] = [1.0, 0.0, 2.0]
    var row1: List[Scalar[DType.float64]] = [0.0, 3.0, 0.0]
    var row2: List[Scalar[DType.float64]] = [4.0, 0.0, 5.0]
    dense.append(row0^)
    dense.append(row1^)
    dense.append(row2^)
    var A = CSRMatrix[DType.float64].from_dense(dense)

    assert_equal(A.nnz(), 5)
    assert_equal(A.nrows, 3)
    assert_equal(A.ncols, 3)

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y = A.spmv(x)

    assert_equal(len(y), 3)
    assert_float_close(y[0], 7.0)
    assert_float_close(y[1], 6.0)
    assert_float_close(y[2], 19.0)


# ---------------------------------------------------------------------------
# COOMatrix.to_csr
# Entries: (0,0,1.0), (0,2,2.0), (1,1,3.0), (2,0,4.0), (2,2,5.0)
# Expected CSR: nnz=5, indptr=[0,2,3,5], indices=[0,2,1,0,2],
#               data=[1,2,3,4,5]
# ---------------------------------------------------------------------------
def test_coo_to_csr() raises:
    var coo = COOMatrix[DType.float64](3, 3)
    coo.append(0, 0, Float64(1.0))
    coo.append(0, 2, Float64(2.0))
    coo.append(1, 1, Float64(3.0))
    coo.append(2, 0, Float64(4.0))
    coo.append(2, 2, Float64(5.0))

    var csr = coo.to_csr()

    assert_equal(csr.nnz(), 5)
    assert_equal(csr.nrows, 3)
    assert_equal(csr.ncols, 3)

    # indptr
    assert_equal(csr.indptr[0], 0)
    assert_equal(csr.indptr[1], 2)
    assert_equal(csr.indptr[2], 3)
    assert_equal(csr.indptr[3], 5)

    # column indices (sorted within each row)
    assert_equal(csr.indices[0], 0)
    assert_equal(csr.indices[1], 2)
    assert_equal(csr.indices[2], 1)
    assert_equal(csr.indices[3], 0)
    assert_equal(csr.indices[4], 2)

    # values
    assert_float_close(csr.data[0], 1.0)
    assert_float_close(csr.data[1], 2.0)
    assert_float_close(csr.data[2], 3.0)
    assert_float_close(csr.data[3], 4.0)
    assert_float_close(csr.data[4], 5.0)


# ---------------------------------------------------------------------------
# DiagMatrix.diag_vec_mul
# diag = [2.0, 3.0, 4.0], x = [1.0, 2.0, 3.0]
# expected = [2.0, 6.0, 12.0]
# ---------------------------------------------------------------------------
def test_diag_vec_mul() raises:
    var D = DiagMatrix[DType.float64](3)
    D.diag[0] = Float64(2.0)
    D.diag[1] = Float64(3.0)
    D.diag[2] = Float64(4.0)

    var x: List[Scalar[DType.float64]] = [1.0, 2.0, 3.0]
    var y = D.diag_vec_mul(x)

    assert_equal(len(y), 3)
    assert_float_close(y[0], 2.0)
    assert_float_close(y[1], 6.0)
    assert_float_close(y[2], 12.0)


# ---------------------------------------------------------------------------
# kron — 2×2 ⊗ 2×2
# A = [[1, 2], [3, 4]]
# B = [[0, 5], [6, 7]]
# kron(A,B) =
#   [[0, 5,  0, 10],
#    [6, 7, 12, 14],
#    [0,15,  0, 20],
#    [18,21,24, 28]]
# ---------------------------------------------------------------------------
def test_kron() raises:
    var dense_A: List[List[Scalar[DType.float64]]] = []
    var a0: List[Scalar[DType.float64]] = [1.0, 2.0]
    var a1: List[Scalar[DType.float64]] = [3.0, 4.0]
    dense_A.append(a0^)
    dense_A.append(a1^)
    var A = CSRMatrix[DType.float64].from_dense(dense_A)

    var dense_B: List[List[Scalar[DType.float64]]] = []
    var b0: List[Scalar[DType.float64]] = [0.0, 5.0]
    var b1: List[Scalar[DType.float64]] = [6.0, 7.0]
    dense_B.append(b0^)
    dense_B.append(b1^)
    var B = CSRMatrix[DType.float64].from_dense(dense_B)

    var C = kron(A, B)
    var result = C.to_dense()

    assert_equal(len(result), 4)
    assert_equal(len(result[0]), 4)

    # Row 0: [0, 5, 0, 10]
    assert_float_close(result[0][0], 0.0)
    assert_float_close(result[0][1], 5.0)
    assert_float_close(result[0][2], 0.0)
    assert_float_close(result[0][3], 10.0)

    # Row 1: [6, 7, 12, 14]
    assert_float_close(result[1][0], 6.0)
    assert_float_close(result[1][1], 7.0)
    assert_float_close(result[1][2], 12.0)
    assert_float_close(result[1][3], 14.0)

    # Row 2: [0, 15, 0, 20]
    assert_float_close(result[2][0], 0.0)
    assert_float_close(result[2][1], 15.0)
    assert_float_close(result[2][2], 0.0)
    assert_float_close(result[2][3], 20.0)

    # Row 3: [18, 21, 24, 28]
    assert_float_close(result[3][0], 18.0)
    assert_float_close(result[3][1], 21.0)
    assert_float_close(result[3][2], 24.0)
    assert_float_close(result[3][3], 28.0)


# ---------------------------------------------------------------------------
# spgemm — sparse × sparse matrix product
# A = [[1, 2], [3, 4]]
# B = [[5, 6], [7, 8]]
# C = A@B = [[19, 22], [43, 50]]
# ---------------------------------------------------------------------------
def test_spgemm() raises:
    var dense_A: List[List[Scalar[DType.float64]]] = []
    var a0: List[Scalar[DType.float64]] = [1.0, 2.0]
    var a1: List[Scalar[DType.float64]] = [3.0, 4.0]
    dense_A.append(a0^)
    dense_A.append(a1^)
    var A = CSRMatrix[DType.float64].from_dense(dense_A)

    var dense_B: List[List[Scalar[DType.float64]]] = []
    var b0: List[Scalar[DType.float64]] = [5.0, 6.0]
    var b1: List[Scalar[DType.float64]] = [7.0, 8.0]
    dense_B.append(b0^)
    dense_B.append(b1^)
    var B = CSRMatrix[DType.float64].from_dense(dense_B)

    var C = spgemm(A, B)
    var result = C.to_dense()

    assert_equal(len(result), 2)
    assert_equal(len(result[0]), 2)

    assert_float_close(result[0][0], 19.0)
    assert_float_close(result[0][1], 22.0)
    assert_float_close(result[1][0], 43.0)
    assert_float_close(result[1][1], 50.0)


# ---------------------------------------------------------------------------
# spmm — sparse × dense matrix product
# A (sparse) = [[1, 0], [0, 2]]
# D (dense)  = [[3, 4], [5, 6]]
# result     = [[3, 4], [10, 12]]
# ---------------------------------------------------------------------------
def test_spmm() raises:
    var dense_A: List[List[Scalar[DType.float64]]] = []
    var a0: List[Scalar[DType.float64]] = [1.0, 0.0]
    var a1: List[Scalar[DType.float64]] = [0.0, 2.0]
    dense_A.append(a0^)
    dense_A.append(a1^)
    var A = CSRMatrix[DType.float64].from_dense(dense_A)

    var D: List[List[Scalar[DType.float64]]] = []
    var d0: List[Scalar[DType.float64]] = [3.0, 4.0]
    var d1: List[Scalar[DType.float64]] = [5.0, 6.0]
    D.append(d0^)
    D.append(d1^)

    var result = spmm(A, D)

    assert_equal(len(result), 2)
    assert_equal(len(result[0]), 2)

    assert_float_close(result[0][0], 3.0)
    assert_float_close(result[0][1], 4.0)
    assert_float_close(result[1][0], 10.0)
    assert_float_close(result[1][1], 12.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
