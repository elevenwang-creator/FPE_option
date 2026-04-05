from sparse.coo import COOMatrix
from sparse.diag import DiagMatrix
from sparse.csr import CSRMatrix
from std.testing import assert_true, TestSuite


def test_coo_append_and_to_csr() raises:
    """COOMatrix should correctly append entries and convert to CSR."""
    var coo = COOMatrix[DType.float64](3, 3)
    coo.append(0, 0, 1.0)
    coo.append(1, 1, 2.0)
    coo.append(2, 2, 3.0)

    var csr = coo.to_csr()
    assert_true(csr.nrows == 3)
    assert_true(csr.ncols == 3)
    assert_true(csr.nnz() == 3)


def test_coo_duplicate_summing() raises:
    """COOMatrix should sum duplicate entries during CSR conversion."""
    var coo = COOMatrix[DType.float64](2, 2)
    coo.append(0, 0, 1.0)
    coo.append(0, 0, 2.0)  # duplicate

    var csr = coo.to_csr()
    assert_true(csr.nnz() == 1, "duplicates should be summed")


def test_diag_matrix() raises:
    """DiagMatrix should create diagonal matrix and multiply."""
    var diag_vals: List[Float64] = [1.0, 2.0, 3.0]
    var dm = DiagMatrix[DType.float64](diag_vals^)

    var x: List[Float64] = [1.0, 1.0, 1.0]
    var y = dm.diag_vec_mul(x)
    assert_true(y[0] == 1.0)
    assert_true(y[1] == 2.0)
    assert_true(y[2] == 3.0)


def test_diag_to_csr() raises:
    """DiagMatrix should convert to CSR correctly."""
    var diag_vals: List[Float64] = [1.0, 2.0]
    var dm = DiagMatrix[DType.float64](diag_vals^)
    var csr = dm.to_csr()
    assert_true(csr.nrows == 2)
    assert_true(csr.ncols == 2)
    assert_true(csr.nnz() == 2)


def test_diag_inverse() raises:
    """DiagMatrix inverse should invert diagonal elements."""
    var diag_vals: List[Float64] = [2.0, 4.0]
    var dm = DiagMatrix[DType.float64](diag_vals^)
    var inv_dm = dm.inverse()
    var x: List[Float64] = [1.0, 1.0]
    var y = inv_dm.diag_vec_mul(x)
    assert_true(y[0] == 0.5)
    assert_true(y[1] == 0.25)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
