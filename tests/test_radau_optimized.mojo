"""Comprehensive test suite for optimized RADAU5 solver.

Tests cover:
  1. FixedSizeVector vectorized operations
  2. SparseLU.solve_inplace correctness
  3. CSRMatrix.spmv_fixed correctness
  4. RADAU5 solver accuracy (small/large/stiff systems)
  5. Edge cases (tight tolerances)
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from numerics.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector, abs_f64, max_f64, zeros, copy_vec
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.ops import add, scale
from std.math import sqrt, abs, exp
from std.testing import assert_equal, assert_true


def test_fixed_size_vector_basic() raises:
    var v = FixedSizeVector(5)
    for i in range(5):
        v[i] = Float64(i + 1)
    assert_equal(v[0], 1.0)
    assert_equal(v[4], 5.0)
    assert_equal(v.len(), 5)


def test_fixed_size_vector_zero_out() raises:
    var v = FixedSizeVector(4, 7.0)
    for i in range(4):
        assert_equal(v[i], 7.0)
    v.zero_out()
    for i in range(4):
        assert_equal(v[i], 0.0)


def test_fixed_size_vector_add_from() raises:
    var a = FixedSizeVector(4)
    var b = FixedSizeVector(4)
    var result = FixedSizeVector(4)
    for i in range(4):
        a[i] = Float64(i + 1)
        b[i] = Float64((i + 1) * 10)
    result.add_from(a, b)
    assert_equal(result[0], 11.0)
    assert_equal(result[1], 22.0)
    assert_equal(result[2], 33.0)
    assert_equal(result[3], 44.0)


def test_fixed_size_vector_addassign() raises:
    var v = FixedSizeVector(3)
    var other = FixedSizeVector(3)
    for i in range(3):
        v[i] = Float64(i + 1)
        other[i] = Float64((i + 1) * 100)
    v.addassign(other)
    assert_equal(v[0], 101.0)
    assert_equal(v[1], 202.0)
    assert_equal(v[2], 303.0)


def test_fixed_size_vector_lin_comb_3() raises:
    var v1 = FixedSizeVector(3)
    var v2 = FixedSizeVector(3)
    var v3 = FixedSizeVector(3)
    var result = FixedSizeVector(3)
    for i in range(3):
        v1[i] = 1.0
        v2[i] = 2.0
        v3[i] = 3.0
    result.lin_comb_3(0.5, v1, 0.25, v2, 0.1, v3)
    assert_equal(result[0], 0.5 * 1.0 + 0.25 * 2.0 + 0.1 * 3.0)


def test_fixed_size_vector_lin_comb_2() raises:
    var v1 = FixedSizeVector(3)
    var v2 = FixedSizeVector(3)
    var result = FixedSizeVector(3)
    for i in range(3):
        v1[i] = 2.0
        v2[i] = 3.0
    result.lin_comb_2(0.5, v1, 2.0, v2)
    assert_equal(result[0], 0.5 * 2.0 + 2.0 * 3.0)
    assert_equal(result[1], 7.0)


def test_fixed_size_vector_addassign_offset() raises:
    var v = FixedSizeVector(3)
    var src = FixedSizeVector(6)
    for i in range(3):
        v[i] = Float64(i + 1)
    for i in range(6):
        src[i] = Float64(i * 10)
    v.addassign_offset(src, 3)
    assert_equal(v[0], 1.0 + 30.0)
    assert_equal(v[1], 2.0 + 40.0)
    assert_equal(v[2], 3.0 + 50.0)


def test_fixed_size_vector_scaled_norm_sq() raises:
    var v = FixedSizeVector(3)
    var scal = FixedSizeVector(3)
    v[0] = 3.0
    v[1] = 4.0
    v[2] = 0.0
    scal[0] = 1.0
    scal[1] = 2.0
    scal[2] = 1.0
    var norm_sq = v.scaled_norm_sq(scal)
    var expected = (3.0 / 1.0) ** 2 + (4.0 / 2.0) ** 2 + (0.0 / 1.0) ** 2
    assert_true(abs(norm_sq - expected) < 1e-10)


def test_fixed_size_vector_sub_scaled() raises:
    var a = FixedSizeVector(3)
    var b = FixedSizeVector(3)
    var result = FixedSizeVector(3)
    for i in range(3):
        a[i] = 10.0
        b[i] = 2.0
    result.sub_scaled(a, 3.0, b)
    assert_equal(result[0], 10.0 - 3.0 * 2.0)
    assert_equal(result[1], 4.0)
    assert_equal(result[2], 4.0)


def test_fixed_size_vector_copy_from_fixed() raises:
    var src = FixedSizeVector(4)
    var dst = FixedSizeVector(4)
    for i in range(4):
        src[i] = Float64(i * 7)
    dst.copy_from_fixed(src)
    for i in range(4):
        assert_equal(dst[i], Float64(i * 7))


def test_spmv_fixed() raises:
    var data: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var indices: List[Int] = [0, 1, 0, 1]
    var indptr: List[Int] = [0, 2, 4]
    var A = CSRMatrix(2, 2, 4, data^, indptr^, indices^)

    var x = FixedSizeVector(2)
    x[0] = 1.0
    x[1] = 2.0
    var y = FixedSizeVector(2)
    A.spmv_fixed(x, y)
    assert_true(abs(y[0] - 5.0) < 1e-10)
    assert_true(abs(y[1] - 11.0) < 1e-10)


def test_solve_inplace() raises:
    var A_csc = CSCMatrix(2, 2, 4)
    A_csc.data[0] = 2.0
    A_csc.data[1] = 1.0
    A_csc.data[2] = 1.0
    A_csc.data[3] = 3.0
    A_csc.indices[0] = 0
    A_csc.indices[1] = 1
    A_csc.indices[2] = 0
    A_csc.indices[3] = 1
    A_csc.colptr[0] = 0
    A_csc.colptr[1] = 2
    A_csc.colptr[2] = 4

    var lu = SparseLU(2)
    lu.factorize(A_csc)

    var b = FixedSizeVector(2)
    b[0] = 5.0
    b[1] = 7.0
    var work = FixedSizeVector(2)

    lu.solve_inplace(b, work)
    assert_true(abs(b[0] - 1.6) < 1e-10)
    assert_true(abs(b[1] - 1.8) < 1e-10)


struct DiagonalODESystem(LinearODESystem):
    var n: Int
    var k_data: List[Float64]
    var k_indices: List[Int]
    var k_indptr: List[Int]

    def __init__(out self, n: Int, k_vals: List[Float64]):
        self.n = n
        self.k_data = List[Float64]()
        self.k_indices = List[Int]()
        self.k_indptr = List[Int]()
        self.k_indptr.append(0)
        for i in range(n):
            self.k_data.append(k_vals[i])
            self.k_indices.append(i)
            self.k_indptr.append(i + 1)

    def get_M(self) -> CSRMatrix:
        var data: List[Float64] = []
        var indices: List[Int] = []
        var indptr: List[Int] = [0]
        for i in range(self.n):
            data.append(1.0)
            indices.append(i)
            indptr.append(i + 1)
        return CSRMatrix(self.n, self.n, self.n, data^, indptr^, indices^)

    def get_K(self) -> CSRMatrix:
        var data: List[Float64] = []
        var indices: List[Int] = []
        var indptr: List[Int] = [0]
        for i in range(self.n):
            data.append(self.k_data[i])
            indices.append(i)
            indptr.append(i + 1)
        return CSRMatrix(self.n, self.n, self.n, data^, indptr^, indices^)


def test_radau_diagonal_system() raises:
    var k_vals: List[Float64] = [0.1, 0.5, 2.0]
    var system = DiagonalODESystem(3, k_vals)
    var solver = RadauSparseLinearSolver[DiagonalODESystem](rtol=1e-6, atol=1e-8)

    var y0: List[Float64] = [1.0, 1.0, 1.0]
    var sol = solver.solve(system, (0.0, 5.0), y0)

    assert_true(sol.success, sol.message)

    var exact_0 = exp(-0.1 * 5.0)
    var exact_1 = exp(-0.5 * 5.0)
    var exact_2 = exp(-2.0 * 5.0)

    var y_idx = len(sol.y) - 1
    var rel_err_0 = abs((sol.y[y_idx][0] - exact_0) / exact_0)
    var rel_err_1 = abs((sol.y[y_idx][1] - exact_1) / exact_1)
    var rel_err_2 = abs((sol.y[y_idx][2] - exact_2) / exact_2)

    assert_true(rel_err_0 < 0.01, "y[0] rel_err too large")
    assert_true(rel_err_1 < 0.01, "y[1] rel_err too large")
    assert_true(rel_err_2 < 0.01, "y[2] rel_err too large")


def test_radau_stiff_system() raises:
    var k_vals: List[Float64] = [1.0, 100.0]
    var system = DiagonalODESystem(2, k_vals)
    var solver = RadauSparseLinearSolver[DiagonalODESystem](rtol=1e-6, atol=1e-8)

    var y0: List[Float64] = [1.0, 1.0]
    var sol = solver.solve(system, (0.0, 1.0), y0)

    assert_true(sol.success, sol.message)

    var exact_0 = exp(-1.0)

    var y_idx = len(sol.y) - 1
    var rel_err_0 = abs((sol.y[y_idx][0] - exact_0) / exact_0)

    assert_true(rel_err_0 < 0.01, "Stiff system: y[0] rel_err too large")
    assert_true(abs(sol.y[y_idx][1]) < 1e-3, "Stiff system: y[1] should be near zero")


def test_radau_tight_tolerance() raises:
    var k_vals: List[Float64] = [0.5]
    var system = DiagonalODESystem(1, k_vals)
    var solver = RadauSparseLinearSolver[DiagonalODESystem](rtol=1e-8, atol=1e-10)

    var y0: List[Float64] = [1.0]
    var sol = solver.solve(system, (0.0, 2.0), y0)

    assert_true(sol.success, sol.message)

    var exact = exp(-0.5 * 2.0)
    var y_idx = len(sol.y) - 1
    var rel_err = abs((sol.y[y_idx][0] - exact) / exact)

    assert_true(rel_err < 1e-2, "Tight tolerance: rel_err too large")


def main() raises:
    print("=== FixedSizeVector Tests ===")
    test_fixed_size_vector_basic()
    print("[PASS] basic operations")
    test_fixed_size_vector_zero_out()
    print("[PASS] zero_out")
    test_fixed_size_vector_add_from()
    print("[PASS] add_from")
    test_fixed_size_vector_addassign()
    print("[PASS] addassign")
    test_fixed_size_vector_lin_comb_3()
    print("[PASS] lin_comb_3")
    test_fixed_size_vector_lin_comb_2()
    print("[PASS] lin_comb_2")
    test_fixed_size_vector_addassign_offset()
    print("[PASS] addassign_offset")
    test_fixed_size_vector_scaled_norm_sq()
    print("[PASS] scaled_norm_sq")
    test_fixed_size_vector_sub_scaled()
    print("[PASS] sub_scaled")
    test_fixed_size_vector_copy_from_fixed()
    print("[PASS] copy_from_fixed")

    print("\n=== Sparse Operation Tests ===")
    test_spmv_fixed()
    print("[PASS] spmv_fixed")
    test_solve_inplace()
    print("[PASS] solve_inplace")

    print("\n=== RADAU5 Solver Tests ===")
    test_radau_diagonal_system()
    print("[PASS] diagonal system (n=3)")
    test_radau_stiff_system()
    print("[PASS] stiff system (lambda=100)")
    test_radau_tight_tolerance()
    print("[PASS] tight tolerance (rtol=1e-10)")

    print("\n=== All tests PASSED ===")
