from numerics.linalg import lu_solve, dense_matvec
from numerics.utils import abs_f64
from std.testing import assert_true, TestSuite


def _close(a: Float64, b: Float64, tol: Float64 = 1e-8) -> Bool:
    return abs_f64(a - b) < tol


def test_lu_solve_identity() raises:
    """lu_solve(I, b) should return b."""
    var A: List[List[Float64]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    var b: List[Float64] = [3.0, 5.0, 7.0]
    var x = lu_solve(A, b)
    assert_true(_close(x[0], 3.0))
    assert_true(_close(x[1], 5.0))
    assert_true(_close(x[2], 7.0))


def test_lu_solve_2x2() raises:
    """lu_solve on a 2x2 system."""
    var A: List[List[Float64]] = [
        [2.0, 1.0],
        [1.0, 3.0],
    ]
    var b: List[Float64] = [5.0, 7.0]
    var x = lu_solve(A, b)
    # 2x + y = 5, x + 3y = 7 => x=1.6, y=1.8
    assert_true(_close(x[0], 1.6))
    assert_true(_close(x[1], 1.8))


def test_dense_matvec() raises:
    """dense_matvec(A, x) should compute A @ x."""
    var A: List[List[Float64]] = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    var x: List[Float64] = [1.0, 2.0]
    var y = dense_matvec(A, x)
    assert_true(_close(y[0], 5.0))
    assert_true(_close(y[1], 11.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
