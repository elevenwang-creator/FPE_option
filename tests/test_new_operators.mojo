"""Test new operator overloads: CSRMatrix.__rmul__, DiagMatrix operators."""

from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix, identity_csr


def assert_eq(actual: Float64, expected: Float64, label: String) raises:
    if abs(actual - expected) > 1e-10:
        print("FAIL:", label, "got", actual, "expected", expected)
        raise Error("assertion failed")
    print("PASS:", label)


def main() raises:
    # --- CSRMatrix.__rmul__ ---
    var A = CSRMatrix.from_dense([[1.0, 2.0], [0.0, 3.0]])
    var C = 2.5 * A
    assert_eq(C.get(0, 0), 2.5, "rmul: 2.5 * A[0,0]")
    assert_eq(C.get(0, 1), 5.0, "rmul: 2.5 * A[0,1]")
    assert_eq(C.get(1, 1), 7.5, "rmul: 2.5 * A[1,1]")
    assert_eq(C.get(1, 0), 0.0, "rmul: 2.5 * A[1,0]")

    # --- DiagMatrix.__add__ ---
    var d1 = DiagMatrix([1.0, 2.0, 3.0])
    var d2 = DiagMatrix([4.0, 5.0, 6.0])
    var d_sum = d1 + d2
    assert_eq(d_sum.values[0], 5.0, "DiagMatrix + [0]")
    assert_eq(d_sum.values[1], 7.0, "DiagMatrix + [1]")
    assert_eq(d_sum.values[2], 9.0, "DiagMatrix + [2]")

    # --- DiagMatrix.__matmul__ ---
    var d_prod = d1 @ d2
    assert_eq(d_prod.values[0], 4.0, "DiagMatrix @ [0]")
    assert_eq(d_prod.values[1], 10.0, "DiagMatrix @ [1]")
    assert_eq(d_prod.values[2], 18.0, "DiagMatrix @ [2]")

    # --- DiagMatrix.__rmul__ ---
    var d_scaled = 3.0 * d1
    assert_eq(d_scaled.values[0], 3.0, "3.0 * DiagMatrix [0]")
    assert_eq(d_scaled.values[1], 6.0, "3.0 * DiagMatrix [1]")
    assert_eq(d_scaled.values[2], 9.0, "3.0 * DiagMatrix [2]")

    # --- DiagMatrix operators return DiagMatrix, .to_csr() works ---
    var d_csr = (d1 + d2).to_csr()
    assert_eq(d_csr.get(0, 0), 5.0, "(d1+d2).to_csr()[0,0]")
    assert_eq(d_csr.get(2, 2), 9.0, "(d1+d2).to_csr()[2,2]")

    print("\nAll operator overload tests passed!")
