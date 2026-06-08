"""Test kron_spmv, kron_T_spmv, kron_T_spmv_dual, weights_spmv against known values."""

from sparse.csr import CSRMatrix
from sparse.kron_spmv import kron_spmv, kron_T_spmv, weights_spmv
from sparse.kron import kron


def assert_close(a: Float64, b: Float64, label: String, tol: Float64 = 1e-12) -> Bool:
    if abs(a - b) > tol:
        print("FAIL:", label, "expected", b, "got", a)
        return False
    return True


def main() raises:
    var all_pass = True

    # Build A = 2x3 CSR: [[1, 0, 2], [0, 3, 0]]
    var a_data: List[Float64] = [1.0, 2.0, 3.0]
    var a_indices: List[Int] = [0, 2, 1]
    var a_indptr: List[Int] = [0, 2, 3]
    var A = CSRMatrix(2, 3, 3, a_data^, a_indptr^, a_indices^)

    # Build B = 2x2 CSR: [[1, 2], [3, 0]]
    var b_data: List[Float64] = [1.0, 2.0, 3.0]
    var b_indices: List[Int] = [0, 1, 0]
    var b_indptr: List[Int] = [0, 2, 3]
    var B = CSRMatrix(2, 2, 3, b_data^, b_indptr^, b_indices^)

    # kron(A,B) is 4x6. x is 3*2=6.
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Test 1: kron_spmv vs kron().spmv_new
    var K = kron(A, B)
    var expected = K.spmv_new(x)
    var result = kron_spmv(A, B, x)

    if len(result) != len(expected):
        print("FAIL: kron_spmv length mismatch")
        all_pass = False
    else:
        for i in range(len(result)):
            if not assert_close(result[i], expected[i], "kron_spmv[" + String(i) + "]"):
                all_pass = False

    if all_pass:
        print("PASS: kron_spmv matches kron().spmv_new")

    # Test 2: kron_T_spmv vs kron(A^T, B^T).spmv_new
    var A_T = A.transpose()
    var B_T = B.transpose()
    var K_T = kron(A_T, B_T)
    var expected_T = K_T.spmv_new(x)
    var result_T = kron_T_spmv(A_T, B_T, x)

    if len(result_T) != len(expected_T):
        print("FAIL: kron_T_spmv length mismatch")
        all_pass = False
    else:
        for i in range(len(result_T)):
            if not assert_close(result_T[i], expected_T[i], "kron_T_spmv[" + String(i) + "]"):
                all_pass = False

    if all_pass:
        print("PASS: kron_T_spmv matches kron(A^T,B^T).spmv_new")

    # Test 3: weights_spmv
    var sw: List[Float64] = [2.0, 3.0]
    var vw: List[Float64] = [1.0, 4.0, 0.5]
    var vv: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var w_result = weights_spmv(sw, vw, vv)

    var w_pass = True
    for i in range(2):
        for j in range(3):
            var idx = i * 3 + j
            var exp_val = sw[i] * vw[j] * vv[idx]
            if not assert_close(w_result[idx], exp_val, "weights_spmv[" + String(idx) + "]"):
                w_pass = False

    if w_pass:
        print("PASS: weights_spmv elementwise correct")

    if all_pass and w_pass:
        print("\nAll kron_spmv tests passed!")
    else:
        print("\nSome tests FAILED")
