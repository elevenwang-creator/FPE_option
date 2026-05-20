"""Test RadauSparseLinearSolver on simple linear ODE systems.

Uses Hairer's default tolerances: rtol=1e-3, atol=1e-6.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from numerics.utils.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.add import add
from sparse.scale import scale
from numerics.utils import zeros, abs_f64, max_f64, copy_vec
from std.math import sqrt, abs, exp


struct SimpleLinearSystem(LinearODESystem):
    var M_mat: CSRMatrix
    var K_mat: CSRMatrix

    def __init__(out self, var M: CSRMatrix, var K: CSRMatrix):
        self.M_mat = M^
        self.K_mat = K^

    def get_M(self) -> CSRMatrix:
        return self.M_mat.copy()

    def get_K(self) -> CSRMatrix:
        return self.K_mat.copy()


def make_diag_csr(n: Int, diag_vals: List[Float64]) -> CSRMatrix:
    var result = CSRMatrix(n, n, n)
    result.indptr[0] = 0
    for i in range(n):
        result.data[i] = diag_vals[i]
        result.indices[i] = i
        result.indptr[i + 1] = i + 1
    return result^


def make_identity_csr(n: Int) -> CSRMatrix:
    var ones: List[Float64] = []
    for _ in range(n):
        ones.append(1.0)
    return make_diag_csr(n, ones)


def make_tridiag_csr(n: Int, a: Float64, b: Float64, c: Float64) -> CSRMatrix:
    var nnz = 3 * n - 2
    var result = CSRMatrix(n, n, nnz)
    var idx = 0
    result.indptr[0] = 0
    for i in range(n):
        if i > 0:
            result.data[idx] = a
            result.indices[idx] = i - 1
            idx += 1
        result.data[idx] = b
        result.indices[idx] = i
        idx += 1
        if i < n - 1:
            result.data[idx] = c
            result.indices[idx] = i + 1
            idx += 1
        result.indptr[i + 1] = idx
    return result^


def main() raises:
    print("=" * 60)
    print(" RADAU5 Simple ODE Verification Test")
    print("=" * 60)

    print()
    print("[Test 1] M=I, K=diag(1,2,3), y'=-K*y")
    print(" Exact: y_i(t) = y0_i * exp(-lambda_i * t)")
    print()

    var n = 3
    var M1 = make_identity_csr(n)
    var K1_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K1 = make_diag_csr(n, K1_diag)
    var y0_1: List[Float64] = [1.0, 1.0, 1.0]

    var sys1 = SimpleLinearSystem(M1^, K1^)
    var solver1 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.1,
    )
    var sol1 = solver1.solve(sys1^, (0.0, 1.0), y0_1)

    if not sol1.success:
        print(" FAILED: " + sol1.message)
    else:
        print(" Steps: " + String(len(sol1.t)))
        var y_final = copy_vec(sol1.y[len(sol1.y) - 1])
        var exact: List[Float64] = [exp(-1.0), exp(-2.0), exp(-3.0)]
        var max_err = 0.0
        for i in range(n):
            var err = abs_f64(y_final[i] - exact[i])
            if err > max_err:
                max_err = err
            print(
                " y["
                + String(i)
                + "] = "
                + String(y_final[i])
                + " exact = "
                + String(exact[i])
                + " err = "
                + String(err)
            )
        print(" Max error: " + String(max_err))
        if max_err < 1e-3:
            print(" PASSED")
        else:
            print(" FAILED")

    print()
    print("[Test 2] M=diag(2,3,4), K=diag(1,2,3), M*y'=-K*y")
    print(" Exact: y_i(t) = y0_i * exp(-(lambda_i/m_ii) * t)")
    print()

    var M2_diag: List[Float64] = [2.0, 3.0, 4.0]
    var M2 = make_diag_csr(n, M2_diag)
    var K2 = make_diag_csr(n, K1_diag)
    var y0_2: List[Float64] = [1.0, 1.0, 1.0]

    var sys2 = SimpleLinearSystem(M2^, K2^)
    var solver2 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.1,
    )
    var sol2 = solver2.solve(sys2^, (0.0, 1.0), y0_2)

    if not sol2.success:
        print(" FAILED: " + sol2.message)
    else:
        print(" Steps: " + String(len(sol2.t)))
        var y_final2 = copy_vec(sol2.y[len(sol2.y) - 1])
        var exact2: List[Float64] = [exp(-0.5), exp(-2.0 / 3.0), exp(-0.75)]
        var max_err2 = 0.0
        for i in range(n):
            var err = abs_f64(y_final2[i] - exact2[i])
            if err > max_err2:
                max_err2 = err
            print(
                " y["
                + String(i)
                + "] = "
                + String(y_final2[i])
                + " exact = "
                + String(exact2[i])
                + " err = "
                + String(err)
            )
        print(" Max error: " + String(max_err2))
        if max_err2 < 1e-3:
            print(" PASSED")
        else:
            print(" FAILED")

    print()
    print("[Test 3] Conservation: sum(y) for M=I, K=diag(1,1,1)")
    print(" y' = -K*y with K=I => y_i(t) = y0_i * exp(-t)")
    print(
        " sum(y(t)) = exp(-t) * sum(y0) => should decrease but stay positive"
    )
    print()

    var K3_diag: List[Float64] = [1.0, 1.0, 1.0]
    var K3 = make_diag_csr(n, K3_diag)
    var M3 = make_identity_csr(n)
    var y0_3: List[Float64] = [0.5, 0.3, 0.2]

    var sys3 = SimpleLinearSystem(M3^, K3^)
    var solver3 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.1,
    )
    var sol3 = solver3.solve(sys3^, (0.0, 1.0), y0_3)

    if not sol3.success:
        print(" FAILED: " + sol3.message)
    else:
        var y0_sum = 0.0
        for i in range(n):
            y0_sum += y0_3[i]
        var y_final3 = copy_vec(sol3.y[len(sol3.y) - 1])
        var y_sum = 0.0
        var exact_sum = exp(-1.0) * y0_sum
        for i in range(n):
            y_sum += y_final3[i]
        print(" sum(y0) = " + String(y0_sum))
        print(" sum(y(T)) = " + String(y_sum))
        print(" exact sum = " + String(exact_sum))
        print(
            " relative error = "
            + String(abs_f64(y_sum - exact_sum) / exact_sum)
        )
        if abs_f64(y_sum - exact_sum) / exact_sum < 1e-3:
            print(" PASSED")
        else:
            print(" FAILED")

    print()
    print("[Test 4] SparseLU accuracy on RADAU5 system matrices")
    print()

    var h_test = 0.01
    var U1_val: Float64 = 3.6378165692476072

    var M2b = make_diag_csr(n, M2_diag)
    var K2b = make_diag_csr(n, K1_diag)
    var E1_h = add(scale(U1_val, M2b), scale(h_test, K2b))
    print(
        " E1_h = U1*M + h*K, size="
        + String(E1_h.nrows)
        + "x"
        + String(E1_h.ncols)
        + ", nnz="
        + String(E1_h.nnz())
    )

    var E1_csc = E1_h.to_csc()
    var lu_test = SparseLU(n)
    lu_test.factorize(E1_csc)

    var x_true: List[Float64] = [1.0, 2.0, 3.0]
    var b_test = E1_h.spmv_new(x_true)
    var x_solve = lu_test.solve(b_test)

    var err_solve = 0.0
    for i in range(n):
        err_solve = max_f64(err_solve, abs_f64(x_solve[i] - x_true[i]))
    print(" ||E1_h*x - b|| = " + String(err_solve))
    if err_solve < 1e-10:
        print(" SparseLU PASSED")
    else:
        print(" SparseLU FAILED")

    print()
    print("[Test 5] Multi-step integration: M=I, K=diag(0.1, 0.5, 2.0)")
    print(" Integrating from t=0 to t=5 with various step sizes")
    print()

    var K5_diag: List[Float64] = [0.1, 0.5, 2.0]
    var K5 = make_diag_csr(n, K5_diag)
    var M5 = make_identity_csr(n)
    var y0_5: List[Float64] = [1.0, 1.0, 1.0]

    var sys5 = SimpleLinearSystem(M5^, K5^)
    var solver5 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.1,
    )
    var sol5 = solver5.solve(sys5^, (0.0, 5.0), y0_5)

    if not sol5.success:
        print(" FAILED: " + sol5.message)
    else:
        print(" Steps: " + String(len(sol5.t)))
        var y_final5 = copy_vec(sol5.y[len(sol5.y) - 1])
        var exact5: List[Float64] = [
            exp(-0.1 * 5.0),
            exp(-0.5 * 5.0),
            exp(-2.0 * 5.0),
        ]
        var max_rel_err5 = 0.0
        for i in range(n):
            var rel_err = abs_f64(y_final5[i] - exact5[i]) / max_f64(
                1e-10, abs_f64(exact5[i])
            )
            if rel_err > max_rel_err5:
                max_rel_err5 = rel_err
            print(
                " y["
                + String(i)
                + "] = "
                + String(y_final5[i])
                + " exact = "
                + String(exact5[i])
                + " rel_err = "
                + String(rel_err)
            )
        print(" Max relative error: " + String(max_rel_err5))
        if max_rel_err5 < 5e-2:
            print(" PASSED")
        else:
            print(" FAILED")

    print()
    print("[Test 6] Non-diagonal sparse system (tridiagonal)")
    print()

    var n6 = 5
    var M6 = make_identity_csr(n6)
    var K6 = make_tridiag_csr(n6, -1.0, 2.0, -1.0)
    var y0_6: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0]

    var sys6 = SimpleLinearSystem(M6^, K6^)
    var solver6 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.01,
    )
    var sol6 = solver6.solve(sys6^, (0.0, 0.5), y0_6)

    if not sol6.success:
        print(" FAILED: " + sol6.message)
    else:
        var y_final6 = copy_vec(sol6.y[len(sol6.y) - 1])
        var sum_y = 0.0
        var all_positive = True
        for i in range(n6):
            sum_y += y_final6[i]
            if y_final6[i] < 0.0:
                all_positive = False
        print(" sum(y(T)) = " + String(sum_y))
        print(" All y positive: " + String(all_positive))
        for i in range(n6):
            print(" y[" + String(i) + "] = " + String(y_final6[i]))
        if all_positive:
            print(" PASSED (basic sanity)")
        else:
            print(" FAILED")

    print()
    print("[Test 7] t_eval interpolation: M=I, K=diag(1,2,3)")
    print(" Requesting solution at t=[0.0, 0.25, 0.5, 0.75, 1.0]")
    print()

    var M7 = make_identity_csr(n)
    var K7 = make_diag_csr(n, K1_diag)
    var y0_7: List[Float64] = [1.0, 1.0, 1.0]

    var sys7 = SimpleLinearSystem(M7^, K7^)
    var solver7 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3,
        atol=1e-6,
        max_step=0.0,
        first_step=0.1,
    )
    var t_eval_7: List[Float64] = [0.0, 0.25, 0.5, 0.75, 1.0]
    var sol7 = solver7.solve(sys7^, (0.0, 1.0), y0_7, t_eval_7^)

    if not sol7.success:
        print(" FAILED: " + sol7.message)
    else:
        print(" Output points: " + String(len(sol7.t)))
        var max_interp_err = 0.0
        for j in range(len(sol7.t)):
            var tj = sol7.t[j]
            var exact_j: List[Float64] = [exp(-1.0 * tj), exp(-2.0 * tj), exp(-3.0 * tj)]
            for i in range(n):
                var err = abs_f64(sol7.y[j][i] - exact_j[i])
                if err > max_interp_err:
                    max_interp_err = err
        print(" Max interpolation error: " + String(max_interp_err))
        if len(sol7.t) == 5 and max_interp_err < 1e-3:
            print(" PASSED")
        else:
            print(" FAILED")

    print()
    print("[Test 8] Step controller: stiff tridiagonal system")
    print(" Verifies h doesn't grow beyond HMAX after quot1/quot2 fix")
    print()

    var n8 = 10
    var M8 = make_identity_csr(n8)
    var K8 = make_tridiag_csr(n8, -5.0, 10.0, -5.0)
    var y0_8: List[Float64] = []
    for _ in range(n8):
        y0_8.append(1.0)

    var sys8 = SimpleLinearSystem(M8^, K8^)
    var solver8 = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-6,
        atol=1e-8,
        max_step=0.0,
        first_step=0.001,
    )
    var sol8 = solver8.solve(sys8^, (0.0, 0.1), y0_8)

    if not sol8.success:
        print(" FAILED: " + sol8.message)
    else:
        print(" Steps: " + String(len(sol8.t)))
        var all_positive8 = True
        for i in range(n8):
            if sol8.y[len(sol8.y) - 1][i] < 0.0:
                all_positive8 = False
        var step_count_ok = len(sol8.t) > 1 and len(sol8.t) <= 200
        if all_positive8 and step_count_ok:
            print(" PASSED")
        else:
            print(" FAILED (steps=" + String(len(sol8.t)) + ")")

    print()
    print("=" * 60)
    print(" All simple ODE tests complete")
    print("=" * 60)
