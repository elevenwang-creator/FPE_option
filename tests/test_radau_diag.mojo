"""Diagnostic test: single RADAU5 step with full intermediate output.

Compares with Fortran RADAU5 algorithm to find the O(h^1) error estimate bug.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import csr_to_csc
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, copy_vec
from std.math import sqrt, abs, exp


comptime SQRT6: Float64 = 2.449489742783178
comptime C1: Float64 = (4.0 - SQRT6) / 10.0
comptime C2: Float64 = (4.0 + SQRT6) / 10.0
comptime DD1: Float64 = (-13.0 - 7.0 * SQRT6) / 3.0
comptime DD2: Float64 = (-13.0 + 7.0 * SQRT6) / 3.0
comptime DD3: Float64 = -1.0 / 3.0
comptime U1: Float64 = 3.6378165692476072
comptime ALPH: Float64 = 2.6753213032678365
comptime BETA: Float64 = 3.0493570545593676

comptime T11: Float64 = 9.1232394870892942792e-02
comptime T12: Float64 = -1.4125529502095420843e-01
comptime T13: Float64 = -3.0029194105147424492e-02
comptime T21: Float64 = 2.4171793270710701896e-01
comptime T22: Float64 = 2.0412935229379993199e-01
comptime T23: Float64 = 3.8294211275726193779e-01
comptime T31: Float64 = 9.6604818261509293619e-01

comptime TI11: Float64 = 4.3255798900631553510e+00
comptime TI12: Float64 = 3.3919925181580986954e-01
comptime TI13: Float64 = 5.4177705399358748719e-01
comptime TI21: Float64 = -4.1787185915519047273e+00
comptime TI22: Float64 = -3.2768282076106238708e-01
comptime TI23: Float64 = 4.7662355450055045196e-01
comptime TI31: Float64 = -5.0287263494578687595e-01
comptime TI32: Float64 = 2.5719269498556054292e+00
comptime TI33: Float64 = -5.9603920482822492497e-01


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


def vec_norm(v: List[Float64], n: Int) -> Float64:
    var s = 0.0
    for i in range(n):
        s += v[i] * v[i]
    return sqrt(s)


def main() raises:
    print("=" * 70)
    print("  RADAU5 Single-Step Diagnostic")
    print("=" * 70)

    var n = 3
    var M = make_identity_csr(n)
    var K_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = make_diag_csr(n, K_diag)
    var y0: List[Float64] = [1.0, 1.0, 1.0]

    var h_values: List[Float64] = [0.1, 0.01, 0.001, 0.0001]

    print()
    print("System: M=I, K=diag(1,2,3), y'=-K*y, y0=[1,1,1]")
    print("Exact: y_i(h) = exp(-lambda_i * h)")
    print()

    for h_idx in range(len(h_values)):
        var h = h_values[h_idx]
        var y = copy_vec(y0)

        print("--- h = " + String(h) + " ---")

        var w = K.spmv(y)
        print("  w = K*y = [" + String(w[0]) + ", " + String(w[1]) + ", " + String(w[2]) + "]")

        var E1_h = add(scale(U1, M), scale(h, K))
        var E1_csc = csr_to_csc(E1_h)
        var lu_real = SparseLU(n)
        lu_real.factorize(E1_csc)

        var E2_h = add(scale(ALPH, M), scale(h, K))
        var n2 = 2 * n
        var E2_full = CSRMatrix(n2, n2, E2_h.nnz() * 2 + M.nnz() * 2)
        var dest = 0
        E2_full.indptr[0] = 0
        for i in range(n):
            for p in range(E2_h.indptr[i], E2_h.indptr[i + 1]):
                E2_full.data[dest] = E2_h.data[p]
                E2_full.indices[dest] = E2_h.indices[p]
                dest += 1
            for p in range(M.indptr[i], M.indptr[i + 1]):
                E2_full.data[dest] = -BETA * M.data[p]
                E2_full.indices[dest] = n + M.indices[p]
                dest += 1
            E2_full.indptr[i + 1] = dest
        for i in range(n):
            for p in range(M.indptr[i], M.indptr[i + 1]):
                E2_full.data[dest] = BETA * M.data[p]
                E2_full.indices[dest] = M.indices[p]
                dest += 1
            for p in range(E2_h.indptr[i], E2_h.indptr[i + 1]):
                E2_full.data[dest] = E2_h.data[p]
                E2_full.indices[dest] = n + E2_h.indices[p]
                dest += 1
            E2_full.indptr[n + i + 1] = dest

        var E2_csc = csr_to_csc(E2_full)
        var lu_complex = SparseLU(n2)
        lu_complex.factorize(E2_csc)

        var Z1 = zeros(n)
        var Z2 = zeros(n)
        var Z3 = zeros(n)
        var F1 = zeros(n)
        var F2 = zeros(n)
        var F3 = zeros(n)

        var nit_max: Int = 7
        var newt: Int = 0
        var converged = False

        for newt_iter in range(nit_max):
            newt = newt_iter + 1

            var KZ1 = K.spmv(Z1)
            var KZ2 = K.spmv(Z2)
            var KZ3 = K.spmv(Z3)

            var MF1 = M.spmv(F1)
            var MF2 = M.spmv(F2)
            var MF3 = M.spmv(F3)

            var rhs_real = zeros(n)
            var rhs_complex = zeros(2 * n)

            for k in range(n):
                var f1_k = -w[k] - KZ1[k]
                var f2_k = -w[k] - KZ2[k]
                var f3_k = -w[k] - KZ3[k]

                var W1_k = TI11 * f1_k + TI12 * f2_k + TI13 * f3_k
                var W2_k = TI21 * f1_k + TI22 * f2_k + TI23 * f3_k
                var W3_k = TI31 * f1_k + TI32 * f2_k + TI33 * f3_k

                rhs_real[k] = h * W1_k - U1 * MF1[k]
                rhs_complex[k] = h * W2_k - ALPH * MF2[k] + BETA * MF3[k]
                rhs_complex[n + k] = h * W3_k - ALPH * MF3[k] - BETA * MF2[k]

            var dF1 = lu_real.solve(rhs_real)
            var dF2_dF3 = lu_complex.solve(rhs_complex)

            var dyno_sq = 0.0
            for k in range(n):
                dyno_sq += dF1[k] * dF1[k] + dF2_dF3[k] * dF2_dF3[k] + dF2_dF3[n + k] * dF2_dF3[n + k]
            var dyno = sqrt(dyno_sq / Float64(3 * n))

            for k in range(n):
                F1[k] = F1[k] + dF1[k]
                F2[k] = F2[k] + dF2_dF3[k]
                F3[k] = F3[k] + dF2_dF3[n + k]
                Z1[k] = T11 * F1[k] + T12 * F2[k] + T13 * F3[k]
                Z2[k] = T21 * F1[k] + T22 * F2[k] + T23 * F3[k]
                Z3[k] = T31 * F1[k] + F2[k]

            if newt_iter == 0:
                print("  Newton iter 1: ||dF|| = " + String(dyno))
                print("    Z1 = [" + String(Z1[0]) + ", " + String(Z1[1]) + ", " + String(Z1[2]) + "]")
                print("    Z2 = [" + String(Z2[0]) + ", " + String(Z2[1]) + ", " + String(Z2[2]) + "]")
                print("    Z3 = [" + String(Z3[0]) + ", " + String(Z3[1]) + ", " + String(Z3[2]) + "]")

            if dyno < 1e-10:
                converged = True
                break

        print("  Newton converged in " + String(newt) + " iterations")
        print("  Z1 = [" + String(Z1[0]) + ", " + String(Z1[1]) + ", " + String(Z1[2]) + "]")
        print("  Z2 = [" + String(Z2[0]) + ", " + String(Z2[1]) + ", " + String(Z2[2]) + "]")
        print("  Z3 = [" + String(Z3[0]) + ", " + String(Z3[1]) + ", " + String(Z3[2]) + "]")

        var y_new = zeros(n)
        for k in range(n):
            y_new[k] = y[k] + Z3[k]

        var exact: List[Float64] = [exp(-1.0 * h), exp(-2.0 * h), exp(-3.0 * h)]
        var sol_err = 0.0
        for k in range(n):
            sol_err += (y_new[k] - exact[k]) ** 2
        sol_err = sqrt(sol_err / Float64(n))
        print("  y_new = [" + String(y_new[0]) + ", " + String(y_new[1]) + ", " + String(y_new[2]) + "]")
        print("  exact = [" + String(exact[0]) + ", " + String(exact[1]) + ", " + String(exact[2]) + "]")
        print("  ||y_new - exact|| = " + String(sol_err))

        var CONT = zeros(n)
        for k in range(n):
            CONT[k] = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]
        print("  CONT = DD1*Z1+DD2*Z2+DD3*Z3 = [" + String(CONT[0]) + ", " + String(CONT[1]) + ", " + String(CONT[2]) + "]")

        var M_CONT = M.spmv(CONT)
        print("  M*CONT = [" + String(M_CONT[0]) + ", " + String(M_CONT[1]) + ", " + String(M_CONT[2]) + "]")

        var rhs_err = zeros(n)
        for k in range(n):
            rhs_err[k] = M_CONT[k] - h * w[k]
        print("  h*w = [" + String(h * w[0]) + ", " + String(h * w[1]) + ", " + String(h * w[2]) + "]")
        print("  rhs_err = M*CONT - h*w = [" + String(rhs_err[0]) + ", " + String(rhs_err[1]) + ", " + String(rhs_err[2]) + "]")

        var error = lu_real.solve(rhs_err)
        print("  error = E1_h \\ rhs_err = [" + String(error[0]) + ", " + String(error[1]) + ", " + String(error[2]) + "]")

        var err_norm_sq = 0.0
        for k in range(n):
            var sc = 1e-8 + 1e-6 * abs_f64(y[k])
            var ratio = error[k] / sc
            err_norm_sq += ratio * ratio
        var err_norm = sqrt(err_norm_sq / Float64(n))
        print("  err_norm (rtol=1e-6, atol=1e-8) = " + String(err_norm))

        var err_raw = 0.0
        for k in range(n):
            err_raw += error[k] * error[k]
        err_raw = sqrt(err_raw / Float64(n))
        print("  ||error|| (raw) = " + String(err_raw))

        print()

    print("=" * 70)
    print("  Error scaling analysis")
    print("=" * 70)
    print()
    print("If error estimate is O(h^p), then ||error|| should scale as h^p")
    print("For RADAU5 order 5, the embedded error estimate should be O(h^3)")
    print("So ||error(h/10)|| / ||error(h)|| should be ~10^(-3) = 0.001")
