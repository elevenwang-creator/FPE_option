"""Debug test: Compare Fortran ESTRAD IJOB=1 vs current Mojo formula.

The error estimate scales as h^1 instead of h^4/h^5.
This test compares the Fortran-style formula (with 1/H factor)
against the current Mojo formula to find the discrepancy.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import csr_to_csc
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, max_f64, min_f64, copy_vec
from std.math import sqrt, abs, min, max


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


def main() raises:
    var n = 3
    var M = make_identity_csr(n)
    var K_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = make_diag_csr(n, K_diag)
    var y0: List[Float64] = [1.0, 1.0, 1.0]

    var SQRT6: Float64 = 2.449489742783178
    var U1: Float64 = 3.6378165692476072
    var ALPH: Float64 = 2.6753213032678365
    var BETA: Float64 = 3.0493570545593676

    var T11: Float64 = 9.1232394870892942792e-02
    var T12: Float64 = -1.4125529502095420843e-01
    var T13: Float64 = -3.0029194105147424492e-02
    var T21: Float64 = 2.4171793270710701896e-01
    var T22: Float64 = 2.0412935229379993199e-01
    var T23: Float64 = 3.8294211275726193779e-01
    var T31: Float64 = 9.6604818261509293619e-01

    var TI11: Float64 = 4.3255798900631553510e+00
    var TI12: Float64 = 3.3919925181580986954e-01
    var TI13: Float64 = 5.4177705399358748719e-01
    var TI21: Float64 = -4.1787185915519047273e+00
    var TI22: Float64 = -3.2768282076106238708e-01
    var TI23: Float64 = 4.7662355450055045196e-01
    var TI31: Float64 = -5.0287263494578687595e-01
    var TI32: Float64 = 2.5719269498556054292e+00
    var TI33: Float64 = -5.9603920482822492497e-01

    var TI_RS1: Float64 = TI11 + TI12 + TI13
    var TI_RS2: Float64 = TI21 + TI22 + TI23
    var TI_RS3: Float64 = TI31 + TI32 + TI33

    var DD1: Float64 = (-13.0 - 7.0 * SQRT6) / 3.0
    var DD2: Float64 = (-13.0 + 7.0 * SQRT6) / 3.0
    var DD3: Float64 = -1.0 / 3.0

    var rtol: Float64 = 1e-8
    var atol: Float64 = 1e-10

    var y = copy_vec(y0)

    print("=== Comparing Fortran ESTRAD IJOB=1 vs Mojo formula ===")
    print("DD1=" + String(DD1) + " DD2=" + String(DD2) + " DD3=" + String(DD3))
    print("TI_RS1=" + String(TI_RS1) + " TI_RS2=" + String(TI_RS2) + " TI_RS3=" + String(TI_RS3))
    print()

    for h_idx in range(5):
        var h_values: List[Float64] = [0.01, 0.001, 0.0001, 1e-5, 1e-6]
        var h = h_values[h_idx]

        var w = K.spmv(y)

        var E1_h = add(scale(U1, M), scale(h, K))
        var E1_csc = csr_to_csc(E1_h)
        var lu_real = SparseLU(n)
        lu_real.factorize(E1_csc)

        var E2_h = add(scale(ALPH, M), scale(h, K))
        var BETA_M = scale(BETA, M)
        var n2 = 2 * n
        var total_nnz = E2_h.nnz() * 2 + BETA_M.nnz() * 2
        var E2_full = CSRMatrix(n2, n2, total_nnz)
        var dest = 0
        E2_full.indptr[0] = 0
        for i in range(n):
            for p in range(E2_h.indptr[i], E2_h.indptr[i + 1]):
                E2_full.data[dest] = E2_h.data[p]
                E2_full.indices[dest] = E2_h.indices[p]
                dest += 1
            for p in range(BETA_M.indptr[i], BETA_M.indptr[i + 1]):
                E2_full.data[dest] = -BETA_M.data[p]
                E2_full.indices[dest] = n + BETA_M.indices[p]
                dest += 1
            E2_full.indptr[i + 1] = dest
        for i in range(n):
            for p in range(BETA_M.indptr[i], BETA_M.indptr[i + 1]):
                E2_full.data[dest] = BETA_M.data[p]
                E2_full.indices[dest] = BETA_M.indices[p]
                dest += 1
            for p in range(E2_h.indptr[i], E2_h.indptr[i + 1]):
                E2_full.data[dest] = E2_h.data[p]
                E2_full.indices[dest] = n + E2_h.indices[p]
                dest += 1
            E2_full.indptr[n + i + 1] = dest

        var E2_csc = csr_to_csc(E2_full)
        var lu_complex = SparseLU(n2)
        lu_complex.factorize(E2_csc)

        var rhs_real = zeros(n)
        for k in range(n):
            rhs_real[k] = h * TI_RS1 * (-w[k])

        var F1 = lu_real.solve(rhs_real)

        var rhs_complex = zeros(2 * n)
        for k in range(n):
            rhs_complex[k] = h * TI_RS2 * (-w[k])
            rhs_complex[n + k] = h * TI_RS3 * (-w[k])

        var z_01 = lu_complex.solve(rhs_complex)
        var F2 = zeros(n)
        var F3 = zeros(n)
        for k in range(n):
            F2[k] = z_01[k]
            F3[k] = z_01[n + k]

        var Z1 = zeros(n)
        var Z2 = zeros(n)
        var Z3 = zeros(n)
        for k in range(n):
            var f1 = F1[k]
            var f2 = F2[k]
            var f3 = F3[k]
            Z1[k] = T11 * f1 + T12 * f2 + T13 * f3
            Z2[k] = T21 * f1 + T22 * f2 + T23 * f3
            Z3[k] = T31 * f1 + f2

        var y_new = zeros(n)
        for k in range(n):
            y_new[k] = y[k] + Z3[k]

        print("--- h = " + String(h) + " ---")
        print("  Z1 = [" + String(Z1[0]) + ", " + String(Z1[1]) + ", " + String(Z1[2]) + "]")
        print("  Z2 = [" + String(Z2[0]) + ", " + String(Z2[1]) + ", " + String(Z2[2]) + "]")
        print("  Z3 = [" + String(Z3[0]) + ", " + String(Z3[1]) + ", " + String(Z3[2]) + "]")

        var DD_Z = zeros(n)
        for k in range(n):
            DD_Z[k] = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]
        print("  DD1*Z1+DD2*Z2+DD3*Z3 = [" + String(DD_Z[0]) + ", " + String(DD_Z[1]) + ", " + String(DD_Z[2]) + "]")

        var hKy = zeros(n)
        for k in range(n):
            hKy[k] = h * w[k]
        print("  h*K*y = [" + String(hKy[0]) + ", " + String(hKy[1]) + ", " + String(hKy[2]) + "]")

        var rhs_err_mojo = zeros(n)
        for k in range(n):
            rhs_err_mojo[k] = DD_Z[k] - hKy[k]
        print("  rhs_err (Mojo) = [" + String(rhs_err_mojo[0]) + ", " + String(rhs_err_mojo[1]) + ", " + String(rhs_err_mojo[2]) + "]")

        var error_mojo = lu_real.solve(rhs_err_mojo)
        print("  error (Mojo) = [" + String(error_mojo[0]) + ", " + String(error_mojo[1]) + ", " + String(error_mojo[2]) + "]")

        var F1_fortran = zeros(n)
        for k in range(n):
            F1_fortran[k] = (DD1 / h) * Z1[k] + (DD2 / h) * Z2[k] + (DD3 / h) * Z3[k]
        print("  F1_fortran = (DD/H)*Z = [" + String(F1_fortran[0]) + ", " + String(F1_fortran[1]) + ", " + String(F1_fortran[2]) + "]")

        var CONT_fortran = zeros(n)
        for k in range(n):
            CONT_fortran[k] = F1_fortran[k] + (-w[k])
        print("  f(x,y) = -K*y = [" + String(-w[0]) + ", " + String(-w[1]) + ", " + String(-w[2]) + "]")
        print("  CONT_fortran = F1 + f(x,y) = [" + String(CONT_fortran[0]) + ", " + String(CONT_fortran[1]) + ", " + String(CONT_fortran[2]) + "]")

        var E1_fortran = add(scale(U1 / h, M), scale(1.0, K))
        var E1_fortran_csc = csr_to_csc(E1_fortran)
        var lu_fortran = SparseLU(n)
        lu_fortran.factorize(E1_fortran_csc)
        var error_fortran = lu_fortran.solve(CONT_fortran)
        print("  error (Fortran) = [" + String(error_fortran[0]) + ", " + String(error_fortran[1]) + ", " + String(error_fortran[2]) + "]")

        var err_norm_mojo = 0.0
        var err_norm_fortran = 0.0
        for k in range(n):
            var sc = atol + rtol * abs_f64(y[k])
            err_norm_mojo += (error_mojo[k] / sc) ** 2
            err_norm_fortran += (error_fortran[k] / sc) ** 2
        err_norm_mojo = sqrt(err_norm_mojo / Float64(n))
        err_norm_fortran = sqrt(err_norm_fortran / Float64(n))

        print("  err_norm (Mojo) = " + String(err_norm_mojo))
        print("  err_norm (Fortran) = " + String(err_norm_fortran))
        print("  ratio Mojo/Fortran = " + String(err_norm_mojo / err_norm_fortran))

        var exact: List[Float64] = [exp(-1.0 * h), exp(-2.0 * h), exp(-3.0 * h)]
        var actual_err = 0.0
        for k in range(n):
            actual_err += ((y_new[k] - exact[k]) / (atol + rtol * abs_f64(y[k]))) ** 2
        actual_err = sqrt(actual_err / Float64(n))
        print("  actual error norm = " + String(actual_err))
        print()
