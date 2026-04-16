"""Debug test: manually trace one RADAU5 step for M=I, K=diag(1,2,3).

This isolates the error estimation issue by printing all intermediate values.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import csr_to_csc
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, copy_vec
from std.math import exp, sqrt, abs


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
    print("=" * 60)
    print("  RADAU5 Single Step Debug Test")
    print("=" * 60)

    var n = 3
    var M = make_identity_csr(n)
    var K_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = make_diag_csr(n, K_diag)
    var y: List[Float64] = [1.0, 1.0, 1.0]
    var h = 0.01

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

    print()
    print("Constants:")
    print("  TI_RS1 = " + String(TI_RS1))
    print("  TI_RS2 = " + String(TI_RS2))
    print("  TI_RS3 = " + String(TI_RS3))
    print("  DD1 = " + String(DD1))
    print("  DD2 = " + String(DD2))
    print("  DD3 = " + String(DD3))

    print()
    print("Step: h = " + String(h) + ", y = [1, 1, 1]")

    var w = K.spmv(y)
    print()
    print("w = K*y = [" + String(w[0]) + ", " + String(w[1]) + ", " + String(w[2]) + "]")

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

    print()
    print("rhs_real = h * TI_RS1 * (-K*y):")
    for k in range(n):
        print("  rhs_real[" + String(k) + "] = " + String(rhs_real[k]))

    var F1 = lu_real.solve(rhs_real)
    print()
    print("F1 (real system solution):")
    for k in range(n):
        print("  F1[" + String(k) + "] = " + String(F1[k]))

    var rhs_complex = zeros(2 * n)
    for k in range(n):
        rhs_complex[k] = h * TI_RS2 * (-w[k])
        rhs_complex[n + k] = h * TI_RS3 * (-w[k])

    print()
    print("rhs_complex:")
    for k in range(n):
        print("  real[" + String(k) + "] = " + String(rhs_complex[k]) +
              "  imag[" + String(k) + "] = " + String(rhs_complex[n + k]))

    var z_01 = lu_complex.solve(rhs_complex)
    var F2 = zeros(n)
    var F3 = zeros(n)
    for k in range(n):
        F2[k] = z_01[k]
        F3[k] = z_01[n + k]

    print()
    print("F2 (complex real part):")
    for k in range(n):
        print("  F2[" + String(k) + "] = " + String(F2[k]))
    print("F3 (complex imag part):")
    for k in range(n):
        print("  F3[" + String(k) + "] = " + String(F3[k]))

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

    print()
    print("Z reconstruction:")
    for k in range(n):
        print("  Z1[" + String(k) + "] = " + String(Z1[k]))
    for k in range(n):
        print("  Z2[" + String(k) + "] = " + String(Z2[k]))
    for k in range(n):
        print("  Z3[" + String(k) + "] = " + String(Z3[k]))

    var y_new = zeros(n)
    for k in range(n):
        y_new[k] = y[k] + Z3[k]

    print()
    print("y_new = y + Z3:")
    for k in range(n):
        var exact_k = exp(-K_diag[k] * h)
        print("  y_new[" + String(k) + "] = " + String(y_new[k]) +
              "  exact = " + String(exact_k) +
              "  err = " + String(abs_f64(y_new[k] - exact_k)))

    var CONT = zeros(n)
    for k in range(n):
        CONT[k] = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]

    print()
    print("Error estimation:")
    print("  CONT = DD1*Z1 + DD2*Z2 + DD3*Z3:")
    for k in range(n):
        print("  CONT[" + String(k) + "] = " + String(CONT[k]))

    var M_CONT = M.spmv(CONT)
    print("  M*CONT (M=I so same as CONT):")
    for k in range(n):
        print("  M_CONT[" + String(k) + "] = " + String(M_CONT[k]))

    var rhs_err = zeros(n)
    for k in range(n):
        rhs_err[k] = M_CONT[k] - h * w[k]

    print("  rhs_err = M*CONT - h*K*y:")
    for k in range(n):
        print("  rhs_err[" + String(k) + "] = " + String(rhs_err[k]))

    var error = lu_real.solve(rhs_err)
    print("  error = E1_h^{-1} * rhs_err:")
    for k in range(n):
        print("  error[" + String(k) + "] = " + String(error[k]))

    var rtol: Float64 = 1e-8
    var atol: Float64 = 1e-10
    var err_norm_sq = 0.0
    for k in range(n):
        var sc = atol + rtol * abs_f64(y[k])
        var ratio = error[k] / sc
        err_norm_sq += ratio * ratio
        print("  sc[" + String(k) + "] = " + String(sc) +
              "  ratio = " + String(ratio))

    var err_norm = sqrt(err_norm_sq / Float64(n))
    print()
    print("  err_norm = " + String(err_norm))
    print("  Step " + ("ACCEPTED" if err_norm < 1.0 else "REJECTED"))

    print()
    print("--- Comparison with simple error estimate ---")
    var e1: Float64 = -2.0 / 9.0
    var e2: Float64 = 4.0 / 9.0
    var e3: Float64 = -2.0 / 9.0
    var simple_err = zeros(n)
    for k in range(n):
        simple_err[k] = e1 * Z1[k] + e2 * Z2[k] + e3 * Z3[k]
    print("  Simple error (e1*Z1+e2*Z2+e3*Z3):")
    for k in range(n):
        print("  simple_err[" + String(k) + "] = " + String(simple_err[k]))

    var simple_err_norm_sq = 0.0
    for k in range(n):
        var sc = atol + rtol * abs_f64(y[k])
        var ratio = simple_err[k] / sc
        simple_err_norm_sq += ratio * ratio
    var simple_err_norm = sqrt(simple_err_norm_sq / Float64(n))
    print("  simple err_norm = " + String(simple_err_norm))

    print()
    print("--- Verify: what should the RADAU5 error estimate be? ---")
    var b1: Float64 = (16.0 - SQRT6) / 36.0
    var b2: Float64 = (16.0 + SQRT6) / 36.0
    var b3: Float64 = 1.0 / 9.0
    print("  Radau IIA weights: b1=" + String(b1) + " b2=" + String(b2) + " b3=" + String(b3))

    var a11: Float64 = (88.0 - 7.0 * SQRT6) / 360.0
    var a12: Float64 = (296.0 - 169.0 * SQRT6) / 1800.0
    var a13: Float64 = (-2.0 + 3.0 * SQRT6) / 225.0
    var a21: Float64 = (296.0 + 169.0 * SQRT6) / 1800.0
    var a22: Float64 = (88.0 + 7.0 * SQRT6) / 360.0
    var a23: Float64 = (-2.0 - 3.0 * SQRT6) / 225.0
    var a31: Float64 = b1
    var a32: Float64 = b2
    var a33: Float64 = b3

    print("  A matrix row 3 (stiffly accurate): a31=" + String(a31) + " a32=" + String(a32) + " a33=" + String(a33))

    print()
    print("  Z3 = " + String(Z3[0]) + " (should be h * b^T * k)")
    print("  h * (b1*k1 + b2*k2 + b3*k3) where k_i = f(y+Z_i)")

    print()
    print("  For linear system, k_i = -K*(y+Z_i)")
    var k1_val = -K_diag[0] * (y[0] + Z1[0])
    var k2_val = -K_diag[0] * (y[0] + Z2[0])
    var k3_val = -K_diag[0] * (y[0] + Z3[0])
    var h_bk = h * (b1 * k1_val + b2 * k2_val + b3 * k3_val)
    print("  h*(b1*k1+b2*k2+b3*k3) for component 0 = " + String(h_bk))
    print("  Z3[0] = " + String(Z3[0]))
    print("  Match: " + String(abs_f64(h_bk - Z3[0]) < 1e-14))

    print()
    print("=" * 60)
