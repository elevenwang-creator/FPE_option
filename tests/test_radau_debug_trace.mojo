"""Debug test: trace RADAU5 step size evolution for M=I case.

This replicates the solver logic with debug prints to find why
M=I cases hit "max steps exceeded".
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
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
    var t0: Float64 = 0.0
    var t1: Float64 = 1.0

    var y = copy_vec(y0)
    var t = t0
    var h: Float64 = 0.01
    var posneg: Float64 = 1.0

    var n_steps: Int = 0
    var n_accepted: Int = 0
    var n_rejected: Int = 0
    var h_old: Float64 = h
    var error_norm_old: Float64 = 1.0e-4
    var reject = False
    var first = True

    var safety: Float64 = 0.9
    var fac1: Float64 = 0.2
    var fac2: Float64 = 8.0
    var uround: Float64 = 1e-16
    var nit: Int = 7
    var cfac = safety * Float64(1 + 2 * nit)

    var h_lu: Float64 = 0.0
    var lu_real = SparseLU(n)
    var lu_complex = SparseLU(2 * n)

    print("=== RADAU5 Step-by-Step Debug (M=I, K=diag(1,2,3)) ===")
    print("rtol=" + String(rtol) + " atol=" + String(atol))
    print()

    var max_debug_steps = 50

    while posneg * (t1 - t) > uround * max_f64(abs(t), abs(t1)):
        if n_steps + n_rejected > max_debug_steps:
            print("DEBUG: stopping after " + String(n_steps + n_rejected) + " attempts")
            break

        if posneg * (t + 1.01 * h - t1) > 0.0:
            h = t1 - t

        var h_abs = abs(h)
        if h_abs < 1e-14:
            print("Step size underflow at t=" + String(t))
            break

        if abs(h - h_lu) > 1e-15 * max_f64(abs(h), abs(h_lu)):
            var E1_h = add(scale(U1, M), scale(h, K))
            var E1_csc = csr_to_csc(E1_h)
            lu_real = SparseLU(n)
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
            lu_complex = SparseLU(n2)
            lu_complex.factorize(E2_csc)

            h_lu = h

        var w = K.spmv(y)

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

        var CONT = zeros(n)
        for k in range(n):
            CONT[k] = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]

        var M_CONT = M.spmv(CONT)
        var rhs_err = zeros(n)
        for k in range(n):
            rhs_err[k] = M_CONT[k] - h * w[k]

        var error = lu_real.solve(rhs_err)

        var err_norm_sq = 0.0
        for k in range(n):
            var sc = atol + rtol * abs_f64(y[k])
            var ratio = error[k] / sc
            err_norm_sq += ratio * ratio
        var err_norm = sqrt(err_norm_sq / Float64(n))
        if err_norm < 1e-10:
            err_norm = 1e-10

        var fac = min_f64(safety, cfac / Float64(1 + 2 * nit))
        var quot = max_f64(1.0 / fac2, min_f64(1.0 / fac1, err_norm ** 0.25 / fac))
        var h_new = h / quot

        var step_type = "REJECT"
        if err_norm < 1.0:
            step_type = "ACCEPT"

        print("step=" + String(n_steps) + " acc=" + String(n_accepted) +
              " rej=" + String(n_rejected) +
              " t=" + String(t) + " h=" + String(h) +
              " err=" + String(err_norm) +
              " h_new=" + String(h_new) +
              " first=" + String(first) + " reject=" + String(reject) +
              " " + step_type)

        if err_norm < 1.0:
            first = False
            n_accepted += 1
            t = t + h
            y = y_new^
            n_steps += 1

            if n_accepted > 1:
                var facgus = (h_old / h) * (err_norm ** 2 / error_norm_old) ** 0.25 / safety
                facgus = max_f64(1.0 / fac2, min_f64(1.0 / fac1, facgus))
                quot = max_f64(quot, facgus)
                h_new = h / quot

            h_old = h
            error_norm_old = max_f64(1e-2, err_norm)

            h_new = posneg * min_f64(abs(h_new), abs(t1 - t))
            if reject:
                h_new = posneg * min_f64(abs(h_new), abs(h))
            reject = False
            h = h_new
        else:
            reject = True
            if first:
                h = h * 0.1
            else:
                h = h_new
            if n_accepted >= 1:
                n_rejected += 1

    print()
    print("Final: t=" + String(t) + " n_steps=" + String(n_steps) +
          " n_accepted=" + String(n_accepted) + " n_rejected=" + String(n_rejected))
    for k in range(n):
        print("  y[" + String(k) + "] = " + String(y[k]))
