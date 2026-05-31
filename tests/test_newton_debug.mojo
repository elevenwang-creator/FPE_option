"""Debug test: trace Newton iteration and error scaling for a single step."""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from numerics.utils.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.ops import add, scale
from numerics.utils import pow_pos
from std.math import sqrt, abs, max, min


struct SimpleSystem(LinearODESystem):
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
    var y: List[Float64] = [1.0, 1.0, 1.0]

    var SQRT6: Float64 = 2.449489742783178
    var DD1: Float64 = (-13.0 - 7.0 * SQRT6) / 3.0
    var DD2: Float64 = (-13.0 + 7.0 * SQRT6) / 3.0
    var DD3: Float64 = -1.0 / 3.0
var U1: Float64 = 3.6378342527444957888e00
var ALPH: Float64 = 2.6810828736277523276e00
var BETA: Float64 = 3.0504301992474109895e00

var T11: Float64 = 9.4438762488975244724e-02
var T12: Float64 = -1.4125529502095421353e-01
var T13: Float64 = 3.0029194105147420657e-02
var T21: Float64 = 2.5021312296533332331e-01
var T22: Float64 = 2.0412935229379994273e-01
var T23: Float64 = -3.8294211275726192101e-01
var T31: Float64 = 1.0

var TI11: Float64 = 4.1787185915519042823e+00
var TI12: Float64 = 3.2768282076106236556e-01
var TI13: Float64 = 5.2337644549944950523e-01
var TI21: Float64 = -4.1787185915519042823e+00
var TI22: Float64 = -3.2768282076106236556e-01
var TI23: Float64 = 4.7662355450055043926e-01
var TI31: Float64 = 5.0287263494578682277e-01
var TI32: Float64 = -2.5719269498556052156e+00
var TI33: Float64 = 5.9603920482822492222e-01

    var rtol: Float64 = 1e-8
    var atol: Float64 = 1e-10
    var uround: Float64 = 1e-16
    var nit: Int = 7
    var fnewt = max(10.0 * uround / rtol, min(0.03, rtol ** 0.5))

    print("fnewt = " + String(fnewt))

    for h_idx in range(5):
        var h_vals: List[Float64] = [0.1, 0.01, 0.001, 0.0001, 1e-5]
        var h = h_vals[h_idx]

        var E1_h = add(scale(U1, M), scale(h, K))
        var E1_csc = csr_to_csc(E1_h)
        var lu_real = SparseLU(n)
        lu_real.factorize(E1_csc)

        var E2_h = add(scale(ALPH, M), scale(h, K))
        var n2 = 2 * n
        var E2_full = CSRMatrix(n2, n2, 4 * n)
        var dest = 0
        E2_full.indptr[0] = 0
        for i in range(n):
            E2_full.data[dest] = ALPH * M.data[i] + h * K.data[i]
            E2_full.indices[dest] = i
            dest += 1
            E2_full.data[dest] = -BETA * M.data[i]
            E2_full.indices[dest] = n + i
            dest += 1
            E2_full.indptr[i + 1] = dest
        for i in range(n):
            E2_full.data[dest] = BETA * M.data[i]
            E2_full.indices[dest] = i
            dest += 1
            E2_full.data[dest] = ALPH * M.data[i] + h * K.data[i]
            E2_full.indices[dest] = n + i
            dest += 1
            E2_full.indptr[n + i + 1] = dest

        var E2_csc = csr_to_csc(E2_full)
        var lu_complex = SparseLU(n2)
        lu_complex.factorize(E2_csc)

        var w = K.spmv(y)

        var Z1 = List[Float64](length=n, fill=0.0)
        var Z2 = List[Float64](length=n, fill=0.0)
        var Z3 = List[Float64](length=n, fill=0.0)
        var F1 = List[Float64](length=n, fill=0.0)
        var F2 = List[Float64](length=n, fill=0.0)
        var F3 = List[Float64](length=n, fill=0.0)

        var faccon: Float64 = 1.0
        var dynold: Float64 = 0.0
        var thqold: Float64 = 0.0
        var theta_loc: Float64 = 0.0

        print("\nh = " + String(h))

        for newt in range(nit):
            var KZ1 = K.spmv(Z1)
            var KZ2 = K.spmv(Z2)
            var KZ3 = K.spmv(Z3)

            var MF1 = M.spmv(F1)
            var MF2 = M.spmv(F2)
            var MF3 = M.spmv(F3)

            var rhs_real = List[Float64](length=n, fill=0.0)
            var rhs_complex = List[Float64](length=n2, fill=0.0)

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
                var s = atol + rtol * abs(y[k])
                dyno_sq += (dF1[k] / s) ** 2 + (dF2_dF3[k] / s) ** 2 + (
                    dF2_dF3[n + k] / s
                ) ** 2
            var dyno = sqrt(dyno_sq / Float64(3 * n))

            var conv_check = faccon * dyno
            print(
                "  newt=" + String(newt + 1) +
                " dyno=" + String(dyno) +
                " faccon=" + String(faccon) +
                " faccon*dyno=" + String(conv_check) +
                " fnewt=" + String(fnewt) +
                " converged=" + String(conv_check <= fnewt)
            )

            if newt >= 1:
                var thq = dyno / max(dynold, uround)
                if newt == 1:
                    theta_loc = thq
                else:
                    theta_loc = sqrt(thq * thqold)
                thqold = thq
                if theta_loc < 0.99:
                    faccon = theta_loc / (1.0 - theta_loc)
                else:
                    print("  DIVERGING theta=" + String(theta_loc))
                    break

            dynold = max(dyno, uround)

            for k in range(n):
                F1[k] = F1[k] + dF1[k]
                F2[k] = F2[k] + dF2_dF3[k]
                F3[k] = F3[k] + dF2_dF3[n + k]
                Z1[k] = T11 * F1[k] + T12 * F2[k] + T13 * F3[k]
                Z2[k] = T21 * F1[k] + T22 * F2[k] + T23 * F3[k]
                Z3[k] = T31 * F1[k] + F2[k]

            if conv_check <= fnewt:
                print("  CONVERGED at newt=" + String(newt + 1))
                break

        var CONT = List[Float64](length=n, fill=0.0)
        for k in range(n):
            CONT[k] = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]

        var M_CONT = M.spmv(CONT)
        var rhs_err = List[Float64](length=n, fill=0.0)
        for k in range(n):
            rhs_err[k] = M_CONT[k] - h * w[k]

        var error = lu_real.solve(rhs_err)

        var err_norm_sq = 0.0
        for k in range(n):
            var sc = atol + rtol * abs(y[k])
            var ratio = error[k] / sc
            err_norm_sq += ratio * ratio
        var err_norm = sqrt(err_norm_sq / Float64(n))

        print("  err_norm = " + String(err_norm))
        print("  Z1 = [" + String(Z1[0]) + ", " + String(Z1[1]) + ", " + String(Z1[2]) + "]")
        print("  Z2 = [" + String(Z2[0]) + ", " + String(Z2[1]) + ", " + String(Z2[2]) + "]")
        print("  Z3 = [" + String(Z3[0]) + ", " + String(Z3[1]) + ", " + String(Z3[2]) + "]")
        print("  CONT = [" + String(CONT[0]) + ", " + String(CONT[1]) + ", " + String(CONT[2]) + "]")
        print("  h*w = [" + String(h * w[0]) + ", " + String(h * w[1]) + ", " + String(h * w[2]) + "]")
        print("  rhs_err = [" + String(rhs_err[0]) + ", " + String(rhs_err[1]) + ", " + String(rhs_err[2]) + "]")
        print("  error = [" + String(error[0]) + ", " + String(error[1]) + ", " + String(error[2]) + "]")
        var exact_y1 = 1.0 - h + h * h / 2.0 - h * h * h / 6.0
        var y_new_0 = y[0] + Z3[0]
        print("  y_new[0] = " + String(y_new_0) + "  exact = " + String(exact_y1) + "  diff = " + String(y_new_0 - exact_y1))

    print("\n--- Error scaling check ---")
    print("If err ~ h^5, then err(0.01)/err(0.001) ~ 10^5 = 100000")
    print("If err ~ h^1, then err(0.01)/err(0.001) ~ 10")
