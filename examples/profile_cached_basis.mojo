"""Profile FPECachedBasis construction step by step."""

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain, FPECachedBasis
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix
from sparse.kron import kron
from std.time import perf_counter_ns as now


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1,
        T=0.6, S0=60.0, V0=0.1, S_min=50.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )

    var domain = FPEDomain[3, 3](params, n_s=38, n_v=38, num_insert=251)

    var t0 = now()
    var basis = domain.build_basis()
    var t1 = now()
    print("build_basis: " + String(Float64(t1 - t0) / 1e9) + "s")

    var t2 = now()
    var weights = domain.integ_weights()
    var t3 = now()
    print("integ_weights (kron of 2 diags): " + String(Float64(t3 - t2) / 1e9) + "s  nnz=" + String(weights.nnz()))

    var t4 = now()
    var Bs = basis.basis_s.eval_all(domain.s_points)
    var t5 = now()
    print("basis_s.eval_all (s_points): " + String(Float64(t5 - t4) / 1e9) + "s  " + String(Bs.nrows) + "x" + String(Bs.ncols) + " nnz=" + String(Bs.nnz()))

    var t6 = now()
    var Bv = basis.basis_v.eval_all(domain.v_points)
    var t7 = now()
    print("basis_v.eval_all (v_points): " + String(Float64(t7 - t6) / 1e9) + "s  " + String(Bv.nrows) + "x" + String(Bv.ncols) + " nnz=" + String(Bv.nnz()))

    var t8 = now()
    var dBs = basis.basis_s.first_derivative_all(domain.s_points)
    var t9 = now()
    print("basis_s.first_derivative_all: " + String(Float64(t9 - t8) / 1e9) + "s  " + String(dBs.nrows) + "x" + String(dBs.ncols) + " nnz=" + String(dBs.nnz()))

    var t10 = now()
    var dBv = basis.basis_v.first_derivative_all(domain.v_points)
    var t11 = now()
    print("basis_v.first_derivative_all: " + String(Float64(t11 - t10) / 1e9) + "s  " + String(dBv.nrows) + "x" + String(dBv.ncols) + " nnz=" + String(dBv.nnz()))

    var t12 = now()
    var Bs_T = Bs.transpose()
    var t13 = now()
    print("Bs.transpose: " + String(Float64(t13 - t12) / 1e9) + "s")

    var t14 = now()
    var Bv_T = Bv.transpose()
    var t15 = now()
    print("Bv.transpose: " + String(Float64(t15 - t14) / 1e9) + "s")

    var t16 = now()
    var dBs_T = dBs.transpose()
    var t17 = now()
    print("dBs.transpose: " + String(Float64(t17 - t16) / 1e9) + "s")

    var t18 = now()
    var dBv_T = dBv.transpose()
    var t19 = now()
    print("dBv.transpose: " + String(Float64(t19 - t18) / 1e9) + "s")

    var t20 = now()
    var two_basis = kron(Bs, Bv)
    var t21 = now()
    print("kron(Bs, Bv) = two_basis: " + String(Float64(t21 - t20) / 1e9) + "s  " + String(two_basis.nrows) + "x" + String(two_basis.ncols) + " nnz=" + String(two_basis.nnz()))

    var t22 = now()
    var s_partial = kron(dBs, Bv)
    var t23 = now()
    print("kron(dBs, Bv) = s_partial: " + String(Float64(t23 - t22) / 1e9) + "s  " + String(s_partial.nrows) + "x" + String(s_partial.ncols) + " nnz=" + String(s_partial.nnz()))

    var t24 = now()
    var v_partial = kron(Bs, dBv)
    var t25 = now()
    print("kron(Bs, dBv) = v_partial: " + String(Float64(t25 - t24) / 1e9) + "s  " + String(v_partial.nrows) + "x" + String(v_partial.ncols) + " nnz=" + String(v_partial.nnz()))

    var t26 = now()
    var two_basis_T = two_basis.transpose()
    var t27 = now()
    print("two_basis.transpose: " + String(Float64(t27 - t26) / 1e9) + "s")

    var t28 = now()
    var s_partial_T = s_partial.transpose()
    var t29 = now()
    print("s_partial.transpose: " + String(Float64(t29 - t28) / 1e9) + "s")

    var t30 = now()
    var v_partial_T = v_partial.transpose()
    var t31 = now()
    print("v_partial.transpose: " + String(Float64(t31 - t30) / 1e9) + "s")

    var total = Float64(t31 - t0) / 1e9
    print()
    print("Total step-by-step: " + String(total) + "s")

    var eval_total = Float64(t5 - t4 + t7 - t6 + t9 - t8 + t11 - t10) / 1e9
    var kron_total = Float64(t21 - t20 + t23 - t22 + t25 - t24) / 1e9
    var trans_total = Float64(t13 - t12 + t15 - t14 + t17 - t16 + t19 - t18 + t27 - t26 + t29 - t28 + t31 - t30) / 1e9
    print()
    print("eval_all total: " + String(eval_total) + "s")
    print("kron total: " + String(kron_total) + "s")
    print("transpose total: " + String(trans_total) + "s")
