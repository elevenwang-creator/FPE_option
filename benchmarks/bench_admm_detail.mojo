"""Benchmark with ADMM iteration count and solve timing."""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from engines.fpe.galerkin import mass_from_cached
from engines.fpe.initial_cond import (
    _delta_from_cached, _diag_inv_sqrt_vec, _elem_mul,
)
from sparse.diag_scale import diag_scale
from sparse.kron_spmv import kron_T_spmv_dual, weights_spmv
from numerics.optim.osqp import OSQPSolver
from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix
from numerics.optim.osqp import _add_alpha_to_diagonal_inplace
from std.math import sqrt, abs, max
from std.sys import simd_width_of
from std.time import perf_counter_ns as now

comptime SIMD_W = simd_width_of[DType.float64]()


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1,
        T=0.1, S0=60.0, V0=0.1, S_min=50.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )
    var domain = FPEDomain(params, n_s=38, n_v=38)
    var cached = FPECachedBasis[3, 3](domain)
    var M = mass_from_cached[3, 3](cached)
    var n = M.ncols

    # Build same as OSQP
    var Mt = M.transpose()
    var MtM = Mt @ M
    var alpha = 1e-6 + 1e-6 + 0.1
    _add_alpha_to_diagonal_inplace(MtM, alpha)
    var KKT_csc = MtM.to_csc()

    var t0 = now()
    var lu = SparseLU(n)
    lu.factorize(KKT_csc)
    var t1 = now()
    print("LU factorize:", Float64(t1 - t0) / 1e9, "s")

    var q_list = Mt.spmv_new(_diag_inv_sqrt_vec(M))

    var q = FixedSizeVector(n)
    q.copy_from(q_list)
    var x = FixedSizeVector(n)
    var z = FixedSizeVector(n)
    var z_old = FixedSizeVector(n)
    var u = FixedSizeVector(n)
    var rhs = FixedSizeVector(n)
    var work = FixedSizeVector(n)

    var rho_val = 0.1
    var sigma_val = 1e-6
    var eps_abs_val = 1e-8
    var eps_rel_val = 1e-8

    var total_solve_time = 0.0
    var iter_count = 0

    for _ in range(50000):
        comptime width = SIMD_W
        var i = 0
        while i + width <= n:
            var sq = (q.ptr() + i).load[width=width]()
            var sz = (z.ptr() + i).load[width=width]()
            var su = (u.ptr() + i).load[width=width]()
            var sx = (x.ptr() + i).load[width=width]()
            var sr = sq + rho_val * (sz - su) + sigma_val * sx
            (rhs.ptr() + i).store[width=width](sr)
            i += width
        while i < n:
            rhs[i] = q[i] + rho_val * (z[i] - u[i]) + sigma_val * x[i]
            i += 1

        var ts0 = now()
        lu.solve_inplace(rhs, work)
        var ts1 = now()
        total_solve_time += Float64(ts1 - ts0) / 1e9

        x.copy_from_fixed(rhs)
        z_old.copy_from_fixed(z)

        var primal_res_sq = 0.0
        var dual_res_sq = 0.0
        var z_norm_sq = 0.0
        var x_norm_sq = 0.0
        var u_norm_sq = 0.0

        for j in range(n):
            var z_new = x[j] + u[j]
            if z_new < 0.0:
                z_new = 0.0
            z[j] = z_new
            u[j] = u[j] + x[j] - z_new
            var pv = x[j] - z_new
            primal_res_sq += pv * pv
            var dv = rho_val * (z_new - z_old[j])
            dual_res_sq += dv * dv
            z_norm_sq += z_new * z_new
            x_norm_sq += x[j] * x[j]
            u_norm_sq += u[j] * u[j]

        iter_count += 1

        var eps_prim = eps_abs_val + eps_rel_val * sqrt(max(z_norm_sq, x_norm_sq))
        var eps_dual = eps_abs_val + eps_rel_val * rho_val * sqrt(u_norm_sq)

        if sqrt(primal_res_sq) < eps_prim and sqrt(dual_res_sq) < eps_dual:
            break

    print("ADMM iterations:", iter_count)
    print("Total solve time:", total_solve_time, "s")
    print("Per solve:", total_solve_time / Float64(iter_count) * 1000.0, "ms")
