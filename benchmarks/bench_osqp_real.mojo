"""Profile OSQP.solve_nnls_sparse internals with REAL inputs."""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from engines.fpe.galerkin import mass_from_cached
from engines.fpe.initial_cond import (
    _delta_from_cached, _diag_inv_sqrt_vec, _elem_mul,
)
from sparse.diag_scale import diag_scale
from sparse.kron_spmv import kron_T_spmv_dual, weights_spmv
from sparse.csc import CSCMatrix
from numerics.utils.sparse_lu import SparseLU
from numerics.optim.osqp import _add_alpha_to_diagonal_inplace, OSQPSolver
from std.time import perf_counter_ns as now


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1,
        T=0.1, S0=60.0, V0=0.1, S_min=50.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )
    var domain = FPEDomain(params, n_s=38, n_v=38)
    var cached = FPECachedBasis[3, 3](domain)
    var M = mass_from_cached[3, 3](cached)
    var Dinv_diag = _diag_inv_sqrt_vec(M)
    var galerkin_scaled = diag_scale(M, Dinv_diag, Dinv_diag)

    var delta_flat = _delta_from_cached[3, 3](cached, params, 0.1)
    var n_quad = cached.n_s * cached.n_v
    var ones: List[Float64] = []
    for _ in range(n_quad):
        ones.append(1.0)
    var w_delta = weights_spmv(cached.s_weights, cached.v_weights, delta_flat)
    var w_ones = weights_spmv(cached.s_weights, cached.v_weights, ones)
    var galerkin_projection: List[Float64] = []
    var m_vec: List[Float64] = []
    kron_T_spmv_dual(
        cached.Bs_T, cached.Bv_T, w_delta, w_ones,
        galerkin_projection, m_vec,
    )
    var galerkin_proj_scaled = _elem_mul(Dinv_diag, galerkin_projection)

    var n = galerkin_scaled.ncols
    var rho_val = 0.1
    var sigma_val = 1e-6
    var lambda_reg = 1e-6

    # 1. M.transpose()
    var t0 = now()
    var Mt = galerkin_scaled.transpose()
    var t1 = now()
    print(" 1. M.transpose():  ", Float64(t1 - t0) / 1e9, "s")

    # 2. Mt @ M (spgemm)
    t0 = now()
    var MtM = Mt @ galerkin_scaled
    t1 = now()
    print(" 2. Mt @ M:         ", Float64(t1 - t0) / 1e9, "s")

    # 3. Diagonal shift
    t0 = now()
    var alpha = lambda_reg + sigma_val + rho_val
    _add_alpha_to_diagonal_inplace(MtM, alpha)
    t1 = now()
    print(" 3. diagonal shift: ", Float64(t1 - t0) / 1e9, "s")

    # 4. CSR->CSC
    t0 = now()
    var KKT_csc = MtM.to_csc()
    t1 = now()
    print(" 4. to_csc:         ", Float64(t1 - t0) / 1e9, "s")

    # 5. SparseLU factorize
    t0 = now()
    var lu = SparseLU(n)
    lu.factorize(KKT_csc)
    t1 = now()
    print(" 5. LU factorize:   ", Float64(t1 - t0) / 1e9, "s")

    # 6. Mt.spmv
    t0 = now()
    var q_list = Mt.spmv_new(galerkin_proj_scaled)
    t1 = now()
    print(" 6. Mt.spmv:        ", Float64(t1 - t0) / 1e9, "s")

    # 7. ADMM via OSQPSolver (use real OSQP to measure)
    var osqp = OSQPSolver(
        max_iter=50000, eps_abs=1e-8, eps_rel=1e-8,
        rho=0.1, sigma=1e-6, lambda_reg=1e-6,
    )
    t0 = now()
    var c_result = osqp.solve_nnls_sparse(galerkin_scaled, galerkin_proj_scaled)
    t1 = now()
    print(" 7. OSQP total:     ", Float64(t1 - t0) / 1e9, "s")

    # Time just the ADMM iterations (excluding setup)
    # Run a second time with already-factored lu
    from numerics.utils import FixedSizeVector
    var q = FixedSizeVector(n)
    q.copy_from(q_list)
    var x = FixedSizeVector(n)
    var z = FixedSizeVector(n)
    var z_old = FixedSizeVector(n)
    var u = FixedSizeVector(n)
    var rhs = FixedSizeVector(n)
    var work = FixedSizeVector(n)
    var eps_abs = 1e-8
    var eps_rel = 1e-8
    var max_iter = 50000
    from std.math import sqrt, max as fmax
    from std.sys import simd_width_of
    comptime width = simd_width_of[DType.float64]()

    t0 = now()
    var iters_used = 0
    for iter in range(max_iter):
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

        lu.solve_inplace(rhs, work)
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

        var eps_prim = eps_abs + eps_rel * sqrt(fmax(z_norm_sq, x_norm_sq))
        var eps_dual = eps_abs + eps_rel * rho_val * sqrt(u_norm_sq)
        iters_used = iter + 1

        if sqrt(primal_res_sq) < eps_prim and sqrt(dual_res_sq) < eps_dual:
            break

    t1 = now()
    print(" 8. ADMM loop only:", Float64(t1 - t0) / 1e9, "s iters:", iters_used)
