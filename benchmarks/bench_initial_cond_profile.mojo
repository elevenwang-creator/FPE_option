"""Profile each step of initial_condition_from_cached to find bottleneck."""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from engines.fpe.galerkin import mass_from_cached
from engines.fpe.initial_cond import (
    _delta_from_cached, _diag_inv_sqrt_vec, _elem_mul,
)
from sparse.diag_scale import diag_scale
from sparse.kron_spmv import kron_T_spmv_dual, weights_spmv, kron_T_spmv
from numerics.optim.osqp import OSQPSolver
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

    # Step 1: delta computation
    var t0 = now()
    var delta_flat = _delta_from_cached[3, 3](cached, params, 0.1)
    var t1 = now()
    print("1. delta:          ", Float64(t1 - t0) / 1e9, "s")

    # Step 2: ones vector + weights_spmv
    var n_quad = cached.n_s * cached.n_v
    var ones: List[Float64] = []
    for _ in range(n_quad):
        ones.append(1.0)

    t0 = now()
    var w_delta = weights_spmv(cached.s_weights, cached.v_weights, delta_flat)
    var w_ones = weights_spmv(cached.s_weights, cached.v_weights, ones)
    t1 = now()
    print("2. weights_spmv:   ", Float64(t1 - t0) / 1e9, "s")

    # Step 3: kron_T_spmv_dual
    t0 = now()
    var galerkin_projection: List[Float64] = []
    var m: List[Float64] = []
    kron_T_spmv_dual(
        cached.Bs_T, cached.Bv_T, w_delta, w_ones,
        galerkin_projection, m,
    )
    t1 = now()
    print("3. kron_T_spmv_dual:", Float64(t1 - t0) / 1e9, "s")

    # Step 4: Dinv computation
    t0 = now()
    var Dinv_diag = _diag_inv_sqrt_vec(M)
    t1 = now()
    print("4. diag_inv_sqrt:  ", Float64(t1 - t0) / 1e9, "s")

    # Step 5: diag_scale (CSR copy)
    t0 = now()
    var galerkin_scaled = diag_scale(M, Dinv_diag, Dinv_diag)
    t1 = now()
    print("5. diag_scale:     ", Float64(t1 - t0) / 1e9, "s")

    # Step 6: elem mul
    t0 = now()
    var galerkin_proj_scaled = _elem_mul(Dinv_diag, galerkin_projection)
    t1 = now()
    print("6. elem_mul:       ", Float64(t1 - t0) / 1e9, "s")

    # Step 7: OSQP solve
    var osqp = OSQPSolver(
        max_iter=50000, eps_abs=1e-8, eps_rel=1e-8,
        rho=0.1, sigma=1e-6, lambda_reg=1e-6,
    )
    t0 = now()
    var c_result = osqp.solve_nnls_sparse(galerkin_scaled, galerkin_proj_scaled)
    t1 = now()
    print("7. OSQP solve:     ", Float64(t1 - t0) / 1e9, "s")

    # Step 8: post-processing
    t0 = now()
    var _ = _elem_mul(Dinv_diag, c_result)
    t1 = now()
    print("8. elem_mul (D*c): ", Float64(t1 - t0) / 1e9, "s")

    print("\nM shape:", M.nrows, "x", M.ncols, "nnz:", M.nnz())
    print("n_basis:", len(galerkin_projection))
