"""Initial condition computation for FPE.

Galerkin projection: kron_T_spmv(Bs_T, Bv_T, weights * delta)
Mass matrix: M (passed in, computed via Kronecker decomposition)
Basis integral: kron_T_spmv(Bs_T, Bv_T, weights * ones)

Uses kron_T_spmv and weights_spmv instead of 2.1M-row kron matrices.
"""

from engines.fpe.domain import FPECachedBasis
from engines.fpe.heston_params import HestonParams
from numerics.optim.osqp import OSQPSolver
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix
from sparse.kron_spmv import kron_T_spmv, weights_spmv
from std.math import exp, pi, sqrt


def _normalize_nonnegative(mut x: List[Float64]):
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = 0.0


def _delta_from_cached[ds: Int, dv: Int](
    cached: FPECachedBasis[ds, dv], params: HestonParams, sigma0: Float64
) -> List[Float64]:
    var s0 = params.S0
    var v0 = params.V0
    var rho_val = 0.0
    var s_range_span = cached.jacobian
    var v_stdev = sigma0 / s_range_span if s_range_span > 0.0 else sigma0
    var s_sigma = sigma0
    var v_sigma = v_stdev
    if s_sigma <= 0.0:
        s_sigma = 1e-3
    if v_sigma <= 0.0:
        v_sigma = 1e-3
    var rho2 = rho_val * rho_val
    var one_minus_rho2 = 1.0 - rho2
    if one_minus_rho2 < 1e-10:
        one_minus_rho2 = 1e-10
    var norm = 1.0 / (2.0 * pi * s_sigma * v_sigma * sqrt(one_minus_rho2))
    var inv_2_omr2 = 0.5 / one_minus_rho2
    var n_s = cached.n_s
    var n_v = cached.n_v
    var delta_flat: List[Float64] = []
    for i in range(n_s):
        var s = cached.s_points_phys[i]
        var ds = (s - s0) / s_sigma
        var ds2 = ds * ds
        for j in range(n_v):
            if ds2 > 50.0:
                delta_flat.append(0.0)
                continue
            var v = cached.v_points_phys[j]
            var dv = (v - v0) / v_sigma
            var dv2 = dv * dv
            var z = ds2 - 2.0 * rho_val * ds * dv + dv2
            if z * inv_2_omr2 > 50.0:
                delta_flat.append(0.0)
            else:
                var val = norm * exp(-z * inv_2_omr2)
                if val < 1e-6:
                    val = 0.0
                delta_flat.append(val)
    return delta_flat^


def initial_condition_from_cached[ds: Int, dv: Int](
    cached: FPECachedBasis[ds, dv],
    params: HestonParams,
    M: CSRMatrix,
    sigma0: Float64 = 0.1,
) raises -> List[Float64]:
    var delta_flat = _delta_from_cached[ds, dv](cached, params, sigma0)
    var galerkin_matrix = M.copy()

    var w_delta = weights_spmv(cached.s_weights, cached.v_weights, delta_flat)
    var galerkin_projection = kron_T_spmv(
        cached.Bs_T, cached.Bv_T, w_delta
    )

    var Dinv_diag: List[Float64] = []
    for i in range(galerkin_matrix.nrows):
        var diag_val = 0.0
        for p in range(
            galerkin_matrix.indptr[i], galerkin_matrix.indptr[i + 1]
        ):
            if galerkin_matrix.indices[p] == i:
                diag_val = galerkin_matrix.data[p]
                break
        if diag_val > 0.0:
            Dinv_diag.append(1.0 / sqrt(diag_val))
        else:
            Dinv_diag.append(1.0)

    var Dinv = DiagMatrix(Dinv_diag.copy()).to_csr()
    var galerkin_scaled = (Dinv @ galerkin_matrix) @ Dinv
    var galerkin_proj_scaled = Dinv.spmv_new(galerkin_projection)

    var n_quad = cached.n_s * cached.n_v
    var ones = List[Float64](length=n_quad, fill=1.0)
    var w_ones = weights_spmv(cached.s_weights, cached.v_weights, ones)
    var m = kron_T_spmv(cached.Bs_T, cached.Bv_T, w_ones)

    var osqp = OSQPSolver(
        max_iter=50000,
        eps_abs=1e-8,
        eps_rel=1e-8,
        rho=0.1,
        sigma=1e-6,
        lambda_reg=1e-6,
    )
    var c_result = osqp.solve_nnls_sparse(
        galerkin_scaled, galerkin_proj_scaled
    )

    var result = Dinv.spmv_new(c_result)

    _normalize_nonnegative(result)

    var m_dot_r = 0.0
    for i in range(len(result)):
        m_dot_r += m[i] * result[i]

    if m_dot_r > 0.0:
        for i in range(len(result)):
            result[i] = result[i] / m_dot_r

    return result^
