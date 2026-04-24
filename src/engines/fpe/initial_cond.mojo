"""Initial condition computation for FPE.

Galerkin projection: basis_matrix.T @ weights @ delta (matrix-vector)
Mass matrix: two_basis.T @ weights @ two_basis
Basis integral: basis_matrix.T @ weights @ ones(n)
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from numerics.optim.osqp import OSQPSolver
from sparse.diag import DiagMatrix
from sparse.ops import spgemm, sparse_transpose
from std.math import exp, pi, sqrt, abs


def _normalize_nonnegative(mut x: List[Float64]):
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = 0.0


def _delta_approx_flat(
    domain: FPEDomain, params: HestonParams, sigma0: Float64
) -> List[Float64]:
    var s0 = params.S0
    var v0 = params.V0
    var rho = 0.0
    var s_range_span = domain.s_max - domain.s_min
    var v_stdev = sigma0 / s_range_span if s_range_span > 0.0 else sigma0
    var s_sigma = sigma0
    var v_sigma = v_stdev
    if s_sigma <= 0.0:
        s_sigma = 1e-3
    if v_sigma <= 0.0:
        v_sigma = 1e-3

    var rho2 = rho * rho
    var one_minus_rho2 = 1.0 - rho2
    if one_minus_rho2 < 1e-10:
        one_minus_rho2 = 1e-10
    var norm = 1.0 / (2.0 * pi * s_sigma * v_sigma * sqrt(one_minus_rho2))
    var inv_2_omr2 = 0.5 / one_minus_rho2

    var n_s = len(domain.s_points_phys)
    var n_v = len(domain.v_points_phys)

    var result: List[Float64] = []
    for i in range(n_s):
        var s = domain.s_points_phys[i]
        var ds = (s - s0) / s_sigma
        var ds2 = ds * ds
        for j in range(n_v):
            if ds2 > 50.0:
                result.append(0.0)
                continue
            var v = domain.v_points_phys[j]
            var dv = (v - v0) / v_sigma
            var dv2 = dv * dv
            var z = ds2 - 2.0 * rho * ds * dv + dv2
            if z * inv_2_omr2 > 50.0:
                result.append(0.0)
            else:
                var val = norm * exp(-z * inv_2_omr2)
                if val < 1e-6:
                    val = 0.0
                result.append(val)
    return result^


struct InitialCondition[B: Int]:
    def __init__(out self):
        pass

    def compute(
        self,
        domain: FPEDomain,
        params: HestonParams,
        sigma0: Float64 = 0.1,
    ) raises -> List[Float64]:
        var basis = domain.build_basis()

        var two_basis = basis.eval_tensor(domain.s_points, domain.v_points)
        var weights = domain.integ_weights()

        var delta_flat = _delta_approx_flat(domain, params, sigma0)

        var basis_matrix_T = sparse_transpose(two_basis)
        var galerkin_matrix = spgemm(spgemm(basis_matrix_T, weights), two_basis)
        
        var w_delta = weights.spmv(delta_flat)
        var galerkin_projection = basis_matrix_T.spmv(w_delta)

        var Dinv_diag: List[Float64] = []
        for i in range(galerkin_matrix.nrows):
            var diag_val = 0.0
            for p in range(galerkin_matrix.indptr[i], galerkin_matrix.indptr[i + 1]):
                if galerkin_matrix.indices[p] == i:
                    diag_val = galerkin_matrix.data[p]
                    break
            if diag_val > 0.0:
                Dinv_diag.append(1.0 / sqrt(diag_val))
            else:
                Dinv_diag.append(1.0)

        var Dinv = DiagMatrix(Dinv_diag.copy()).to_csr()
        var galerkin_scaled = spgemm(spgemm(Dinv, galerkin_matrix), Dinv)
        var galerkin_proj_scaled = Dinv.spmv(galerkin_projection)

        var ones = List[Float64]()
        for _ in range(two_basis.nrows):
            ones.append(1.0)
        var m = basis_matrix_T.spmv(weights.spmv(ones))

        var osqp = OSQPSolver(
            max_iter=50000,
            eps_abs=1e-8,
            eps_rel=1e-8,
            rho=0.1,
            sigma=1e-6,
            lambda_reg=1e-6,
        )
        var c_result = osqp.solve_nnls_sparse(galerkin_scaled, galerkin_proj_scaled)

        var result = Dinv.spmv(c_result)

        _normalize_nonnegative(result)

        var m_dot_r = 0.0
        for i in range(len(result)):
            m_dot_r += m[i] * result[i]

        if m_dot_r > 0.0:
            for i in range(len(result)):
                result[i] = result[i] / m_dot_r

        return result^
