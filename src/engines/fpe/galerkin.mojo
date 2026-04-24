"""Galerkin assembler for FPE stiffness and mass matrices.

Strictly follows FPE_Solver_Final_Version.py logic:
  weights = kron(diag(sw), diag(vw))
  s_diag = kron(diag(s), eye(n_v))
  v_diag = kron(eye(n_s), diag(v))
  s_sq_diag = kron(diag(s**2), eye(n_v))
  s_v_diag = s_diag @ v_diag
  s_s_v_diag = s_sq_diag @ v_diag

  M = two_basis.T @ weights @ two_basis
  K = s_partial.T @ integ_sbw @ two_basis
    + s_partial.T @ integ_svw @ v_partial
    + s_partial.T @ integ_ssvw @ s_partial
    + v_partial.T @ integ_vbw @ two_basis
    + v_partial.T @ integ_vw @ v_partial
    + K_sv.T
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix, identity_csr
from sparse.ops import add, kron, scale, spgemm, sparse_transpose


struct GalerkinAssembler[B: Int]:
    def __init__(out self):
        pass

    def mass_matrix(self, domain: FPEDomain) -> CSRMatrix:
        var basis = domain.build_basis()
        var two_basis = basis.eval_tensor(domain.s_points, domain.v_points)
        var weights = domain.integ_weights()
        var two_basis_T = sparse_transpose(two_basis)
        return spgemm(spgemm(two_basis_T, weights), two_basis)

    def stiffness_matrix(
        self, domain: FPEDomain, params: HestonParams
    ) -> CSRMatrix:
        var basis = domain.build_basis()

        var two_basis = basis.eval_tensor(domain.s_points, domain.v_points)
        var s_partial = basis.partial_s(domain.s_points, domain.v_points)
        var v_partial = basis.partial_v(domain.s_points, domain.v_points)

        var n_s = len(domain.s_points)
        var n_v = len(domain.v_points)
        var s = domain.s_points_phys.copy()
        var v = domain.v_points_phys.copy()
        var j = domain.jacobian_factor()

        var r = params.r
        var kappa = params.kappa
        var theta = params.theta
        var eta = params.sigma
        var rho = params.rho

        var s_sq: List[Float64] = []
        for i in range(len(s)):
            s_sq.append(s[i] * s[i])

        var s_diag = kron(DiagMatrix(s.copy()).to_csr(), identity_csr(n_v))
        var v_diag = kron(identity_csr(n_s), DiagMatrix(v.copy()).to_csr())
        var s_sq_diag = kron(DiagMatrix(s_sq.copy()).to_csr(), identity_csr(n_v))
        var s_v_diag = spgemm(s_diag, v_diag)
        var s_s_v_diag = spgemm(s_sq_diag, v_diag)

        var weights = domain.integ_weights()

        var k1 = (-r + 0.5 * rho * eta) / j
        var k2 = 1.0 / j
        var k3 = 0.5 / (j * j)
        var k4 = 0.5 * rho * eta / j
        var k5 = 0.5 * eta * eta - kappa * theta
        var k6 = kappa + 0.5 * rho * eta
        var k8 = 0.5 * eta * eta

        var integ_ssvw = spgemm(scale(k3, s_s_v_diag), weights)
        var integ_svw = spgemm(scale(k4, s_v_diag), weights)
        var integ_sbw = spgemm(
            add(scale(k1, s_diag), scale(k2, s_v_diag)), weights
        )

        var s_partial_T = sparse_transpose(s_partial)
        var K_sb = spgemm(spgemm(s_partial_T, integ_sbw), two_basis)
        var K_sv = spgemm(spgemm(s_partial_T, integ_svw), v_partial)
        var K_ssv = spgemm(spgemm(s_partial_T, integ_ssvw), s_partial)

        var integ_vbw = add(
            scale(k5, weights), scale(k6, spgemm(v_diag, weights))
        )
        var integ_vw = spgemm(scale(k8, v_diag), weights)

        var v_partial_T = sparse_transpose(v_partial)
        var K_vb = spgemm(spgemm(v_partial_T, integ_vbw), two_basis)
        var K_vv = spgemm(spgemm(v_partial_T, integ_vw), v_partial)
        var K_vs = sparse_transpose(K_sv)

        var K = add(K_sb, K_vb)
        var K2 = add(K, K_ssv)
        var K3 = add(K2, K_vv)
        var K4 = add(K3, K_vs)
        var K5 = add(K4, K_sv)

        return K5^
