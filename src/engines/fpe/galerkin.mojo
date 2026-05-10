"""Galerkin assembler using Kronecker decomposition.

kron(A,B)^T @ kron(D1,D2) @ kron(C,E) = kron(A^T@D1@C, B^T@D2@E)
Replaces 5 large SpGEMMs (~16s) with small 1D SpGEMMs + kron (~0.01s).
"""

from engines.fpe.domain import FPECachedBasis
from engines.fpe.heston_params import HestonParams
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix
from sparse.diag_mul import diag_row_scale
from sparse.kron import kron
from sparse.scale import scale


def mass_from_cached[ds: Int, dv: Int](
    cached: FPECachedBasis[ds, dv]
) -> CSRMatrix:
    var Bs_T = cached.Bs_T.copy()
    var Bv_T = cached.Bv_T.copy()
    var Bs = cached.Bs.copy()
    var Bv = cached.Bv.copy()
    var sw_d = DiagMatrix(cached.s_weights.copy()).to_csr()
    var vw_d = DiagMatrix(cached.v_weights.copy()).to_csr()
    var Ms = Bs_T @ diag_row_scale(sw_d, Bs)
    var Mv = Bv_T @ diag_row_scale(vw_d, Bv)
    return kron(Ms, Mv)


def _scaled_vec(n: Int, alpha: Float64, x: List[Float64]) -> List[Float64]:
    var out: List[Float64] = []
    for i in range(n):
        out.append(alpha * x[i])
    return out^


def _mul_vecs(n: Int, x: List[Float64], y: List[Float64]) -> List[Float64]:
    var out: List[Float64] = []
    for i in range(n):
        out.append(x[i] * y[i])
    return out^


def stiffness_from_cached[ds: Int, dv: Int](
    cached: FPECachedBasis[ds, dv], params: HestonParams
) -> CSRMatrix:
    var Bs = cached.Bs.copy()
    var Bv = cached.Bv.copy()
    var dBs = cached.dBs.copy()
    var dBv = cached.dBv.copy()
    var Bs_T = cached.Bs_T.copy()
    var Bv_T = cached.Bv_T.copy()
    var dBs_T = cached.dBs_T.copy()
    var dBv_T = cached.dBv_T.copy()

    var sw = cached.s_weights.copy()
    var vw = cached.v_weights.copy()
    var s = cached.s_points_phys.copy()
    var v = cached.v_points_phys.copy()
    var j = cached.jacobian

    var r = params.r
    var kappa = params.kappa
    var theta = params.theta
    var eta = params.sigma
    var rho = params.rho

    var k1 = (-r + 0.5 * rho * eta) / j
    var k2 = 1.0 / j
    var k3 = 0.5 / (j * j)
    var k4 = 0.5 * rho * eta / j
    var k5 = 0.5 * eta * eta - kappa * theta
    var k6 = kappa + 0.5 * rho * eta
    var k8 = 0.5 * eta * eta

    var nq_s = len(sw)
    var nq_v = len(vw)

    var s_sw = _mul_vecs(nq_s, s, sw)
    var v_vw = _mul_vecs(nq_v, v, vw)
    var s_sq_sw: List[Float64] = []
    for i in range(nq_s):
        s_sq_sw.append(s[i] * s[i] * sw[i])

    var sw_d = DiagMatrix(sw.copy()).to_csr()
    var vw_d = DiagMatrix(vw.copy()).to_csr()
    var v_vw_d = DiagMatrix(v_vw.copy()).to_csr()

    var Ns = Bs_T @ diag_row_scale(sw_d, Bs)
    var Nv = Bv_T @ diag_row_scale(vw_d, Bv)

    var S1s = dBs_T @ diag_row_scale(
        DiagMatrix(_scaled_vec(nq_s, k1, s_sw)).to_csr(), Bs
    )
    var S2s = dBs_T @ diag_row_scale(
        DiagMatrix(_scaled_vec(nq_s, k2, s_sw)).to_csr(), Bs
    )
    var V1v = Nv^
    var V2v = Bv_T @ diag_row_scale(v_vw_d, Bv)

    var K_sb = kron(S1s, V1v) + kron(S2s, V2v)

    var S3s = dBs_T @ diag_row_scale(
        DiagMatrix(_scaled_vec(nq_s, k4, s_sw)).to_csr(), Bs
    )
    var V3v = Bv_T @ diag_row_scale(v_vw_d, dBv)
    var K_sv = kron(S3s, V3v)

    var S4s = dBs_T @ diag_row_scale(
        DiagMatrix(_scaled_vec(nq_s, k3, s_sq_sw)).to_csr(), dBs
    )
    var K_ssv = kron(S4s, V2v)

    var V4v_base = dBv_T @ diag_row_scale(
        DiagMatrix(vw.copy()).to_csr(), Bv
    )
    var V5v_base = dBv_T @ diag_row_scale(v_vw_d, Bv)
    var K_vb = kron(Ns, scale(k5, V4v_base) + scale(k6, V5v_base))

    var V6v = dBv_T @ diag_row_scale(v_vw_d, dBv)
    var K_vv = kron(scale(k8, Ns), V6v)

    var K_vs = K_sv.transpose()

    return (K_sb + K_vb + K_ssv + K_vv + K_vs + K_sv)
