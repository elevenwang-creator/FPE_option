from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from numerics.optim.osqp import OSQP
from sparse.csr import CSRMatrix
from std.math import exp, pi


def _csr_to_dense(Phi: CSRMatrix[DType.float64]) -> List[List[Float64]]:
    var in_dense = Phi.to_dense()
    var out: List[List[Float64]] = []
    for i in range(Phi.nrows):
        var row: List[Float64] = []
        for j in range(Phi.ncols):
            row.append(in_dense[i][j])
        out.append(row^)
    return out^


def _normalize_nonnegative(mut x: List[Float64]):
    var sum_x = 0.0
    for i in range(len(x)):
        if x[i] < 0.0:
            x[i] = 0.0
        sum_x += x[i]

    if sum_x > 0.0:
        for i in range(len(x)):
            x[i] = x[i] / sum_x


def _bivariate_gaussian(domain: FPEDomain, params: HestonParams, sigma0: Float64) -> List[Float64]:
    var out: List[Float64] = []

    var s_width = params.S_max - params.S_min
    var v_width = params.V_max - params.V_min
    if s_width <= 0.0:
        s_width = 1.0
    if v_width <= 0.0:
        v_width = 1.0

    var s0 = (params.S0 - params.S_min) / s_width
    var v0 = (params.V0 - params.V_min) / v_width

    var s_sigma = sigma0 / s_width
    var v_sigma = 0.1 * sigma0 / v_width
    if s_sigma <= 0.0:
        s_sigma = 1e-3
    if v_sigma <= 0.0:
        v_sigma = 1e-3

    var norm = 1.0 / (2.0 * pi * s_sigma * v_sigma)
    var sum_val = 0.0
    for i in range(len(domain.s_points)):
        var s = domain.s_points[i]
        for j in range(len(domain.v_points)):
            var v = domain.v_points[j]
            var ds = (s - s0) / s_sigma
            var dv = (v - v0) / v_sigma
            var val = norm * exp(-0.5 * (ds * ds + dv * dv))
            out.append(val)
            sum_val += val

    if sum_val > 0.0:
        for k in range(len(out)):
            out[k] = out[k] / sum_val

    return out^


struct InitialCondition[B: Int]:
    def __init__(out self):
        pass

    def compute(
        self,
        domain: FPEDomain,
        params: HestonParams,
        sigma0: Float64 = 2.0,
    ) raises -> List[Float64]:
        var basis = domain.build_basis()
        var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
        var delta = _bivariate_gaussian(domain, params, sigma0)
        var A = _csr_to_dense(Phi)
        var osqp = OSQP(max_iter=5000, tol=1e-8)
        var c = osqp.solve_nnls(A, delta)
        _normalize_nonnegative(c)
        return c^
