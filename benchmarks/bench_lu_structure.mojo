"""Check SparseLU structure sizes."""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from engines.fpe.galerkin import mass_from_cached
from engines.fpe.initial_cond import _diag_inv_sqrt_vec
from sparse.diag_scale import diag_scale
from sparse.csc import CSCMatrix
from numerics.utils.sparse_lu import SparseLU
from numerics.optim.osqp import _add_alpha_to_diagonal_inplace
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

    var Mt = galerkin_scaled.transpose()
    var MtM = Mt @ galerkin_scaled
    _add_alpha_to_diagonal_inplace(MtM, 0.1 + 1e-6 + 1e-6)
    var KKT_csc = MtM.to_csc()

    print("KKT nnz:", KKT_csc.nnz(), "of", KKT_csc.nrows * KKT_csc.ncols)
    print("KKT density:", Float64(KKT_csc.nnz()) / Float64(KKT_csc.nrows * KKT_csc.ncols))

    var lu = SparseLU(KKT_csc.nrows)
    var t0 = now()
    lu.factorize(KKT_csc)
    var t1 = now()
    print("Factorize time:", Float64(t1 - t0) / 1e9, "s")

    print("L nnz:", len(lu.Lx))
    print("U nnz:", len(lu.Ux))
    print("L fill:", Float64(len(lu.Lx)) / Float64(KKT_csc.nnz()))
    print("U fill:", Float64(len(lu.Ux)) / Float64(KKT_csc.nnz()))

    var avg_l_col = Float64(len(lu.Lx)) / Float64(KKT_csc.nrows)
    var avg_u_col = Float64(len(lu.Ux)) / Float64(KKT_csc.nrows)
    print("Avg L entries/col:", avg_l_col)
    print("Avg U entries/col:", avg_u_col)

    var max_l_col = 0
    var max_u_col = 0
    for k in range(KKT_csc.nrows):
        var l_nn = lu.Lp[k + 1] - lu.Lp[k]
        var u_nn = lu.Up[k + 1] - lu.Up[k]
        if l_nn > max_l_col:
            max_l_col = l_nn
        if u_nn > max_u_col:
            max_u_col = u_nn
    print("Max L entries/col:", max_l_col)
    print("Max U entries/col:", max_u_col)

    # Time solve_inplace
    from numerics.utils import FixedSizeVector
    var n = KKT_csc.nrows
    var b = FixedSizeVector(n)
    var work = FixedSizeVector(n)
    for i in range(n):
        b[i] = 1.0

    var iters = 1000
    t0 = now()
    for _ in range(iters):
        lu.solve_inplace(b, work)
        for i in range(n):
            b[i] = 1.0
    t1 = now()
    var per_solve = Float64(t1 - t0) / 1e9 / Float64(iters)
    print("solve_inplace avg:", per_solve * 1e6, "us")
