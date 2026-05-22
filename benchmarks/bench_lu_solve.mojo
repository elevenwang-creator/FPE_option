"""Benchmark SparseLU.solve_inplace with realistic KKT matrix."""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from engines.fpe.galerkin import mass_from_cached
from numerics.optim.osqp import OSQPSolver
from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csr import CSRMatrix
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
    var n = M.ncols

    # Build KKT matrix same way OSQP does
    var Mt = M.transpose()
    var MtM = Mt @ M

    from sparse.csc import CSCMatrix
    from numerics.optim.osqp import _add_alpha_to_diagonal_inplace

    var alpha = 1e-6 + 1e-6 + 0.1
    _add_alpha_to_diagonal_inplace(MtM, alpha)
    var KKT_csc = MtM.to_csc()

    # Time factorize
    var lu = SparseLU(n)
    var t0 = now()
    lu.factorize(KKT_csc)
    var t1 = now()
    var factorize_time = Float64(t1 - t0) / 1e9
    print("LU factorize:", factorize_time, "s")

    # Time solve_inplace (100 iterations)
    var b = FixedSizeVector(n)
    var work = FixedSizeVector(n)
    var rhs = FixedSizeVector(n)
    for i in range(n):
        b[i] = 1.0 / Float64(i + 1)

    var num_iters = 100
    t0 = now()
    for _ in range(num_iters):
        rhs.copy_from_fixed(b)
        lu.solve_inplace(rhs, work)
    t1 = now()
    var total_solve = Float64(t1 - t0) / 1e9
    var per_solve = total_solve / Float64(num_iters)
    print("Solve x", num_iters, ":", total_solve, "s")
    print("Per solve:", per_solve * 1000.0, "ms")

    # Estimated ADMM time
    print("\nEstimated ADMM (1370 iters):", per_solve * 1370.0, "s")
    print("Factorize + ADMM:", factorize_time + per_solve * 1370.0, "s")

    # Full OSQP for comparison
    var osqp = OSQPSolver(
        max_iter=50000, eps_abs=1e-8, eps_rel=1e-8,
        rho=0.1, sigma=1e-6, lambda_reg=1e-6,
    )
    var M2 = mass_from_cached[3, 3](cached)
    var b_list: List[Float64] = []
    for i in range(n):
        b_list.append(1.0 / Float64(i + 1))

    t0 = now()
    var _ = osqp.solve_nnls_sparse(M2, b_list)
    t1 = now()
    print("\nFull OSQP:", Float64(t1 - t0) / 1e9, "s")
