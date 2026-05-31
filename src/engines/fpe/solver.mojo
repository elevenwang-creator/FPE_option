"""FPE solver with Kronecker Galerkin + RadauIIA time integration.

Key design:
- Uses sparse CSRMatrix spmv for ODE RHS (O(nnz) vs O(n^2))
- Uses proper RadauIIA solver (order 5 vs order 1 BackwardEuler)
- D-scaling (diagonal preconditioning) for well-conditioned ODE system
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.heston_params import HestonParams
from engines.fpe.initial_cond import initial_condition_from_cached
from numerics.ode.radau import RadauSparseLinearSolver
from numerics.ode.radau import LinearODESystem

from sparse.csr import CSRMatrix
from sparse.diag_scale import diag_scale
from std.math import sqrt


struct FPESparseLinearSystem(LinearODESystem):
    var M: CSRMatrix
    var K: CSRMatrix

    def __init__(out self, var M: CSRMatrix, var K: CSRMatrix):
        self.M = M^
        self.K = K^

    def get_M(self) -> CSRMatrix:
        return self.M.copy()

    def get_K(self) -> CSRMatrix:
        return self.K.copy()


@fieldwise_init
struct FPESolver[B: Int]:
    var rtol: Float64
    var atol: Float64
    var max_step: Float64
    var first_step: Float64

    def solve(
        self,
        domain: FPEDomain,
        params: HestonParams,
        t_eval: Optional[List[Float64]] = None,
    ) raises -> List[List[Float64]]:
        var cached = domain.cached_basis()
        var M = mass_from_cached(cached)
        var K = stiffness_from_cached(cached, params)
        var q0 = initial_condition_from_cached(cached, params, M.copy())
        var t_end = params.T

        return self._solve(M^, K^, q0, t_eval, t_end)

    def _solve(
        self,
        var M: CSRMatrix,
        var K: CSRMatrix,
        q0: List[Float64],
        t_eval: Optional[List[Float64]],
        t_end: Float64,
    ) raises -> List[List[Float64]]:
        var n = M.nrows
        var D = List[Float64](length=n, fill=0.0)

        var Dinv = List[Float64](length=n, fill=0.0)
        for i in range(n):
            var m_ii = 0.0
            for p in range(M.indptr[i], M.indptr[i + 1]):
                if M.indices[p] == i:
                    m_ii = M.data[p]
                    break
            if m_ii > 1e-30:
                D[i] = sqrt(m_ii)
                Dinv[i] = 1.0 / D[i]
            else:
                D[i] = 1.0
                Dinv[i] = 1.0

        var M_s = diag_scale(M, Dinv, Dinv)
        var K_s = diag_scale(K, Dinv, Dinv)

        var q0_s = List[Float64](length=n, fill=0.0)
        for i in range(n):
            q0_s[i] = D[i] * q0[i]

        var ode = RadauSparseLinearSolver[FPESparseLinearSystem](
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            first_step=self.first_step,
        )
        var system = FPESparseLinearSystem(M_s^, K_s^)
        var sol = ode.solve(system, (0.0, t_end), q0_s, t_eval)

        var y_out: List[List[Float64]] = []
        for idx in range(len(sol.y)):
            var q_unscaled = List[Float64](length=n, fill=0.0)
            for i in range(n):
                q_unscaled[i] = Dinv[i] * sol.y[idx][i]
            y_out.append(q_unscaled^)

        return y_out^
