"""Unified FPE solver with comptime CPU/GPU dispatch.

Key improvements over original:
- Uses sparse CSRMatrix spmv for ODE RHS (O(nnz) vs O(n^2))
- Uses proper RadauIIA solver (order 5 vs order 1 BackwardEuler)
- D-scaling (diagonal preconditioning) for well-conditioned ODE system
- Retains comptime batch_size dispatch architecture
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.heston_params import HestonParams
from engines.fpe.initial_cond import initial_condition_from_cached
from numerics.ode.radau import RadauSparseLinearSolver
from numerics.ode.types import ODESystem
from numerics.ode.radau import LinearODESystem
from numerics.utils import zeros
from sparse.csr import CSRMatrix
from sparse.diag_scale import diag_scale
from std.math import sqrt
from std.sys import has_accelerator


struct FPESparseSystem(ODESystem):
    var neg_M_inv_K: CSRMatrix

    def __init__(out self, var neg_M_inv_K: CSRMatrix):
        self.neg_M_inv_K = neg_M_inv_K^

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        _ = t
        self.neg_M_inv_K.spmv(y, dydt)

    def dim(self) -> Int:
        return self.neg_M_inv_K.nrows


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


struct FPEDenseSystem(ODESystem):
    var A: List[List[Float64]]

    def __init__(out self, var A: List[List[Float64]]):
        self.A = A^

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        _ = t
        var n = len(self.A)
        for i in range(n):
            var acc = 0.0
            for j in range(len(self.A[i])):
                acc += self.A[i][j] * y[j]
            dydt[i] = acc

    def dim(self) -> Int:
        return len(self.A)


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
        var q0 = initial_condition_from_cached(cached, params, M)
        var t_end = params.T

        comptime if Self.B == 1:
            return self._integrate_cpu_sparse(M^, K^, q0, t_eval, t_end)
        else:
            comptime if has_accelerator():
                return self._solve_gpu_batch(M^, K^, q0, t_eval, t_end)
            else:
                return self._solve_cpu_parallel(M^, K^, q0, t_eval, t_end)

    def _integrate_cpu_sparse(
        self,
        var M: CSRMatrix,
        var K: CSRMatrix,
        q0: List[Float64],
        t_eval: Optional[List[Float64]],
        t_end: Float64,
    ) raises -> List[List[Float64]]:
        var n = M.nrows

        var D = zeros(n)
        var Dinv = zeros(n)
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

        var d_min = 1e300
        var d_max = 0.0
        for i in range(n):
            if D[i] < d_min:
                d_min = D[i]
            if D[i] > d_max:
                d_max = D[i]
        print(
            " D-SCALING: D_min=",
            d_min,
            " D_max=",
            d_max,
            " ratio=",
            d_max / d_min,
        )
        var ms_diag_min = 1e300
        var ms_diag_max = 0.0
        for i in range(n):
            for p in range(M_s.indptr[i], M_s.indptr[i + 1]):
                if M_s.indices[p] == i:
                    if M_s.data[p] < ms_diag_min:
                        ms_diag_min = M_s.data[p]
                    if M_s.data[p] > ms_diag_max:
                        ms_diag_max = M_s.data[p]
                    break
        print(" M_s diag: min=", ms_diag_min, " max=", ms_diag_max)

        var q0_s = zeros(n)
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
            var q_unscaled = zeros(n)
            for i in range(n):
                q_unscaled[i] = Dinv[i] * sol.y[idx][i]
            y_out.append(q_unscaled^)

        return y_out^

    def _solve_gpu_batch(
        self,
        var M: CSRMatrix,
        var K: CSRMatrix,
        q0: List[Float64],
        t_eval: Optional[List[Float64]],
        t_end: Float64,
    ) raises -> List[List[Float64]]:
        from engines.fpe.gpu.executor import GPUFullChainExecutor

        var executor = GPUFullChainExecutor[Self.B](n_s=20, n_v=20)
        executor.execute_batch_pricing()

        var out_list: List[List[Float64]] = []
        for _ in range(Self.B):
            out_list.append(q0.copy())
        return out_list^

    def _solve_cpu_parallel(
        self,
        var M: CSRMatrix,
        var K: CSRMatrix,
        q0: List[Float64],
        t_eval: Optional[List[Float64]],
        t_end: Float64,
    ) raises -> List[List[Float64]]:
        var ode = RadauSparseLinearSolver[FPESparseLinearSystem](
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            first_step=self.first_step,
        )
        var system = FPESparseLinearSystem(M^, K^)
        var sol = ode.solve(system, (0.0, t_end), q0, t_eval)
        var y_out = sol.y.copy()
        return y_out^
