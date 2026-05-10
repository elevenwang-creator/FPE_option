from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from sparse.csr import CSRMatrix
from sparse.diag_scale import diag_scale
from std.math import sqrt
from std.time import perf_counter_ns as now


struct FPESystem(LinearODESystem):
    var M_mat: CSRMatrix
    var K_mat: CSRMatrix

    def __init__(out self, var M: CSRMatrix, var K: CSRMatrix):
        self.M_mat = M^
        self.K_mat = K^

    def get_M(self) -> CSRMatrix:
        return self.M_mat.copy()

    def get_K(self) -> CSRMatrix:
        return self.K_mat.copy()


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var n_s = 38
    var n_v = 38
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v, num_insert=251)
    print("s_knots: ", len(domain.s_knots), " v_knots: ", len(domain.v_knots))

    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)
    var q0 = InitialCondition[1]().compute(domain, params, sigma0=0.1)

    var n = M.nrows
    print("n=", n)

    # D-scaling
    var D = List[Float64]()
    var Dinv = List[Float64]()
    for i in range(n):
        var m_ii = 0.0
        for p in range(M.indptr[i], M.indptr[i + 1]):
            if M.indices[p] == i:
                m_ii = M.data[p]
                break
        if m_ii > 1e-30:
            D.append(sqrt(m_ii))
            Dinv.append(1.0 / sqrt(m_ii))
        else:
            D.append(1.0)
            Dinv.append(1.0)

    var M_s = diag_scale(M, Dinv, Dinv)
    var K_s = diag_scale(K, Dinv, Dinv)

    var q0_s = List[Float64]()
    for i in range(n):
        q0_s.append(D[i] * q0[i])

    print("D-scaling done, starting Radau solve...")

    var system = FPESystem(M_s^, K_s^)
    var solver = RadauSparseLinearSolver[FPESystem](
        rtol=1e-4, atol=1e-6, max_step=0.1, first_step=1e-6
    )

    var t0 = now()
    var sol = solver.solve(system, (0.0, 0.6), q0_s)
    var t1 = now()
    print("Solve time: ", Float64(t1 - t0) / 1e9, " s")
    print("Success: ", sol.success)
    print("Message: ", sol.message)
    if sol.success:
        print("Steps: ", len(sol.t))
        print("t_final: ", sol.t[len(sol.t) - 1])
        var y_final = sol.y[len(sol.y) - 1].copy()
        var y_sum = 0.0
        for i in range(n):
            y_sum += y_final[i]
        print("y_sum: ", y_sum)
