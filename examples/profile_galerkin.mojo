"""Profile cached_basis construction and full pipeline."""

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.initial_cond import initial_condition_from_cached
from engines.fpe.pdf import pdf_from_cached
from engines.fpe.solver import FPESolver
from sparse.csr import CSRMatrix
from std.time import perf_counter_ns as now


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1,
        T=0.6, S0=60.0, V0=0.1, S_min=50.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )

    var t0 = now()
    var domain = FPEDomain[3, 3](params, n_s=38, n_v=38, num_insert=251)
    var t1 = now()
    print("Domain construction: " + String(Float64(t1 - t0) / 1e9) + "s")

    var t_cb = now()
    var cached = domain.cached_basis()
    var t_cb_end = now()
    print("Cached basis: " + String(Float64(t_cb_end - t_cb) / 1e9) + "s")

    print()
    print("=== Mass matrix (Kronecker) ===")
    var t_m = now()
    var M = mass_from_cached[3, 3](cached)
    var t_m_end = now()
    print("Mass: " + String(Float64(t_m_end - t_m) / 1e9) + "s")
    print(" M: " + String(M.nrows) + "x" + String(M.ncols) + " nnz=" + String(M.nnz()))

    print()
    print("=== Stiffness matrix (Kronecker) ===")
    var t_k = now()
    var K = stiffness_from_cached[3, 3](cached, params)
    var t_k_end = now()
    print("Stiffness: " + String(Float64(t_k_end - t_k) / 1e9) + "s")
    print(" K: " + String(K.nrows) + "x" + String(K.ncols) + " nnz=" + String(K.nnz()))

    print()
    print("=== Initial condition ===")
    var t_ic = now()
    var q0 = initial_condition_from_cached[3, 3](cached, params, M, sigma0=0.1)
    var t_ic_end = now()
    print("Initial condition: " + String(Float64(t_ic_end - t_ic) / 1e9) + "s")
    print(" q0 length: " + String(len(q0)))

    print()
    print("=== ODE solve ===")
    var t_eval = [0.0, params.T]
    var fpe_solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1, first_step=1e-6)
    var t_ode = now()
    var q_t = fpe_solver.solve(domain, params, t_eval^)
    var t_ode_end = now()
    print("ODE solve: " + String(Float64(t_ode_end - t_ode) / 1e9) + "s")
    print(" Steps: " + String(len(q_t)))

    print()
    print("=== PDF ===")
    var q_final = q_t[len(q_t) - 1].copy()
    var t_pdf = now()
    var pdf_grid = pdf_from_cached[3, 3](cached, q_final)
    var t_pdf_end = now()
    print("PDF: " + String(Float64(t_pdf_end - t_pdf) / 1e9) + "s")

    var pdf_norm = 0.0
    for i in range(len(pdf_grid)):
        for j in range(len(pdf_grid[i])):
            pdf_norm += pdf_grid[i][j] * domain.s_weights[i] * domain.v_weights[j]
    print(" PDF integral: " + String(pdf_norm))

    var t_total = now() - t0
    print()
    print("Total: " + String(Float64(t_total) / 1e9) + "s")
