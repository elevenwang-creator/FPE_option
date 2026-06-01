"""Benchmark FPE solve performance."""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from std.time import perf_counter_ns as now


def bench_fpe_solve() raises:
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var domain = FPEDomain(params, n_s=8, n_v=8)
    var solver = FPESolver(rtol=1e-4, atol=1e-6, max_step=0.05, first_step=0.0)
    var t_eval: List[Float64] = [0.0, 0.1]

    var start = now()
    var sol = solver.solve(domain, params, t_eval)
    var end = now()
    var elapsed = Float64(end - start) / 1e9

    print("FPE Solve (8x8 grid, degree 3)")
    print("  Solution points:", len(sol))
    print("  Solution dim:", len(sol[0]))
    print("  Total time:", elapsed, "s")


def main() raises:
    print("=== FPE Solve Benchmark ===")
    bench_fpe_solve()
