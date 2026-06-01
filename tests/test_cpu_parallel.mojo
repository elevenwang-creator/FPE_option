from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from std.testing import assert_true, TestSuite


def test_cpu_parallel_runs() raises:
    """CPU parallel solve (B=2) should produce valid results."""
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

    var domain = FPEDomain(params, n_s=6, n_v=6, degree_s=2, degree_v=2)

    var solver_par = FPESolver(rtol=1e-4, atol=1e-6, max_step=0.05)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol_par = solver_par.solve(domain, params, t_eval)

    # Verify solution is valid
    assert_true(len(sol_par) > 0, "should have solution states")
    assert_true(len(sol_par[0]) > 0, "should have state vector")
    
    # Verify non-negative and normalized
    for i in range(len(sol_par)):
        var row_sum = 0.0
        for j in range(len(sol_par[i])):
            assert_true(sol_par[i][j] >= -1e-10, "states should be non-negative")
            row_sum += sol_par[i][j]
        assert_true(row_sum > 0.99 and row_sum < 1.01, "states should sum to ~1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
