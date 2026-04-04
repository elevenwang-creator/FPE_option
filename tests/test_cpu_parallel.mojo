from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from std.testing import assert_true, TestSuite


def test_cpu_parallel_matches_sequential() raises:
    """CPU parallel solve should produce same results as sequential."""
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

    var solver_seq = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.05)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol_seq = solver_seq.solve(domain, params, t_eval)

    var solver_par = FPESolver[2](rtol=1e-4, atol=1e-6, max_step=0.05)
    var sol_par = solver_par.solve(domain, params, t_eval)

    assert_true(len(sol_seq) == len(sol_par), "solution lengths should match")
    for i in range(len(sol_seq)):
        assert_true(len(sol_seq[i]) == len(sol_par[i]), "row lengths should match")
        for j in range(len(sol_seq[i])):
            var diff = sol_seq[i][j] - sol_par[i][j]
            if diff < 0.0:
                diff = -diff
            assert_true(diff < 1e-8, "values should match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
