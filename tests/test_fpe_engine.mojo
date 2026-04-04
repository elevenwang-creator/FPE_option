from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from engines.fpe.solver import FPESolver
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def test_fpe_pipeline_small_grid() raises:
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
    assert_true(params.is_valid())

    var domain = FPEDomain(params, n_s=8, n_v=8, degree_s=3, degree_v=3)
    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)

    assert_true(M.nrows == M.ncols)
    assert_true(M.nrows > 0)

    var q0 = InitialCondition[1]().compute(domain, params)
    var q0_sum = 0.0
    for i in range(len(q0)):
        q0_sum += q0[i]
    assert_true(_abs(q0_sum - 1.0) < 1e-2)

    var solver = FPESolver[1](rtol=1e-6, atol=1e-8, max_step=0.02)
    var sol = solver.solve(domain, params, [0.0, 0.1])
    assert_true(len(sol) >= 1)

    for i in range(len(sol[len(sol) - 1])):
        assert_true(sol[len(sol) - 1][i] >= -1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
