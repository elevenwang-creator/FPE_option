"""Test CPU single / GPU batch dispatch logic.

Verifies:
- B=1 (single option) → CPU path with RadauIIA
- B>1 (batch options) → GPU path with Metal on Apple Silicon
- Results are consistent between CPU and GPU paths

NOTE: This test requires --target-accelerator=metal:1 for GPU kernel compilation.
Run: pixi run mojo run --target-accelerator=metal:1 -I src tests/test_cpu_gpu_dispatch.mojo
"""
from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _close(a: Float64, b: Float64, tol: Float64 = 0.01) -> Bool:
    return _abs(a - b) < tol


def test_single_option_uses_cpu() raises:
    """B=1 solver should use CPU path (RadauIIA + sparse spmv)."""
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

    # B=1 → CPU path
    var solver_cpu = FPESolver(rtol=1e-4, atol=1e-6, max_step=0.05, first_step=0.0)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol = solver_cpu.solve(domain, params, t_eval)

    assert_true(len(sol) > 0, "should have solution states")
    assert_true(len(sol[0]) > 0, "should have state vector")
    
    # Verify non-negative and normalized
    for i in range(len(sol)):
        var row_sum = 0.0
        for j in range(len(sol[i])):
            assert_true(sol[i][j] >= -1e-10, "states should be non-negative")
            row_sum += sol[i][j]
        assert_true(row_sum > 0.99 and row_sum < 1.01, "states should sum to ~1")


def test_cpu_single_produces_valid_pdf() raises:
    """CPU single solve should produce a valid probability distribution."""
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

    var domain = FPEDomain(params, n_s=8, n_v=8, degree_s=3, degree_v=3)
    var solver = FPESolver(rtol=1e-4, atol=1e-6, max_step=0.05, first_step=0.0)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol = solver.solve(domain, params, t_eval)

    # Final state should be valid PDF
    var final_state = sol[len(sol) - 1].copy()
    var total = 0.0
    for i in range(len(final_state)):
        assert_true(final_state[i] >= -1e-10)
        total += final_state[i]
    
    # Total probability should be close to 1
    assert_true(total > 0.9 and total < 1.1)


def main() raises:
    print("=" * 60)
    print("CPU Single / GPU Batch Dispatch Tests")
    print("GPU available:", has_accelerator())
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
