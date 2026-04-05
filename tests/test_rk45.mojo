from numerics.ode.rk45 import RungeKutta45
from numerics.ode.types import ODESystem, ODESolution
from numerics.utils import abs_f64, zeros
from std.testing import assert_true, TestSuite


def _close(a: Float64, b: Float64, tol: Float64 = 0.1) -> Bool:
    return abs_f64(a - b) < tol


struct LinearGrowthSystem(ODESystem):
    def __init__(out self):
        pass
    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        dydt[0] = 1.0
    def dim(self) -> Int:
        return 1


def test_rk45_linear_growth() raises:
    """RK45 should solve dy/dt = 1 => y(t) = y0 + t."""
    var sys = LinearGrowthSystem()
    var solver = RungeKutta45[LinearGrowthSystem](rtol=1e-6, atol=1e-8, max_step=0.1, min_step=1e-10)
    var t_eval: List[Float64] = [0.0, 1.0]
    var sol = solver.solve(sys, (0.0, 1.0), [0.0], t_eval^)
    assert_true(sol.success, "should succeed")
    assert_true(len(sol.t) > 0, "should have time points")
    # Check first value is y(0) = 0
    assert_true(_close(sol.y[0][0], 0.0), "y(0) should be 0")
    # Check last value is approximately y(1) = 1
    var last_idx = len(sol.y) - 1
    assert_true(_close(sol.y[last_idx][0], 1.0, 0.01), "y(t_end) should be ~1")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
