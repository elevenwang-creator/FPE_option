from numerics.ode.types import ODESystem
from numerics.ode.rk45 import RungeKutta45
from numerics.ode.radau import BackwardEuler
from std.math import exp
from std.testing import assert_true, TestSuite


def assert_float_close(a: Float64, b: Float64, atol: Float64) raises:
    var diff = a - b
    if diff < 0:
        diff = -diff
    assert_true(diff <= atol, "Expected " + String(b) + " got " + String(a))


struct ExpDecaySystem(ODESystem):
    def __init__(out self):
        pass

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        _ = t
        dydt[0] = -y[0]

    def dim(self) -> Int:
        return 1


struct LinearGrowthSystem(ODESystem):
    def __init__(out self):
        pass

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        _ = y
        dydt[0] = 2.0 * t

    def dim(self) -> Int:
        return 1


struct StiffDecaySystem(ODESystem):
    def __init__(out self):
        pass

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        _ = t
        dydt[0] = -100.0 * y[0]

    def dim(self) -> Int:
        return 1


def test_rk45_exponential_decay() raises:
    var solver = RungeKutta45[ExpDecaySystem](
        rtol=1e-8,
        atol=1e-10,
        max_step=0.25,
        min_step=1e-8,
    )
    var sys = ExpDecaySystem()
    var sol = solver.solve(sys, (0.0, 1.0), [1.0])
    assert_true(sol.success)
    var y_end = sol.y[len(sol.y) - 1][0]
    assert_float_close(y_end, exp(Float64(-1.0)), 1e-4)


def test_rk45_linear_growth() raises:
    var solver = RungeKutta45[LinearGrowthSystem](
        rtol=1e-8,
        atol=1e-10,
        max_step=0.25,
        min_step=1e-8,
    )
    var sys = LinearGrowthSystem()
    var sol = solver.solve(sys, (0.0, 2.0), [0.0])
    assert_true(sol.success)
    var y_end = sol.y[len(sol.y) - 1][0]
    assert_float_close(y_end, 4.0, 1e-4)


def test_backward_euler_stiff_decay() raises:
    var solver = BackwardEuler[StiffDecaySystem](
        rtol=1e-7,
        atol=1e-9,
        max_step=0.05,
    )
    var sys = StiffDecaySystem()
    var sol = solver.solve(sys, (0.0, 1.0), [1.0])
    assert_true(sol.success)
    var y_end = sol.y[len(sol.y) - 1][0]
    assert_float_close(y_end, exp(Float64(-100.0)), 1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
