from numerics.optim.osqp import OSQP, ProjectedGradient
from numerics.optim.lm import LevenbergMarquardt, ResidualCallable, JacobianCallable
from std.testing import assert_true, TestSuite


struct ResidualLine(ResidualCallable):
    def __init__(out self):
        pass

    def __call__(self, x: List[Float64]) raises -> List[Float64]:
        var t: List[Float64] = [0.0, 1.0, 2.0]
        var y: List[Float64] = [1.0, 3.0, 5.0]

        var out: List[Float64] = []
        for i in range(len(t)):
            out.append(x[0] * t[i] + x[1] - y[i])
        return out^


struct JacobianLine(JacobianCallable):
    def __init__(out self):
        pass

    def __call__(self, x: List[Float64]) raises -> List[List[Float64]]:
        _ = x
        var J: List[List[Float64]] = []
        J.append([0.0, 1.0])
        J.append([1.0, 1.0])
        J.append([2.0, 1.0])
        return J^


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_float_close(a: Float64, b: Float64, atol: Float64 = 1e-6) raises:
    assert_true(_abs(a - b) <= atol, "Expected " + String(b) + " got " + String(a))


def test_nnls_identity_with_sum_row() raises:
    var A: List[List[Float64]] = []
    A.append([1.0, 0.0])
    A.append([0.0, 1.0])
    A.append([1.0, 1.0])
    var b: List[Float64] = [1.0, 1.0, 2.0]

    var solver = ProjectedGradient(max_iter=2000, tol=1e-10, step_size=-1.0)
    var c = solver.solve(A, b)

    assert_float_close(c[0], 1.0, 1e-6)
    assert_float_close(c[1], 1.0, 1e-6)

    var osqp = OSQP(max_iter=2000, tol=1e-10)
    var c2 = osqp.solve_nnls(A, b)
    assert_float_close(c2[0], 1.0, 1e-6)
    assert_float_close(c2[1], 1.0, 1e-6)


def test_nnls_projects_negative_target_to_zero() raises:
    var A: List[List[Float64]] = []
    A.append([1.0, 0.0])
    A.append([0.0, 1.0])
    var b: List[Float64] = [-1.0, -1.0]

    var solver = ProjectedGradient(max_iter=2000, tol=1e-10, step_size=-1.0)
    var c = solver.solve(A, b)

    assert_float_close(c[0], 0.0, 1e-8)
    assert_float_close(c[1], 0.0, 1e-8)


def test_lm_linear_fit() raises:
    var lm = LevenbergMarquardt(
        max_iter=100,
        tol=1e-12,
        lambda_init=1e-3,
        lambda_up=10.0,
        lambda_down=0.1,
    )
    var x = lm.solve(ResidualLine(), JacobianLine(), [0.0, 0.0])

    assert_float_close(x[0], 2.0, 1e-6)
    assert_float_close(x[1], 1.0, 1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
