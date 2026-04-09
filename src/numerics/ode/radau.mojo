"""Stiff ODE solvers: BackwardEuler and RadauIIA.

RadauIIA is a 3-stage implicit Runge-Kutta method of order 5, designed
for stiff ODE systems like the FPE. Uses comptime Butcher tableau
constants and simplified Newton iteration for implicit stage resolution.

BackwardEuler is retained as a simpler fallback (order 1 + Richardson).
"""

from numerics.ode.types import ODESolution, ODESystem
from numerics.utils import abs_f64, max_f64, min_f64, zeros, copy_vec, copy_mat, swap_rows
from numerics.linalg import lu_solve
from std.math import exp, log


def _estimate_jacobian_linear[System: ODESystem](
    system: System, t: Float64, y: List[Float64]
) raises -> List[List[Float64]]:
    var n = len(y)
    var A: List[List[Float64]] = []
    for _ in range(n):
        A.append(zeros(n))

    var f0 = zeros(n)
    system.rhs(t, y, f0)

    for j in range(n):
        var yp = copy_vec(y)
        var eps = 1e-8 * (1.0 + abs_f64(y[j]))
        yp[j] = yp[j] + eps

        var fj = zeros(n)
        system.rhs(t, yp, fj)
        for i in range(n):
            A[i][j] = (fj[i] - f0[i]) / eps

    return A^


def _backward_euler_step(
    A: List[List[Float64]], h: Float64, y: List[Float64]
) raises -> List[Float64]:
    var n = len(y)
    var B: List[List[Float64]] = []
    for i in range(n):
        var row = zeros(n)
        for j in range(n):
            row[j] = -h * A[i][j]
        row[i] = row[i] + 1.0
        B.append(row^)

    return lu_solve(B, y)


@fieldwise_init
struct BackwardEuler[System: ODESystem]:
    """Simplified stiff solver using backward Euler + Richardson extrapolation."""

    var rtol: Float64
    var atol: Float64
    var max_step: Float64

    def solve(
        self,
        system: Self.System,
        t_span: Tuple[Float64, Float64],
        y0: List[Float64],
        t_eval: Optional[List[Float64]] = None,
    ) raises -> ODESolution:
        _ = t_eval

        comptime safety = 0.9
        comptime min_factor = 0.2
        comptime max_factor = 5.0

        var t0 = t_span[0]
        var t1 = t_span[1]
        var n = len(y0)
        if n != system.dim():
            return ODESolution([], [], False, "Initial state dimension mismatch")

        var jac = _estimate_jacobian_linear(system, t0, y0)

        var t_values: List[Float64] = [t0]
        var y_values: List[List[Float64]] = []
        y_values.append(copy_vec(y0))

        var y = copy_vec(y0)
        var t = t0
        var h = self.max_step
        if h <= 0.0:
            h = (t1 - t0) / 50.0
        var min_step = 1e-12

        while t < t1:
            if t + h > t1:
                h = t1 - t

            var y_full = _backward_euler_step(jac, h, y)
            var y_half = _backward_euler_step(jac, h * 0.5, y)
            var y_half2 = _backward_euler_step(jac, h * 0.5, y_half)

            var y_rich = zeros(n)
            var err_norm = 0.0
            for i in range(n):
                y_rich[i] = 2.0 * y_half2[i] - y_full[i]
                var err_i = abs_f64(y_rich[i] - y_half2[i])
                var sc = self.atol + self.rtol * max_f64(abs_f64(y[i]), abs_f64(y_rich[i]))
                var ratio = err_i / sc
                if ratio > err_norm:
                    err_norm = ratio

            if err_norm <= 1.0:
                t = t + h
                y = y_rich^
                t_values.append(t)
                y_values.append(copy_vec(y))

                var factor = max_factor
                if err_norm > 0.0:
                    factor = safety * exp(log(1.0 / err_norm) * 0.5)
                    factor = max_f64(min_factor, min_f64(max_factor, factor))
                h = min_f64(self.max_step, h * factor)
            else:
                var factor = safety * exp(log(1.0 / err_norm) * 0.5)
                factor = max_f64(min_factor, min_f64(1.0, factor))
                h = h * factor
                if h < min_step:
                    return ODESolution(
                        t_values^,
                        y_values^,
                        False,
                        "BackwardEuler step size underflow",
                    )

        return ODESolution(
            t_values^,
            y_values^,
            True,
            "BackwardEuler integration successful",
        )


struct RadauIIA[System: ODESystem]:
    """3-stage implicit Radau IIA method. Order 5.

    Proper implicit RK solver for stiff ODE systems like the FPE.
    Uses comptime Butcher tableau constants and simplified Newton
    iteration for resolving the implicit stages.

    References:
    - Hairer & Wanner, "Solving Ordinary Differential Equations II", Ch. IV
    - The Butcher tableau coefficients are exact for a 3-stage Radau IIA method.
    """

    var rtol: Float64
    var atol: Float64
    var max_step: Float64
    var newton_tol: Float64
    var newton_max_iter: Int

    def __init__(
        out self,
        rtol: Float64 = 1e-6,
        atol: Float64 = 1e-8,
        max_step: Float64 = 0.0,
        newton_tol: Float64 = 1e-8,
        newton_max_iter: Int = 12,
    ):
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter

    def solve(
        self,
        system: Self.System,
        t_span: Tuple[Float64, Float64],
        y0: List[Float64],
        t_eval: Optional[List[Float64]] = None,
    ) raises -> ODESolution:
        _ = t_eval

        # ---- comptime Butcher tableau for 3-stage Radau IIA ----
        # Abscissae c
        comptime c1: Float64 = 0.15505102572168219018  # (4 - √6) / 10
        comptime c2: Float64 = 0.64494897427831780982  # (4 + √6) / 10
        comptime c3: Float64 = 1.0

        # A matrix (3×3) — implicit stages
        comptime a11: Float64 = 0.11208445195653073473
        comptime a12: Float64 = -0.04067796433082320083
        comptime a13: Float64 = 0.02581692768754036853
        comptime a21: Float64 = 0.23402839165419385511
        comptime a22: Float64 = 0.20686796081962466466
        comptime a23: Float64 = -0.04783262767800705297
        comptime a31: Float64 = 0.21668178412381825484
        comptime a32: Float64 = 0.40612326386737472080
        comptime a33: Float64 = 0.18903606908424243706

        # b weights = last row of A for Radau IIA
        comptime b1: Float64 = a31
        comptime b2: Float64 = a32
        comptime b3: Float64 = a33

        # Adaptive step control
        comptime safety: Float64 = 0.9
        comptime min_factor: Float64 = 0.2
        comptime max_factor: Float64 = 5.0
        comptime order: Float64 = 5.0

        var t0 = t_span[0]
        var t1 = t_span[1]
        var n = len(y0)
        if n != system.dim():
            return ODESolution([], [], False, "RadauIIA: dimension mismatch")

        var t_values: List[Float64] = [t0]
        var y_values: List[List[Float64]] = []
        y_values.append(copy_vec(y0))

        var y = copy_vec(y0)
        var t = t0
        var h = self.max_step
        if h <= 0.0:
            h = (t1 - t0) / 20.0
        var min_step = 1e-14

        while t < t1 - 1e-14:
            if t + h > t1:
                h = t1 - t

            # ---- Coupled Newton iteration for implicit stages ----
            # Solve the full 3n x 3n block system:
            # [I-h*a11*J, -h*a12*J, -h*a13*J][dk1]   [f1-k1]
            # [-h*a21*J, I-h*a22*J, -h*a23*J][dk2] = [f2-k2]
            # [-h*a31*J, -h*a32*J, I-h*a33*J][dk3]   [f3-k3]
            var f0 = zeros(n)
            system.rhs(t, y, f0)
            var k1 = copy_vec(f0)
            var k2 = copy_vec(f0)
            var k3 = copy_vec(f0)

            var J = _estimate_jacobian_linear(system, t, y)

            var converged = False
            for _ in range(self.newton_max_iter):
                var y1 = zeros(n)
                var y2 = zeros(n)
                var y3 = zeros(n)
                for i in range(n):
                    y1[i] = y[i] + h * (a11 * k1[i] + a12 * k2[i] + a13 * k3[i])
                    y2[i] = y[i] + h * (a21 * k1[i] + a22 * k2[i] + a23 * k3[i])
                    y3[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i] + a33 * k3[i])

                var f1 = zeros(n)
                var f2 = zeros(n)
                var f3 = zeros(n)
                system.rhs(t + c1 * h, y1, f1)
                system.rhs(t + c2 * h, y2, f2)
                system.rhs(t + c3 * h, y3, f3)

                var max_res = 0.0
                for i in range(n):
                    max_res = max_f64(max_res, abs_f64(f1[i] - k1[i]))
                    max_res = max_f64(max_res, abs_f64(f2[i] - k2[i]))
                    max_res = max_f64(max_res, abs_f64(f3[i] - k3[i]))

                if max_res < self.newton_tol:
                    converged = True
                    break

                # Build 3n x 3n block Jacobian system
                var N3 = 3 * n
                var block_system: List[List[Float64]] = []
                for _ in range(N3):
                    block_system.append(zeros(N3)^)

                for i_block in range(n):
                    for j_block in range(n):
                        var Jij = J[i_block][j_block]
                        # Block (0,0): I - h*a11*J
                        block_system[i_block][j_block] = (
                            -h * a11 * Jij if i_block != j_block
                            else 1.0 - h * a11 * Jij
                        )
                        # Block (0,1): -h*a12*J
                        block_system[i_block][n + j_block] = -h * a12 * Jij
                        # Block (0,2): -h*a13*J
                        block_system[i_block][2 * n + j_block] = -h * a13 * Jij
                        # Block (1,0): -h*a21*J
                        block_system[n + i_block][j_block] = -h * a21 * Jij
                        # Block (1,1): I - h*a22*J
                        block_system[n + i_block][n + j_block] = (
                            -h * a22 * Jij if i_block != j_block
                            else 1.0 - h * a22 * Jij
                        )
                        # Block (1,2): -h*a23*J
                        block_system[n + i_block][2 * n + j_block] = -h * a23 * Jij
                        # Block (2,0): -h*a31*J
                        block_system[2 * n + i_block][j_block] = -h * a31 * Jij
                        # Block (2,1): -h*a32*J
                        block_system[2 * n + i_block][n + j_block] = -h * a32 * Jij
                        # Block (2,2): I - h*a33*J
                        block_system[2 * n + i_block][2 * n + j_block] = (
                            -h * a33 * Jij if i_block != j_block
                            else 1.0 - h * a33 * Jij
                        )

                var rhs_block = zeros(N3)
                for i in range(n):
                    rhs_block[i] = f1[i] - k1[i]
                    rhs_block[n + i] = f2[i] - k2[i]
                    rhs_block[2 * n + i] = f3[i] - k3[i]

                var dk = lu_solve(block_system, rhs_block)
                for i in range(n):
                    k1[i] = k1[i] + dk[i]
                    k2[i] = k2[i] + dk[n + i]
                    k3[i] = k3[i] + dk[2 * n + i]

            if not converged:
                # Reduce step size and retry
                h = h * 0.5
                if h < min_step:
                    return ODESolution(
                        t_values^, y_values^, False,
                        "RadauIIA: Newton iteration failed to converge"
                    )
                continue

            # ---- Compute solution and embedded error estimate ----
            # y_{n+1} = y_n + h * (b1*k1 + b2*k2 + b3*k3)
            var y_new = zeros(n)
            for i in range(n):
                y_new[i] = y[i] + h * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i])

            # Error estimate: difference between order-5 and embedded order-3
            # Using the stage 3 value as the lower-order estimate
            var err_norm = 0.0
            for i in range(n):
                var err_i = abs_f64(y_new[i] - (y[i] + h * k3[i]))
                var sc = self.atol + self.rtol * max_f64(abs_f64(y[i]), abs_f64(y_new[i]))
                var ratio = err_i / sc
                err_norm = max_f64(err_norm, ratio)

            if err_norm <= 1.0:
                # Accept step
                t = t + h
                y = y_new^
                t_values.append(t)
                y_values.append(copy_vec(y))

                # Adjust step size
                var factor = max_factor
                if err_norm > 1e-10:
                    factor = safety * exp(log(1.0 / err_norm) / order)
                    factor = max_f64(min_factor, min_f64(max_factor, factor))
                h = min_f64(self.max_step if self.max_step > 0 else (t1 - t0), h * factor)
            else:
                # Reject step, reduce h
                var factor = safety * exp(log(1.0 / err_norm) / order)
                factor = max_f64(min_factor, min_f64(1.0, factor))
                h = h * factor
                if h < min_step:
                    return ODESolution(
                        t_values^, y_values^, False,
                        "RadauIIA: step size underflow"
                    )

        return ODESolution(
            t_values^,
            y_values^,
            True,
            "RadauIIA integration successful",
        )
