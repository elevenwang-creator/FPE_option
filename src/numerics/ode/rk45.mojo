from numerics.utils import abs_f64, copy_vec, max_f64, min_f64, zeros

from numerics.ode.types import ODESolution, ODESystem
from std.math import exp, log












@fieldwise_init
struct RungeKutta45[System: ODESystem]:
    var rtol: Float64
    var atol: Float64
    var max_step: Float64
    var min_step: Float64

    def solve(
        self,
        system: Self.System,
        t_span: Tuple[Float64, Float64],
        y0: List[Float64],
        t_eval: Optional[List[Float64]] = None,
    ) raises -> ODESolution:
        _ = t_eval

        comptime c2 = 1.0 / 5.0
        comptime c3 = 3.0 / 10.0
        comptime c4 = 4.0 / 5.0
        comptime c5 = 8.0 / 9.0
        comptime c6 = 1.0
        comptime c7 = 1.0

        comptime a21 = 1.0 / 5.0

        comptime a31 = 3.0 / 40.0
        comptime a32 = 9.0 / 40.0

        comptime a41 = 44.0 / 45.0
        comptime a42 = -56.0 / 15.0
        comptime a43 = 32.0 / 9.0

        comptime a51 = 19372.0 / 6561.0
        comptime a52 = -25360.0 / 2187.0
        comptime a53 = 64448.0 / 6561.0
        comptime a54 = -212.0 / 729.0

        comptime a61 = 9017.0 / 3168.0
        comptime a62 = -355.0 / 33.0
        comptime a63 = 46732.0 / 5247.0
        comptime a64 = 49.0 / 176.0
        comptime a65 = -5103.0 / 18656.0

        comptime b1 = 35.0 / 384.0
        comptime b3 = 500.0 / 1113.0
        comptime b4 = 125.0 / 192.0
        comptime b5 = -2187.0 / 6784.0
        comptime b6 = 11.0 / 84.0

        comptime e1 = 71.0 / 57600.0
        comptime e3 = -71.0 / 16695.0
        comptime e4 = 71.0 / 1920.0
        comptime e5 = -17253.0 / 339200.0
        comptime e6 = 22.0 / 525.0
        comptime e7 = -1.0 / 40.0

        comptime safety = 0.9
        comptime min_factor = 0.2
        comptime max_factor = 5.0

        var t0 = t_span[0]
        var t1 = t_span[1]
        var n = len(y0)
        if n != system.dim():
            return ODESolution([], [], False, "Initial state dimension mismatch")

        var t_values: List[Float64] = [t0]
        var y_values: List[List[Float64]] = []
        y_values.append(copy_vec(y0))

        var y = copy_vec(y0)
        var t = t0
        var h = self.max_step
        if h <= 0.0:
            h = (t1 - t0) / 100.0
        if h < self.min_step:
            h = self.min_step

        while t < t1:
            if t + h > t1:
                h = t1 - t

            var k1 = zeros(n)
            var k2 = zeros(n)
            var k3 = zeros(n)
            var k4 = zeros(n)
            var k5 = zeros(n)
            var k6 = zeros(n)
            var k7 = zeros(n)
            var ytmp = zeros(n)
            var y5 = zeros(n)

            system.rhs(t, y, k1)

            for i in range(n):
                ytmp[i] = y[i] + h * (a21 * k1[i])
            system.rhs(t + c2 * h, ytmp, k2)

            for i in range(n):
                ytmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i])
            system.rhs(t + c3 * h, ytmp, k3)

            for i in range(n):
                ytmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i])
            system.rhs(t + c4 * h, ytmp, k4)

            for i in range(n):
                ytmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i])
            system.rhs(t + c5 * h, ytmp, k5)

            for i in range(n):
                ytmp[i] = y[i] + h * (
                    a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]
                )
            system.rhs(t + c6 * h, ytmp, k6)

            for i in range(n):
                y5[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
            system.rhs(t + c7 * h, y5, k7)

            var err_norm = 0.0
            for i in range(n):
                var err_i = h * (
                    e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]
                )
                var scale = self.atol + self.rtol * max_f64(abs_f64(y[i]), abs_f64(y5[i]))
                var ratio = abs_f64(err_i) / scale
                if ratio > err_norm:
                    err_norm = ratio

            if err_norm <= 1.0:
                t = t + h
                y = y5^
                t_values.append(t)
                y_values.append(copy_vec(y))

                var factor = max_factor
                if err_norm > 0.0:
                    factor = safety * exp(log(1.0 / err_norm) * 0.2)
                    factor = max_f64(min_factor, min_f64(max_factor, factor))
                h = min_f64(self.max_step, h * factor)
                if h < self.min_step:
                    h = self.min_step
            else:
                var factor = safety * exp(log(1.0 / err_norm) * 0.2)
                factor = max_f64(min_factor, min_f64(1.0, factor))
                h = h * factor
                if h < self.min_step:
                    return ODESolution(
                        t_values^,
                        y_values^,
                        False,
                        "RK45 step size underflow",
                    )

        return ODESolution(t_values^, y_values^, True, "RK45 integration successful")
