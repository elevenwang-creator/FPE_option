"""Test: compute error estimate directly as DD·Z - Z3 to check scaling."""

from std.math import sqrt, abs, exp


comptime SQRT6: Float64 = 2.449489742783178
comptime DD1: Float64 = (-13.0 - 7.0 * SQRT6) / 3.0
comptime DD2: Float64 = (-13.0 + 7.0 * SQRT6) / 3.0
comptime DD3: Float64 = -1.0 / 3.0


def main():
    print("=" * 70)
    print("  Direct Error Estimate Test: DD·Z - Z3")
    print("=" * 70)
    print()

    var h_values = [0.1, 0.01, 0.001, 0.0001]

    var Z1_list = [
        [-0.015375844391430237, -0.03051553491552798, -0.045419503427764796],
        [-0.0015482602286591616, -0.003094140860963842, -0.004637645029109222],
        [-0.00015493319944779867, -0.0003098425726878116, -0.0004647281232140189],
        [-1.549439222427793e-05, -3.098854615481929e-05, -4.648246179515528e-05],
    ]

    var Z2_list = [
        [-0.0625217978087713, -0.12113839925336518, -0.17609606535128813],
        [-0.0064352309974636, -0.012829081506104715, -0.019181817571260047],
        [-0.0006453928353614703, -0.0012903694597103982, -0.001934930141141615],
        [-6.455802067556227e-05, -0.00012911187682696857, -0.00019366156872251957],
    ]

    var Z3_list = [
        [-0.09520127499051974, -0.18133429690918335, -0.2592622187207434],
        [-0.009954632552283895, -0.019810122594834785, -0.029567456778536214],
        [-0.0009999529919821068, -0.001998905602574729, -0.0029968588318150364],
        [-0.00010004034488244802, -0.00020007067694294514, -0.00030009099718287846],
    ]

    print("Computing err_est = (DD1*Z1 + DD2*Z2 + DD3*Z3) - Z3")
    print("(Embedded 2nd order - 5th order difference)")
    print()

    var prev_norm: Float64 = 0.0
    for idx in range(len(h_values)):
        var h = h_values[idx]
        var Z1 = Z1_list[idx]
        var Z2 = Z2_list[idx]
        var Z3 = Z3_list[idx]

        var err_est: List[Float64] = []
        for k in range(3):
            var DDZ = DD1 * Z1[k] + DD2 * Z2[k] + DD3 * Z3[k]
            var err = DDZ - Z3[k]
            err_est.append(err)

        var norm = 0.0
        for k in range(3):
            norm += err_est[k] * err_est[k]
        norm = sqrt(norm / 3.0)

        print("h = " + String(h) + ": ||err_est|| = " + String(norm))
        if idx > 0:
            var ratio = norm / prev_norm
            print("  ratio = " + String(ratio))
        prev_norm = norm

    print()
    print("=" * 70)
    print("  Scaling expectations:")
    print("  ratio ~0.1    → O(h^1)")
    print("  ratio ~0.01   → O(h^2)")
    print("  ratio ~0.001  → O(h^3) (correct embedded estimate!)")
    print("=" * 70)
