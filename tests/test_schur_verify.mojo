"""Verify Schur decomposition values and test dense LU approach."""

from numerics.utils import abs_f64, zeros
from std.math import sqrt


def mat3_mul(A: List[Float64], B: List[Float64]) -> List[Float64]:
    var C: List[Float64] = []
    for i in range(3):
        for j in range(3):
            var s: Float64 = 0.0
            for k in range(3):
                s = s + A[i * 3 + k] * B[k * 3 + j]
            C.append(s)
    return C^


def mat3_transpose(A: List[Float64]) -> List[Float64]:
    var T: List[Float64] = []
    for i in range(3):
        for j in range(3):
            T.append(A[j * 3 + i])
    return T^


def main() raises:
    print("=== Verify Schur Decomposition ===")
    print()

    var sqrt6 = sqrt(6.0)

    var a11: Float64 = (88.0 - 7.0 * sqrt6) / 360.0
    var a12: Float64 = (296.0 - 169.0 * sqrt6) / 1800.0
    var a13: Float64 = (-2.0 + 3.0 * sqrt6) / 225.0
    var a21: Float64 = (296.0 + 169.0 * sqrt6) / 1800.0
    var a22: Float64 = (88.0 + 7.0 * sqrt6) / 360.0
    var a23: Float64 = (-2.0 - 3.0 * sqrt6) / 225.0
    var a31: Float64 = (16.0 - sqrt6) / 36.0
    var a32: Float64 = (16.0 + sqrt6) / 36.0
    var a33: Float64 = 1.0 / 9.0

    var A_flat: List[Float64] = [a11, a12, a13, a21, a22, a23, a31, a32, a33]

    print("A (Butcher matrix):")
    for i in range(3):
        print("  [", end="")
        for j in range(3):
            print(String(A_flat[i * 3 + j]), end=" ")
        print("]")
    print()

    var T_flat: List[Float64] = [
        0.16255558520216112, 0.51074439865923390, -0.47719467969124402,
        -0.06697332890760048, 0.16255558520216112, -0.28529656780973917,
        0.0, 0.0, 0.27488882959567795,
    ]

    var Q_flat: List[Float64] = [
        0.13866510875190752, 0.04627814930949071, 0.98925745916384644,
        -0.22964124235174019, -0.97017888655183304, 0.07757466016809490,
        -0.96334671195056865, 0.23793121061671349, 0.12390258911134427,
    ]

    print("T (Schur form):")
    for i in range(3):
        print("  [", end="")
        for j in range(3):
            print(String(T_flat[i * 3 + j]), end=" ")
        print("]")
    print()

    print("Q (Schur vectors):")
    for i in range(3):
        print("  [", end="")
        for j in range(3):
            print(String(Q_flat[i * 3 + j]), end=" ")
        print("]")
    print()

    var QT = mat3_transpose(Q_flat)
    var Q_T = mat3_mul(Q_flat, T_flat)
    var Q_T_QT = mat3_mul(Q_T, QT)

    print("Q @ T @ Q^T (should equal A):")
    for i in range(3):
        print("  [", end="")
        for j in range(3):
            print(String(Q_T_QT[i * 3 + j]), end=" ")
        print("]")
    print()

    var max_err = 0.0
    for i in range(9):
        var err = abs_f64(Q_T_QT[i] - A_flat[i])
        if err > max_err:
            max_err = err
    print("Max error |Q*T*Q^T - A| = ", max_err)
    print()

    print("Q @ Q^T (should be I):")
    var QQ_T = mat3_mul(Q_flat, QT)
    for i in range(3):
        print("  [", end="")
        for j in range(3):
            print(String(QQ_T[i * 3 + j]), end=" ")
        print("]")
    print()

    var max_orth_err = 0.0
    for i in range(3):
        for j in range(3):
            var expected = 1.0 if i == j else 0.0
            var err = abs_f64(QQ_T[i * 3 + j] - expected)
            if err > max_orth_err:
                max_orth_err = err
    print("Max orthogonality error = ", max_orth_err)
    print()

    print("Eigenvalues of T:")
    print("  T[0,0] = ", T_flat[0], " (real eigenvalue)")
    print("  2x2 block: T[1,1]=", T_flat[4], " T[1,2]=", T_flat[5])
    print("             T[2,1]=", T_flat[7], " T[2,2]=", T_flat[8])
    var tr_2x2 = T_flat[4] + T_flat[8]
    var det_2x2 = T_flat[4] * T_flat[8] - T_flat[5] * T_flat[7]
    var disc = tr_2x2 * tr_2x2 - 4.0 * det_2x2
    print("  2x2 eigenvalues: ", tr_2x2 / 2.0, " +/- ", sqrt(abs_f64(disc)) / 2.0, "i")
    print()

    print("Column sums of Q (= row sums of Q^T):")
    for j in range(3):
        var col_sum = Q_flat[0 * 3 + j] + Q_flat[1 * 3 + j] + Q_flat[2 * 3 + j]
        print("  Q[:,", j, "] sum = ", col_sum)
    print()

    print("=== Verification complete ===")
