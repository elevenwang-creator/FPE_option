from std.algorithm import parallelize
from std.algorithm.backend.vectorize import vectorize
from std.atomic import Atomic
from std.memory import Span
from std.math import abs
from std.sys import simd_width_of
from numerics.utils import FixedSizeVector
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def main() raises:
    var A = CSRMatrix.from_dense([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    var AT = A.transpose()
    var AT_dense = AT.to_dense()

    var expected = [[1.0, 0.0, 4.0], [0.0, 3.0, 0.0], [2.0, 0.0, 5.0]]

    var ok = True
    for i in range(3):
        for j in range(3):
            if abs(AT_dense[i][j] - expected[i][j]) > 1e-12:
                print("MISMATCH at [", i, "][", j, "]: got", AT_dense[i][j], "expected", expected[i][j])
                ok = False

    if ok:
        print("PASS: 3x3 transpose correct")

    var x: List[Float64] = [1.0, 2.0, 3.0]
    var y: List[Float64] = []
    for _ in range(3):
        y.append(0.0)
    A.spmv(x, y)

    var yt: List[Float64] = []
    for _ in range(3):
        yt.append(0.0)
    AT.spmv(x, yt)

    print("A*x  =", y)
    print("AT*x =", yt)

    ok = True
    var a_times_x = [7.0, 6.0, 19.0]
    var at_times_x = [13.0, 6.0, 17.0]
    for i in range(3):
        if abs(y[i] - a_times_x[i]) > 1e-12:
            print("SpMV MISMATCH A*x[", i, "]: got", y[i], "expected", a_times_x[i])
            ok = False
        if abs(yt[i] - at_times_x[i]) > 1e-12:
            print("SpMV MISMATCH AT*x[", i, "]: got", yt[i], "expected", at_times_x[i])
            ok = False
    if ok:
        print("PASS: SpMV on transposed matrix correct")

    var A4 = CSRMatrix.from_dense([[1.0, 0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 5.0, 6.0]])
    var AT4 = A4.transpose()
    var AT4_dense = AT4.to_dense()
    var expected4 = [[1.0, 0.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 6.0]]
    ok = True
    for i in range(5):
        for j in range(4):
            if abs(AT4_dense[i][j] - expected4[i][j]) > 1e-12:
                print("MISMATCH 4x5 at [", i, "][", j, "]: got", AT4_dense[i][j], "expected", expected4[i][j])
                ok = False
    if ok:
        print("PASS: 4x5 transpose correct")

    print("All transpose tests passed!")
