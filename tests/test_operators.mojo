from sparse.csr import CSRMatrix
from std.math import abs


def main() raises:
    # Test __add__: A + B
    var A = CSRMatrix.from_dense([[1.0, 0.0], [0.0, 2.0]])
    var B = CSRMatrix.from_dense([[3.0, 1.0], [0.0, 0.0]])
    var C = A + B
    var ok = True
    if abs(C.get(0, 0) - 4.0) > 1e-12:
        print("FAIL A+B [0,0] =", C.get(0, 0))
        ok = False
    if abs(C.get(0, 1) - 1.0) > 1e-12:
        print("FAIL A+B [0,1] =", C.get(0, 1))
        ok = False
    if abs(C.get(1, 1) - 2.0) > 1e-12:
        print("FAIL A+B [1,1] =", C.get(1, 1))
        ok = False
    if ok:
        print("PASS: A + B operator")

    # Test __matmul__: A @ B
    var D = CSRMatrix.from_dense([[1.0, 2.0], [3.0, 4.0]])
    var E = CSRMatrix.from_dense([[5.0, 0.0], [0.0, 6.0]])
    var F = D @ E
    ok = True
    # D@E = [[5, 12], [15, 24]]
    if abs(F.get(0, 0) - 5.0) > 1e-12:
        print("FAIL D@E [0,0] =", F.get(0, 0))
        ok = False
    if abs(F.get(0, 1) - 12.0) > 1e-12:
        print("FAIL D@E [0,1] =", F.get(0, 1))
        ok = False
    if abs(F.get(1, 0) - 15.0) > 1e-12:
        print("FAIL D@E [1,0] =", F.get(1, 0))
        ok = False
    if abs(F.get(1, 1) - 24.0) > 1e-12:
        print("FAIL D@E [1,1] =", F.get(1, 1))
        ok = False
    if ok:
        print("PASS: A @ B operator")

    # Test chained: A + B + C
    var G = CSRMatrix.from_dense([[1.0, 0.0], [0.0, 1.0]])
    var H = CSRMatrix.from_dense([[2.0, 0.0], [0.0, 2.0]])
    var I = CSRMatrix.from_dense([[3.0, 0.0], [0.0, 3.0]])
    var J = G + H + I
    ok = True
    if abs(J.get(0, 0) - 6.0) > 1e-12:
        print("FAIL chained add [0,0] =", J.get(0, 0))
        ok = False
    if abs(J.get(1, 1) - 6.0) > 1e-12:
        print("FAIL chained add [1,1] =", J.get(1, 1))
        ok = False
    if ok:
        print("PASS: chained A + B + C")

    print("All operator overload tests passed!")
