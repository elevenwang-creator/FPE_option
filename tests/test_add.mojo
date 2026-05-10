from sparse.csr import CSRMatrix
from sparse.add import add


def main() raises:
    var A = CSRMatrix.from_dense([[1.0, 0.0], [0.0, 2.0]])
    var B = CSRMatrix.from_dense([[3.0, 1.0], [0.0, 0.0]])
    var C = add(A, B)
    var d = C.to_dense()
    print("C[0][0] =", d[0][0], "expect 4.0")
    print("C[0][1] =", d[0][1], "expect 1.0")
    print("C[1][0] =", d[1][0], "expect 0.0")
    print("C[1][1] =", d[1][1], "expect 2.0")
    print("add test done")
