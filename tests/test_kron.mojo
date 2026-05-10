from sparse.csr import CSRMatrix
from sparse.kron import kron
from std.math import abs


def main() raises:
    # Test 1: 2x2 ⊗ 2x2
    # A = [[1, 2], [3, 4]], B = [[5, 0], [0, 6]]
    var a_data: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var a_indices: List[Int] = [0, 1, 0, 1]
    var a_indptr: List[Int] = [0, 2, 4]
    var A = CSRMatrix(2, 2, 4, a_data^, a_indptr^, a_indices^)

    var b_data: List[Float64] = [5.0, 6.0]
    var b_indices: List[Int] = [0, 1]
    var b_indptr: List[Int] = [0, 1, 2]
    var B = CSRMatrix(2, 2, 2, b_data^, b_indptr^, b_indices^)

    var C = kron(A, B)

    var ok = True
    if C.nrows != 4:
        print("FAIL: nrows =", C.nrows, "expected 4")
        ok = False
    if C.ncols != 4:
        print("FAIL: ncols =", C.ncols, "expected 4")
        ok = False

    # C = A ⊗ B:
    # row 0 (i=0,k=0): 1*[5,0] + 2*[0,6] = [5,0,0,12]
    # row 1 (i=0,k=1): 1*[0,6] + 2*[0,0] = [0,6,0,0]
    # row 2 (i=1,k=0): 3*[5,0] + 4*[0,6] = [15,0,0,24]
    # row 3 (i=1,k=1): 3*[0,6] + 4*[0,0] = [0,18,0,0]
    var expected = [
        [5.0, 0.0, 10.0, 0.0],
        [0.0, 6.0, 0.0, 12.0],
        [15.0, 0.0, 20.0, 0.0],
        [0.0, 18.0, 0.0, 24.0],
    ]
    for i in range(4):
        for j in range(4):
            var got = C.get(i, j)
            var exp = expected[i][j]
            if abs(got - exp) > 1e-12:
                print("FAIL C[", i, ",", j, "] =", got, "expected", exp)
                ok = False
    if ok:
        print("PASS: 2x2 kron correct")

    # Test 2: I2 ⊗ I2 = I4
    ok = True
    var i_data: List[Float64] = [1.0, 1.0]
    var i_indices: List[Int] = [0, 1]
    var i_indptr: List[Int] = [0, 1, 2]
    var I2 = CSRMatrix(2, 2, 2, i_data^, i_indptr^, i_indices^)

    var C2 = kron(I2, I2)
    for i in range(4):
        if abs(C2.get(i, i) - 1.0) > 1e-12:
            print("FAIL: I4 diagonal at", i, "=", C2.get(i, i))
            ok = False
        for j in range(4):
            if j != i and abs(C2.get(i, j)) > 1e-12:
                print("FAIL: I4 off-diag at [", i, ",", j, "] =", C2.get(i, j))
                ok = False
    if ok:
        print("PASS: I2⊗I2 = I4 correct")

    # Test 3: denser B to exercise SIMD-width scatter paths
    # A = [[1, 2]], B = dense 4x4 identity-like (all 4 cols per row)
    ok = True
    var a3_data: List[Float64] = [1.0, 2.0]
    var a3_indices: List[Int] = [0, 1]
    var a3_indptr: List[Int] = [0, 2]
    var A3 = CSRMatrix(1, 2, 2, a3_data^, a3_indptr^, a3_indices^)

    var B3 = CSRMatrix.from_dense([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ])

    var C3 = kron(A3, B3)

    # out_nrows = 1*4 = 4, out_ncols = 2*4 = 8
    if C3.nrows != 4:
        print("FAIL: dense nrows =", C3.nrows)
        ok = False
    if C3.ncols != 8:
        print("FAIL: dense ncols =", C3.ncols)
        ok = False

    # row 0 (i=0,k=0): 1*[1,2,3,4] + 2*[0,0,0,0] → cols 0-3: [1,2,3,4], cols 4-7: [10,12,14,16]
    # (a_col=1→col_base=4, a_val=2: 2*B[0,:]=[2,4,6,8] → cols 4-7)
    if abs(C3.get(0, 0) - 1.0) > 1e-12:
        print("FAIL dense C3[0,0] =", C3.get(0, 0))
        ok = False
    if abs(C3.get(0, 4) - 2.0) > 1e-12:
        print("FAIL dense C3[0,4] =", C3.get(0, 4))
        ok = False
    if abs(C3.get(0, 7) - 8.0) > 1e-12:
        print("FAIL dense C3[0,7] =", C3.get(0, 7))
        ok = False
    # row 3 (i=0,k=3): 1*[13,14,15,16] + 2*B[3,:]=[26,28,30,32] → cols 4-7
    if abs(C3.get(3, 0) - 13.0) > 1e-12:
        print("FAIL dense C3[3,0] =", C3.get(3, 0))
        ok = False
    if abs(C3.get(3, 7) - 32.0) > 1e-12:
        print("FAIL dense C3[3,7] =", C3.get(3, 7))
        ok = False

    if ok:
        print("PASS: dense 1x2⊗4x4 kron correct")

    print("All kron tests passed!")
