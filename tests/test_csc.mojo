from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix
from std.math import abs

def main() raises:
    var A = CSRMatrix.from_dense([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    var C = A.to_csc()

    print("CSC colptr:", C.colptr)
    print("CSC indices:", C.indices)
    print("CSC data:", C.data)

    var ok = True
    var expected_colptr = [0, 2, 3, 5]
    var expected_indices = [0, 2, 1, 0, 2]
    var expected_data = [1.0, 4.0, 3.0, 2.0, 5.0]

    for j in range(4):
        if C.colptr[j] != expected_colptr[j]:
            print("colptr MISMATCH [", j, "]: got", C.colptr[j], "expected", expected_colptr[j])
            ok = False
    for p in range(5):
        if C.indices[p] != expected_indices[p]:
            print("indices MISMATCH [", p, "]: got", C.indices[p], "expected", expected_indices[p])
            ok = False
        if abs(C.data[p] - expected_data[p]) > 1e-12:
            print("data MISMATCH [", p, "]: got", C.data[p], "expected", expected_data[p])
            ok = False

    if ok:
        print("PASS: 3x3 CSR->CSC correct")

    var A2 = CSRMatrix.from_dense([[1.0, 0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 5.0, 6.0]])
    var C2 = A2.to_csc()
    var expected2_colptr = [0, 2, 3, 4, 5, 6]
    var expected2_indices = [0, 3, 1, 0, 3, 3]
    var expected2_data = [1.0, 4.0, 3.0, 2.0, 5.0, 6.0]
    ok = True
    for j in range(6):
        if C2.colptr[j] != expected2_colptr[j]:
            print("colptr2 MISMATCH [", j, "]: got", C2.colptr[j], "expected", expected2_colptr[j])
            ok = False
    for p in range(6):
        if C2.indices[p] != expected2_indices[p]:
            print("indices2 MISMATCH [", p, "]: got", C2.indices[p], "expected", expected2_indices[p])
            ok = False
        if abs(C2.data[p] - expected2_data[p]) > 1e-12:
            print("data2 MISMATCH [", p, "]: got", C2.data[p], "expected", expected2_data[p])
            ok = False
    if ok:
        print("PASS: 4x5 CSR->CSC correct")

    print("All CSC tests passed!")
