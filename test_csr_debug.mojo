from sparse.csr import CSRMatrix

def main():
    var A = CSRMatrix(3, 3, 4)
    A.indptr[0] = 0
    A.indptr[1] = 1
    A.indptr[2] = 2
    A.indptr[3] = 4
    A.data[0] = 1.0
    A.indices[0] = 0
    A.data[1] = 2.0
    A.indices[1] = 1
    A.data[2] = 3.0
    A.indices[2] = 0
    A.data[3] = 4.0
    A.indices[3] = 2

    print("nnz=" + String(A.nnz()))
    print("indptr: " + String(A.indptr[0]) + " " + String(A.indptr[1]) + " " + String(A.indptr[2]) + " " + String(A.indptr[3]))

    for i in range(3):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            print("  row=" + String(i) + " col=" + String(A.indices[p]) + " val=" + String(A.data[p]))

    var n = A.ncols
    var result: List[List[Float64]] = []
    for _ in range(n):
        var row: List[Float64] = []
        for _ in range(n):
            row.append(0.0)
        result.append(row^)

    for row_idx in range(A.nrows):
        var w_val = 1.0
        for p1 in range(A.indptr[row_idx], A.indptr[row_idx + 1]):
            var j1 = A.indices[p1]
            var v1 = A.data[p1]
            for p2 in range(p1, A.indptr[row_idx + 1]):
                var j2 = A.indices[p2]
                var v2 = A.data[p2]
                result[j1][j2] += w_val * v1 * v2
                if j1 != j2:
                    result[j2][j1] += w_val * v1 * v2

    for i in range(n):
        var s = ""
        for j in range(n):
            if j > 0:
                s += ", "
            s += String(result[i][j])
        print(s)
