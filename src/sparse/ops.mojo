"""Sparse matrix operations with direct CSR construction.

All operations produce CSR directly using UnsafePointer workspace.
No COO intermediate. O(nnz) workspace.
"""

from sparse.csr import CSRMatrix


def spgemm(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    if A.ncols != B.nrows:
        return CSRMatrix(A.nrows, B.ncols)

    var nrows = A.nrows
    var ncols = B.ncols

    var marker = alloc[Int](ncols)
    for j in range(ncols):
        marker[j] = -1

    var row_nnz = alloc[Int](nrows)
    for i in range(nrows):
        var count = 0
        for ap in range(A.indptr[i], A.indptr[i + 1]):
            var k = A.indices[ap]
            if k < 0 or k >= B.nrows:
                continue
            for bp in range(B.indptr[k], B.indptr[k + 1]):
                var j = B.indices[bp]
                if marker[j] != i:
                    marker[j] = i
                    count += 1
        row_nnz[i] = count

    var total_nnz = 0
    for i in range(nrows):
        total_nnz += row_nnz[i]

    var result = CSRMatrix(nrows, ncols, total_nnz)
    result.indptr[0] = 0
    for i in range(nrows):
        result.indptr[i + 1] = result.indptr[i] + row_nnz[i]

    for j in range(ncols):
        marker[j] = -1

    for i in range(nrows):
        var row_start = result.indptr[i]
        var dest = row_start
        for ap in range(A.indptr[i], A.indptr[i + 1]):
            var k = A.indices[ap]
            var a_val = A.data[ap]
            if k < 0 or k >= B.nrows:
                continue
            for bp in range(B.indptr[k], B.indptr[k + 1]):
                var j = B.indices[bp]
                var b_val = B.data[bp]
                if marker[j] < row_start:
                    marker[j] = dest
                    result.indices[dest] = j
                    result.data[dest] = a_val * b_val
                    dest += 1
                else:
                    result.data[marker[j]] += a_val * b_val

        var row_end = dest
        for p in range(row_start + 1, row_end):
            var key_j = result.indices[p]
            var key_v = result.data[p]
            var q = p - 1
            while q >= row_start and result.indices[q] > key_j:
                result.indices[q + 1] = result.indices[q]
                result.data[q + 1] = result.data[q]
                q -= 1
            result.indices[q + 1] = key_j
            result.data[q + 1] = key_v

    marker.free()
    row_nnz.free()
    return result^


def add(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    var row_nnz = alloc[Int](A.nrows)
    for i in range(A.nrows):
        var count = 0
        var a = A.indptr[i]
        var a_end = A.indptr[i + 1]
        var b = B.indptr[i]
        var b_end = B.indptr[i + 1]

        while a < a_end and b < b_end:
            if A.indices[a] < B.indices[b]:
                count += 1
                a += 1
            elif A.indices[a] > B.indices[b]:
                count += 1
                b += 1
            else:
                var s = A.data[a] + B.data[b]
                if s != 0:
                    count += 1
                a += 1
                b += 1
        count += (a_end - a) + (b_end - b)
        row_nnz[i] = count

    var total_nnz = 0
    for i in range(A.nrows):
        total_nnz += row_nnz[i]

    var result = CSRMatrix(A.nrows, A.ncols, total_nnz)
    result.indptr[0] = 0
    for i in range(A.nrows):
        result.indptr[i + 1] = result.indptr[i] + row_nnz[i]
    var dest = 0
    for i in range(A.nrows):
        var a = A.indptr[i]
        var a_end = A.indptr[i + 1]
        var b = B.indptr[i]
        var b_end = B.indptr[i + 1]

        while a < a_end and b < b_end:
            if A.indices[a] < B.indices[b]:
                result.data[dest] = A.data[a]
                result.indices[dest] = A.indices[a]
                dest += 1
                a += 1
            elif A.indices[a] > B.indices[b]:
                result.data[dest] = B.data[b]
                result.indices[dest] = B.indices[b]
                dest += 1
                b += 1
            else:
                var s = A.data[a] + B.data[b]
                if s != 0:
                    result.data[dest] = s
                    result.indices[dest] = A.indices[a]
                    dest += 1
                a += 1
                b += 1

        while a < a_end:
            result.data[dest] = A.data[a]
            result.indices[dest] = A.indices[a]
            dest += 1
            a += 1
        while b < b_end:
            result.data[dest] = B.data[b]
            result.indices[dest] = B.indices[b]
            dest += 1
            b += 1

    row_nnz.free()
    return result^


def scale(alpha: Float64, A: CSRMatrix) -> CSRMatrix:
    var nnz_val = A.nnz()
    var result = CSRMatrix(A.nrows, A.ncols, nnz_val)
    for i in range(A.nrows + 1):
        result.indptr[i] = A.indptr[i]
    for p in range(nnz_val):
        result.data[p] = alpha * A.data[p]
        result.indices[p] = A.indices[p]
    return result^


def diag_scale(A: CSRMatrix, row_scale: List[Float64], col_scale: List[Float64]) -> CSRMatrix:
    var nnz_val = A.nnz()
    var result = CSRMatrix(A.nrows, A.ncols, nnz_val)
    for i in range(A.nrows + 1):
        result.indptr[i] = A.indptr[i]
    for i in range(A.nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            result.data[p] = A.data[p] * row_scale[i] * col_scale[A.indices[p]]
            result.indices[p] = A.indices[p]
    return result^


def kron(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    var total_nnz = 0
    for i in range(A.nrows):
        var a_nnz = A.indptr[i + 1] - A.indptr[i]
        for k in range(B.nrows):
            var b_nnz = B.indptr[k + 1] - B.indptr[k]
            total_nnz += a_nnz * b_nnz

    var out_nrows = A.nrows * B.nrows
    var out_ncols = A.ncols * B.ncols

    var result = CSRMatrix(out_nrows, out_ncols, total_nnz)
    var dest = 0
    result.indptr[0] = 0

    for i in range(A.nrows):
        for k in range(B.nrows):
            for ap in range(A.indptr[i], A.indptr[i + 1]):
                var a_col = A.indices[ap]
                var a_val = A.data[ap]
                for bp in range(B.indptr[k], B.indptr[k + 1]):
                    result.data[dest] = a_val * B.data[bp]
                    result.indices[dest] = a_col * B.ncols + B.indices[bp]
                    dest += 1
            result.indptr[i * B.nrows + k + 1] = dest

    return result^


def sparse_transpose(A: CSRMatrix) -> CSRMatrix:
    return A.transpose()
