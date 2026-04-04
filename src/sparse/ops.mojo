from sparse.coo import COOMatrix
from sparse.csr import CSRMatrix


def kron[dtype: DType](A: CSRMatrix[dtype], B: CSRMatrix[dtype]) -> CSRMatrix[dtype]:
    var out = COOMatrix[dtype](A.nrows * B.nrows, A.ncols * B.ncols)

    for i in range(A.nrows):
        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        for ap in range(a_start, a_end):
            var a_col = A.indices[ap]
            var a_val = A.data[ap]
            for k in range(B.nrows):
                var b_start = B.indptr[k]
                var b_end = B.indptr[k + 1]
                for bp in range(b_start, b_end):
                    var c_row = i * B.nrows + k
                    var c_col = a_col * B.ncols + B.indices[bp]
                    out.append(c_row, c_col, a_val * B.data[bp])

    return out.to_csr()


def spgemm[dtype: DType](A: CSRMatrix[dtype], B: CSRMatrix[dtype]) -> CSRMatrix[dtype]:
    if A.ncols != B.nrows:
        return CSRMatrix[dtype](A.nrows, B.ncols)

    var out = COOMatrix[dtype](A.nrows, B.ncols)

    var row_acc: List[Scalar[dtype]] = []
    for _ in range(B.ncols):
        row_acc.append(0)

    for i in range(A.nrows):
        var touched: List[Int] = []

        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        for ap in range(a_start, a_end):
            var k = A.indices[ap]
            var a_val = A.data[ap]
            if k < 0 or k >= B.nrows:
                continue

            var b_start = B.indptr[k]
            var b_end = B.indptr[k + 1]
            for bp in range(b_start, b_end):
                var j = B.indices[bp]
                if j < 0 or j >= B.ncols:
                    continue

                if row_acc[j] == 0:
                    touched.append(j)
                row_acc[j] += a_val * B.data[bp]

        for t in range(len(touched)):
            var j = touched[t]
            var v = row_acc[j]
            if v != 0:
                out.append(i, j, v)
            row_acc[j] = 0

    return out.to_csr()


def spmm[dtype: DType](A: CSRMatrix[dtype], D: List[List[Scalar[dtype]]]) -> List[List[Scalar[dtype]]]:
    var out: List[List[Scalar[dtype]]] = []
    var dense_rows = len(D)
    if dense_rows != A.ncols:
        return out^

    var p = 0
    if dense_rows > 0:
        p = len(D[0])

    for _ in range(A.nrows):
        var row: List[Scalar[dtype]] = []
        for _ in range(p):
            row.append(0)
        out.append(row^)

    for i in range(A.nrows):
        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        for ap in range(a_start, a_end):
            var k = A.indices[ap]
            if k < 0 or k >= dense_rows:
                continue

            var a_val = A.data[ap]
            var width = len(D[k])
            if width > p:
                width = p
            for j in range(width):
                out[i][j] += a_val * D[k][j]

    return out^


def add[dtype: DType](
    A: CSRMatrix[dtype], B: CSRMatrix[dtype]
) -> CSRMatrix[dtype]:
    """Sparse add: C = A + B. O(nnz_A + nnz_B) via merge of sorted indices.

    Eliminates the O(n²) dense round-trip previously used in galerkin.mojo.
    """
    var coo = COOMatrix[dtype](A.nrows, A.ncols)

    for i in range(A.nrows):
        var a_start = A.indptr[i]
        var a_end = A.indptr[i + 1]
        var b_start = B.indptr[i]
        var b_end = B.indptr[i + 1]

        var a = a_start
        var b = b_start

        # Merge-sort style: both index lists are sorted within each row
        while a < a_end and b < b_end:
            if A.indices[a] < B.indices[b]:
                coo.append(i, A.indices[a], A.data[a])
                a += 1
            elif A.indices[a] > B.indices[b]:
                coo.append(i, B.indices[b], B.data[b])
                b += 1
            else:
                # Same column: add values
                var summed = A.data[a] + B.data[b]
                if summed != 0:
                    coo.append(i, A.indices[a], summed)
                a += 1
                b += 1

        # Flush remaining entries
        while a < a_end:
            coo.append(i, A.indices[a], A.data[a])
            a += 1
        while b < b_end:
            coo.append(i, B.indices[b], B.data[b])
            b += 1

    return coo.to_csr()


def scale[dtype: DType](
    alpha: Scalar[dtype], A: CSRMatrix[dtype]
) -> CSRMatrix[dtype]:
    """Sparse scale: B = α·A. O(nnz) — operates directly on data values.

    Eliminates O(n²) dense round-trip previously used in galerkin.mojo.
    """
    var out = CSRMatrix[dtype](A.nrows, A.ncols)
    # Copy structure
    out.indptr = A.indptr.copy()
    out.indices = A.indices.copy()
    # Scale data values
    out.data = []
    for p in range(len(A.data)):
        out.data.append(alpha * A.data[p])
    return out^


def sparse_transpose[dtype: DType](
    A: CSRMatrix[dtype],
) -> CSRMatrix[dtype]:
    """Sparse transpose: A^T. O(nnz) without dense round-trip.

    Builds COO in transposed order, then converts to CSR.
    """
    return A.transpose()
