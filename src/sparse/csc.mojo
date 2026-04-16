"""Compressed Sparse Column matrix with direct construction from CSR.

Direct CSC construction using UnsafePointer workspace.
No COO intermediate, no implicit copy errors.
"""

from sparse.csr import CSRMatrix


struct CSCMatrix(Movable):
    var data: List[Float64]
    var indices: List[Int]
    var colptr: List[Int]
    var _nnz: Int
    var nrows: Int
    var ncols: Int

    def __init__(out self, nrows: Int, ncols: Int, nnz: Int = 0):
        self.nrows = nrows
        self.ncols = ncols
        self._nnz = nnz
        self.data = List[Float64]()
        self.indices = List[Int]()
        self.colptr = List[Int]()
        for _ in range(nnz if nnz > 0 else 1):
            self.data.append(0.0)
            self.indices.append(0)
        for _ in range(ncols + 1):
            self.colptr.append(0)

    def nnz(self) -> Int:
        return self._nnz


def csr_to_csc(A: CSRMatrix) -> CSCMatrix:
    var nnz_val = A.nnz()
    if nnz_val == 0:
        return CSCMatrix(A.nrows, A.ncols)

    var col_count = alloc[Int](A.ncols)
    for j in range(A.ncols):
        col_count[j] = 0
    for p in range(nnz_val):
        col_count[A.indices[p]] += 1

    var result = CSCMatrix(A.nrows, A.ncols, nnz_val)
    result.colptr[0] = 0
    for j in range(A.ncols):
        result.colptr[j + 1] = result.colptr[j] + col_count[j]

    var pos = alloc[Int](A.ncols)
    for j in range(A.ncols):
        pos[j] = result.colptr[j]

    for i in range(A.nrows):
        for p in range(A.indptr[i], A.indptr[i + 1]):
            var j = A.indices[p]
            var dest = pos[j]
            result.indices[dest] = i
            result.data[dest] = A.data[p]
            pos[j] = dest + 1

    col_count.free()
    pos.free()
    return result^
