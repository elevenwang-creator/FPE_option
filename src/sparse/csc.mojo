"""Compressed Sparse Column matrix format."""


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
