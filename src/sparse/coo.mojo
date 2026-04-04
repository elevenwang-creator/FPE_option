from sparse.csr import CSRMatrix


struct COOMatrix[dtype: DType](Movable):
    var row: List[Int]
    var col: List[Int]
    var data: List[Scalar[Self.dtype]]
    var nrows: Int
    var ncols: Int

    def __init__(out self, nrows: Int, ncols: Int):
        self.row = []
        self.col = []
        self.data = []
        self.nrows = nrows
        self.ncols = ncols

    def append(mut self, r: Int, c: Int, v: Scalar[Self.dtype]):
        if r < 0 or r >= self.nrows:
            return
        if c < 0 or c >= self.ncols:
            return
        if v == 0:
            return
        self.row.append(r)
        self.col.append(c)
        self.data.append(v)

    def to_csr(self) -> CSRMatrix[Self.dtype]:
        var nnz = len(self.data)
        var order: List[Int] = []
        for i in range(nnz):
            order.append(i)

        for i in range(1, nnz):
            var key = order[i]
            var j = i - 1
            while j >= 0:
                var lhs = order[j]
                var left_r = self.row[lhs]
                var left_c = self.col[lhs]
                var right_r = self.row[key]
                var right_c = self.col[key]
                if left_r < right_r or (left_r == right_r and left_c <= right_c):
                    break
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        var out = CSRMatrix[Self.dtype](self.nrows, self.ncols)
        out.indptr[0] = 0
        var current_row = 0
        var has_pending = False
        var pending_r = 0
        var pending_c = 0
        var pending_v: Scalar[Self.dtype] = 0

        for k in range(nnz):
            var idx = order[k]
            var r = self.row[idx]
            var c = self.col[idx]
            var v = self.data[idx]

            if not has_pending:
                pending_r = r
                pending_c = c
                pending_v = v
                has_pending = True
                continue

            if r == pending_r and c == pending_c:
                pending_v += v
                continue

            if pending_v != 0:
                while current_row < pending_r:
                    current_row += 1
                    out.indptr[current_row] = len(out.data)
                out.indices.append(pending_c)
                out.data.append(pending_v)

            pending_r = r
            pending_c = c
            pending_v = v

        if has_pending and pending_v != 0:
            while current_row < pending_r:
                current_row += 1
                out.indptr[current_row] = len(out.data)
            out.indices.append(pending_c)
            out.data.append(pending_v)

        while current_row < self.nrows:
            current_row += 1
            out.indptr[current_row] = len(out.data)

        return out^
