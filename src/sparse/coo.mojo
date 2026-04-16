from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix


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
        if nnz == 0:
            return CSRMatrix[Self.dtype](self.nrows, self.ncols)

        # Build index array and sort by (row, col) using merge sort - O(nnz log nnz)
        var order: List[Int] = []
        for i in range(nnz):
            order.append(i)

        order = self._merge_sort(order)

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

    def _merge_sort(self, order: List[Int]) -> List[Int]:
        """Stable merge sort by (row, col) pair - O(nnz log nnz)."""
        var n = len(order)
        if n <= 1:
            var result: List[Int] = []
            for i in range(len(order)):
                result.append(order[i])
            return result^

        var mid = n // 2
        var left: List[Int] = []
        var right: List[Int] = []
        for i in range(mid):
            left.append(order[i])
        for i in range(mid, n):
            right.append(order[i])

        var sorted_left = self._merge_sort(left)
        var sorted_right = self._merge_sort(right)
        return self._merge(sorted_left, sorted_right)

    def _merge(self, left: List[Int], right: List[Int]) -> List[Int]:
        """Merge two sorted index lists by (row, col)."""
        var result: List[Int] = []
        var i = 0
        var j = 0
        while i < len(left) and j < len(right):
            var li = left[i]
            var ri = right[j]
            if self.row[li] < self.row[ri]:
                result.append(li)
                i += 1
            elif self.row[li] > self.row[ri]:
                result.append(ri)
                j += 1
            else:
                if self.col[li] <= self.col[ri]:
                    result.append(li)
                    i += 1
                else:
                    result.append(ri)
                    j += 1
        while i < len(left):
            result.append(left[i])
            i += 1
        while j < len(right):
            result.append(right[j])
            j += 1
        return result^

    def to_csc(self) -> CSCMatrix[Self.dtype]:
        var nnz = len(self.data)
        if nnz == 0:
            return CSCMatrix[Self.dtype](self.nrows, self.ncols)

        var order: List[Int] = []
        for i in range(nnz):
            order.append(i)
        order = self._sort_by_col(order)

        var out = CSCMatrix[Self.dtype](self.nrows, self.ncols)
        out.indptr[0] = 0
        var current_col = 0

        var k = 0
        while k < nnz:
            var idx = order[k]
            var r = self.row[idx]
            var c = self.col[idx]
            var v = self.data[idx]

            while current_col < c:
                current_col += 1
                out.indptr[current_col] = len(out.data)

            out.row.append(r)
            out.data.append(v)
            k += 1

        while current_col < self.ncols:
            current_col += 1
            out.indptr[current_col] = len(out.data)

        return out^

    def _sort_by_col(ref self, order: List[Int]) -> List[Int]:
        var n = len(order)
        if n <= 1:
            var result: List[Int] = []
            for i in range(n):
                result.append(order[i])
            return result^

        var mid = n // 2
        var left: List[Int] = []
        var right: List[Int] = []
        for i in range(mid):
            left.append(order[i])
        for i in range(mid, n):
            right.append(order[i])

        var sorted_left = self._sort_by_col(left)
        var sorted_right = self._sort_by_col(right)
        return self._merge_by_col(sorted_left, sorted_right)

    def _merge_by_col(ref self, left: List[Int], right: List[Int]) -> List[Int]:
        var result: List[Int] = []
        var i = 0
        var j = 0
        while i < len(left) and j < len(right):
            var li = left[i]
            var ri = right[j]
            if self.col[li] < self.col[ri]:
                result.append(li)
                i += 1
            elif self.col[li] > self.col[ri]:
                result.append(ri)
                j += 1
            else:
                if self.row[li] <= self.row[ri]:
                    result.append(li)
                    i += 1
                else:
                    result.append(ri)
                    j += 1
        while i < len(left):
            result.append(left[i])
            i += 1
        while j < len(right):
            result.append(right[j])
            j += 1
        return result^