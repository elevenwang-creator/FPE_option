"""Compressed Sparse Row matrix with direct construction and SIMD spmv.

Key design:
- Direct CSR construction via two-pass algorithm (no COO intermediate)
- SIMD-vectorized spmv
- All matrix ops (spgemm, add, kron, transpose) produce CSR directly
- Move semantics with ownership transfer (^)
- List[Float64] storage (UnsafePointer with origin not available in this Mojo version)
"""

from std.sys import simd_width_of
from numerics.utils import FixedSizeVector

comptime SIMD_W: Int = simd_width_of[DType.float64]()


struct CSRMatrix(Movable, Writable):
    var data: List[Float64]
    var indices: List[Int]
    var indptr: List[Int]
    var _nnz: Int
    var nrows: Int
    var ncols: Int

    def __init__(out self, nrows: Int, ncols: Int, nnz: Int = 0):
        self.nrows = nrows
        self.ncols = ncols
        self._nnz = nnz
        self.data = List[Float64]()
        self.indices = List[Int]()
        self.indptr = List[Int]()
        for _ in range(nnz):
            self.data.append(0.0)
            self.indices.append(0)
        for _ in range(nrows + 1):
            self.indptr.append(0)

    def __init__(
        out self,
        nrows: Int,
        ncols: Int,
        nnz: Int,
        var data: List[Float64],
        var indptr: List[Int],
        var indices: List[Int],
    ):
        self.nrows = nrows
        self.ncols = ncols
        self._nnz = nnz
        self.data = data^
        self.indptr = indptr^
        self.indices = indices^

    def nnz(self) -> Int:
        return self._nnz

    def copy(self) -> Self:
        var result = Self(self.nrows, self.ncols, self._nnz)
        for i in range(self._nnz):
            result.data[i] = self.data[i]
            result.indices[i] = self.indices[i]
        for i in range(self.nrows + 1):
            result.indptr[i] = self.indptr[i]
        return result^

    def get(self, row: Int, col: Int) -> Float64:
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            return 0
        for p in range(self.indptr[row], self.indptr[row + 1]):
            if self.indices[p] == col:
                return self.data[p]
        return 0

    def spmv(self, x: List[Float64]) -> List[Float64]:
        comptime width = SIMD_W
        var y: List[Float64] = []
        for _ in range(self.nrows):
            y.append(0)

        if len(x) != self.ncols:
            return y^

        for i in range(self.nrows):
            var row_start = self.indptr[i]
            var row_end = self.indptr[i + 1]
            var acc: Float64 = 0

            var p = row_start
            while p + width <= row_end:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = self.data[p + k]
                    x_vals[k] = x[self.indices[p + k]]
                acc += (vals * x_vals).reduce_add()
                p += width

            while p < row_end:
                acc += self.data[p] * x[self.indices[p]]
                p += 1

            y[i] = acc
        return y^

    def spmv_into(self, x: List[Float64], mut y: List[Float64]):
        comptime width = SIMD_W

        for i in range(self.nrows):
            var row_start = self.indptr[i]
            var row_end = self.indptr[i + 1]
            var acc: Float64 = 0

            var p = row_start
            while p + width <= row_end:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = self.data[p + k]
                    x_vals[k] = x[self.indices[p + k]]
                acc += (vals * x_vals).reduce_add()
                p += width

            while p < row_end:
                acc += self.data[p] * x[self.indices[p]]
                p += 1

            y[i] = acc

    def spmv_inplace_fixed(self, x: List[Float64], mut y: FixedSizeVector):
        comptime width = SIMD_W
        y.zero_out()

        for i in range(self.nrows):
            var row_start = self.indptr[i]
            var row_end = self.indptr[i + 1]
            var acc: Float64 = 0

            var p = row_start
            while p + width <= row_end:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = self.data[p + k]
                    x_vals[k] = x[self.indices[p + k]]
                acc += (vals * x_vals).reduce_add()
                p += width

            while p < row_end:
                acc += self.data[p] * x[self.indices[p]]
                p += 1

            y[i] = acc

    def spmv_fixed(self, x: FixedSizeVector, mut y: FixedSizeVector):
        comptime width = SIMD_W
        y.zero_out()

        for i in range(self.nrows):
            var row_start = self.indptr[i]
            var row_end = self.indptr[i + 1]
            var acc: Float64 = 0

            var p = row_start
            while p + width <= row_end:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = self.data[p + k]
                    x_vals[k] = x[self.indices[p + k]]
                acc += (vals * x_vals).reduce_add()
                p += width

            while p < row_end:
                acc += self.data[p] * x[self.indices[p]]
                p += 1

            y[i] = acc

    def transpose(self) -> Self:
        var nnz_val = self._nnz
        if nnz_val == 0:
            return Self(self.ncols, self.nrows)

        var col_count = alloc[Int](self.ncols)
        for j in range(self.ncols):
            col_count[j] = 0
        for p in range(nnz_val):
            col_count[self.indices[p]] += 1

        var result = Self(self.ncols, self.nrows, nnz_val)
        result.indptr[0] = 0
        for j in range(self.ncols):
            result.indptr[j + 1] = result.indptr[j] + col_count[j]

        var pos = alloc[Int](self.ncols)
        for j in range(self.ncols):
            pos[j] = result.indptr[j]

        for i in range(self.nrows):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                var j = self.indices[p]
                var dest = pos[j]
                result.indices[dest] = i
                result.data[dest] = self.data[p]
                pos[j] = dest + 1

        col_count.free()
        pos.free()
        return result^

    def to_dense(self) -> List[List[Float64]]:
        var dense: List[List[Float64]] = []
        for _ in range(self.nrows):
            var row: List[Float64] = []
            for _ in range(self.ncols):
                row.append(0)
            dense.append(row^)

        for i in range(self.nrows):
            for p in range(self.indptr[i], self.indptr[i + 1]):
                dense[i][self.indices[p]] = self.data[p]
        return dense^

    @staticmethod
    def from_dense(dense: List[List[Float64]]) -> CSRMatrix:
        var nrows = len(dense)
        if nrows == 0:
            return CSRMatrix(0, 0)

        var ncols = len(dense[0])
        var row_counts = alloc[Int](nrows)
        for i in range(nrows):
            row_counts[i] = 0
        for i in range(nrows):
            for j in range(ncols):
                if dense[i][j] != 0:
                    row_counts[i] += 1

        var total_nnz = 0
        for i in range(nrows):
            total_nnz += row_counts[i]

        var result = CSRMatrix(nrows, ncols, total_nnz)
        var offset = 0
        result.indptr[0] = 0
        for i in range(nrows):
            for j in range(ncols):
                if dense[i][j] != 0:
                    result.data[offset] = dense[i][j]
                    result.indices[offset] = j
                    offset += 1
            result.indptr[i + 1] = offset

        row_counts.free()
        return result^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "CSRMatrix(", self.nrows, "x", self.ncols, ", nnz=", self._nnz, ")"
        )
