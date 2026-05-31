"""Compressed Sparse Row matrix with direct construction and SIMD spmv.

Key design:
- Direct CSR construction via two-pass algorithm (no COO intermediate)
- SIMD-vectorized spmv via parallelize + vectorize closures
- All matrix ops (spgemm, add, kron, transpose) produce CSR directly
- Move semantics with ownership transfer (^)
- List[Float64] storage with Span zero-copy views in spmv
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span, UnsafePointer
from std.sys import simd_width_of
from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from sparse.add import add as sparse_add
from sparse.scale import scale as sparse_scale
from sparse.spgemm import spgemm as sparse_spgemm

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

    def __add__(self, rhs: Self) -> Self:
        return sparse_add(self, rhs)

    def __matmul__(self, rhs: Self) -> Self:
        return sparse_spgemm(self, rhs)

    def __rmul__(self, alpha: Float64) -> Self:
        return sparse_scale(alpha, self)

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

    def spmv_new(self, x: List[Float64]) -> List[Float64]:
        var y = List[Float64](length=self.nrows, fill=0.0)
        self.spmv(x, y)
        return y^

    def spmv(self, x: List[Float64], mut y: List[Float64]):
        var vals_span = Span(self.data)
        var cols_span = Span(self.indices)
        var x_span = Span(x)
        var y_span = Span[mut=True](y)
        var rp_ptr = self.indptr.unsafe_ptr()

        @parameter
        def process_row(i: Int):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            var nnz_count = r_end - r_start
            if nnz_count == 0:
                y_span[i] = 0
                return
            var row_vals = vals_span[r_start:r_end]
            var row_cols = cols_span[r_start:r_end]
            var dot: Float64 = 0

            def process_nnz[
                width: Int
            ](p_offset: Int) {mut dot, row_vals, row_cols, x_span}:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = row_vals[p_offset + k]
                    x_vals[k] = x_span[row_cols[p_offset + k]]
                dot += (vals * x_vals).reduce_add()

            vectorize[SIMD_W](nnz_count, process_nnz)
            y_span[i] = dot

        for _i in range(self.nrows):
            process_row(_i)

    def spmv(self, x: FixedSizeVector, mut y: FixedSizeVector):
        y.zero_out()
        var vals_span = Span(self.data)
        var cols_span = Span(self.indices)
        var x_ptr = x.ptr()
        var y_ptr = y.ptr()
        var rp_ptr = self.indptr.unsafe_ptr()

        @parameter
        def process_row(i: Int):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            var nnz_count = r_end - r_start
            if nnz_count == 0:
                y_ptr[i] = 0
                return
            var row_vals = vals_span[r_start:r_end]
            var row_cols = cols_span[r_start:r_end]
            var dot: Float64 = 0

            def process_nnz[
                width: Int
            ](p_offset: Int) {mut dot, row_vals, row_cols, x_ptr}:
                var vals = SIMD[DType.float64, width]()
                var x_vals = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = row_vals[p_offset + k]
                    x_vals[k] = x_ptr[row_cols[p_offset + k]]
                dot += (vals * x_vals).reduce_add()

            vectorize[SIMD_W](nnz_count, process_nnz)
            y_ptr[i] = dot

        for _i in range(self.nrows):
            process_row(_i)

    def spmv_triple(
        self,
        x1: List[Float64],
        x2: List[Float64],
        x3: List[Float64],
        mut y1: List[Float64],
        mut y2: List[Float64],
        mut y3: List[Float64],
    ):
        """Fused triple SpMV: iterate matrix once, produce 3 outputs.

        Computes y1 = A*x1, y2 = A*x2, y3 = A*x3 with one pass over
        the CSR structure. Halves memory traffic vs 3 separate spmv.
        """
        var vals_span = Span(self.data)
        var cols_span = Span(self.indices)
        var x1_span = Span(x1)
        var x2_span = Span(x2)
        var x3_span = Span(x3)
        var y1_span = Span[mut=True](y1)
        var y2_span = Span[mut=True](y2)
        var y3_span = Span[mut=True](y3)
        var rp_ptr = self.indptr.unsafe_ptr()

        @parameter
        def process_row(i: Int):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            var nnz_count = r_end - r_start
            if nnz_count == 0:
                y1_span[i] = 0
                y2_span[i] = 0
                y3_span[i] = 0
                return
            var row_vals = vals_span[r_start:r_end]
            var row_cols = cols_span[r_start:r_end]
            var dot1: Float64 = 0
            var dot2: Float64 = 0
            var dot3: Float64 = 0

            def process_nnz[
                width: Int
            ](p_offset: Int) {
                mut dot1,
                mut dot2,
                mut dot3,
                row_vals,
                row_cols,
                x1_span,
                x2_span,
                x3_span,
            }:
                var vals = SIMD[DType.float64, width]()
                var xv1 = SIMD[DType.float64, width]()
                var xv2 = SIMD[DType.float64, width]()
                var xv3 = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = row_vals[p_offset + k]
                    var col = row_cols[p_offset + k]
                    xv1[k] = x1_span[col]
                    xv2[k] = x2_span[col]
                    xv3[k] = x3_span[col]
                var v = vals
                dot1 += (v * xv1).reduce_add()
                dot2 += (v * xv2).reduce_add()
                dot3 += (v * xv3).reduce_add()

            vectorize[SIMD_W](nnz_count, process_nnz)
            y1_span[i] = dot1
            y2_span[i] = dot2
            y3_span[i] = dot3

        for _i in range(self.nrows):
            process_row(_i)

    def spmv_triple(
        self,
        x1: FixedSizeVector,
        x2: FixedSizeVector,
        x3: FixedSizeVector,
        mut y1: FixedSizeVector,
        mut y2: FixedSizeVector,
        mut y3: FixedSizeVector,
    ):
        y1.zero_out()
        y2.zero_out()
        y3.zero_out()
        var vals_span = Span(self.data)
        var cols_span = Span(self.indices)
        var x1_ptr = x1.ptr()
        var x2_ptr = x2.ptr()
        var x3_ptr = x3.ptr()
        var y1_ptr = y1.ptr()
        var y2_ptr = y2.ptr()
        var y3_ptr = y3.ptr()
        var rp_ptr = self.indptr.unsafe_ptr()

        @parameter
        def process_row(i: Int):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            var nnz_count = r_end - r_start
            if nnz_count == 0:
                y1_ptr[i] = 0
                y2_ptr[i] = 0
                y3_ptr[i] = 0
                return
            var row_vals = vals_span[r_start:r_end]
            var row_cols = cols_span[r_start:r_end]
            var dot1: Float64 = 0
            var dot2: Float64 = 0
            var dot3: Float64 = 0

            def process_nnz[
                width: Int
            ](p_offset: Int) {
                mut dot1,
                mut dot2,
                mut dot3,
                row_vals,
                row_cols,
                x1_ptr,
                x2_ptr,
                x3_ptr,
            }:
                var vals = SIMD[DType.float64, width]()
                var xv1 = SIMD[DType.float64, width]()
                var xv2 = SIMD[DType.float64, width]()
                var xv3 = SIMD[DType.float64, width]()
                for k in range(width):
                    vals[k] = row_vals[p_offset + k]
                    var col = row_cols[p_offset + k]
                    xv1[k] = x1_ptr[col]
                    xv2[k] = x2_ptr[col]
                    xv3[k] = x3_ptr[col]
                var v = vals
                dot1 += (v * xv1).reduce_add()
                dot2 += (v * xv2).reduce_add()
                dot3 += (v * xv3).reduce_add()

            vectorize[SIMD_W](nnz_count, process_nnz)
            y1_ptr[i] = dot1
            y2_ptr[i] = dot2
            y3_ptr[i] = dot3

        for _i in range(self.nrows):
            process_row(_i)

    def transpose(self) -> Self:
        var nnz_val = self._nnz
        if nnz_val == 0:
            return Self(self.ncols, self.nrows)

        var ncols = self.ncols
        var nrows = self.nrows
        var cols_span = Span(self.indices)
        var data_span = Span(self.data)
        var rp_ptr = self.indptr.unsafe_ptr()

        var col_count = alloc[Int](ncols)
        for j in range(ncols):
            col_count[j] = 0
        for p in range(nnz_val):
            col_count[cols_span[p]] += 1

        var result = Self(ncols, nrows, nnz_val)
        result.indptr[0] = 0
        for j in range(ncols):
            result.indptr[j + 1] = result.indptr[j] + col_count[j]
            col_count[j] = result.indptr[j]

        var r_data_ptr = result.data.unsafe_ptr()
        var r_idx_ptr = result.indices.unsafe_ptr()

        for i in range(nrows):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            for p in range(r_start, r_end):
                var j = cols_span[p]
                var dest = col_count[j]
                r_idx_ptr[dest] = i
                r_data_ptr[dest] = data_span[p]
                col_count[j] = dest + 1

        col_count.free()
        return result^

    def to_csc(self) -> CSCMatrix:
        var nnz_val = self._nnz
        if nnz_val == 0:
            return CSCMatrix(self.nrows, self.ncols)

        var ncols = self.ncols
        var nrows = self.nrows
        var cols_span = Span(self.indices)
        var data_span = Span(self.data)
        var rp_ptr = self.indptr.unsafe_ptr()

        var col_count = alloc[Int](ncols)
        for j in range(ncols):
            col_count[j] = 0
        for p in range(nnz_val):
            col_count[cols_span[p]] += 1

        var result = CSCMatrix(nrows, ncols, nnz_val)
        result.colptr[0] = 0
        for j in range(ncols):
            result.colptr[j + 1] = result.colptr[j] + col_count[j]
            col_count[j] = result.colptr[j]

        var r_data_ptr = result.data.unsafe_ptr()
        var r_idx_ptr = result.indices.unsafe_ptr()

        for i in range(nrows):
            var r_start = rp_ptr[i]
            var r_end = rp_ptr[i + 1]
            for p in range(r_start, r_end):
                var j = cols_span[p]
                var dest = col_count[j]
                r_idx_ptr[dest] = i
                r_data_ptr[dest] = data_span[p]
                col_count[j] = dest + 1

        col_count.free()
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
