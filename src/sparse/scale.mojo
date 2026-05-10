"""Parallelized scalar multiply: C = alpha * A.

Algorithm: parallelize over rows with vectorized alpha * A.data scatter.
1. Count phase: trivial — C has same sparsity as A (nnz per row unchanged)
2. Fill phase: parallelize over rows, vectorize within each row
   - Copy indptr directly (identical structure)
   - Copy column indices directly
   - Vectorized: alpha * A.data[p..p+width] -> C.data[p..p+width]
No sort needed — structure identical to input.
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def scale(alpha: Float64, A: CSRMatrix) -> CSRMatrix:
    var nnz_val = A.nnz()
    if nnz_val == 0:
        return CSRMatrix(A.nrows, A.ncols)

    var nrows = A.nrows

    var result = CSRMatrix(A.nrows, A.ncols, nnz_val)

    for i in range(nrows + 1):
        result.indptr[i] = A.indptr[i]

    var r_data_ptr = result.data.unsafe_ptr()
    var r_idx_ptr = result.indices.unsafe_ptr()
    var a_data_span = Span(A.data)
    var a_cols_span = Span(A.indices)
    var a_rp_ptr = A.indptr.unsafe_ptr()

    @parameter
    def fill_row(i: Int):
        var r_start = a_rp_ptr[i]
        var r_end = a_rp_ptr[i + 1]
        var row_len = r_end - r_start
        if row_len == 0:
            return

        var row_vals = a_data_span[r_start:r_end]
        var row_cols = a_cols_span[r_start:r_end]

        def vscale[
            width: Int
        ](p_off: Int) {
            read alpha,
            read row_vals,
            read row_cols,
            read r_data_ptr,
            read r_idx_ptr,
            read r_start,
        }:
            for w in range(width):
                r_data_ptr[r_start + p_off + w] = alpha * row_vals[p_off + w]
                r_idx_ptr[r_start + p_off + w] = row_cols[p_off + w]

        vectorize[SIMD_W](row_len, vscale)

    for _i in range(nrows):
        fill_row(_i)

    return result^
