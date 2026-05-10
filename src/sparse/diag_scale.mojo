"""Parallelized diagonal scaling: C = D_row * A * D_col.

Computes C[i,j] = row_scale[i] * A[i,j] * col_scale[j].
Preserves sparsity pattern — no zeros introduced or removed.

Algorithm: parallelize over rows with vectorized scaling.
1. Count phase: trivial — C has same sparsity as A (nnz per row unchanged)
2. Fill phase: parallelize over rows, vectorize within each row
   - For each nonzero A[i,j]: C[i,j] = row_scale[i] * col_scale[j] * A[i,j]
   - Column indices copied directly
   - Indptr copied directly (identical structure)
No sort needed — structure identical to input.
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def diag_scale(
    A: CSRMatrix, row_scale: List[Float64], col_scale: List[Float64]
) -> CSRMatrix:
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
    var rs_ptr = row_scale.unsafe_ptr()
    var cs_ptr = col_scale.unsafe_ptr()

    @parameter
    def fill_row(i: Int):
        var r_start = a_rp_ptr[i]
        var r_end = a_rp_ptr[i + 1]
        var row_len = r_end - r_start
        if row_len == 0:
            return

        var row_vals = a_data_span[r_start:r_end]
        var row_cols = a_cols_span[r_start:r_end]
        var rs_val = rs_ptr[i]

        def vscale[
            width: Int
        ](p_off: Int) {
            read rs_val,
            read cs_ptr,
            read row_vals,
            read row_cols,
            read r_data_ptr,
            read r_idx_ptr,
            read r_start,
        }:
            for w in range(width):
                var col = row_cols[p_off + w]
                r_data_ptr[r_start + p_off + w] = (
                    rs_val * cs_ptr[col] * row_vals[p_off + w]
                )
                r_idx_ptr[r_start + p_off + w] = col

        vectorize[SIMD_W](row_len, vscale)

    for _i in range(nrows):
        fill_row(_i)

    return result^
