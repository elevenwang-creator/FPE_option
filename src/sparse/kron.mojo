"""Parallelized Kronecker product: C = A ⊗ B.

Algorithm: row-wise Kronecker with parallelized output row processing.
1. Count phase: parallelize over output rows to compute nnz per row
   - Each output row (i,k) has nnz = nnz(A[i,:]) * nnz(B[k,:])
2. Fill phase: parallelize over output rows with fused vectorize
   - vectorize for SIMD batch multiply a_val * B[k,:] + scatter
     into pre-allocated sequential slots (no marker/accumulate —
     each (i,k,j_a,j_b) produces a unique column index)
3. No sort needed — output is already sorted by construction
   (A columns ascending, B columns ascending within each A column)
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix
from sparse.scratch import ScratchBuffer

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def kron(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    var out_nrows = A.nrows * B.nrows
    var out_ncols = A.ncols * B.ncols

    var a_rp_ptr = A.indptr.unsafe_ptr()
    var a_cols_span = Span(A.indices)
    var a_data_span = Span(A.data)
    var b_rp_ptr = B.indptr.unsafe_ptr()
    var b_cols_span = Span(B.indices)
    var b_data_span = Span(B.data)
    var b_ncols = B.ncols

    # --- Count phase: compute nnz per output row ---
    var row_nnz = ScratchBuffer[Int](out_nrows)

    @parameter
    def count_row(out_i: Int):
        var i = out_i // B.nrows
        var k = out_i % B.nrows
        var a_nnz = a_rp_ptr[i + 1] - a_rp_ptr[i]
        var b_nnz = b_rp_ptr[k + 1] - b_rp_ptr[k]
        row_nnz[out_i] = a_nnz * b_nnz

    for _i in range(out_nrows):
        count_row(_i)

    var total_nnz = 0
    for out_i in range(out_nrows):
        total_nnz += row_nnz[out_i]

    var result = CSRMatrix(out_nrows, out_ncols, total_nnz)
    result.indptr[0] = 0
    for out_i in range(out_nrows):
        result.indptr[out_i + 1] = result.indptr[out_i] + row_nnz[out_i]

    # --- Fill phase ---
    var r_idx_ptr = result.indices.unsafe_ptr()
    var r_data_ptr = result.data.unsafe_ptr()
    var r_rp_ptr = result.indptr.unsafe_ptr()

    @parameter
    def fill_row(out_i: Int):
        var i = out_i // B.nrows
        var k = out_i % B.nrows
        var dest = r_rp_ptr[out_i]

        var a_start = a_rp_ptr[i]
        var a_end = a_rp_ptr[i + 1]
        var b_start = b_rp_ptr[k]
        var b_end = b_rp_ptr[k + 1]
        var b_len = b_end - b_start

        if b_len == 0:
            return

        var b_row_cols = b_cols_span[b_start:b_end]
        var b_row_vals = b_data_span[b_start:b_end]

        for ap in range(a_start, a_end):
            var a_col = a_cols_span[ap]
            var a_val = a_data_span[ap]
            var col_base = a_col * b_ncols

            def vscatter[
                width: Int
            ](p_off: Int) {
                read a_val,
                read b_row_vals,
                read b_row_cols,
                read col_base,
                read r_data_ptr,
                read r_idx_ptr,
                read dest,
            }:
                for w in range(width):
                    r_data_ptr[dest + p_off + w] = a_val * b_row_vals[p_off + w]
                    r_idx_ptr[dest + p_off + w] = (
                        col_base + b_row_cols[p_off + w]
                    )

            vectorize[SIMD_W](b_len, vscatter)
            dest += b_len

    for _i in range(out_nrows):
        fill_row(_i)

    return result^
