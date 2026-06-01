"""Parallelized sparse matrix addition: C = A + B.

Algorithm: row-wise merge of two CSR matrices with parallelized row processing.
1. Count phase: parallelize over rows to count nnz per output row
   - Two-pointer merge of sorted column indices per row
   - Overlapping columns: sum values, skip if zero
2. Fill phase: parallelize over rows with fused vectorize scatter
   - Two-pointer merge with vectorized copy for A-only or B-only tails
   - Overlap entries: scalar sum (unique column, no accumulation needed)
3. No sort needed — output is already sorted by construction
   (merge of two sorted sequences produces sorted output)
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix
from sparse.scratch import ScratchBuffer

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def add(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    var nrows = A.nrows
    var ncols = A.ncols

    var a_rp_ptr = A.indptr.unsafe_ptr()
    var a_cols_span = Span(A.indices)
    var a_data_span = Span(A.data)
    var b_rp_ptr = B.indptr.unsafe_ptr()
    var b_cols_span = Span(B.indices)
    var b_data_span = Span(B.data)

    # --- Count phase: compute nnz per output row ---
    var row_nnz = ScratchBuffer[Int](nrows)

    @parameter
    def count_row(i: Int):
        var a = a_rp_ptr[i]
        var a_end = a_rp_ptr[i + 1]
        var b = b_rp_ptr[i]
        var b_end = b_rp_ptr[i + 1]
        var count = 0

        while a < a_end and b < b_end:
            if a_cols_span[a] < b_cols_span[b]:
                count += 1
                a += 1
            elif a_cols_span[a] > b_cols_span[b]:
                count += 1
                b += 1
            else:
                var s = a_data_span[a] + b_data_span[b]
                if s != 0:
                    count += 1
                a += 1
                b += 1

        count += (a_end - a) + (b_end - b)
        row_nnz[i] = count

    for _i in range(nrows):
        count_row(_i)

    var total_nnz = 0
    for i in range(nrows):
        total_nnz += row_nnz[i]

    var result = CSRMatrix(nrows, ncols, total_nnz)
    result.indptr[0] = 0
    for i in range(nrows):
        result.indptr[i + 1] = result.indptr[i] + row_nnz[i]

    # --- Fill phase ---
    var r_idx_ptr = result.indices.unsafe_ptr()
    var r_data_ptr = result.data.unsafe_ptr()
    var r_rp_ptr = result.indptr.unsafe_ptr()

    @parameter
    def fill_row(i: Int):
        var a = a_rp_ptr[i]
        var a_end = a_rp_ptr[i + 1]
        var b = b_rp_ptr[i]
        var b_end = b_rp_ptr[i + 1]
        var dest = r_rp_ptr[i]

        while a < a_end and b < b_end:
            if a_cols_span[a] < b_cols_span[b]:
                r_data_ptr[dest] = a_data_span[a]
                r_idx_ptr[dest] = a_cols_span[a]
                dest += 1
                a += 1
            elif a_cols_span[a] > b_cols_span[b]:
                r_data_ptr[dest] = b_data_span[b]
                r_idx_ptr[dest] = b_cols_span[b]
                dest += 1
                b += 1
            else:
                var s = a_data_span[a] + b_data_span[b]
                if s != 0:
                    r_data_ptr[dest] = s
                    r_idx_ptr[dest] = a_cols_span[a]
                    dest += 1
                a += 1
                b += 1

        # A-only tail: vectorized scatter
        var a_tail = a_end - a
        if a_tail > 0:
            var a_tail_cols = a_cols_span[a:a_end]
            var a_tail_vals = a_data_span[a:a_end]

            def vscatter_a[
                width: Int
            ](p_off: Int) {
                read a_tail_vals,
                read a_tail_cols,
                read r_data_ptr,
                read r_idx_ptr,
                read dest,
            }:
                for w in range(width):
                    r_data_ptr[dest + p_off + w] = a_tail_vals[p_off + w]
                    r_idx_ptr[dest + p_off + w] = a_tail_cols[p_off + w]

            vectorize[SIMD_W](a_tail, vscatter_a)
            dest += a_tail

        # B-only tail: vectorized scatter
        var b_tail = b_end - b
        if b_tail > 0:
            var b_tail_cols = b_cols_span[b:b_end]
            var b_tail_vals = b_data_span[b:b_end]

            def vscatter_b[
                width: Int
            ](p_off: Int) {
                read b_tail_vals,
                read b_tail_cols,
                read r_data_ptr,
                read r_idx_ptr,
                read dest,
            }:
                for w in range(width):
                    r_data_ptr[dest + p_off + w] = b_tail_vals[p_off + w]
                    r_idx_ptr[dest + p_off + w] = b_tail_cols[p_off + w]

            vectorize[SIMD_W](b_tail, vscatter_b)
            dest += b_tail

    for _i in range(nrows):
        fill_row(_i)

    return result^
