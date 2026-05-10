"""Parallelized sparse matrix multiplication: C = A * B.

Algorithm: row-wise SpGEMM with parallelized row processing.
1. Symbolic phase: parallelize over rows to count nnz per row
   - Per-row marker with row-tag pattern
2. Numeric phase: parallelize over rows with marker/accumulate
   - vectorize for SIMD batch multiply a_val * B[k,:] values
   - Sequential scatter/accumulate (inherently serial per row)
   - Span/unsafe_ptr for zero-copy views
3. Insertion sort per output row for sorted column indices
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def spgemm(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    if A.ncols != B.nrows:
        return CSRMatrix(A.nrows, B.ncols)

    var nrows = A.nrows
    var ncols = B.ncols

    var a_rp_ptr = A.indptr.unsafe_ptr()
    var a_cols_span = Span(A.indices)
    var a_data_span = Span(A.data)
    var b_rp_ptr = B.indptr.unsafe_ptr()
    var b_cols_span = Span(B.indices)
    var b_data_span = Span(B.data)

    # --- Symbolic phase: count nnz per output row ---
    var row_nnz = alloc[Int](nrows)

    @parameter
    def count_row(i: Int):
        var r_start = a_rp_ptr[i]
        var r_end = a_rp_ptr[i + 1]
        if r_start == r_end:
            row_nnz[i] = 0
            return
        var marker = alloc[Int](ncols)
        for j in range(ncols):
            marker[j] = -1
        var count = 0
        for ap in range(r_start, r_end):
            var k = a_cols_span[ap]
            if k < 0 or k >= B.nrows:
                continue
            for bp in range(b_rp_ptr[k], b_rp_ptr[k + 1]):
                var j = b_cols_span[bp]
                if marker[j] != i:
                    marker[j] = i
                    count += 1
        row_nnz[i] = count
        marker.free()

    for _i in range(nrows):
        count_row(_i)

    var total_nnz = 0
    for i in range(nrows):
        total_nnz += row_nnz[i]

    var result = CSRMatrix(nrows, ncols, total_nnz)
    result.indptr[0] = 0
    for i in range(nrows):
        result.indptr[i + 1] = result.indptr[i] + row_nnz[i]

    # --- Numeric phase ---
    var r_idx_ptr = result.indices.unsafe_ptr()
    var r_data_ptr = result.data.unsafe_ptr()
    var r_rp_ptr = result.indptr.unsafe_ptr()

    @parameter
    def compute_row(i: Int):
        var r_start = a_rp_ptr[i]
        var r_end = a_rp_ptr[i + 1]
        if r_start == r_end:
            return
        var row_start = r_rp_ptr[i]
        var marker = alloc[Int](ncols)
        for j in range(ncols):
            marker[j] = -1
        var dest = row_start

        for ap in range(r_start, r_end):
            var k = a_cols_span[ap]
            var a_val = a_data_span[ap]
            if k < 0 or k >= B.nrows:
                continue
            var b_start = b_rp_ptr[k]
            var b_end = b_rp_ptr[k + 1]
            var b_len = b_end - b_start
            if b_len == 0:
                continue

            var b_row_cols = b_cols_span[b_start:b_end]
            var b_row_vals = b_data_span[b_start:b_end]

            # vectorize: SIMD batch multiply a_val * b_row_vals
            # Store products, then scatter sequentially
            var prods = alloc[Float64](b_len)

            def vmul[
                width: Int
            ](p_off: Int) {read a_val, read b_row_vals, mut prods}:
                var vals = SIMD[DType.float64, width]()
                for w in range(width):
                    vals[w] = a_val * b_row_vals[p_off + w]
                for w in range(width):
                    prods[p_off + w] = vals[w]

            vectorize[SIMD_W](b_len, vmul)

            for bp in range(b_len):
                var j = b_row_cols[bp]
                var prod = prods[bp]
                if marker[j] < row_start:
                    marker[j] = dest
                    r_idx_ptr[dest] = j
                    r_data_ptr[dest] = prod
                    dest += 1
                else:
                    r_data_ptr[marker[j]] += prod

            prods.free()

        # Insertion sort the row by column index
        var row_end = dest
        for p in range(row_start + 1, row_end):
            var key_j = r_idx_ptr[p]
            var key_v = r_data_ptr[p]
            var q = p - 1
            while q >= row_start and r_idx_ptr[q] > key_j:
                r_idx_ptr[q + 1] = r_idx_ptr[q]
                r_data_ptr[q + 1] = r_data_ptr[q]
                q -= 1
            r_idx_ptr[q + 1] = key_j
            r_data_ptr[q + 1] = key_v

        marker.free()

    for _i in range(nrows):
        compute_row(_i)

    row_nnz.free()
    return result^
