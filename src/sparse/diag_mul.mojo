"""Diagonal-sparse multiply: D @ S and S @ D without spgemm.

When D is a diagonal matrix (stored as CSR with 1 nnz per row),
D @ S just scales each row of S by the corresponding diagonal value,
and S @ D just scales each column of S by the corresponding diagonal value.

Both are O(nnz) operations — no marker array needed.
"""

from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
from sparse.csr import CSRMatrix

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def diag_row_scale(D: CSRMatrix, S: CSRMatrix) -> CSRMatrix:
    """Compute D @ S where D is diagonal CSR.

    Scales each row i of S by D's diagonal value at row i.
    If D has no entry at row i (zero diagonal), the row is zeroed.
    Output has same sparsity pattern as S.
    """
    var nnz_val = S.nnz()
    if nnz_val == 0:
        return CSRMatrix(S.nrows, S.ncols)

    var n_diag = D.nrows
    var diag_vals = alloc[Float64](n_diag)
    for j in range(n_diag):
        var d_start = D.indptr[j]
        var d_end = D.indptr[j + 1]
        if d_start < d_end and D.indices[d_start] == j:
            diag_vals[j] = D.data[d_start]
        else:
            diag_vals[j] = 0.0

    var result = CSRMatrix(S.nrows, S.ncols, nnz_val)
    for i in range(S.nrows + 1):
        result.indptr[i] = S.indptr[i]

    var s_data_span = Span(S.data)
    var s_cols_span = Span(S.indices)
    var s_rp_ptr = S.indptr.unsafe_ptr()
    var r_data_ptr = result.data.unsafe_ptr()
    var r_idx_ptr = result.indices.unsafe_ptr()

    @parameter
    def fill_row(i: Int):
        var r_start = s_rp_ptr[i]
        var r_end = s_rp_ptr[i + 1]
        var row_len = r_end - r_start
        if row_len == 0:
            return

        var d_val = diag_vals[i]

        var row_vals = s_data_span[r_start:r_end]
        var row_cols = s_cols_span[r_start:r_end]

        def vscale[
            width: Int
        ](p_off: Int) {
            read d_val,
            read row_vals,
            read row_cols,
            read r_data_ptr,
            read r_idx_ptr,
            read r_start,
        }:
            for w in range(width):
                r_data_ptr[r_start + p_off + w] = d_val * row_vals[p_off + w]
                r_idx_ptr[r_start + p_off + w] = row_cols[p_off + w]

        vectorize[SIMD_W](row_len, vscale)

    for _i in range(S.nrows):
        fill_row(_i)

    diag_vals.free()
    return result^


def diag_col_scale(S: CSRMatrix, D: CSRMatrix) -> CSRMatrix:
    """Compute S @ D where D is diagonal CSR.

    Scales each column j of S by D's diagonal value at row j.
    If D has no entry at row j (zero diagonal), the column is zeroed.
    Output has same sparsity pattern as S.
    """
    var nnz_val = S.nnz()
    if nnz_val == 0:
        return CSRMatrix(S.nrows, S.ncols)

    var n_diag = D.nrows
    var diag_vals = alloc[Float64](n_diag)
    for j in range(n_diag):
        var d_start = D.indptr[j]
        var d_end = D.indptr[j + 1]
        if d_start < d_end and D.indices[d_start] == j:
            diag_vals[j] = D.data[d_start]
        else:
            diag_vals[j] = 0.0

    var result = CSRMatrix(S.nrows, S.ncols, nnz_val)
    for i in range(S.nrows + 1):
        result.indptr[i] = S.indptr[i]

    var s_data_span = Span(S.data)
    var s_cols_span = Span(S.indices)
    var s_rp_ptr = S.indptr.unsafe_ptr()
    var r_data_ptr = result.data.unsafe_ptr()
    var r_idx_ptr = result.indices.unsafe_ptr()

    @parameter
    def fill_row(i: Int):
        var r_start = s_rp_ptr[i]
        var r_end = s_rp_ptr[i + 1]
        var row_len = r_end - r_start
        if row_len == 0:
            return

        var row_vals = s_data_span[r_start:r_end]
        var row_cols = s_cols_span[r_start:r_end]

        def vscale[
            width: Int
        ](p_off: Int) {
            read diag_vals,
            read row_vals,
            read row_cols,
            read r_data_ptr,
            read r_idx_ptr,
            read r_start,
        }:
            for w in range(width):
                var col = row_cols[p_off + w]
                r_data_ptr[r_start + p_off + w] = (
                    row_vals[p_off + w] * diag_vals[col]
                )
                r_idx_ptr[r_start + p_off + w] = col

        vectorize[SIMD_W](row_len, vscale)

    for _i in range(S.nrows):
        fill_row(_i)

    diag_vals.free()
    return result^


def diag_diag_mul(A: CSRMatrix, B: CSRMatrix) -> CSRMatrix:
    """Compute A @ B where both A and B are diagonal CSR.

    Element-wise multiply of diagonal values.
    Output is diagonal CSR (only nonzero entries stored).
    """
    var n = A.nrows
    if n == 0:
        return CSRMatrix(0, 0)

    var a_vals = alloc[Float64](n)
    for i in range(n):
        var d_start = A.indptr[i]
        var d_end = A.indptr[i + 1]
        if d_start < d_end and A.indices[d_start] == i:
            a_vals[i] = A.data[d_start]
        else:
            a_vals[i] = 0.0

    var b_vals = alloc[Float64](n)
    for i in range(n):
        var d_start = B.indptr[i]
        var d_end = B.indptr[i + 1]
        if d_start < d_end and B.indices[d_start] == i:
            b_vals[i] = B.data[d_start]
        else:
            b_vals[i] = 0.0

    var nnz_count = 0
    for i in range(n):
        if a_vals[i] * b_vals[i] != 0.0:
            nnz_count += 1

    var result = CSRMatrix(n, n, nnz_count)
    result.indptr[0] = 0
    var dest = 0
    for i in range(n):
        var prod = a_vals[i] * b_vals[i]
        if prod != 0.0:
            result.data[dest] = prod
            result.indices[dest] = i
            dest += 1
        result.indptr[i + 1] = dest

    a_vals.free()
    b_vals.free()
    return result^
