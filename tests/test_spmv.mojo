from std.algorithm import parallelize
from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of
comptime W = simd_width_of[DType.float64]()
def main() raises:
    var data: List[Float64] = [2.0, 1.0, 3.0, 1.0, 4.0]
    var indices: List[Int] = [0, 2, 1, 3, 4]
    var indptr: List[Int] = [0, 2, 3, 3, 4, 5]
    var x: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var y = List[Float64](length=5, fill=0.0)
    var vals_span = Span(data)
    var cols_span = Span(indices)
    var x_span = Span(x)
    var y_span = Span[mut=True](y)
    var rp_ptr = indptr.unsafe_ptr()
    @parameter
    def process_row(i: Int):
        var r_start = rp_ptr[i]
        var r_end = rp_ptr[i + 1]
        var nnz = r_end - r_start
        if nnz == 0:
            y_span[i] = 0
            return
        var row_vals = vals_span[r_start:r_end]
        var row_cols = cols_span[r_start:r_end]
        var dot: Float64 = 0
        def process_nnz[width: Int](p_offset: Int) {mut dot, row_vals, row_cols, x_span}:
            var vals = SIMD[DType.float64, width]()
            var x_vals = SIMD[DType.float64, width]()
            for k in range(width):
                vals[k] = row_vals[p_offset + k]
                x_vals[k] = x_span[row_cols[p_offset + k]]
            dot += (vals * x_vals).reduce_add()
        vectorize[W](nnz, process_nnz)
        y_span[i] = dot
    parallelize[process_row](5)
    for i in range(5):
        print("y[", i, "] =", y[i])