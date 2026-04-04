from std.gpu import barrier, global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor


def spmv_kernel[
    dtype: DType,
    data_layout: Layout,
    idx_layout: Layout,
    ptr_layout: Layout,
    vec_layout: Layout,
](
    data: LayoutTensor[dtype, data_layout, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, idx_layout, MutAnyOrigin],
    indptr: LayoutTensor[DType.int32, ptr_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, vec_layout, MutAnyOrigin],
    y: LayoutTensor[dtype, vec_layout, MutAnyOrigin],
    nrows: Int,
):
    """GPU SpMV: one thread per row."""
    var row_u = global_idx.x
    if row_u < UInt(nrows):
        var row = Int(row_u)
        var start = Int(indptr[row])
        var end = Int(indptr[row + 1])
        var sum: Scalar[dtype] = 0
        for j in range(start, end):
            sum += rebind[Scalar[dtype]](data[j]) * rebind[Scalar[dtype]](x[Int(indices[j])])
        y[row] = rebind[y.element_type](sum)


def batch_spmv_kernel[
    dtype: DType,
    data_layout: Layout,
    idx_layout: Layout,
    ptr_layout: Layout,
    x_layout: Layout,
    y_layout: Layout,
](
    data: LayoutTensor[dtype, data_layout, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, idx_layout, MutAnyOrigin],
    indptr: LayoutTensor[DType.int32, ptr_layout, MutAnyOrigin],
    X: LayoutTensor[dtype, x_layout, MutAnyOrigin],
    Y: LayoutTensor[dtype, y_layout, MutAnyOrigin],
    nrows: Int,
    batch_size: Int,
):
    """Batch SpMV: one thread per (row, batch) pair."""
    var row_u = global_idx.x
    var b_u = global_idx.y
    if row_u < UInt(nrows) and b_u < UInt(batch_size):
        var row = Int(row_u)
        var b = Int(b_u)
        var start = Int(indptr[row])
        var end = Int(indptr[row + 1])
        var sum: Scalar[dtype] = 0
        for j in range(start, end):
            sum += rebind[Scalar[dtype]](data[j]) * rebind[Scalar[dtype]](X[b, Int(indices[j])])
        Y[b, row] = rebind[Y.element_type](sum)
