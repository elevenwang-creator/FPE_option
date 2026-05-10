from std.gpu import barrier, block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT

comptime SPARSE_DTYPE = GPU_DTYPE
comptime SPARSE_VEC_LAYOUT = GPU_VEC_LAYOUT


def spmv_kernel(
    data: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    indptr: LayoutTensor[DType.int32, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    x: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    y: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    nrows: Int,
):
    """GPU SpMV: block_idx limits scope, thread_idx distributes rows.

    Architecture:
    grid_dim.x covers total elements via block_dim.x blocks.
    Threads cooperate sequentially or explicitly.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if Int(row) < nrows:
        var r = Int(row)
        var start = Int(indptr[r])
        var end = Int(indptr[r + 1])
        var sum: Scalar[SPARSE_DTYPE] = 0
        for j in range(start, end):
            sum += rebind[Scalar[SPARSE_DTYPE]](data[j]) * rebind[
                Scalar[SPARSE_DTYPE]
            ](x[Int(indices[j])])
        y[r] = rebind[y.element_type](sum)


def batch_spmv_kernel(
    data: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    indptr: LayoutTensor[DType.int32, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    X: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    Y: LayoutTensor[SPARSE_DTYPE, SPARSE_VEC_LAYOUT, MutAnyOrigin],
    nrows: Int,
    batch_size: Int,
):
    """Batch SpMV: one block per batch map, threads cooperate over matrix rows.
    """
    var b = block_idx.x
    var tid = thread_idx.x
    var threads = block_dim.x

    if Int(b) >= batch_size:
        return

    var i = Int(tid)
    while i < nrows:
        var start = Int(indptr[i])
        var end = Int(indptr[i + 1])
        var sum: Scalar[SPARSE_DTYPE] = 0
        for j in range(start, end):
            sum += rebind[Scalar[SPARSE_DTYPE]](data[j]) * rebind[
                Scalar[SPARSE_DTYPE]
            ](X[Int(b), Int(indices[j])])
        Y[Int(b), i] = rebind[Y.element_type](sum)
        i += Int(threads)
