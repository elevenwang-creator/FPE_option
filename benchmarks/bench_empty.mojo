from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime dtype = DType.float32
comptime N = 1024
comptime BLOCK = 256
comptime layout = Layout.row_major(N)

def add_kernel(
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    b: LayoutTensor[dtype, layout, MutAnyOrigin],
    c: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if Int(tid) < size:
        c[tid] = a[tid] + b[tid]

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = LayoutTensor[dtype, layout](a_buf)
    var b = LayoutTensor[dtype, layout](b_buf)
    var c = LayoutTensor[dtype, layout](c_buf)

    ctx.enqueue_function[add_kernel, add_kernel](
        a, b, c, N,
        grid_dim=ceildiv(N, BLOCK),
        block_dim=BLOCK,
    )

    with c_buf.map_to_host() as host:
        var result = LayoutTensor[dtype, layout](host)
        print("First result element:", result[0])

