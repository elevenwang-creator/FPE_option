"""M1 Pro Metal GPU kernel tests.

Tests GPU kernel compilation and execution on Apple Silicon.
Requires: --target-accelerator=metal:1

Run: pixi run gpu-test
"""
from std.sys import has_accelerator, has_apple_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import global_idx
from layout import Layout, LayoutTensor
from std.testing import assert_true, assert_equal, TestSuite


comptime N = 8
comptime vec_layout = Layout.row_major(N)


def double_kernel(
    inp: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
    result: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
):
    """GPU kernel: result[i] = inp[i] * 2.0."""
    var idx = global_idx.x
    if idx < N:
        result[idx] = rebind[result.element_type](inp[idx] * 2.0)


def add_kernel(
    a: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
    b: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
    result: LayoutTensor[DType.float32, vec_layout, MutAnyOrigin],
):
    """GPU kernel: result[i] = a[i] + b[i]."""
    var idx = global_idx.x
    if idx < N:
        result[idx] = rebind[result.element_type](a[idx] + b[idx])


def run_gpu_tests() raises:
    """Run all GPU tests from main() to avoid TestSuite compilation issues."""
    print("=" * 60)
    print("M1 Pro Metal GPU Tests")
    print("Mojo: 0.26.3.0.dev2026040405 | Target: metal:1")
    print("=" * 60)
    
    # Test 1: GPU detection
    print("\n=== Test 1: GPU Detection ===")
    assert_true(has_accelerator(), "should have accelerator")
    assert_true(has_apple_gpu_accelerator(), "should have Apple GPU accelerator")
    print("PASS: GPU detected")
    
    # Test 2: DeviceContext
    print("\n=== Test 2: DeviceContext ===")
    var ctx = DeviceContext(api="metal")
    print("PASS: Metal DeviceContext created")
    
    # Test 3: Buffer roundtrip
    print("\n=== Test 3: Buffer Roundtrip ===")
    var host_buf = ctx.enqueue_create_host_buffer[DType.float32](N)
    var dev_buf = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    
    for i in range(N):
        host_buf[i] = Float32(i + 1)
    
    ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
    ctx.synchronize()
    
    var result_buf = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=result_buf, src_buf=dev_buf)
    ctx.synchronize()
    
    for i in range(N):
        assert_true(Float64(result_buf[i]) == Float64(i + 1))
    print("PASS: Buffer roundtrip correct")
    
    # Test 4: Double kernel
    print("\n=== Test 4: Double Kernel ===")
    var inp_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var inp_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    
    for i in range(N):
        inp_host[i] = Float32(i + 1)
    
    ctx.enqueue_copy(dst_buf=inp_dev, src_buf=inp_host)
    ctx.synchronize()
    
    var inp_tensor = LayoutTensor[DType.float32, vec_layout](inp_dev)
    var out_tensor = LayoutTensor[DType.float32, vec_layout](out_dev)
    
    ctx.enqueue_function[double_kernel, double_kernel](
        inp_tensor, out_tensor,
        grid_dim=N, block_dim=256,
    )
    ctx.synchronize()
    
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()
    
    print("Input:  ", end="")
    for i in range(N):
        print(Float64(inp_host[i]), end=" ")
    print()
    print("Output: ", end="")
    for i in range(N):
        var val = Float64(out_host[i])
        var expected = Float64((i + 1) * 2)
        print(val, end=" ")
        assert_true(val == expected)
    print()
    print("PASS: Double kernel correct")
    
    # Test 5: Add kernel
    print("\n=== Test 5: Add Kernel ===")
    var a_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var b_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var add_out_host = ctx.enqueue_create_host_buffer[DType.float32](N)
    var a_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var b_dev = ctx.enqueue_create_buffer[DType.float32](N)
    var add_out_dev = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.synchronize()
    
    for i in range(N):
        a_host[i] = Float32(i + 1)
        b_host[i] = Float32(i * 10)
    
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.synchronize()
    
    var a_tensor = LayoutTensor[DType.float32, vec_layout](a_dev)
    var b_tensor = LayoutTensor[DType.float32, vec_layout](b_dev)
    var add_out_tensor = LayoutTensor[DType.float32, vec_layout](add_out_dev)
    
    ctx.enqueue_function[add_kernel, add_kernel](
        a_tensor, b_tensor, add_out_tensor,
        grid_dim=N, block_dim=256,
    )
    ctx.synchronize()
    
    ctx.enqueue_copy(dst_buf=add_out_host, src_buf=add_out_dev)
    ctx.synchronize()
    
    print("A: ", end="")
    for i in range(N):
        print(Float64(a_host[i]), end=" ")
    print()
    print("B: ", end="")
    for i in range(N):
        print(Float64(b_host[i]), end=" ")
    print()
    print("A+B: ", end="")
    for i in range(N):
        var val = Float64(add_out_host[i])
        var expected = Float64((i + 1) + i * 10)
        print(val, end=" ")
        assert_true(val == expected)
    print()
    print("PASS: Add kernel correct")
    
    # Test 6: Large buffer
    print("\n=== Test 6: Large Buffer (100MB) ===")
    var n_large = 25_000_000
    var large_buf = ctx.enqueue_create_buffer[DType.float32](n_large)
    ctx.synchronize()
    print("PASS: 100MB buffer allocated")
    
    print("\n" + "=" * 60)
    print("ALL GPU TESTS PASSED")
    print("=" * 60)


def main() raises:
    run_gpu_tests()
