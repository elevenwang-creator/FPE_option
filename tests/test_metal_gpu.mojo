"""M1 Pro Metal GPU kernel tests.

Tests GPU kernel compilation and execution on Apple Silicon.
Requires: --target-accelerator=metal:1

Run: pixi run gpu-test
"""
from std.sys import has_accelerator, has_apple_gpu_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite


def test_gpu_detection() raises:
    """Test GPU detection functions."""
    assert_true(has_accelerator(), "should have accelerator")
    assert_true(has_apple_gpu_accelerator(), "should have Apple GPU")


def test_device_context() raises:
    """Test Metal DeviceContext creation."""
    var ctx = DeviceContext(api="metal")
    assert_true(True)


def test_buffer_roundtrip() raises:
    """Test buffer roundtrip on Metal."""
    var ctx = DeviceContext(api="metal")
    comptime N = 8
    
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


def test_large_buffer() raises:
    """Test large buffer allocation (100MB)."""
    var ctx = DeviceContext(api="metal")
    var n_large = 25_000_000
    var large_buf = ctx.enqueue_create_buffer[DType.float32](n_large)
    ctx.synchronize()
    assert_true(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
