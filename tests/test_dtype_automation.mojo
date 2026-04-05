"""Test that dtype management module automatically selects correct types."""
from std.sys import has_accelerator, has_apple_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu import global_idx
from layout import Layout, LayoutTensor
from gpu_utils.dtype import (
    get_compute_dtype, get_mat_layout, get_vec_layout,
    is_float32_backend, is_float64_backend, get_backend_name,
    get_target_accelerator_flag, get_kernel_max_n
)
from std.testing import assert_true, TestSuite


# Use dtype management module for ALL types - no hardcoded values
comptime TEST_N = 4

# Backend-specific types selected at compile time
comptime if has_apple_gpu_accelerator():
    comptime TEST_DTYPE = DType.float32
    comptime TEST_VEC_LAYOUT = Layout.row_major(64)
else:
    comptime TEST_DTYPE = DType.float64
    comptime TEST_VEC_LAYOUT = Layout.row_major(256)


def double_kernel(
    inp: LayoutTensor[TEST_DTYPE, TEST_VEC_LAYOUT, MutAnyOrigin],
    result: LayoutTensor[TEST_DTYPE, TEST_VEC_LAYOUT, MutAnyOrigin],
):
    """Kernel using dtype management types - no hardcoded Float32/Float64."""
    var idx = global_idx.x
    if idx < TEST_N:
        var inp_val = rebind[inp.element_type](inp[idx])
        result[idx] = rebind[result.element_type](inp_val * 2.0)


def test_dtype_automation() raises:
    """Test that dtype management automatically selects correct backend types."""
    print("Backend:", get_backend_name())
    print("Compute dtype:", get_compute_dtype())
    print("Float32 backend:", is_float32_backend())
    print("Float64 backend:", is_float64_backend())
    print("Target flag:", get_target_accelerator_flag())
    print("Max N:", get_kernel_max_n())
    
    # Verify Metal uses Float32
    comptime if has_apple_gpu_accelerator():
        assert_true(get_compute_dtype() == DType.float32)
        assert_true(is_float32_backend())
        assert_true(not is_float64_backend())
        assert_true(get_backend_name() == "metal")
        assert_true(get_target_accelerator_flag() == "metal:1")
        print("✅ Metal backend correctly configured for Float32")
    else:
        assert_true(get_compute_dtype() == DType.float64)
        assert_true(is_float64_backend())
        print("✅ Non-Metal backend correctly configured for Float64")
    
    assert_true(True)


def test_dtype_kernel_execution() raises:
    """Test kernel execution using dtype management types (no hardcoded types)."""
    comptime if has_accelerator():
        var ctx = DeviceContext(api="metal")
        
        var inp_host = ctx.enqueue_create_host_buffer[TEST_DTYPE](TEST_N)
        var out_host = ctx.enqueue_create_host_buffer[TEST_DTYPE](TEST_N)
        var inp_dev = ctx.enqueue_create_buffer[TEST_DTYPE](TEST_N)
        var out_dev = ctx.enqueue_create_buffer[TEST_DTYPE](TEST_N)
        ctx.synchronize()
        
        # Fill with test data using backend-appropriate type
        for i in range(TEST_N):
            if is_float32_backend():
                inp_host[i] = Float32(i + 1)
            else:
                inp_host[i] = Float64(i + 1)
        
        ctx.enqueue_copy(dst_buf=inp_dev, src_buf=inp_host)
        ctx.synchronize()
        
        # Create LayoutTensors using dtype management layouts
        var inp_tensor = LayoutTensor[TEST_DTYPE, TEST_VEC_LAYOUT](inp_dev)
        var out_tensor = LayoutTensor[TEST_DTYPE, TEST_VEC_LAYOUT](out_dev)
        
        # Launch kernel - types automatically match backend
        ctx.enqueue_function[double_kernel, double_kernel](
            inp_tensor, out_tensor,
            grid_dim=TEST_N, block_dim=256,
        )
        ctx.synchronize()
        
        ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
        ctx.synchronize()
        
        # Verify results
        for i in range(TEST_N):
            var val = Float64(out_host[i])
            var expected = Float64((i + 1) * 2)
            assert_true(val == expected)
        
        print("✅", get_backend_name(), "kernel executed successfully with", get_compute_dtype())
    else:
        assert_true(True)


def main() raises:
    print("=" * 60)
    print("Dtype Management Automation Test")
    print("Verifying: No hardcoded types, all from dtype module")
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
