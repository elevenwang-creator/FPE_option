"""Tests for GPU data type management module."""
from gpu_utils.dtype import (
    get_compute_dtype, get_mat_layout, get_vec_layout,
    is_float32_backend, is_float64_backend, get_backend_name,
    get_target_accelerator_flag, get_kernel_max_n
)
from std.sys import has_accelerator, has_apple_gpu_accelerator
from std.testing import assert_true, TestSuite


def test_get_compute_dtype_returns_valid_dtype() raises:
    """Get_compute_dtype() should return a valid DType."""
    var dtype = get_compute_dtype()
    # Should be either float32 (Metal) or float64 (CUDA/HIP/CPU)
    var is_valid = (dtype == DType.float32) or (dtype == DType.float64)
    assert_true(is_valid)


def test_metal_uses_float32() raises:
    """On Apple Silicon, compute dtype should be Float32."""
    comptime if has_apple_gpu_accelerator():
        var dtype = get_compute_dtype()
        assert_true(dtype == DType.float32)
        assert_true(is_float32_backend())
        assert_true(not is_float64_backend())
    else:
        assert_true(True)


def test_backend_name_is_valid() raises:
    """Get_backend_name() should return a valid backend name."""
    var name = get_backend_name()
    var valid = (name == "metal") or (name == "cuda") or (name == "hip") or (name == "generic") or (name == "cpu")
    assert_true(valid)


def test_target_accelerator_flag() raises:
    """Get_target_accelerator_flag() should return appropriate flag."""
    var flag = get_target_accelerator_flag()
    comptime if has_apple_gpu_accelerator():
        assert_true(flag == "metal:1")
    else:
        assert_true(flag == "")


def test_dtype_consistency() raises:
    """Get_compute_dtype and get_mat_layout should return consistent types."""
    var compute = get_compute_dtype()
    var mat = get_mat_layout()
    var vec = get_vec_layout()
    # Just verify they don't crash
    assert_true(compute == DType.float32 or compute == DType.float64)


def main() raises:
    print("=" * 60)
    print("GPU Data Type Management Tests")
    print("Backend:", get_backend_name())
    print("Compute dtype:", get_compute_dtype())
    print("Float32 backend:", is_float32_backend())
    print("Float64 backend:", is_float64_backend())
    print("Target flag:", get_target_accelerator_flag())
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
