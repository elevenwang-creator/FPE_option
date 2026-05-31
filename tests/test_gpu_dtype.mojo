"""Tests for GPU data type management module."""
from gpu_utils.dtype import (
    GPU_DTYPE,
    GPU_MAX_N,
    GPU_VEC_LAYOUT,
    METAL_DTYPE,
    METAL_MAX_N,
    CUDA_DTYPE,
    CUDA_MAX_N,
)
from std.sys import has_apple_gpu_accelerator
from std.testing import assert_true, TestSuite


def test_gpu_dtype_is_valid() raises:
    """GPU_DTYPE should be either float32 (Metal) or float64 (CUDA)."""
    var is_valid = (GPU_DTYPE == DType.float32) or (GPU_DTYPE == DType.float64)
    assert_true(is_valid)


def test_gpu_dtype_matches_backend() raises:
    """GPU_DTYPE should match the expected backend dtype."""
    comptime if has_apple_gpu_accelerator():
        assert_true(GPU_DTYPE == METAL_DTYPE)
    else:
        assert_true(GPU_DTYPE == CUDA_DTYPE)


def test_gpu_max_n_matches_backend() raises:
    """GPU_MAX_N should match the expected backend limit."""
    comptime if has_apple_gpu_accelerator():
        assert_true(GPU_MAX_N == METAL_MAX_N)
    else:
        assert_true(GPU_MAX_N == CUDA_MAX_N)


def main() raises:
    print("=" * 60)
    print("GPU Data Type Management Tests")
    print("GPU_DTYPE:", GPU_DTYPE)
    print("GPU_MAX_N:", GPU_MAX_N)
    print("GPU_VEC_LAYOUT: <comptime>")
    print("=" * 60)

    TestSuite.discover_tests[__functions_in_module()]().run()
