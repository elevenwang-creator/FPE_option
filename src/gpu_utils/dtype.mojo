"""Cross-platform GPU data type management.

Provides automatic dtype selection based on backend GPU capabilities:
- Metal (Apple Silicon): Float32 only (Metal doesn't support Float64 kernels)
- CUDA (NVIDIA): Float64 preferred, Float32 for performance
- HIP (AMD): Float64 preferred, Float32 for performance
- CPU: Float64

Usage:
    from gpu_utils.dtype import GpuDType, get_compute_dtype, get_layout_dtype

    # Get the best dtype for current backend
    var dtype = get_compute_dtype()  # DType.float32 on Metal, DType.float64 on CUDA/HIP/CPU

    # Use in generic code
    def my_kernel[T: DType](...):
        ...
"""

from std.sys import has_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator


def get_compute_dtype() -> DType:
    """Get the best compute dtype for the current GPU backend.

    Returns:
        DType.float32 for Metal (Apple Silicon) - Metal doesn't support Float64 kernels
        DType.float64 for CUDA/HIP - full Float64 support
        DType.float64 for CPU - default precision
    """
    comptime if has_apple_gpu_accelerator():
        return DType.float32
    else:
        return DType.float64


def get_layout_dtype() -> DType:
    """Get the dtype for LayoutTensor operations.

    Same as get_compute_dtype() - ensures consistency between
    kernel computation and memory layout.
    """
    return get_compute_dtype()


def is_float32_backend() -> Bool:
    """Check if current backend requires Float32."""
    comptime if has_apple_gpu_accelerator():
        return True
    else:
        return False


def is_float64_backend() -> Bool:
    """Check if current backend supports Float64."""
    return not is_float32_backend()


def get_backend_name() -> String:
    """Get the name of the current GPU backend."""
    comptime if has_apple_gpu_accelerator():
        return "metal"
    elif has_nvidia_gpu_accelerator():
        return "cuda"
    elif has_amd_gpu_accelerator():
        return "hip"
    elif has_accelerator():
        return "generic"
    else:
        return "cpu"


def get_target_accelerator_flag() -> String:
    """Get the --target-accelerator flag value for compilation.

    Returns:
        "metal:1" for M1
        "metal:2" for M2
        "metal:3" for M3
        "metal:4" for M4
        "" for non-Apple GPU (use auto-detection)
    """
    comptime if has_apple_gpu_accelerator():
        # M1 Pro/Max = metal:1
        # For newer chips, we'd need runtime detection
        return "metal:1"
    else:
        return ""
