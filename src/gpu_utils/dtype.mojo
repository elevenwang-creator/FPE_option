"""Cross-platform GPU data type and layout management.

Provides backend-specific constants for automatic type selection.
Each GPU file uses these constants with ternary expressions to select
the appropriate types at compile time.

Backend configuration:
- Metal (Apple Silicon): Float32, max 64 elements
- CUDA (NVIDIA): Float64, max 256 elements
- HIP (AMD): Float64, max 256 elements
- CPU: Float64, max 64 elements

Usage in GPU files:
    from gpu_utils.dtype import METAL_MAX_N, METAL_VEC_LAYOUT, CUDA_MAX_N, CUDA_VEC_LAYOUT
    from std.sys import has_apple_gpu_accelerator
    
    comptime GPU_DTYPE = DType.float32 if has_apple_gpu_accelerator() else DType.float64
    comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
    comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
"""

from std.sys import has_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from layout import Layout


# Backend-specific configuration constants - always exported
# GPU files use these with ternary expressions to select types automatically

# Metal (Apple Silicon) - Float32 only, smaller max size
comptime METAL_DTYPE = DType.float32
comptime METAL_MAX_N = 64
comptime METAL_MAT_LAYOUT = Layout.row_major(METAL_MAX_N, METAL_MAX_N)
comptime METAL_VEC_LAYOUT = Layout.row_major(METAL_MAX_N)

# CUDA (NVIDIA) - Float64, larger max size
comptime CUDA_DTYPE = DType.float64
comptime CUDA_MAX_N = 256
comptime CUDA_MAT_LAYOUT = Layout.row_major(CUDA_MAX_N, CUDA_MAX_N)
comptime CUDA_VEC_LAYOUT = Layout.row_major(CUDA_MAX_N)

# HIP (AMD) - Float64, larger max size
comptime HIP_DTYPE = DType.float64
comptime HIP_MAX_N = 256
comptime HIP_MAT_LAYOUT = Layout.row_major(HIP_MAX_N, HIP_MAX_N)
comptime HIP_VEC_LAYOUT = Layout.row_major(HIP_MAX_N)

# CPU - Float64, moderate size
comptime CPU_DTYPE = DType.float64
comptime CPU_MAX_N = 64
comptime CPU_MAT_LAYOUT = Layout.row_major(CPU_MAX_N, CPU_MAX_N)
comptime CPU_VEC_LAYOUT = Layout.row_major(CPU_MAX_N)


# Convenience functions for runtime queries
def get_compute_dtype() -> DType:
    """Get the best compute dtype for the current GPU backend."""
    comptime if has_apple_gpu_accelerator():
        return METAL_DTYPE
    elif has_nvidia_gpu_accelerator():
        return CUDA_DTYPE
    elif has_amd_gpu_accelerator():
        return HIP_DTYPE
    else:
        return CPU_DTYPE


def get_kernel_max_n() -> Int:
    """Get maximum matrix dimension for current backend."""
    comptime if has_apple_gpu_accelerator():
        return METAL_MAX_N
    elif has_nvidia_gpu_accelerator():
        return CUDA_MAX_N
    elif has_amd_gpu_accelerator():
        return HIP_MAX_N
    else:
        return CPU_MAX_N


def get_mat_layout() -> Layout:
    """Get matrix layout for current backend."""
    comptime if has_apple_gpu_accelerator():
        return materialize[METAL_MAT_LAYOUT]()
    elif has_nvidia_gpu_accelerator():
        return materialize[CUDA_MAT_LAYOUT]()
    elif has_amd_gpu_accelerator():
        return materialize[HIP_MAT_LAYOUT]()
    else:
        return materialize[CPU_MAT_LAYOUT]()


def get_vec_layout() -> Layout:
    """Get vector layout for current backend."""
    comptime if has_apple_gpu_accelerator():
        return materialize[METAL_VEC_LAYOUT]()
    elif has_nvidia_gpu_accelerator():
        return materialize[CUDA_VEC_LAYOUT]()
    elif has_amd_gpu_accelerator():
        return materialize[HIP_VEC_LAYOUT]()
    else:
        return materialize[CPU_VEC_LAYOUT]()


def is_float32_backend() -> Bool:
    """Check if current backend requires Float32."""
    return get_compute_dtype() == DType.float32


def is_float64_backend() -> Bool:
    """Check if current backend supports Float64."""
    return get_compute_dtype() == DType.float64


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
    """Get the --target-accelerator flag value for compilation."""
    comptime if has_apple_gpu_accelerator():
        return "metal:1"
    else:
        return ""
