"""Cross-platform GPU data type and layout management.

Provides automatic dtype and layout selection based on backend GPU capabilities.
Uses parametric kernels for true "write once, deploy anywhere" - the same source
code compiles to different backends with appropriate types at compile time.

Backend dtype selection:
- Metal (Apple Silicon): Float32 only (Metal doesn't support Float64 kernels)
- CUDA (NVIDIA): Float64 preferred, Float32 for performance
- HIP (AMD): Float64 preferred, Float32 for performance  
- CPU: Float64

Layout management:
- Comptime layouts required by Mojo GPU model
- Backend-specific max sizes for memory efficiency
- Automatic layout generation from backend capabilities

Usage:
    from gpu_utils.dtype import GpuConfig
    
    # Get backend configuration
    var config = GpuConfig()
    # config.dtype, config.mat_layout, config.vec_layout are all set correctly
    
    # Or use convenience functions
    var dtype = get_compute_dtype()  # DType value
"""

from std.sys import has_accelerator, has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from layout import Layout


# Backend-specific configuration - ALL comptime for kernel compatibility
# These constants are exported for use in kernel files
comptime METAL_MAX_N = 64
comptime METAL_MAT_LAYOUT = Layout.row_major(METAL_MAX_N, METAL_MAX_N)
comptime METAL_VEC_LAYOUT = Layout.row_major(METAL_MAX_N)

comptime CUDA_MAX_N = 256
comptime CUDA_MAT_LAYOUT = Layout.row_major(CUDA_MAX_N, CUDA_MAX_N)
comptime CUDA_VEC_LAYOUT = Layout.row_major(CUDA_MAX_N)

comptime HIP_MAX_N = 256
comptime HIP_MAT_LAYOUT = Layout.row_major(HIP_MAX_N, HIP_MAX_N)
comptime HIP_VEC_LAYOUT = Layout.row_major(HIP_MAX_N)

comptime CPU_MAX_N = 64
comptime CPU_MAT_LAYOUT = Layout.row_major(CPU_MAX_N, CPU_MAX_N)
comptime CPU_VEC_LAYOUT = Layout.row_major(CPU_MAX_N)


# Backend-specific dtype and layouts - selected at compile time
# These are exported for direct use in kernels
comptime if has_apple_gpu_accelerator():
    comptime GPU_DTYPE = DType.float32
    comptime GPU_MAT_LAYOUT = METAL_MAT_LAYOUT
    comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT
    comptime GPU_MAX_N = METAL_MAX_N
elif has_nvidia_gpu_accelerator():
    comptime GPU_DTYPE = DType.float64
    comptime GPU_MAT_LAYOUT = CUDA_MAT_LAYOUT
    comptime GPU_VEC_LAYOUT = CUDA_VEC_LAYOUT
    comptime GPU_MAX_N = CUDA_MAX_N
elif has_amd_gpu_accelerator():
    comptime GPU_DTYPE = DType.float64
    comptime GPU_MAT_LAYOUT = HIP_MAT_LAYOUT
    comptime GPU_VEC_LAYOUT = HIP_VEC_LAYOUT
    comptime GPU_MAX_N = HIP_MAX_N
else:
    comptime GPU_DTYPE = DType.float64
    comptime GPU_MAT_LAYOUT = CPU_MAT_LAYOUT
    comptime GPU_VEC_LAYOUT = CPU_VEC_LAYOUT
    comptime GPU_MAX_N = CPU_MAX_N


def get_compute_dtype() -> DType:
    """Get the best compute dtype for the current GPU backend."""
    return GPU_DTYPE


def get_kernel_max_n() -> Int:
    """Get maximum matrix dimension for current backend."""
    return GPU_MAX_N


def get_mat_layout() -> Layout:
    """Get matrix layout for current backend."""
    return GPU_MAT_LAYOUT


def get_vec_layout() -> Layout:
    """Get vector layout for current backend."""
    return GPU_VEC_LAYOUT


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
    """Get the --target-accelerator flag value for compilation."""
    comptime if has_apple_gpu_accelerator():
        return "metal:1"
    else:
        return ""
