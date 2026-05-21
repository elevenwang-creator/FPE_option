"""Cross-platform GPU data type and layout management.

Provides backend-specific constants and shared convenience aliases
for automatic type selection at compile time.

Backend configuration:
- Metal (Apple Silicon): Float32, max 64 elements
- CUDA (NVIDIA): Float64, max 256 elements
- HIP (AMD): Float64, max 256 elements
- CPU: Float64, max 64 elements

Usage in GPU files:
from gpu_utils.dtype import GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT

comptime MY_DTYPE = GPU_DTYPE
comptime MY_MAX_N = GPU_MAX_N
comptime MY_VEC_LAYOUT = GPU_VEC_LAYOUT
"""

from gpu_utils.detect import detect_gpu_backend
from layout import Layout
from std.sys import has_apple_gpu_accelerator


comptime METAL_DTYPE = DType.float32
comptime METAL_MAX_N = 64
comptime METAL_MAT_LAYOUT = Layout.row_major(METAL_MAX_N, METAL_MAX_N)
comptime METAL_VEC_LAYOUT = Layout.row_major(METAL_MAX_N)

comptime CUDA_DTYPE = DType.float64
comptime CUDA_MAX_N = 256
comptime CUDA_MAT_LAYOUT = Layout.row_major(CUDA_MAX_N, CUDA_MAX_N)
comptime CUDA_VEC_LAYOUT = Layout.row_major(CUDA_MAX_N)

comptime HIP_DTYPE = DType.float64
comptime HIP_MAX_N = 256
comptime HIP_MAT_LAYOUT = Layout.row_major(HIP_MAX_N, HIP_MAX_N)
comptime HIP_VEC_LAYOUT = Layout.row_major(HIP_MAX_N)

comptime CPU_DTYPE = DType.float64
comptime CPU_MAX_N = 64
comptime CPU_MAT_LAYOUT = Layout.row_major(CPU_MAX_N, CPU_MAX_N)
comptime CPU_VEC_LAYOUT = Layout.row_major(CPU_MAX_N)


def get_compute_dtype() -> DType:
    var backend = detect_gpu_backend()
    if backend == "metal":
        return METAL_DTYPE
    elif backend == "cuda":
        return CUDA_DTYPE
    elif backend == "hip":
        return HIP_DTYPE
    else:
        return CPU_DTYPE


def get_kernel_max_n() -> Int:
    var backend = detect_gpu_backend()
    if backend == "metal":
        return METAL_MAX_N
    elif backend == "cuda":
        return CUDA_MAX_N
    elif backend == "hip":
        return HIP_MAX_N
    else:
        return CPU_MAX_N


def get_mat_layout() -> Layout:
    var backend = detect_gpu_backend()
    if backend == "metal":
        return materialize[METAL_MAT_LAYOUT]()
    elif backend == "cuda":
        return materialize[CUDA_MAT_LAYOUT]()
    elif backend == "hip":
        return materialize[HIP_MAT_LAYOUT]()
    else:
        return materialize[CPU_MAT_LAYOUT]()


def get_vec_layout() -> Layout:
    var backend = detect_gpu_backend()
    if backend == "metal":
        return materialize[METAL_VEC_LAYOUT]()
    elif backend == "cuda":
        return materialize[CUDA_VEC_LAYOUT]()
    elif backend == "hip":
        return materialize[HIP_VEC_LAYOUT]()
    else:
        return materialize[CPU_VEC_LAYOUT]()


def is_float32_backend() -> Bool:
    return get_compute_dtype() == DType.float32


def is_float64_backend() -> Bool:
    return get_compute_dtype() == DType.float64


def get_backend_name() -> String:
    return detect_gpu_backend()


def get_target_accelerator_flag() -> String:
    var api_name = detect_gpu_backend()
    if api_name == "metal":
        return "metal:1"
    else:
        return ""


comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
