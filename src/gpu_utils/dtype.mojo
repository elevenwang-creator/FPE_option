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

from layout import Layout
from std.sys import has_apple_gpu_accelerator


comptime METAL_DTYPE = DType.float32
comptime METAL_MAX_N = 64
comptime METAL_VEC_LAYOUT = Layout.row_major(METAL_MAX_N)

comptime CUDA_DTYPE = DType.float64
comptime CUDA_MAX_N = 256
comptime CUDA_VEC_LAYOUT = Layout.row_major(CUDA_MAX_N)

comptime HIP_DTYPE = DType.float64
comptime HIP_MAX_N = 256
comptime HIP_VEC_LAYOUT = Layout.row_major(HIP_MAX_N)

comptime CPU_DTYPE = DType.float64
comptime CPU_MAX_N = 64
comptime CPU_VEC_LAYOUT = Layout.row_major(CPU_MAX_N)

comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
