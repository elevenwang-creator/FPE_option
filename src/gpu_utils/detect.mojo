"""Multi-backend GPU detection with automatic fallback.

Detects available GPU backends in priority order:
1. Apple Silicon Metal (has_apple_gpu_accelerator)
2. NVIDIA CUDA (has_nvidia_gpu_accelerator)
3. AMD ROCm/HIP (has_amd_gpu_accelerator)
4. Generic accelerator
5. CPU fallback

Usage:
    from gpu_utils.detect import detect_gpu_backend, is_gpu_available, get_device_api_name

    var backend = detect_gpu_backend()  # "metal", "cuda", "hip", "generic", "cpu"
    if is_gpu_available():
        # Use GPU path
        pass
"""

from std.sys import (
    has_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
)


def detect_gpu_backend() -> String:
    """Detect the best available GPU backend.

    Returns: 'metal', 'cuda', 'hip', 'generic', or 'cpu'
    """
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


def is_gpu_available() -> Bool:
    """Check if any GPU accelerator is available."""
    return has_accelerator()


def get_device_api_name() -> String:
    """Get the API name to pass to DeviceContext.

    Returns 'metal' for Apple Silicon, 'cuda' for NVIDIA, 'hip' for AMD,
    empty string for generic backend or CPU.
    """
    comptime if has_apple_gpu_accelerator():
        return "metal"
    elif has_nvidia_gpu_accelerator():
        return "cuda"
    elif has_amd_gpu_accelerator():
        return "hip"
    elif has_accelerator():
        return ""  # Generic backend uses auto-detection
    else:
        return ""
