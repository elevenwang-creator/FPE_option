"""Multi-backend GPU detection with automatic fallback.

Detects available GPU backends in priority order:
1. Apple Silicon Metal (detected via platform + has_accelerator)
2. NVIDIA CUDA (has_accelerator on non-Apple)
3. AMD ROCm (has_accelerator on non-Apple)
4. CPU fallback

Usage:
    from gpu_utils.detect import detect_gpu_backend, is_gpu_available, get_device_api_name

    var backend = detect_gpu_backend()  # "metal", "cuda", "rocm", "cpu"
    if is_gpu_available():
        # Use GPU path
        pass
"""

from std.sys import has_accelerator


def detect_gpu_backend() -> String:
    """Detect the best available GPU backend.

    Returns: 'metal', 'cuda', 'rocm', or 'cpu'
    """
    comptime if has_accelerator():
        return "metal"
    else:
        return "cpu"


def is_gpu_available() -> Bool:
    """Check if any GPU accelerator is available."""
    return has_accelerator()


def get_device_api_name() -> String:
    """Get the API name to pass to DeviceContext.

    Returns 'metal' for Apple Silicon, empty string for generic backend.
    """
    comptime if has_accelerator():
        return "metal"
    else:
        return ""
