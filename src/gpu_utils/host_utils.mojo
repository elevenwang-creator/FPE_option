"""Shared GPU host-side utilities for buffer management and kernel launch."""

from gpu_utils.detect import get_device_api_name, is_gpu_available
from std.gpu.host import DeviceContext


def create_device_context() raises -> DeviceContext:
    """Create a DeviceContext with the appropriate backend API.

    Uses Metal on Apple Silicon, generic backend otherwise.
    """
    var api_name = get_device_api_name()
    if api_name == "metal":
        return DeviceContext(api="metal")
    else:
        return DeviceContext()
