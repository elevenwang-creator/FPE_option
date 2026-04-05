"""Shared GPU host-side utilities for buffer management and kernel launch.

Provides automatic backend detection for true cross-platform deployment.
No hardcoded API strings - all backend selection is automatic.
"""

from gpu_utils.detect import get_device_api_name, is_gpu_available
from std.gpu.host import DeviceContext


def create_device_context() raises -> DeviceContext:
    """Create a DeviceContext with automatic backend detection.

    Automatically selects the correct backend API:
    - Apple Silicon → DeviceContext(api="metal")
    - NVIDIA → DeviceContext(api="cuda")
    - AMD → DeviceContext(api="hip")
    - Generic → DeviceContext() (auto-detect)
    - CPU → DeviceContext() (fallback)

    No hardcoded API strings - fully automatic.
    """
    var api_name = get_device_api_name()
    if api_name != "":
        return DeviceContext(api=api_name)
    else:
        return DeviceContext()
