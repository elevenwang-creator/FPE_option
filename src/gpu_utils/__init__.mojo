from gpu_utils.detect import detect_gpu_backend, is_gpu_available, get_device_api_name
from gpu_utils.host_utils import create_device_context
from gpu_utils.dtype import get_compute_dtype, is_float32_backend, is_float64_backend, get_backend_name, get_target_accelerator_flag, GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT
