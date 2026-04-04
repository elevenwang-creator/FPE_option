from std.testing import assert_true, TestSuite
from gpu_utils.detect import detect_gpu_backend, is_gpu_available, get_device_api_name


def test_detect_gpu_backend_returns_valid_string() raises:
    var backend = detect_gpu_backend()
    var valid = (backend == "metal" or backend == "cuda" or
                 backend == "rocm" or backend == "cpu")
    assert_true(valid, "backend should be metal/cuda/rocm/cpu, got: " + backend)


def test_is_gpu_available_returns_bool() raises:
    var result = is_gpu_available()
    assert_true(result == True or result == False, "should return bool")


def test_get_device_api_name_returns_valid_string() raises:
    var api = get_device_api_name()
    var valid = (api == "metal" or api == "")
    assert_true(valid, "api should be 'metal' or '', got: " + api)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
