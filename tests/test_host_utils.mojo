from std.testing import assert_true, TestSuite
from gpu_utils.host_utils import create_device_context


def test_create_device_context_returns_context() raises:
    var ctx = create_device_context()
    assert_true(True, "context created")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
