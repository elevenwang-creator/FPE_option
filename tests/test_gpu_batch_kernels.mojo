from std.testing import assert_true, TestSuite


def test_gpu_batch_kernels_importable() raises:
    from engines.fpe.gpu_batch_kernels import batch_euler_step
    assert_true(True, "module importable")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
