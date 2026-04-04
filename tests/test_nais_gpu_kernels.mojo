from std.testing import assert_true, TestSuite


def test_nais_gpu_forward_kernels_importable() raises:
    from engines.nais.gpu_forward_kernels import nais_forward_kernel
    assert_true(True, "module importable")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
