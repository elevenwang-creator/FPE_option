from std.testing import assert_true, TestSuite


def test_gpu_trainer_module_importable() raises:
    from engines.nais.gpu_trainer import GPUTrainer
    assert_true(True, "module importable")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
