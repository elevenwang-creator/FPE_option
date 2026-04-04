from numerics.utils import zeros_3d
from std.testing import assert_true, TestSuite


def test_zeros_3d_creates_correct_shape() raises:
    var result = zeros_3d(2, 3, 4)
    assert_true(len(result) == 2, "outer dimension should be 2")
    assert_true(len(result[0]) == 3, "middle dimension should be 3")
    assert_true(len(result[0][0]) == 4, "inner dimension should be 4")


def test_zeros_3d_all_zeros() raises:
    var result = zeros_3d(2, 2, 2)
    for i in range(len(result)):
        for j in range(len(result[i])):
            for k in range(len(result[i][j])):
                assert_true(result[i][j][k] == 0.0, "all elements should be 0.0")


def test_zeros_3d_independent_slices() raises:
    var result = zeros_3d(2, 2, 2)
    result[0][0][0] = 1.0
    assert_true(result[1][0][0] == 0.0, "slices should be independent")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
