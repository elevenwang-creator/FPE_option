from engines.nais.utils import _generate_brownian_paths
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def test_brownian_paths_not_constant() raises:
    """Brownian paths should not all be 0.5 - they should be random normal."""
    var M = 100
    var N = 10
    var D = 2
    var bm = _generate_brownian_paths(M, N, D)

    # Check that not all values are 0.5
    var all_same = True
    for i in range(M):
        for j in range(N):
            for k in range(D):
                if _abs(bm[i][j][k] - 0.5) > 0.01:
                    all_same = False
    assert_true(not all_same, "Brownian paths should have variation, not all 0.5")


def test_brownian_paths_shape() raises:
    """Brownian paths should have correct dimensions."""
    var M = 50
    var N = 20
    var D = 3
    var bm = _generate_brownian_paths(M, N, D)

    assert_true(len(bm) == M, "outer dimension should be M")
    assert_true(len(bm[0]) == N + 1, "middle dimension should be N+1 (including t=0)")
    assert_true(len(bm[0][0]) == D, "inner dimension should be D")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
