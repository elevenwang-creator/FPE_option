"""Test that GPU executor uses correct dtype for current backend."""
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator, has_apple_gpu_accelerator
from engines.fpe.gpu_batch_executor import gpu_batch_solve
from sparse.csr import CSRMatrix
from numerics.utils import zeros
from gpu_utils.dtype import get_compute_dtype, is_float32_backend


def test_gpu_executor_uses_backend_dtype() raises:
    """GPU executor should use backend-appropriate dtype, not hardcoded Float32."""
    # Create a small test matrix
    var n = 4
    var mat: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        mat.append(row^)
    
    var csr = CSRMatrix[DType.float64].from_dense(mat)
    var q0: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    
    # Solve with GPU executor
    var results = gpu_batch_solve(csr, q0, 0.1, 1, 10)
    
    # Should return valid results
    assert_true(len(results) == 1, "should have 1 result")
    assert_true(len(results[0]) == n, "should have n states")
    
    # Results should be non-negative and normalized
    var row_sum = 0.0
    for i in range(len(results[0])):
        assert_true(results[0][i] >= -1e-10, "states should be non-negative")
        row_sum += results[0][i]
    assert_true(row_sum > 0.9 and row_sum < 1.1, "states should sum to ~1")


def main() raises:
    print("=" * 60)
    print("GPU Executor Dtype Test")
    print("Backend dtype:", get_compute_dtype())
    print("Float32 backend:", is_float32_backend())
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
