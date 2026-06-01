"""Test GPU vs CPU results match within tolerance."""
from std.testing import assert_true, TestSuite
from engines.fpe.gpu_batch_executor import gpu_batch_solve, _cpu_euler_solve
from sparse.csr import CSRMatrix
from std.sys import has_accelerator


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def test_gpu_vs_cpu_results_match() raises:
    """GPU and CPU solvers should produce similar results."""
    # Create a small test matrix (identity scaled)
    var n = 4
    var mat: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        for j in range(n):
            if i == j:
                row.append(-0.1)  # Small negative diagonal for decay
            else:
                row.append(0.0)
        mat.append(row^)
    
    var csr = CSRMatrix[DType.float64].from_dense(mat)
    var q0: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var t_end = 0.1
    var num_steps = 100
    
    # GPU solve (includes _project_nonnegative which normalizes)
    var gpu_results = gpu_batch_solve(csr, q0, t_end, 1, num_steps)
    
    # CPU solve (need to normalize manually to match GPU)
    var mat_dense = mat.copy()
    var cpu_result = _cpu_euler_solve(mat_dense, q0, t_end, num_steps)
    
    # Normalize CPU result to match GPU's _project_nonnegative
    var cpu_sum = 0.0
    for i in range(n):
        if cpu_result[i] < 0.0:
            cpu_result[i] = 0.0
        cpu_sum += cpu_result[i]
    if cpu_sum > 0.0:
        for i in range(n):
            cpu_result[i] = cpu_result[i] / cpu_sum
    
    # Compare results - print for debugging
    print("GPU results:", end=" ")
    for i in range(n):
        print(gpu_results[0][i], end=" ")
    print()
    print("CPU results:", end=" ")
    for i in range(n):
        print(cpu_result[i], end=" ")
    print()
    
    for i in range(n):
        var gpu_val = gpu_results[0][i]
        var cpu_val = cpu_result[i]
        var diff = _abs(gpu_val - cpu_val)
        print("i=", i, " GPU=", gpu_val, " CPU=", cpu_val, " diff=", diff)
        # Allow tolerance for Float32 vs Float64 and different execution paths
        assert_true(diff < 0.1)


def main() raises:
    print("=" * 60)
    print("GPU vs CPU Comparison Test")
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
