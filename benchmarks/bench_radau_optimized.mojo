"""Mojo RADAU5 performance benchmark (optimized version).

Uses std.benchmark.run for accurate timing.
Compares small (n=3), medium (n=10), and large (n=100) systems.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.utils import FixedSizeVector, abs_f64, max_f64, zeros, copy_vec
from sparse.csr import CSRMatrix
from std.math import sqrt, abs, exp
from std.benchmark import run as bench_run
from std.benchmark import Unit


struct DiagonalODESystem(LinearODESystem):
    var n: Int
    var k_data: List[Float64]
    var k_indices: List[Int]
    var k_indptr: List[Int]

    def __init__(out self, n: Int, k_vals: List[Float64]):
        self.n = n
        self.k_data = List[Float64]()
        self.k_indices = List[Int]()
        self.k_indptr = List[Int]()
        self.k_indptr.append(0)
        for i in range(n):
            self.k_data.append(k_vals[i])
            self.k_indices.append(i)
            self.k_indptr.append(i + 1)

    def get_M(self) -> CSRMatrix:
        var data: List[Float64] = []
        var indices: List[Int] = []
        var indptr: List[Int] = [0]
        for i in range(self.n):
            data.append(1.0)
            indices.append(i)
            indptr.append(i + 1)
        return CSRMatrix(self.n, self.n, self.n, data^, indptr^, indices^)

    def get_K(self) -> CSRMatrix:
        var data: List[Float64] = []
        var indices: List[Int] = []
        var indptr: List[Int] = [0]
        for i in range(self.n):
            data.append(self.k_data[i])
            indices.append(i)
            indptr.append(i + 1)
        return CSRMatrix(self.n, self.n, self.n, data^, indptr^, indices^)


struct TridiagODESystem(LinearODESystem):
    var n: Int
    var alpha: Float64
    var beta: Float64

    def __init__(out self, n: Int, alpha: Float64 = 1.0, beta: Float64 = 0.01):
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def get_M(self) -> CSRMatrix:
        var data: List[Float64] = List[Float64]()
        var indices: List[Int] = List[Int]()
        var indptr: List[Int] = List[Int]()
        indptr.append(0)
        
        for i in range(self.n):
            if i > 0:
                data.append(1.0)
                indices.append(i - 1)
            data.append(2.0)
            indices.append(i)
            if i < self.n - 1:
                data.append(1.0)
                indices.append(i + 1)
            indptr.append(len(data))
        
        return CSRMatrix(self.n, self.n, len(data), data^, indptr^, indices^)

    def get_K(self) -> CSRMatrix:
        var data: List[Float64] = List[Float64]()
        var indices: List[Int] = List[Int]()
        var indptr: List[Int] = List[Int]()
        indptr.append(0)
        
        for i in range(self.n):
            if i > 0:
                data.append(-self.beta)
                indices.append(i - 1)
            data.append(self.alpha + 2.0 * self.beta)
            indices.append(i)
            if i < self.n - 1:
                data.append(-self.beta)
                indices.append(i + 1)
            indptr.append(len(data))
        
        return CSRMatrix(self.n, self.n, len(data), data^, indptr^, indices^)


def main() raises:
    print("=" * 70)
    print("  RADAU5 Optimized Performance Benchmark")
    print("=" * 70)
    print()

    print("[1] Small Diagonal System (n=3, t=0->5)")
    var k_vals: List[Float64] = [0.1, 0.5, 2.0]
    var sys_small = DiagonalODESystem(3, k_vals)
    var solver_small = RadauSparseLinearSolver[DiagonalODESystem](
        rtol=1e-6, atol=1e-8,
    )
    var y0_small: List[Float64] = [1.0, 1.0, 1.0]

    def bench_small() raises capturing:
        _ = solver_small.solve(sys_small, (0.0, 5.0), y0_small)

    var report1 = bench_run[bench_small](
        num_warmup_iters=3,
        max_iters=500,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report1.mean(Unit.ms), "ms")
    print("  Min time:", report1.min(Unit.ms), "ms")
    print("  Max time:", report1.max(Unit.ms), "ms")
    print("  Iterations:", report1.iters())

    var sol1 = solver_small.solve(sys_small, (0.0, 5.0), y0_small)
    print("  Steps:", len(sol1.t) - 1)
    var y_idx1 = len(sol1.y) - 1
    print("  y_final:", sol1.y[y_idx1][0], sol1.y[y_idx1][1], sol1.y[y_idx1][2])
    print()

    print("[2] Medium Tridiagonal System (n=10, t=0->1)")
    var sys_med = TridiagODESystem(10, 1.0, 0.01)
    var solver_med = RadauSparseLinearSolver[TridiagODESystem](
        rtol=1e-6, atol=1e-8,
    )
    var y0_med: List[Float64] = []
    for i in range(10):
        y0_med.append(1.0)

    def bench_medium() raises capturing:
        _ = solver_med.solve(sys_med, (0.0, 1.0), y0_med)

    var report2 = bench_run[bench_medium](
        num_warmup_iters=3,
        max_iters=200,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report2.mean(Unit.ms), "ms")
    print("  Min time:", report2.min(Unit.ms), "ms")
    print("  Max time:", report2.max(Unit.ms), "ms")
    print("  Iterations:", report2.iters())

    var sol2 = solver_med.solve(sys_med, (0.0, 1.0), y0_med)
    print("  Steps:", len(sol2.t) - 1)
    print()

    print("[3] Large Tridiagonal System (n=100, t=0->1)")
    var sys_large = TridiagODESystem(100, 1.0, 0.01)
    var solver_large = RadauSparseLinearSolver[TridiagODESystem](
        rtol=1e-6, atol=1e-8,
    )
    var y0_large: List[Float64] = []
    for i in range(100):
        y0_large.append(1.0)

    def bench_large() raises capturing:
        _ = solver_large.solve(sys_large, (0.0, 1.0), y0_large)

    var report3 = bench_run[bench_large](
        num_warmup_iters=2,
        max_iters=50,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report3.mean(Unit.ms), "ms")
    print("  Min time:", report3.min(Unit.ms), "ms")
    print("  Max time:", report3.max(Unit.ms), "ms")
    print("  Iterations:", report3.iters())

    var sol3 = solver_large.solve(sys_large, (0.0, 1.0), y0_large)
    print("  Steps:", len(sol3.t) - 1)
    print()

    print("[4] Very Large Tridiagonal System (n=1000, t=0->0.01)")
    var sys_xlarge = TridiagODESystem(1000, 1.0, 0.01)
    var solver_xlarge = RadauSparseLinearSolver[TridiagODESystem](
        rtol=1e-6, atol=1e-8,
    )
    var y0_xlarge: List[Float64] = []
    for i in range(1000):
        y0_xlarge.append(1.0)

    def bench_xlarge() raises capturing:
        _ = solver_xlarge.solve(sys_xlarge, (0.0, 0.01), y0_xlarge)

    var report4 = bench_run[bench_xlarge](
        num_warmup_iters=1,
        max_iters=2,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report4.mean(Unit.ms), "ms")
    print("  Iterations:", report4.iters())

    var sol4 = solver_xlarge.solve(sys_xlarge, (0.0, 0.01), y0_xlarge)
    print("  Steps:", len(sol4.t) - 1)
    print()

    print("=" * 70)
    print("  Benchmark complete")
    print("=" * 70)
