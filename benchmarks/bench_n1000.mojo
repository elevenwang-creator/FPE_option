"""Mojo RADAU5 performance benchmark for n=1000.

Compares with Fortran test_radau5_very_large.f
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.utils import FixedSizeVector
from sparse.csr import CSRMatrix
from std.benchmark import run as bench_run
from std.benchmark import Unit


def make_tridiag_csr(n: Int, a: Float64, b: Float64, c: Float64) -> CSRMatrix:
    var nnz = 3 * n - 2
    var result = CSRMatrix(n, n, nnz)
    var idx = 0
    result.indptr[0] = 0
    for i in range(n):
        if i > 0:
            result.data[idx] = a
            result.indices[idx] = i - 1
            idx += 1
        result.data[idx] = b
        result.indices[idx] = i
        idx += 1
        if i < n - 1:
            result.data[idx] = c
            result.indices[idx] = i + 1
            idx += 1
        result.indptr[i + 1] = idx
    return result^


struct TridiagODESystem(LinearODESystem):
    var n: Int
    var alpha: Float64
    var beta: Float64
    var M_cached: CSRMatrix
    var K_cached: CSRMatrix

    def __init__(out self, n: Int, alpha: Float64 = 1.0, beta: Float64 = 0.01):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.M_cached = make_tridiag_csr(n, 1.0, 2.0, 1.0)
        self.K_cached = make_tridiag_csr(n, -beta, alpha + 2.0 * beta, -beta)

    def get_M(self) -> CSRMatrix:
        return self.M_cached.copy()

    def get_K(self) -> CSRMatrix:
        return self.K_cached.copy()


def main() raises:
    print("=" * 70)
    print("  Mojo RADAU5 Benchmark (n=1000, t=0->0.1)")
    print("=" * 70)
    print()

    var sys = TridiagODESystem(1000, 1.0, 0.01)
    var solver = RadauSparseLinearSolver[TridiagODESystem](
        rtol=1e-3, atol=1e-6,
    )
    var y0: List[Float64] = []
    for i in range(1000):
        y0.append(1.0)

    def bench_n1000() raises capturing:
        _ = solver.solve(sys, (0.0, 0.1), y0)

    var report = bench_run[bench_n1000](
        num_warmup_iters=1,
        max_iters=3,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report.mean(Unit.ms), "ms")
    print("  Min time:", report.min(Unit.ms), "ms")
    print("  Iterations:", report.iters())

    var sol = solver.solve(sys, (0.0, 0.1), y0)
    print("  Steps:", len(sol.t) - 1)
    print()

    print("=" * 70)
    print("  Benchmark complete")
    print("=" * 70)
