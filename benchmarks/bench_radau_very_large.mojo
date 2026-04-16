"""Mojo benchmark with very large matrix (n=1000).

Uses a tridiagonal K matrix (n=1000) for large-scale performance testing.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from sparse.csr import CSRMatrix
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, copy_vec
from std.math import sqrt, abs
from std.benchmark import run as bench_run
from std.benchmark import Unit


struct SimpleLinearSystem(LinearODESystem):
    var M_mat: CSRMatrix
    var K_mat: CSRMatrix

    def __init__(out self, var M: CSRMatrix, var K: CSRMatrix):
        self.M_mat = M^
        self.K_mat = K^

    def get_M(self) -> CSRMatrix:
        return self.M_mat.copy()

    def get_K(self) -> CSRMatrix:
        return self.K_mat.copy()


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


def make_identity_csr(n: Int) -> CSRMatrix:
    var ones: List[Float64] = []
    for _ in range(n):
        ones.append(1.0)
    return make_diag_csr(n, ones)


def make_diag_csr(n: Int, diag_vals: List[Float64]) -> CSRMatrix:
    var result = CSRMatrix(n, n, n)
    result.indptr[0] = 0
    for i in range(n):
        result.data[i] = diag_vals[i]
        result.indices[i] = i
        result.indptr[i + 1] = i + 1
    return result^


def main() raises:
    print("=" * 70)
    print("  Mojo Very Large Matrix Benchmark (n=1000)")
    print("  (Tridiagonal K matrix, complete integration)")
    print("=" * 70)
    print()

    var n = 1000
    var M = make_identity_csr(n)
    var K = make_tridiag_csr(n, -1.0, 2.0, -1.0)
    var y0: List[Float64] = []
    for _ in range(n):
        y0.append(1.0)

    var sys = SimpleLinearSystem(M^, K^)
    var solver = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-6, atol=1e-8, first_step=0.001,
    )

    def bench_very_large_mojo() raises capturing:
        var sol = solver.solve(sys, (0.0, 0.1), y0)
        _ = sol

    print("[1] Very large system (n=1000, t=0→0.1):")
    var report = bench_run[bench_very_large_mojo](
        num_warmup_iters=1,
        max_iters=100,
        min_runtime_secs=0.0,
    )
    print("  Iterations: ", report.iters())
    print("  Total time: ", report.duration(Unit.s), " s")
    print("  Per iteration: ", report.mean(Unit.ms), " ms")
    print("  Throughput: ", 1000.0 / report.mean(Unit.ms), " iters/s")
    print()

    print("=" * 70)
    print("  Very large matrix benchmark complete")
    print("=" * 70)
