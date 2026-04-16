"""Benchmark RadauSparseLinearSolver performance using Mojo std.benchmark.

Uses Mojo standard library benchmark module as required.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from sparse.csr import CSRMatrix
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, copy_vec
from std.math import sqrt, abs, exp
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


def make_diag_csr(n: Int, diag_vals: List[Float64]) -> CSRMatrix:
    var result = CSRMatrix(n, n, n)
    result.indptr[0] = 0
    for i in range(n):
        result.data[i] = diag_vals[i]
        result.indices[i] = i
        result.indptr[i + 1] = i + 1
    return result^


def make_identity_csr(n: Int) -> CSRMatrix:
    var ones: List[Float64] = []
    for _ in range(n):
        ones.append(1.0)
    return make_diag_csr(n, ones)


def main() raises:
    print("=" * 70)
    print("  RADAU5 Performance Benchmark (using Mojo std.benchmark)")
    print("=" * 70)
    print()

    print("[1] Small system (n=3, t=0→1):")

    var n = 3
    var M = make_identity_csr(n)
    var K_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = make_diag_csr(n, K_diag)
    var y0: List[Float64] = [1.0, 1.0, 1.0]

    var sys = SimpleLinearSystem(M^, K^)
    var solver = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3, atol=1e-6, first_step=0.1,
    )

    def bench_radau_small_capturing() raises capturing:
        var sol = solver.solve(sys, (0.0, 1.0), y0)
        _ = sol

    var report1 = bench_run[bench_radau_small_capturing](
        num_warmup_iters=3,
        max_iters=1000,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report1.mean(Unit.ms), "ms")
    print("  Min time:", report1.min(Unit.ms), "ms")
    print("  Max time:", report1.max(Unit.ms), "ms")
    print("  Throughput:", 1000.0 / report1.mean(Unit.ms), "iters/s")
    print()

    print("=" * 70)
    print("  Benchmark complete")
    print("=" * 70)
