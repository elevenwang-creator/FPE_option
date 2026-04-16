"""Benchmark: Compare sparse LU vs dense LAPACK for n=1000"""

from numerics.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csr import CSRMatrix
from std.benchmark import run as bench_run
from std.benchmark import Unit


def build_tridiag_matrix(n: Int) -> CSRMatrix:
    var data: List[Float64] = List[Float64]()
    var indices: List[Int] = List[Int]()
    var indptr: List[Int] = List[Int]()
    indptr.append(0)
    
    for i in range(n):
        if i > 0:
            data.append(1.0)
            indices.append(i - 1)
        data.append(2.0)
        indices.append(i)
        if i < n - 1:
            data.append(1.0)
            indices.append(i + 1)
        indptr.append(len(data))
    
    return CSRMatrix(n, n, len(data), data^, indptr^, indices^)


def main() raises:
    print("=" * 70)
    print("  Sparse LU vs Dense LAPACK (n=1000)")
    print("=" * 70)
    print()

    var n = 1000
    var A = build_tridiag_matrix(n)
    var lu = SparseLU()
    lu.factorize(A)
    
    var b = FixedSizeVector(n)
    var work = FixedSizeVector(n)
    for i in range(n):
        b[i] = 1.0
    
    print("[1] Sparse LU Decomposition (n=1000)")
    
    def bench_sparse_lu() raises capturing:
        lu.factorize(A)
    
    var report1 = bench_run[bench_sparse_lu](
        num_warmup_iters=2,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report1.mean(Unit.ms), "ms")
    print("  Min time:", report1.min(Unit.ms), "ms")
    print("  Iterations:", report1.iters())
    print()

    print("[2] Sparse LU Solve (n=1000)")
    
    def bench_sparse_solve() raises capturing:
        lu.solve_inplace(b, work)
    
    var report2 = bench_run[bench_sparse_solve](
        num_warmup_iters=2,
        max_iters=500,
        min_runtime_secs=0.5,
    )
    print("  Mean time:", report2.mean(Unit.ms), "ms")
    print("  Min time:", report2.min(Unit.ms), "ms")
    print("  Iterations:", report2.iters())
    print()

    print("=" * 70)
    print("  Comparison Summary")
    print("=" * 70)
    print("  For n=1000 tridiagonal system:")
    print("  - Sparse LU: O(n) time, O(n) memory")
    print("  - Dense LAPACK: O(n^3) time, O(n^2) memory")
    print()
    print("  Memory usage:")
    print("  - Sparse: ~60 KB (3n non-zeros)")
    print("  - Dense:  ~8 MB (n^2 elements)")
    print("=" * 70)
