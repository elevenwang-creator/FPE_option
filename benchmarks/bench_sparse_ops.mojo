"""Benchmark sparse matrix operations."""

from sparse.csr import CSRMatrix
from numerics.utils import zeros
from std.math import sin
from std.python import Python


def _make_sparse_matrix(n: Int, density: Float64) -> CSRMatrix:
    var dense: List[List[Float64]] = []
    for i in range(n):
        var row: List[Float64] = []
        for j in range(n):
            var val = sin(Float64(i * 17 + j * 13)) * 0.1
            if val > (1.0 - density):
                row.append(val)
            else:
                row.append(0.0)
        dense.append(row^)
    return CSRMatrix.from_dense(dense)


def bench_csr_spmv() raises:
    var A = _make_sparse_matrix(100, 0.1)
    var x = zeros(100)
    var y = zeros(100)
    for i in range(100):
        x[i] = Float64(i) * 0.01

    var time_mod = Python.import_module("time")
    var start = time_mod.perf_counter()
    var iterations = 10000
    for _ in range(iterations):
        A.spmv(x, y)
    var end = time_mod.perf_counter()
    var elapsed = Float64(py=end) - Float64(py=start)
    var per_op = elapsed / Float64(iterations) * 1e6

    print("CSR SpMV (100x100, 10% density)")
    print(" Iterations:", iterations)
    print(" Total time:", elapsed, "s")
    print(" Per operation:", per_op, "μs")


def main() raises:
    print("=== Sparse Operations Benchmark ===")
    bench_csr_spmv()
