"""Benchmark SparseLU factorize scaling on banded matrices."""

from numerics.utils.sparse_lu import SparseLU
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.time import perf_counter_ns as now


def build_banded_csc(n: Int, bandwidth: Int) raises -> CSCMatrix:
    var A_csr = CSRMatrix(n, n, 0)
    var indptr: List[Int] = [0]
    var indices: List[Int] = []
    var data: List[Float64] = []
    var half_bw = bandwidth // 2
    for i in range(n):
        for j in range(max(0, i - half_bw), min(n, i + half_bw + 1)):
            indices.append(j)
            if j == i:
                data.append(4.0)
            else:
                data.append(-1.0 / Float64(abs(i - j)))
        indptr.append(len(indices))
    A_csr.indptr = indptr^
    A_csr.indices = indices^
    A_csr.data = data^
    A_csr._nnz = len(A_csr.data)
    return A_csr.to_csc()


def main() raises:
    print("=== SparseLU Scaling Benchmark ===")
    print()
    print("n      bw    symbolic_us  numeric_us  ratio")

    for n in [64, 128, 256, 512, 1024]:
        var bw = min(n, max(5, n // 10))
        var A = build_banded_csc(n, bw)

        var lu = SparseLU(n)

        var t0 = now()
        lu.factorize_symbolic(A)
        var t_sym = Float64(now() - t0) / 1e3

        var t1 = now()
        lu.factorize_numeric(A)
        var t_num = Float64(now() - t1) / 1e3

        var ratio = t_sym / max(t_num, 0.001)
        print(n, "   ", bw, "   ", t_sym, "      ", t_num, "      ", ratio)
