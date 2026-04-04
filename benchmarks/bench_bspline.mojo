"""Benchmark BSpline basis evaluation."""

from numerics.bspline.knots import GenerateKnots
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis
from numerics.bspline.tensor_product import TensorProductBasis
from std.python import Python


def bench_bspline_eval() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    var basis = BSplineBasis[2](knots^)

    var time_mod = Python.import_module("time")
    var start = time_mod.perf_counter()
    var iterations = 100000
    var total = 0.0
    for _ in range(iterations):
        for i in range(basis.num_basis):
            total += basis.de_boor_cox(0.5, i)
    var end = time_mod.perf_counter()
    var elapsed = Float64(py=end) - Float64(py=start)
    var per_eval = elapsed / Float64(iterations * basis.num_basis) * 1e6

    print("BSpline Basis Evaluation (degree 2, 6 basis functions)")
    print("  Iterations:", iterations)
    print("  Total evaluations:", iterations * basis.num_basis)
    print("  Total time:", elapsed, "s")
    print("  Per evaluation:", per_eval, "μs")
    print("  Sum at 0.5:", total / Float64(iterations))


def main() raises:
    print("=== BSpline Benchmark ===")
    bench_bspline_eval()
