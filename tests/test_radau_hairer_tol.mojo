"""Quick test with Hairer's default tolerances to see if steps are reasonable.
"""

from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.ode.types import ODESolution
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import csr_to_csc
from sparse.ops import add, scale
from numerics.utils import zeros, abs_f64, copy_vec
from std.math import sqrt, abs, exp


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
    print("  Quick RADAU5 Test with Hairer's Default Tolerances")
    print("  rtol=1e-3, atol=1e-6")
    print("=" * 70)
    print()

    var n = 3
    var M = make_identity_csr(n)
    var K_diag: List[Float64] = [1.0, 2.0, 3.0]
    var K = make_diag_csr(n, K_diag)
    var y0: List[Float64] = [1.0, 1.0, 1.0]

    var sys = SimpleLinearSystem(M^, K^)
    var solver = RadauSparseLinearSolver[SimpleLinearSystem](
        rtol=1e-3, atol=1e-6, first_step=0.1,
    )
    var sol = solver.solve(sys, (0.0, 1.0), y0)

    if not sol.success:
        print("FAILED: " + sol.message)
    else:
        print("SUCCESS!")
        print("Steps taken: " + String(len(sol.t)))
        print()

        var y_final = copy_vec(sol.y[len(sol.y) - 1])
        var exact: List[Float64] = [exp(-1.0), exp(-2.0), exp(-3.0)]
        print("y_final = [" + String(y_final[0]) + ", " + String(y_final[1]) + ", " + String(y_final[2]) + "]")
        print("exact    = [" + String(exact[0]) + ", " + String(exact[1]) + ", " + String(exact[2]) + "]")
        print()

        var max_err = 0.0
        for i in range(n):
            var err = abs_f64(y_final[i] - exact[i])
            if err > max_err:
                max_err = err
        print("Max absolute error: " + String(max_err))

        print()
        print("=" * 70)
