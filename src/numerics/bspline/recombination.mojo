"""Recombination basis with direct CSR construction.

Matches Python: Rcol = num_basis - 2, matrix is n x Rcol.
Row 0: left boundary condition
Rows 1 to n-2: identity (R[i, i-1] = 1)
Row n-1: right boundary condition
Direct CSR output, no COO intermediate.
"""

from numerics.bspline.basis import BSplineBasis
from sparse.csr import CSRMatrix


def _build_recombination_matrix(
    n: Int, left_cond: String, right_cond: String
) -> CSRMatrix:
    var Rcol = n - 2
    if Rcol < 0:
        Rcol = 0

    var total_nnz = Rcol
    if left_cond == "neumann":
        if Rcol > 0:
            total_nnz += 1
    if right_cond == "neumann":
        if Rcol > 0:
            total_nnz += 1

    var result = CSRMatrix(n, Rcol, total_nnz)
    var dest = 0
    result.indptr[0] = 0

    if left_cond == "neumann" and Rcol > 0:
        result.data[dest] = 1.0
        result.indices[dest] = 0
        dest += 1
        result.indptr[1] = dest

    for i in range(1, n - 1):
        result.data[dest] = 1.0
        result.indices[dest] = i - 1
        dest += 1
        result.indptr[i + 1] = dest

    if right_cond == "neumann" and Rcol > 0:
        result.data[dest] = 1.0
        result.indices[dest] = Rcol - 1
        dest += 1
        result.indptr[n] = dest

    return result^


struct RecombinationBasis[degree: Int](Movable):
    var basis: BSplineBasis[Self.degree]
    var left_cond: String
    var right_cond: String
    var _R: CSRMatrix

    def __init__(
        out self,
        var basis: BSplineBasis[Self.degree],
        left_cond: String = "dirichlet",
        right_cond: String = "neumann",
    ):
        self.basis = basis^
        self.left_cond = left_cond
        self.right_cond = right_cond
        self._R = _build_recombination_matrix(
            self.basis.num_basis, left_cond, right_cond
        )


def recomb_eval_all(rb: RecombinationBasis, points: List[Float64]) -> CSRMatrix:
    var B = rb.basis.eval_all(points)
    return B @ rb._R.copy()


def recomb_first_derivative_all(
    rb: RecombinationBasis, points: List[Float64]
) -> CSRMatrix:
    var dB = rb.basis.first_derivative_all(points)
    return dB @ rb._R.copy()
