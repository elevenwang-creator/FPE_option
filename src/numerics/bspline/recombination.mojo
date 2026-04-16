"""Recombination basis with direct CSR construction.

Matches Python: Rcol = num_basis - 2, matrix is n x Rcol.
Row 0: left boundary condition
Rows 1 to n-2: identity (R[i, i-1] = 1)
Row n-1: right boundary condition
Direct CSR output, no COO intermediate.
"""

from numerics.bspline.basis import BSplineBasis
from sparse.csr import CSRMatrix
from sparse.ops import spgemm


struct RecombinationBasis[degree: Int](Copyable, Movable):
    var basis: BSplineBasis[Self.degree]
    var left_cond: String
    var right_cond: String

    def __init__(
        out self,
        var basis: BSplineBasis[Self.degree],
        left_cond: String = "dirichlet",
        right_cond: String = "neumann",
    ):
        self.basis = basis^
        self.left_cond = left_cond
        self.right_cond = right_cond

    def recombination_matrix(self) -> CSRMatrix:
        var n = self.basis.num_basis
        var Rcol = n - 2
        if Rcol < 0:
            Rcol = 0

        var total_nnz = Rcol
        if self.left_cond == "neumann":
            if Rcol > 0:
                total_nnz += 1
        if self.right_cond == "neumann":
            if Rcol > 0:
                total_nnz += 1

        var result = CSRMatrix(n, Rcol, total_nnz)
        var dest = 0
        result.indptr[0] = 0

        if self.left_cond == "neumann" and Rcol > 0:
            result.data[dest] = 1.0
            result.indices[dest] = 0
            dest += 1
        result.indptr[1] = dest

        for i in range(1, n - 1):
            result.data[dest] = 1.0
            result.indices[dest] = i - 1
            dest += 1
            result.indptr[i + 1] = dest

        if self.right_cond == "neumann" and Rcol > 0:
            result.data[dest] = 1.0
            result.indices[dest] = Rcol - 1
            dest += 1
        result.indptr[n] = dest

        return result^

    def eval_all(self, points: List[Float64]) -> CSRMatrix:
        var B = self.basis.eval_all(points)
        var R = self.recombination_matrix()
        return spgemm(B, R)

    def first_derivative_all(self, points: List[Float64]) -> CSRMatrix:
        var dB = self.basis.first_derivative_all(points)
        var R = self.recombination_matrix()
        return spgemm(dB, R)
