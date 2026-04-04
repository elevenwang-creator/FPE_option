from numerics.bspline.basis import BSplineBasis
from sparse.coo import COOMatrix
from sparse.csr import CSRMatrix
from sparse.ops import spgemm


@align(64)
struct RecombinationBasis[degree: Int](Copyable, Movable):
    var basis: BSplineBasis[Self.degree]
    var left_cond: String
    var right_cond: String

    def __init__(
        out self,
        var basis: BSplineBasis[Self.degree],
        left_cond: String = "dirichlet",
        right_cond: String = "newmann",
    ):
        self.basis = basis^
        self.left_cond = left_cond
        self.right_cond = right_cond

    def recombination_matrix(self) -> CSRMatrix[DType.float64]:
        var n = self.basis.num_basis
        var remove_left = self.left_cond == "dirichlet"
        var remove_right = self.right_cond == "dirichlet"

        var num_removed = 0
        if remove_left:
            num_removed += 1
        if remove_right:
            num_removed += 1

        var ncols = n - num_removed
        if ncols < 0:
            ncols = 0

        var out = COOMatrix[DType.float64](n, ncols)

        if self.left_cond == "dirichlet" and self.right_cond == "dirichlet":
            for j in range(ncols):
                out.append(j + 1, j, 1.0)

        elif self.left_cond == "newmann" and self.right_cond == "dirichlet":
            # Remove last only; add +1 in top-left corner.
            for j in range(ncols):
                out.append(j + 1, j, 1.0)
            if ncols > 0:
                out.append(0, 0, 1.0)

        elif self.left_cond == "dirichlet" and self.right_cond == "newmann":
            # Remove first only; add +1 in bottom-right corner.
            for j in range(ncols):
                out.append(j, j, 1.0)
            if ncols > 0:
                out.append(n - 1, ncols - 1, 1.0)

        elif self.left_cond == "newmann" and self.right_cond == "newmann":
            # Keep all and add +1 in both corners.
            for j in range(n):
                out.append(j, j, 1.0)
            if n > 0:
                out.append(0, 0, 1.0)
                out.append(n - 1, n - 1, 1.0)

        else:
            # Fallback to dirichlet-dirichlet behavior for unknown strings.
            for j in range(ncols):
                out.append(j + 1, j, 1.0)

        return out.to_csr()

    def eval_all(self, points: List[Float64]) -> CSRMatrix[DType.float64]:
        var B = self.basis.eval_all(points)
        var R = self.recombination_matrix()
        return spgemm(B, R)

    def first_derivative_all(self, points: List[Float64]) -> CSRMatrix[DType.float64]:
        var dB = self.basis.first_derivative_all(points)
        var R = self.recombination_matrix()
        return spgemm(dB, R)
