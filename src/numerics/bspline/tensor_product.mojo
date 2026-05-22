from numerics.bspline.recombination import RecombinationBasis, recomb_eval_all
from sparse.csr import CSRMatrix
from sparse.kron import kron


@align(64)
struct TensorProductBasis[degree_s: Int, degree_v: Int](Movable):
    var basis_s: RecombinationBasis[Self.degree_s]
    var basis_v: RecombinationBasis[Self.degree_v]

    def __init__(
        out self,
        var basis_s: RecombinationBasis[Self.degree_s],
        var basis_v: RecombinationBasis[Self.degree_v],
    ):
        self.basis_s = basis_s^
        self.basis_v = basis_v^

    def eval_tensor(
        self, s_points: List[Float64], v_points: List[Float64]
    ) -> CSRMatrix:
        var Bs = recomb_eval_all(self.basis_s, s_points)
        var Bv = recomb_eval_all(self.basis_v, v_points)
        return kron(Bs, Bv)
