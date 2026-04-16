from numerics.bspline.recombination import RecombinationBasis
from sparse.csr import CSRMatrix
from sparse.ops import kron


@align(64)
struct TensorProductBasis[degree_s: Int, degree_v: Int](Copyable, Movable):
    var basis_s: RecombinationBasis[Self.degree_s]
    var basis_v: RecombinationBasis[Self.degree_v]

    def __init__(
        out self,
        var basis_s: RecombinationBasis[Self.degree_s],
        var basis_v: RecombinationBasis[Self.degree_v],
    ):
        self.basis_s = basis_s^
        self.basis_v = basis_v^

    def eval_tensor(self, s_points: List[Float64], v_points: List[Float64]) -> CSRMatrix:
        var Bs = self.basis_s.eval_all(s_points)
        var Bv = self.basis_v.eval_all(v_points)
        return kron(Bs, Bv)

    def partial_s(self, s_points: List[Float64], v_points: List[Float64]) -> CSRMatrix:
        var dBs = self.basis_s.first_derivative_all(s_points)
        var Bv = self.basis_v.eval_all(v_points)
        return kron(dBs, Bv)

    def partial_v(self, s_points: List[Float64], v_points: List[Float64]) -> CSRMatrix:
        var Bs = self.basis_s.eval_all(s_points)
        var dBv = self.basis_v.first_derivative_all(v_points)
        return kron(Bs, dBv)
