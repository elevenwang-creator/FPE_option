from numerics.utils import linspace
from numerics.bspline.knots import GenerateKnots

from engines.fpe.heston_params import HestonParams
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis
from numerics.bspline.tensor_product import TensorProductBasis





def _uniform_weights(points: List[Float64]) -> List[Float64]:
    var n = len(points)
    var w: List[Float64] = []
    if n == 0:
        return w^
    if n == 1:
        w.append(1.0)
        return w^

    var h = (points[n - 1] - points[0]) / Float64(n - 1)
    for i in range(n):
        if i == 0 or i == n - 1:
            w.append(0.5 * h)
        else:
            w.append(h)
    return w^


def _build_uniform_knots(n_internal: Int, degree: Int) -> List[Float64]:
    var knots: List[Float64] = []
    for _ in range(degree):
        knots.append(0.0)

    if n_internal > 0:
        var internal = linspace(0.0, 1.0, n_internal + 2)
        for i in range(1, len(internal) - 1):
            knots.append(internal[i])

    for _ in range(degree):
        knots.append(1.0)

    return knots^


struct FPEDomain(Copyable, Movable):
    var s_knots: List[Float64]
    var v_knots: List[Float64]
    var s_degree: Int
    var v_degree: Int
    var s_points: List[Float64]
    var v_points: List[Float64]
    var s_weights: List[Float64]
    var v_weights: List[Float64]
    var s_min: Float64
    var s_max: Float64
    var v_min: Float64
    var v_max: Float64

    def __init__(
        out self,
        params: HestonParams,
        n_s: Int = 20,
        n_v: Int = 20,
        degree_s: Int = 3,
        degree_v: Int = 3,
    ):
        self.s_degree = degree_s
        self.v_degree = degree_v
        self.s_min = params.S_min
        self.s_max = params.S_max
        self.v_min = params.V_min
        self.v_max = params.V_max

        # Align with Python: Generate non-uniform knots centered at S0/V0
        var s_gen = GenerateKnots(
            n=n_s, degree=degree_s, method="non-uniform", 
            center=0.1, 
            boundary=(self.s_min, self.s_max),
            mean=params.S0, std=0.1
        )
        self.s_knots = s_gen.generate_knots()

        var v_gen = GenerateKnots(
            n=n_v, degree=degree_v, method="non-uniform",
            center=0.1,
            boundary=(self.v_min, self.v_max),
            mean=params.V0, std=0.001
        )
        self.v_knots = v_gen.generate_knots()

        # Points for integration over the generated knots
        self.s_points = self.s_knots.copy()
        self.v_points = self.v_knots.copy()
        self.s_weights = _uniform_weights(self.s_points)
        self.v_weights = _uniform_weights(self.v_points)

    def jacobian_factor(self) -> Float64:
        return self.s_max - self.s_min

    def map_s_to_physical(self, s_norm: Float64) -> Float64:
        return self.s_min + s_norm * (self.s_max - self.s_min)

    def map_v_to_physical(self, v_norm: Float64) -> Float64:
        return self.v_min + v_norm * (self.v_max - self.v_min)

    def build_basis(self) -> TensorProductBasis[3, 3]:
        var b_s = BSplineBasis[3](self.s_knots.copy())
        var b_v = BSplineBasis[3](self.v_knots.copy())
        var r_s = RecombinationBasis[3](b_s^, left_cond="dirichlet", right_cond="newmann")
        var r_v = RecombinationBasis[3](b_v^, left_cond="dirichlet", right_cond="newmann")
        return TensorProductBasis[3, 3](basis_s=r_s^, basis_v=r_v^)
