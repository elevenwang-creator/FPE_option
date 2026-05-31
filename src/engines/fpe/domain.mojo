"""FPE domain: knots, quadrature, B-spline basis caching.

FPECachedBasis stores only 1D factor matrices + weights.
Kronecker-structured operations use kron_spmv / kron_T_spmv
instead of building 2.1M-row kron matrices.
"""

from numerics.bspline.knots import GenerateKnots, GaussLegendre
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis, recomb_eval_all, recomb_first_derivative_all
from numerics.bspline.tensor_product import TensorProductBasis
from engines.fpe.heston_params import HestonParams
from numerics.utils.helpers import linspace, normalize
from std.memory import Span
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix
from sparse.kron import kron


def _grid_create(
    center: Float64,
    std: Float64,
    boundary: Tuple[Float64, Float64],
    num_insert: Int,
    is_v: Bool,
) -> List[Float64]:
    var lo = boundary[0]
    var hi = boundary[1]
    var lb_interm = center - 5.0 * std
    var ub_interm = center + 5.0 * std
    var left_trail = Int(Float64(num_insert) * 0.2)
    var num_interm = Int(Float64(num_insert) * 0.3)
    var right_trail = Int(Float64(num_insert) * 0.5)
    if left_trail < 1:
        left_trail = 1
    if num_interm < 3:
        num_interm = 3
    if right_trail < 1:
        right_trail = 1

    var grid: List[Float64] = []

    # Coarse lower tail: lo to lb_interm
    grid += linspace(lo, lb_interm, left_trail)

    # Approaching mean: lb_interm to center - std/5
    grid += linspace(lb_interm, center - std / 5.0, max(1, num_interm // 3))

    # Dense near mean: center - std/5 to center + std/5
    grid += linspace(center - std / 5.0, center + std / 5.0, max(2, num_interm // 3))

    # Ensure center is included
    grid.append(center)

    # Leaving mean: center + std/5 to ub_interm
    grid += linspace(center + std / 5.0, ub_interm, max(1, num_interm // 3))

    # Coarse upper tail: ub_interm to hi
    grid += linspace(ub_interm, hi, right_trail)

    # Extra points near upper boundary
    grid += linspace(hi - 0.1, hi, max(2, Int(Float64(num_insert) * 0.1)))

    # For v: add extra points near v=0
    if is_v:
        grid += linspace(0.0, 0.01, max(2, Int(Float64(num_insert) * 0.2)))

    sort(Span(grid))
    var n = len(grid)
    var result: List[Float64] = []
    if n > 0:
        result.append(grid[0])
        for i in range(1, n):
            if grid[i] != result[len(result) - 1]:
                result.append(grid[i])
    return result^


def _compute_quad_points(
    grid: List[Float64], num_gauss: Int
) -> List[Float64]:
    var result: List[Float64] = []
    var gl = GaussLegendre(num_gauss)
    for i in range(len(grid) - 1):
        var a = grid[i]
        var b = grid[i + 1]
        if b <= a:
            continue
        var mid = (a + b) * 0.5
        var half_span = (b - a) * 0.5
        result.append(a)
        for j in range(gl.order):
            result.append(mid + half_span * gl.nodes[j])
    return result^


def _compute_quad_weights(
    grid: List[Float64], num_gauss: Int
) -> List[Float64]:
    var result: List[Float64] = []
    var gl = GaussLegendre(num_gauss)
    for i in range(len(grid) - 1):
        var a = grid[i]
        var b = grid[i + 1]
        if b <= a:
            continue
        var half_span = (b - a) * 0.5
        result.append(0.0)
        for j in range(gl.order):
            result.append(half_span * gl.weights[j])
    return result^


struct FPECachedBasis[degree_s: Int, degree_v: Int](Movable):
    var Bs: CSRMatrix
    var Bv: CSRMatrix
    var dBs: CSRMatrix
    var dBv: CSRMatrix
    var Bs_T: CSRMatrix
    var Bv_T: CSRMatrix
    var dBs_T: CSRMatrix
    var dBv_T: CSRMatrix
    var n_s: Int
    var n_v: Int
    var s_points_phys: List[Float64]
    var v_points_phys: List[Float64]
    var jacobian: Float64
    var s_weights: List[Float64]
    var v_weights: List[Float64]

    def __init__(
        out self,
        domain: FPEDomain[Self.degree_s, Self.degree_v],
    ):
        var basis = domain.build_basis()
        self.Bs = recomb_eval_all(basis.basis_s, domain.s_points)
        self.Bv = recomb_eval_all(basis.basis_v, domain.v_points)
        self.dBs = recomb_first_derivative_all(basis.basis_s, domain.s_points)
        self.dBv = recomb_first_derivative_all(basis.basis_v, domain.v_points)
        self.Bs_T = self.Bs.transpose()
        self.Bv_T = self.Bv.transpose()
        self.dBs_T = self.dBs.transpose()
        self.dBv_T = self.dBv.transpose()
        self.n_s = len(domain.s_points)
        self.n_v = len(domain.v_points)
        self.s_points_phys = domain.s_points_phys.copy()
        self.v_points_phys = domain.v_points_phys.copy()
        self.jacobian = domain.jacobian_factor()
        self.s_weights = domain.s_weights.copy()
        self.v_weights = domain.v_weights.copy()


struct FPEDomain[degree_s: Int = 3, degree_v: Int = 3](Copyable, Movable):
    var s_knots: List[Float64]
    var v_knots: List[Float64]
    var s_points: List[Float64]
    var v_points: List[Float64]
    var s_weights: List[Float64]
    var v_weights: List[Float64]
    var s_points_phys: List[Float64]
    var v_points_phys: List[Float64]
    var s_min: Float64
    var s_max: Float64
    var v_min: Float64
    var v_max: Float64
    var s_left_cond: String
    var s_right_cond: String

    def __init__(
        out self,
        params: HestonParams,
        n_s: Int = 38,
        n_v: Int = 38,
        num_insert: Int = 251,
        s_left_cond: String = "dirichlet",
        s_right_cond: String = "neumann",
    ):
        self.s_min = params.S_min
        self.s_max = params.S_max
        self.v_min = params.V_min
        self.v_max = params.V_max
        self.s_left_cond = s_left_cond
        self.s_right_cond = s_right_cond
        var center_s = (params.S0 - self.s_min) / (self.s_max - self.s_min)

        var s_gen = GenerateKnots(
            n=n_s,
            degree=Self.degree_s,
            method="non-uniform",
            center=center_s,
            boundary=(self.s_min, self.s_max),
            mean=params.S0,
            std=0.1,
        )
        self.s_knots = s_gen.generate_knots()

        var v_gen = GenerateKnots(
            n=n_v,
            degree=Self.degree_v,
            method="non-uniform",
            center=params.V0,
            boundary=(self.v_min, self.v_max),
            mean=params.V0,
            std=0.001,
        )
        self.v_knots = v_gen.generate_knots()

        var s_grid_phys = _grid_create(
            params.S0, 0.1, (self.s_min, self.s_max), num_insert, is_v=False
        )
        var v_grid_phys = _grid_create(
            params.V0, 0.001, (self.v_min, self.v_max), num_insert, is_v=True
        )

        var grid_s = normalize(s_grid_phys)
        var grid_v = v_grid_phys.copy()

        var num_gauss = (Self.degree_s + Self.degree_v + 1 + 1) // 2

        self.s_points = _compute_quad_points(grid_s, num_gauss)
        self.s_weights = _compute_quad_weights(grid_s, num_gauss)

        self.v_points = _compute_quad_points(grid_v, num_gauss)
        self.v_weights = _compute_quad_weights(grid_v, num_gauss)

        var jacobian_s = self.s_max - self.s_min
        for i in range(len(self.s_weights)):
            self.s_weights[i] = self.s_weights[i] * jacobian_s

        self.s_points_phys = List[Float64]()
        for i in range(len(self.s_points)):
            self.s_points_phys.append(
                self.s_min + self.s_points[i] * (self.s_max - self.s_min)
            )
        self.v_points_phys = List[Float64]()
        for i in range(len(self.v_points)):
            self.v_points_phys.append(self.v_points[i])

    def jacobian_factor(self) -> Float64:
        return self.s_max - self.s_min

    def integ_weights(self) -> CSRMatrix:
        var sw_diag = DiagMatrix(self.s_weights.copy()).to_csr()
        var vw_diag = DiagMatrix(self.v_weights.copy()).to_csr()
        return kron(sw_diag, vw_diag)

    def build_basis(self) -> TensorProductBasis[Self.degree_s, Self.degree_v]:
        var b_s = BSplineBasis[Self.degree_s](self.s_knots.copy())
        var b_v = BSplineBasis[Self.degree_v](self.v_knots.copy())
        var r_s = RecombinationBasis[Self.degree_s](
            b_s^, left_cond=self.s_left_cond, right_cond=self.s_right_cond
        )
        var r_v = RecombinationBasis[Self.degree_v](
            b_v^, left_cond="neumann", right_cond="neumann"
        )
        return TensorProductBasis[Self.degree_s, Self.degree_v](
            basis_s=r_s^, basis_v=r_v^
        )

    def cached_basis(self) -> FPECachedBasis[Self.degree_s, Self.degree_v]:
        return FPECachedBasis[Self.degree_s, Self.degree_v](self)
