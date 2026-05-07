from numerics.bspline.knots import GenerateKnots, GaussLegendre
from engines.fpe.heston_params import HestonParams
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis
from numerics.bspline.tensor_product import TensorProductBasis
from numerics.utils import linspace
from sparse.csr import CSRMatrix
from sparse.diag import DiagMatrix
from sparse.kron import kron


def _sort_unique(x: List[Float64]) -> List[Float64]:
    var n = len(x)
    if n == 0:
        return []
    var arr = x.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                var tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
    var out: List[Float64] = []
    out.append(arr[0])
    for i in range(1, n):
        if arr[i] > out[len(out) - 1] + 1e-9:
            out.append(arr[i])
    return out^


def _normalize_list(x: List[Float64]) -> List[Float64]:
    var gen = GenerateKnots(1, 1)
    return gen.normalize(x)


def _grid_create(
    mean: Float64, std_dev: Float64, bound: Tuple[Float64, Float64],
    num_insert: Int = 251, is_v: Bool = False,
) -> List[Float64]:
    var lb = bound[0]
    var ub = bound[1]
    var lb_interm = mean - 5.0 * std_dev
    var ub_interm = mean + 5.0 * std_dev
    var num_interm = num_insert * 30 // 100
    var right_trail = num_insert * 50 // 100
    var left_trail = num_insert * 20 // 100

    var x: List[Float64] = []
    var seg1 = linspace(lb, lb_interm, left_trail)
    for i in range(len(seg1)):
        x.append(seg1[i])
    var seg2 = linspace(lb_interm, mean - std_dev / 5.0, num_interm // 3)
    for i in range(len(seg2)):
        x.append(seg2[i])
    var seg3 = linspace(
        mean - std_dev / 5.0, mean + std_dev / 5.0, num_interm // 3
    )
    for i in range(len(seg3)):
        x.append(seg3[i])
    x.append(mean)
    var seg4 = linspace(mean + std_dev / 5.0, ub_interm, num_interm // 3)
    for i in range(len(seg4)):
        x.append(seg4[i])
    var seg5 = linspace(ub_interm, ub, right_trail)
    for i in range(len(seg5)):
        x.append(seg5[i])
    var seg6 = linspace(ub - 0.1, ub, num_insert * 10 // 100)
    for i in range(len(seg6)):
        x.append(seg6[i])

    if is_v:
        var zero_seg = linspace(0.0, 0.01, num_insert * 20 // 100)
        var new_x: List[Float64] = []
        for i in range(len(zero_seg)):
            new_x.append(zero_seg[i])
        for i in range(len(x)):
            new_x.append(x[i])
        x = new_x^

    return _sort_unique(x)^


def _compute_quad_points(grid: List[Float64], num_gauss: Int) -> List[Float64]:
    var unique: List[Float64] = []
    if len(grid) > 0:
        unique.append(grid[0])
        for i in range(1, len(grid)):
            if grid[i] > unique[len(unique) - 1] + 1e-9:
                unique.append(grid[i])

    var n_intervals = len(unique) - 1
    var gl = GaussLegendre(num_gauss)

    var points: List[Float64] = []

    for i in range(n_intervals):
        var a = unique[i]
        var b = unique[i + 1]
        if b <= a:
            continue
        var half_span = 0.5 * (b - a)
        var mid = 0.5 * (a + b)

        points.append(a)
        for j in range(gl.order):
            points.append(half_span * gl.nodes[j] + mid)

    return points^


def _compute_quad_weights(grid: List[Float64], num_gauss: Int) -> List[Float64]:
    var unique: List[Float64] = []
    if len(grid) > 0:
        unique.append(grid[0])
        for i in range(1, len(grid)):
            if grid[i] > unique[len(unique) - 1] + 1e-9:
                unique.append(grid[i])

    var n_intervals = len(unique) - 1
    var gl = GaussLegendre(num_gauss)

    var weights: List[Float64] = []

    for i in range(n_intervals):
        var a = unique[i]
        var b = unique[i + 1]
        if b <= a:
            continue
        var half_span = 0.5 * (b - a)

        weights.append(0.0)
        for j in range(gl.order):
            weights.append(half_span * gl.weights[j])

    return weights^


struct FPECachedBasis[degree_s: Int, degree_v: Int](Movable):
    var basis: TensorProductBasis[Self.degree_s, Self.degree_v]
    var weights: CSRMatrix
    var two_basis: CSRMatrix
    var s_partial: CSRMatrix
    var v_partial: CSRMatrix
    var two_basis_T: CSRMatrix
    var n_s: Int
    var n_v: Int

    def __init__(
        out self,
        var domain: FPEDomain[Self.degree_s, Self.degree_v],
    ):
        self.basis = domain.build_basis()
        self.weights = domain.integ_weights()
        self.two_basis = self.basis.eval_tensor(domain.s_points, domain.v_points)
        self.s_partial = self.basis.partial_s(domain.s_points, domain.v_points)
        self.v_partial = self.basis.partial_v(domain.s_points, domain.v_points)
        self.two_basis_T = self.two_basis.transpose()
        self.n_s = len(domain.s_points)
        self.n_v = len(domain.v_points)


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

    def __init__(
        out self,
        params: HestonParams,
        n_s: Int = 38,
        n_v: Int = 38,
        num_insert: Int = 251,
    ):
        self.s_min = params.S_min
        self.s_max = params.S_max
        self.v_min = params.V_min
        self.v_max = params.V_max

        var s_gen = GenerateKnots(
            n=n_s,
            degree=Self.degree_s,
            method="non-uniform",
            center=0.2,
            boundary=(self.s_min, self.s_max),
            mean=params.S0,
            std=0.1,
        )
        self.s_knots = s_gen.generate_knots()

        var v_gen = GenerateKnots(
            n=n_v,
            degree=Self.degree_v,
            method="non-uniform",
            center=0.2,
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

        var grid_s = _normalize_list(s_grid_phys)
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
            b_s^, left_cond="dirichlet", right_cond="neumann"
        )
        var r_v = RecombinationBasis[Self.degree_v](
            b_v^, left_cond="neumann", right_cond="neumann"
        )
        return TensorProductBasis[Self.degree_s, Self.degree_v](
            basis_s=r_s^, basis_v=r_v^
        )

    def cached_basis(self) -> FPECachedBasis[Self.degree_s, Self.degree_v]:
        return FPECachedBasis[Self.degree_s, Self.degree_v](self)
