from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.initial_cond import initial_condition_from_cached
from engines.fpe.pdf import pdf_from_cached
from engines.fpe.solver import FPESolver
from server.option_types import FpeParams
from server.payoffs import BarrierPayoff
from server.pricer import PDFGrid, _price_at
from server.greeks import Greeks
from numerics.utils import vec_scale
from sparse.csr import CSRMatrix
from sparse.kron import kron
from std.math import exp


struct ComputePipeline(Movable):
    var fp: FpeParams
    var heston: HestonParams
    var _num_insert: Int
    var domain: FPEDomain[3, 3]
    var cached: FPECachedBasis[3, 3]
    var M: CSRMatrix
    var K_mat: CSRMatrix
    var q0: Optional[List[Float64]]
    var solution: Optional[List[List[Float64]]]
    var pdf_grid: Optional[List[List[Float64]]]

    def __init__(out self, var fp: FpeParams, num_insert: Int = 251) raises:
        if not fp.is_valid():
            raise Error("invalid FPE parameters")
        self.fp = fp^
        self._num_insert = num_insert
        self.heston = self.fp.revised_heston()
        self.domain = FPEDomain[3, 3](
            self.heston,
            n_s=self.fp.n_s,
            n_v=self.fp.n_v,
            num_insert=self._num_insert,
            s_left_cond=self.fp.s_left_cond(),
            s_right_cond=self.fp.s_right_cond(),
        )
        self.cached = self.domain.cached_basis()
        self.M = mass_from_cached(self.cached)
        self.K_mat = stiffness_from_cached(self.cached, self.heston)
        self.q0 = None
        self.solution = None
        self.pdf_grid = None

    def knots(self) -> Tuple[List[Float64], List[Float64]]:
        return (self.domain.s_knots.copy(), self.domain.v_knots.copy())

    def grid_points(
        self,
    ) -> Tuple[List[Float64], List[Float64], List[Float64], List[Float64]]:
        return (
            self.cached.s_points_phys.copy(),
            self.cached.v_points_phys.copy(),
            self.cached.s_weights.copy(),
            self.cached.v_weights.copy(),
        )

    def basis_1d(
        self,
    ) -> Tuple[CSRMatrix, CSRMatrix, CSRMatrix, CSRMatrix]:
        return (
            self.cached.Bs.copy(),
            self.cached.dBs.copy(),
            self.cached.Bv.copy(),
            self.cached.dBv.copy(),
        )

    def basis_2d(self) -> CSRMatrix:
        return kron(self.cached.Bs, self.cached.Bv)

    def initial_condition(mut self) raises -> List[Float64]:
        if self.q0 == None:
            self.q0 = initial_condition_from_cached(
                self.cached, self.heston, self.M.copy()
            )
        return self.q0.value().copy()

    def _ensure_solve(mut self) raises:
        if self.solution == None:
            if self.q0 == None:
                self.q0 = initial_condition_from_cached(
                    self.cached, self.heston, self.M.copy()
                )
            var solver = FPESolver[1](
                rtol=1e-4,
                atol=1e-6,
                max_step=self.heston.T / 5.0,
                first_step=1e-6,
            )
            self.solution = solver.solve(self.domain, self.heston)

    def solve(mut self) raises -> List[List[Float64]]:
        self._ensure_solve()
        var result: List[List[Float64]] = []
        for i in range(len(self.solution.value())):
            result.append(self.solution.value()[i].copy())
        return result^

    def pdf(mut self) raises -> List[List[Float64]]:
        self._ensure_solve()
        if self.pdf_grid == None:
            self.pdf_grid = pdf_from_cached(
                self.cached,
                self.solution.value()[len(self.solution.value()) - 1],
            )
        var result: List[List[Float64]] = []
        for i in range(len(self.pdf_grid.value())):
            result.append(self.pdf_grid.value()[i].copy())
        return result^

    def price_at(mut self, strikes: List[Float64]) raises -> List[Float64]:
        self._ensure_solve()
        if self.pdf_grid == None:
            self.pdf_grid = pdf_from_cached(
                self.cached,
                self.solution.value()[len(self.solution.value()) - 1],
            )
        var payoff = BarrierPayoff(
            option_type=self.fp.option_type,
            strikes=strikes.copy(),
            barrier=self.fp.barrier,
        )
        var grid = PDFGrid(
            pdf=self.pdf_grid.value().copy(),
            s_points=self.cached.s_points_phys.copy(),
            v_points=self.cached.v_points_phys.copy(),
            T=self.heston.T,
            ds_weights=self.cached.s_weights.copy(),
            dv_weights=self.cached.v_weights.copy(),
        )
        var prices = _price_at(grid, payoff)
        var discount = exp(-self.heston.r * self.heston.T)
        return vec_scale(prices, discount)

    def greeks(
        mut self,
        strikes: List[Float64],
        rel_s: Float64 = 0.01,
        rel_v: Float64 = 0.1,
    ) raises -> Tuple[List[Float64], List[Float64], List[Float64]]:
        var base_price = self.price_at(strikes)
        var h_s = rel_s * self.heston.S0
        var h_v = rel_v * self.heston.V0
        var fp_bumped = FpeParams(
            heston=self.heston.copy(),
            n_s=self.fp.n_s,
            n_v=self.fp.n_v,
            barrier=self.fp.barrier,
            option_type=self.fp.option_type,
            strikes=strikes.copy(),
        )
        var g = Greeks(h_s=h_s, h_v=h_v, num_insert=self._num_insert)
        return g.compute(fp_bumped^, base_price)
