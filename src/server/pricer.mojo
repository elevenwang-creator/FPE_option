from std.math import exp
from std.sys import simd_width_of
from std.algorithm.backend.vectorize import vectorize

from engines.fpe.domain import FPEDomain
from engines.fpe.solver import FPESolver
from engines.fpe.pdf import pdf_from_cached
from server.option_types import FpeParams
from server.payoffs import BarrierPayoff
from numerics.utils import vec_scale, mat_vec_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major

comptime SIMD_W: Int = simd_width_of[DType.float64]()


@fieldwise_init
struct PDFGrid(Copyable, Movable):
    var pdf: List[List[Float64]]
    var s_points: List[Float64]
    var v_points: List[Float64]
    var T: Float64
    var ds_weights: List[Float64]
    var dv_weights: List[Float64]


def _price_at(grid: PDFGrid, payoff: BarrierPayoff) -> List[Float64]:
    var n_strikes = len(payoff.strikes)
    var n_s = len(grid.s_points)
    var n_v = len(grid.v_points)

    var pdf_buf = List[Float64](length=n_s * n_v, fill=0.0)
    for i in range(n_s):
        var base = i * n_v
        var row_ptr = grid.pdf[i].unsafe_ptr()
        var buf_ptr = pdf_buf.unsafe_ptr()
        def cp_row[width: Int](j_off: Int) {read row_ptr, read buf_ptr, read base}:
            buf_ptr.store[width=width](base + j_off, row_ptr.load[width=width](j_off))
        vectorize[SIMD_W](n_v, cp_row)

    var pdf_layout = row_major(coord[DType.int64]((n_s, n_v)))
    var pdf_tensor = TileTensor(pdf_buf, pdf_layout)

    var dot = List[Float64](length=n_s, fill=0.0)
    var dot_span = Span[mut=True, Float64](dot)
    mat_vec_mul(pdf_tensor, Span[Float64](grid.dv_weights), dot_span)

    var weight = List[Float64](length=n_s, fill=0.0)
    for i in range(n_s):
        weight[i] = dot[i] * grid.ds_weights[i]

    var payoff_buf_T = List[Float64](length=n_strikes * n_s, fill=0.0)
    for i in range(n_s):
        var vals = payoff.evaluate(grid.s_points[i])
        for k in range(n_strikes):
            payoff_buf_T[k * n_s + i] = vals[k]

    var payoff_T_layout = row_major(coord[DType.int64]((n_strikes, n_s)))
    var payoff_T_tensor = TileTensor(payoff_buf_T, payoff_T_layout)
    var prices = List[Float64](length=n_strikes, fill=0.0)
    var prices_span = Span[mut=True, Float64](prices)
    mat_vec_mul(payoff_T_tensor, Span[Float64](weight), prices_span)

    return prices^


struct Pricer(Copyable, Movable):
    var rtol: Float64
    var atol: Float64
    var num_insert: Int

    def __init__(
        out self,
        rtol: Float64 = 1e-4,
        atol: Float64 = 1e-6,
        num_insert: Int = 50,
    ):
        self.rtol = rtol
        self.atol = atol
        self.num_insert = num_insert

    def price(self, fpe_params: FpeParams) raises -> List[Float64]:
        if not fpe_params.is_valid():
            var err = List[Float64](length=len(fpe_params.strikes), fill=0.0)
            return err^

        var revised = fpe_params.revised_heston()
        var domain = FPEDomain[3, 3](
            revised,
            n_s=fpe_params.n_s,
            n_v=fpe_params.n_v,
            num_insert=self.num_insert,
            s_left_cond=fpe_params.s_left_cond(),
            s_right_cond=fpe_params.s_right_cond(),
        )
        var solver = FPESolver[1](
            rtol=self.rtol,
            atol=self.atol,
            max_step=revised.T / 5.0,
            first_step=1e-6,
        )
        var sol = solver.solve(domain, revised)
        var cached = domain.cached_basis()
        var pdf_grid = pdf_from_cached[3, 3](cached, sol[len(sol) - 1])
        var payoff = BarrierPayoff(
            option_type=fpe_params.option_type,
            strikes=fpe_params.strikes.copy(),
            barrier=fpe_params.barrier,
        )
        var grid = PDFGrid(
            pdf=pdf_grid^,
            s_points=domain.s_points_phys.copy(),
            v_points=domain.v_points_phys.copy(),
            T=revised.T,
            ds_weights=domain.s_weights.copy(),
            dv_weights=domain.v_weights.copy(),
        )
        var prices = _price_at(grid, payoff)
        var discount = exp(-revised.r * revised.T)
        return vec_scale(prices, discount)
