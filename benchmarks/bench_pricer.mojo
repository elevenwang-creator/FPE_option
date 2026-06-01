"""Benchmark _price_at: original vs mat_vec_mul."""
from std.time import perf_counter_ns as now
from std.sys import simd_width_of
from std.algorithm.backend.vectorize import vectorize
from server.payoffs import BarrierPayoff
from server.pricer import PDFGrid
from numerics.utils import mat_vec_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def make_grid(n_s: Int, n_v: Int) raises -> PDFGrid:
    var pdf = List[List[Float64]]()
    for i in range(n_s):
        var row = List[Float64]()
        for j in range(n_v):
            row.append(Float64((i + 1) * (j + 1)) / 100.0)
        pdf.append(row^)

    var s_points = List[Float64]()
    for i in range(n_s):
        s_points.append(50.0 + Float64(i) * 100.0 / Float64(n_s - 1))

    var v_points = List[Float64]()
    for j in range(n_v):
        v_points.append(Float64(j + 1) * 0.1 / Float64(n_v))

    var ds = List[Float64]()
    for i in range(n_s - 1):
        ds.append(s_points[i + 1] - s_points[i])
    ds.append(ds[len(ds) - 1])

    var dv = List[Float64]()
    for j in range(n_v - 1):
        dv.append(v_points[j + 1] - v_points[j])
    dv.append(dv[len(dv) - 1])

    return PDFGrid(
        pdf=pdf^,
        s_points=s_points^,
        v_points=v_points^,
        T=0.1,
        ds_weights=ds^,
        dv_weights=dv^,
    )


def _price_at_old(grid: PDFGrid, payoff: BarrierPayoff) raises -> List[Float64]:
    var n_strikes = len(payoff.strikes)
    var n_s = len(grid.s_points)
    var n_v = len(grid.v_points)
    var prices: List[Float64] = []
    for _ in range(n_strikes):
        prices.append(0.0)
    var dv_ptr = grid.dv_weights.unsafe_ptr()

    for i in range(n_s):
        var S = grid.s_points[i]
        var payoff_vals = payoff.evaluate(S)
        var ds_w = grid.ds_weights[i]
        var pdf_row_ptr = grid.pdf[i].unsafe_ptr()

        var weighted_pdf = List[Float64]()
        for _ in range(n_v):
            weighted_pdf.append(0.0)
        var wp_ptr = weighted_pdf.unsafe_ptr()

        def vw[width: Int](j_off: Int) {read pdf_row_ptr, read dv_ptr, read ds_w, read wp_ptr}:
            for w in range(width):
                wp_ptr[j_off + w] = pdf_row_ptr[j_off + w] * dv_ptr[j_off + w] * ds_w

        vectorize[SIMD_W](n_v, vw)

        var pdf_sum = 0.0
        for j in range(n_v):
            pdf_sum += weighted_pdf[j]

        for k in range(n_strikes):
            prices[k] += payoff_vals[k] * pdf_sum

    return prices^


def _price_at_new(grid: PDFGrid, payoff: BarrierPayoff) raises -> List[Float64]:
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
    var pdf_tensor = TileTensor(Span[Float64](pdf_buf), pdf_layout)

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
    var payoff_T_tensor = TileTensor(Span[Float64](payoff_buf_T), payoff_T_layout)
    var prices = List[Float64](length=n_strikes, fill=0.0)
    var prices_span = Span[mut=True, Float64](prices)
    mat_vec_mul(payoff_T_tensor, Span[Float64](weight), prices_span)

    return prices^


def main() raises:
    print("=== _price_at Benchmark ===")
    var grid = make_grid(64, 64)
    var payoff = BarrierPayoff(option_type=8, strikes=[60.0, 65.0, 70.0], barrier=0.0)
    var n_iter = 2000

    _price_at_old(grid, payoff)
    _price_at_new(grid, payoff)

    var start = now()
    for _ in range(n_iter):
        _price_at_old(grid, payoff)
    var t1 = Float64(now() - start) / 1e9

    start = now()
    for _ in range(n_iter):
        _price_at_new(grid, payoff)
    var t2 = Float64(now() - start) / 1e9

    print("Grid:", len(grid.s_points), "x", len(grid.v_points))
    print("Strikes:", len(payoff.strikes))
    print("Iterations:", n_iter)
    print()
    print("Original (per-row SIMD):", t1, "s  avg:", t1 / Float64(n_iter) * 1e6, "us")
    print("New (mat_vec_mul x2):    ", t2, "s  avg:", t2 / Float64(n_iter) * 1e6, "us")
    print("Speedup:                 ", t1 / t2, "x")
