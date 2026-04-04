"""Benchmark pricing performance."""

from server.pdf_cache import PDFGrid
from server.interpolator import Interpolator
from server.payoffs import EuropeanCall
from server.pricer import PricingRequest, Pricer
from server.greeks import Greeks
from std.python import Python


def bench_pricing() raises:
    var pdf: List[List[Float64]] = []
    for _ in range(20):
        var row: List[Float64] = []
        for _ in range(20):
            row.append(0.0025)
        pdf.append(row^)
    var s_points: List[Float64] = []
    var v_points: List[Float64] = []
    for i in range(20):
        s_points.append(50.0 + Float64(i) * 5.0)
        v_points.append(0.01 + Float64(i) * 0.025)

    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_points^, v_points=v_points^, T=1.0, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()

    var requests: List[PricingRequest] = []
    for i in range(100):
        var K = 80.0 + Float64(i) * 0.5
        requests.append(PricingRequest(
            S=100.0,
            K=K,
            V=0.15,
            barrier=200.0,
            payoff_type=1,
            param_hash=UInt64(i),
        ))

    var pricer = Pricer[1](interpolator=Interpolator(), greeks_computer=Greeks[1](h_s=1e-2, h_v=1e-3))

    var time_mod = Python.import_module("time")
    var start = time_mod.perf_counter()
    var results = pricer.price(grid, requests)
    var end = time_mod.perf_counter()
    var elapsed = Float64(py=end) - Float64(py=start)
    var per_price = elapsed / Float64(len(results)) * 1e6

    print("Pricing (20x20 grid, 100 options)")
    print("  Total time:", elapsed, "s")
    print("  Per option:", per_price, "μs")
    print("  First price:", results[0].price)


def main() raises:
    print("=== Pricing Benchmark ===")
    bench_pricing()
