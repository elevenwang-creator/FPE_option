"""Benchmark: native Mojo FPE full pipeline (std.benchmark).
"""

from std.benchmark import run, Unit
from server.pricer import Pricer
from server.option_types import FpeParams
from engines.fpe.heston_params import HestonParams


comptime N_S = 38
comptime N_V = 38


def _make_heston() -> HestonParams:
    return HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6,
        S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )


def _make_strikes() -> List[Float64]:
    var s = List[Float64]()
    for k in range(8):
        s.append(60.0 + Float64(k) * 5.0 + 5.0)
    return s^


def bench_pricer():
    try:
        var fp = FpeParams(
            heston=_make_heston(), n_s=N_S, n_v=N_V,
            barrier=50.0, option_type=2, strikes=_make_strikes(),
        )
        var pricer = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
        _ = pricer.price(fp^)
    except:
        pass


def main() raises:
    print("=== Mojo FPE Engine Benchmark (std.benchmark) ===")
    print()

    # Warm-up
    print("Warm-up run (1 solve)...")
    var fp = FpeParams(
        heston=_make_heston(), n_s=N_S, n_v=N_V,
        barrier=50.0, option_type=2, strikes=_make_strikes(),
    )
    var pw = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
    _ = pw.price(fp^)

    # Benchmark via std.benchmark
    print("Benchmark run (1 solve, std.benchmark)...")
    var report = run[func2=bench_pricer](0, 1, 0.0, 120.0)
    report.print(Unit.ms)

    # Print prices from a fresh solve
    print()
    var prices = pw.price(fp^)
    print("  Grid:                 ", N_S, "x", N_V)
    print("  Strikes:               8")
    print("  num_insert:            251")
    print("  Option type:           down_and_out_call (barrier=50.0)")
    print()
    print("  Prices:")
    for i in range(len(prices)):
        print("    K=", 60.0 + Float64(i) * 5.0 + 5.0, ":", prices[i])
    print()
    var t = report.mean()
    print("  Full pipeline time:   ", t, "s")
    print()
    print("=== Benchmark complete ===")
