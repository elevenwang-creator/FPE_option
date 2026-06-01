"""Benchmark: native Mojo FPE full pipeline.
"""

from std.time import perf_counter_ns as now
from server.pricer import Pricer
from server.option_types import FpeParams
from engines.fpe.heston_params import HestonParams


def make_params(
    n_s: Int, n_v: Int,
) raises -> FpeParams:
    var heston = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6,
        S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0,
        V_min=0.0, V_max=1.0,
    )
    var strikes = List[Float64]()
    for k in range(8):
        strikes.append(60.0 + Float64(k) * 5.0 + 5.0)
    return FpeParams(
        heston=heston^,
        n_s=n_s,
        n_v=n_v,
        barrier=50.0,
        option_type=2,
        strikes=strikes^,
    )


def main() raises:
    print("=== Mojo FPE Engine Benchmark ===")
    print()

    var n_s = 38
    var n_v = 38
    var pricer = Pricer(rtol=1e-4, atol=1e-6, num_insert=50)

    # Warm-up: run once to warm caches/JIT
    print("Warm-up run (1 solve)...")
    _ = pricer.price(make_params(n_s, n_v))

    # Timed run: measure full pipeline
    print("Timed run (1 solve)...")
    var start = now()
    var prices = pricer.price(make_params(n_s, n_v))
    var elapsed = Float64(now() - start) / 1e9

    print()
    print("  Grid:                 ", n_s, "x", n_v)
    print("  Strikes:               8")
    print("  Option type:           down_and_out_call (barrier=50.0)")
    print("  Full pipeline time:   ", elapsed, "s")
    print()
    print("  Prices:")
    for i in range(len(prices)):
        print("    K=", 60.0 + Float64(i) * 5.0 + 5.0, ":", prices[i])
    print()
    print("=== Benchmark complete ===")
