"""Benchmark: in-option pricing overhead vs vanilla and out-option.

Uses std.benchmark for timing and outputs detailed report.
"""

from std.benchmark import run, Unit
from server.pricer import Pricer
from server.option_types import FpeParams
from engines.fpe.heston_params import HestonParams

comptime N_S = 38
comptime N_V = 38


def _make_strikes() -> List[Float64]:
    var s = List[Float64]()
    for k in range(8):
        s.append(60.0 + Float64(k) * 5.0 + 5.0)
    return s^


def _make_heston() -> HestonParams:
    return HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )


def bench_vanilla_call():
    try:
        var fp = FpeParams(
            heston=_make_heston(), n_s=N_S, n_v=N_V,
            barrier=0.0, option_type=8, strikes=_make_strikes(),
        )
        var pricer = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
        _ = pricer.price(fp^)
    except:
        pass


def bench_down_and_out_call():
    try:
        var fp = FpeParams(
            heston=_make_heston(), n_s=N_S, n_v=N_V,
            barrier=50.0, option_type=2, strikes=_make_strikes(),
        )
        var pricer = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
        _ = pricer.price(fp^)
    except:
        pass


def bench_down_and_in_call():
    try:
        var fp = FpeParams(
            heston=_make_heston(), n_s=N_S, n_v=N_V,
            barrier=50.0, option_type=0, strikes=_make_strikes(),
        )
        var pricer = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
        _ = pricer.price(fp^)
    except:
        pass


def main() raises:
    # Warmup: solve DOC once to prime JIT/caches
    try:
        var fp = FpeParams(
            heston=_make_heston(), n_s=N_S, n_v=N_V,
            barrier=50.0, option_type=2, strikes=_make_strikes(),
        )
        var w = Pricer(rtol=1e-4, atol=1e-6, num_insert=251)
        _ = w.price(fp^)
    except:
        pass

    print("")
    print("============================================")
    print("  In-Option Pricing Overhead Benchmark")
    print("============================================")
    print("  Config: n_s=38, n_v=38, S0=60, T=0.6")
    print("         8 strikes [65, 70, ..., 100]")
    print("         Domain: [0, 150], Barrier: 50")
    print("  Pricer: num_insert=251, rtol=1e-4, atol=1e-6")
    print("  Warmed up: 1 DOC solve")
    print("")

    var r1 = run[func2=bench_vanilla_call](0, 1, 0.0, 120.0)
    print("  [1/3] Vanilla call (1 solve, full domain [0, 150]):")
    r1.print(Unit.ms)
    print("")

    var r2 = run[func2=bench_down_and_out_call](0, 1, 0.0, 120.0)
    print("  [2/3] DOC (1 solve, truncated domain [50, 150]):")
    r2.print(Unit.ms)
    print("")

    var r3 = run[func2=bench_down_and_in_call](0, 1, 0.0, 120.0)
    print("  [3/3] DIC (2 solves via in-out parity):")
    r3.print(Unit.ms)
    print("")

    var t_van = r1.mean()
    var t_doc = r2.mean()
    var t_dic = r3.mean()
    print("============================================")
    print("  Summary")
    print("============================================")
    print("  Vanilla call:   ", t_van * 1000.0, " ms  (", t_van, " s)")
    print("  DOC:            ", t_doc * 1000.0, " ms  (", t_doc, " s)")
    print("  DIC:            ", t_dic * 1000.0, " ms  (", t_dic, " s)")
    print("  DIC/DOC ratio:  ", t_dic / t_doc, "x")
    print("")
