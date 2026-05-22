"""Heston Model Single Option Pricing via FPE.

Pricing pipeline:
  1. HestonParams - set Heston stochastic volatility parameters
  2. FpeParams - wraps Heston + grid size, barrier, option_type, strikes
  3. PricingEngine.price() - solve FPE, compute PDF, price all strikes
  4. PricingResult - price, delta, gamma, vega, success per strike
"""

from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from std.time import perf_counter_ns as now


def _print_result(label: String, r: PricingResult, strike: Float64) raises:
    print(
        "  K="
        + String(strike)
        + "  price="
        + String(r.price)
        + "  delta="
        + String(r.delta)
        + "  gamma="
        + String(r.gamma)
        + "  vega="
        + String(r.vega)
        + "  success="
        + String(r.success)
    )


def main() raises:
    print("=" * 60)
    print(" Heston Model Option Pricing via FPE (PricingEngine API)")
    print("=" * 60)

    var t_start = now()

    var heston = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.6,
        S0=60.0,
        V0=0.1,
        S_min=0.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )
    heston.validate()
    print()
    print("[1/4] Heston parameters:")
    print(
        "  kappa="
        + String(heston.kappa)
        + " theta="
        + String(heston.theta)
        + " sigma="
        + String(heston.sigma)
        + " rho="
        + String(heston.rho)
    )
    print(
        "  r="
        + String(heston.r)
        + " T="
        + String(heston.T)
        + " S0="
        + String(heston.S0)
        + " V0="
        + String(heston.V0)
    )
    print(
        "  Feller: 2*kappa*theta/sigma^2 - 1 = "
        + String(heston.feller_condition())
    )

    var strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]

    var engine = PricingEngine(num_insert=251)

    print()
    print("[2/4] European Call (option_type=8):")
    var t2 = now()
    var fp_call = FpeParams(
        heston=heston.copy(),
        n_s=38,
        n_v=38,
        barrier=0.0,
        option_type=8,
        strikes=strikes.copy(),
    )
    var call_results = engine.price(fp_call)
    for i in range(len(call_results)):
        _print_result("Call", call_results[i], strikes[i])
    print("  Time: " + String(Float64(now() - t2) / 1e9) + "s")

    print()
    print("[3/4] Up-and-Out Call (option_type=6, barrier=80):")
    var t3 = now()
    var fp_uoc = FpeParams(
        heston=heston.copy(),
        n_s=38,
        n_v=38,
        barrier=80.0,
        option_type=6,
        strikes=strikes.copy(),
    )
    var uoc_results = engine.price(fp_uoc)
    for i in range(len(uoc_results)):
        _print_result("UOC", uoc_results[i], strikes[i])
    print("  Time: " + String(Float64(now() - t3) / 1e9) + "s")

    print()
    print("[4/4] Down-and-Out Call (option_type=2, barrier=50):")
    var t4 = now()
    var fp_doc = FpeParams(
        heston=heston.copy(),
        n_s=38,
        n_v=38,
        barrier=50.0,
        option_type=2,
        strikes=strikes.copy(),
    )

    var doc_results = engine.price(fp_doc)
    for i in range(len(doc_results)):
        _print_result("DOC", doc_results[i], strikes[i])
    print("  Time: " + String(Float64(now() - t4) / 1e9) + "s")

    print()
    var all_valid = True
    for i in range(len(call_results)):
        if not call_results[i].success:
            all_valid = False
        if not uoc_results[i].success:
            all_valid = False
        if not doc_results[i].success:
            all_valid = False
        if call_results[i].price <= uoc_results[i].price:
            all_valid = False
    print("=" * 60)
    print(" Functional test: " + ("PASSED" if all_valid else "FAILED"))
    print(" Total time: " + String(Float64(now() - t_start) / 1e9) + "s")
    print("=" * 60)
