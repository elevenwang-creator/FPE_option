"""Facade module: single-call API for FPE option pricing.

NOTE: `price_batch` is a convenience wrapper for multi-strike pricing
sharing the same barrier and payoff type. True batch pricing with
per-option barrier/type is not yet implemented — each option in the
list must share the same barrier and payoff_type (taken from the first
element)."""

from engines.fpe.heston_params import HestonParams
from server.pricing_engine import PricingEngine
from server.option_types import FpeParams, PricingResult


def price(
    heston: HestonParams,
    K: Float64,
    barrier: Float64 = 0.0,
    payoff_type: Int = 8,
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> PricingResult:
    var strikes: List[Float64] = [K]
    var h_copy = heston.copy()
    var fp = FpeParams(
        heston=h_copy^,
        n_s=n_s,
        n_v=n_v,
        barrier=barrier,
        option_type=payoff_type,
        strikes=strikes^,
    )
    var engine = PricingEngine(rtol=rtol, atol=atol)
    var results = engine.price(fp)
    return results[0].copy()


def price_batch(
    heston: HestonParams,
    options: List[Tuple[Float64, Float64, Int]],
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> List[PricingResult]:
    var all_strikes: List[Float64] = []
    for i in range(len(options)):
        all_strikes.append(options[i][0])
    var barrier = 0.0
    var payoff_type = 8
    if len(options) > 0:
        barrier = options[0][1]
        payoff_type = options[0][2]
    var h_copy = heston.copy()
    var fp = FpeParams(
        heston=h_copy^,
        n_s=n_s,
        n_v=n_v,
        barrier=barrier,
        option_type=payoff_type,
        strikes=all_strikes^,
    )
    var engine = PricingEngine(rtol=rtol, atol=atol)
    return engine.price(fp)
