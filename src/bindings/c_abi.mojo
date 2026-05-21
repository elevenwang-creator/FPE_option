from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from engines.fpe.heston_params import HestonParams


@fieldwise_init
struct CPriceResult(TrivialRegisterPassable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool


def fpe_price(
    kappa: Float64,
    theta: Float64,
    sigma: Float64,
    rho: Float64,
    r_rate: Float64,
    T: Float64,
    S0: Float64,
    V0: Float64,
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_strikes: Int,
    barrier: Float64,
    option_type: Int,
    n_s: Int,
    n_v: Int,
    rtol: Float64,
    atol: Float64,
    out_ptr: UnsafePointer[CPriceResult, MutAnyOrigin],
) raises -> Int:
    var strikes: List[Float64] = []
    for i in range(n_strikes):
        strikes.append(K_ptr[i])

    var heston = HestonParams(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        r=r_rate,
        T=T,
        S0=S0,
        V0=V0,
        S_min=0.0,
        S_max=S0 * 3.0,
        V_min=0.0,
        V_max=1.0,
    )

    var fpe_params = FpeParams(
        heston=heston^,
        n_s=n_s,
        n_v=n_v,
        barrier=barrier,
        option_type=option_type,
        strikes=strikes^,
    )

    var engine = PricingEngine(rtol=rtol, atol=atol)
    var results = engine.price(fpe_params)

    for i in range(len(results)):
        out_ptr[i] = CPriceResult(
            price=results[i].price,
            delta=results[i].delta,
            gamma=results[i].gamma,
            vega=results[i].vega,
            success=results[i].success,
        )

    return len(results)
