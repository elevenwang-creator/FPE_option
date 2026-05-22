from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from engines.fpe.heston_params import HestonParams


comptime MAX_STRIKES: Int = 1024


comptime ERR_OK: Int32 = 0
comptime ERR_NULL_K: Int32 = -1
comptime ERR_NULL_OUT: Int32 = -2
comptime ERR_INVALID_NSTRIKES: Int32 = -3
comptime ERR_BUFFER_TOO_SMALL: Int32 = -4
comptime ERR_INVALID_PARAMS: Int32 = -5
comptime ERR_SOLVER_FAILED: Int32 = -6
comptime ERR_TOO_MANY_STRIKES: Int32 = -7


@fieldwise_init
struct FpePriceResult(TrivialRegisterPassable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Int32


@export("fpe_price", ABI="C")
def fpe_price(
    kappa: Float64,
    theta: Float64,
    sigma: Float64,
    rho: Float64,
    r_rate: Float64,
    T: Float64,
    S0: Float64,
    V0: Float64,
    K_ptr: Optional[UnsafePointer[Float64, MutAnyOrigin]],
    n_strikes: Int32,
    barrier: Float64,
    option_type: Int32,
    n_s: Int32,
    n_v: Int32,
    rtol: Float64,
    atol: Float64,
    out_ptr: Optional[UnsafePointer[FpePriceResult, MutAnyOrigin]],
    out_capacity: Int32,
) abi("C") -> Int32:
    if K_ptr == None:
        return ERR_NULL_K

    if out_ptr == None:
        return ERR_NULL_OUT

    if n_strikes <= 0:
        return ERR_INVALID_NSTRIKES

    if n_strikes > out_capacity:
        return ERR_BUFFER_TOO_SMALL

    if Int(n_strikes) > MAX_STRIKES:
        return ERR_TOO_MANY_STRIKES

    var k_ptr = K_ptr.value()
    var o_ptr = out_ptr.value()

    var strikes: List[Float64] = []
    for i in range(Int(n_strikes)):
        strikes.append(k_ptr[i])

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
        #S_max=S0 * 3.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var fpe_params = FpeParams(
        heston=heston.copy(),
        n_s=Int(n_s),
        n_v=Int(n_v),
        barrier=barrier,
        option_type=Int(option_type),
        strikes=strikes^,
    )

    if not fpe_params.is_valid():
        o_ptr[0] = FpePriceResult(
            price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=0
        )
        return ERR_INVALID_PARAMS

    var engine = PricingEngine(rtol=rtol, atol=atol)
    try:
        var results = engine.price(fpe_params)
        var count = min(len(results), Int(out_capacity))
        for i in range(count):
            o_ptr[i] = FpePriceResult(
                price=results[i].price,
                delta=results[i].delta,
                gamma=results[i].gamma,
                vega=results[i].vega,
                success=Int32(1) if results[i].success else Int32(0),
            )
        return Int32(count)
    except:
        o_ptr[0] = FpePriceResult(
            price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=0
        )
        return ERR_SOLVER_FAILED
