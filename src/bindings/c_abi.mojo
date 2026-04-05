from server.pricer import PricingRequest
from server.pdf_cache import PDFGrid
from server.pricing_engine import PricingEngine


def _uniform_pdf(n_s: Int, n_v: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    var w = 1.0 / Float64(n_s * n_v)
    for _ in range(n_s):
        var row: List[Float64] = []
        for _ in range(n_v):
            row.append(w)
        out.append(row^)
    return out^


def _seed_grid(mut engine: PricingEngine, param_hash: UInt64):
    var s_points: List[Float64] = [80.0, 90.0, 100.0, 110.0, 120.0]
    var v_points: List[Float64] = [0.02, 0.05, 0.1, 0.2, 0.4]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=_uniform_pdf(5, 5), s_points=s_points^, v_points=v_points^, T=0.5, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    engine.store_pdf(param_hash, grid^)


@export
def fpe_init() -> Int32:
    """Initialize the pricing engine. Returns 0 on success."""
    return 0


@export
def fpe_destroy():
    """Cleanup resources."""
    pass


@export
def fpe_price_single(
    S: Float64,
    K: Float64,
    V: Float64,
    barrier: Float64,
    payoff_type: Int32,
    param_hash: UInt64,
    out_price: UnsafePointer[Float64, MutAnyOrigin],
    out_delta: UnsafePointer[Float64, MutAnyOrigin],
    out_gamma: UnsafePointer[Float64, MutAnyOrigin],
) -> Int32:
    """Price a single option. Returns 0 on success, 1 on error."""
    # Null pointer checks
    if out_price == None or out_delta == None or out_gamma == None:
        return 1

    var engine = PricingEngine()
    _seed_grid(engine, param_hash)

    var req = PricingRequest(
        S=S,
        K=K,
        V=V,
        barrier=barrier,
        payoff_type=Int(payoff_type),
        param_hash=param_hash,
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)
    if len(results) == 0 or not results[0].success:
        return 1

    out_price[] = results[0].price
    out_delta[] = results[0].delta
    out_gamma[] = results[0].gamma
    return 0


@export
def fpe_price_batch(
    S: UnsafePointer[Float64, MutAnyOrigin],
    K: UnsafePointer[Float64, MutAnyOrigin],
    T: UnsafePointer[Float64, MutAnyOrigin],
    barrier: UnsafePointer[Float64, MutAnyOrigin],
    payoff_type: UnsafePointer[Int32, MutAnyOrigin],
    count: Int32,
    param_hash: UInt64,
    out_prices: UnsafePointer[Float64, MutAnyOrigin],
    out_deltas: UnsafePointer[Float64, MutAnyOrigin],
    out_gammas: UnsafePointer[Float64, MutAnyOrigin],
) -> Int32:
    """Price batch of options. Returns 0 on success, 1 on error."""
    # Null pointer checks
    if S == None or K == None or T == None or barrier == None or payoff_type == None:
        return 1
    if out_prices == None or out_deltas == None or out_gammas == None:
        return 1
    if count <= 0:
        return 1

    var engine = PricingEngine()
    _seed_grid(engine, param_hash)
    
    var reqs: List[PricingRequest] = []
    for i in range(Int(count)):
        reqs.append(PricingRequest(
            S=S[i], K=K[i], V=0.1, barrier=barrier[i], payoff_type=Int(payoff_type[i]), param_hash=param_hash
        ))
    
    var results = engine.price[1](reqs)
    
    for i in range(Int(count)):
        if i < len(results):
            out_prices[i] = results[i].price
            out_deltas[i] = results[i].delta
            out_gammas[i] = results[i].gamma
    return 0

