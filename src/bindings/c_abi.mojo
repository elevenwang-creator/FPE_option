from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from server.pricer import PricingRequest
from server.pdf_cache import PDFGrid
from server.pricing_engine import PricingEngine


def _seed_grid(mut engine: PricingEngine, param_hash: UInt64, T: Float64) raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=T,
        S0=100.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var domain = FPEDomain[3, 3](params, n_s=8, n_v=8)
    var solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1, first_step=0.01)
    var t_eval: List[Float64] = [0.0, T]
    var sol = solver.solve(domain, params, t_eval)

    var n_s = len(domain.s_points)
    var n_v = len(domain.v_points)
    var pdf: List[List[Float64]] = []
    for i in range(n_s):
        var row: List[Float64] = []
        for j in range(n_v):
            row.append(sol[len(sol) - 1][i * n_v + j])
        pdf.append(row^)

    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=domain.s_points.copy(),
        v_points=domain.v_points.copy(),
        T=T,
        ds_weights=ds^,
        dv_weights=dv^,
    )
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
) raises -> Int32:
    """Price a single option. Returns 0 on success, 1 on error."""
    var engine = PricingEngine()
    _seed_grid(engine, param_hash, 0.5)

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
) raises -> Int32:
    """Price batch of options. Returns 0 on success, 1 on error."""
    if count <= 0:
        return 1

    var engine = PricingEngine()
    _seed_grid(engine, param_hash, 0.5)

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
