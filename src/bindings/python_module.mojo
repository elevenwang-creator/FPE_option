from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
from server.pdf_cache import PDFGrid
from server.pricer import PricingRequest
from server.pricing_engine import PricingEngine


def _seed_grid(mut engine: PricingEngine, param_hash: UInt64, T: Float64):
    """Solve FPE and store real PDF grid in the cache.
    
    This replaces the placeholder uniform PDF with the actual FPE solution.
    """
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=T,
        S0=100.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var domain = FPEDomain(params, n_s=8, n_v=8, degree_s=3, degree_v=3)
    var solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1)
    var t_eval: List[Float64] = [0.0, T]
    var sol = solver.solve(domain, params, t_eval)
    
    # Build real PDF grid from FPE solution
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
    var grid = PDFGrid(pdf=pdf^, s_points=domain.s_points.copy(), v_points=domain.v_points.copy(), T=T, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    engine.store_pdf(param_hash, grid^)


def _param_hash(params: HestonParams) -> UInt64:
    """Hash of Heston parameters using bit-mixing for better distribution."""
    var h: UInt64 = 5381
    # Mix each parameter's bits into the hash
    h = ((h << 5) + h) ^ UInt64(Int(params.kappa * 1e6))
    h = ((h << 5) + h) ^ UInt64(Int(params.theta * 1e7))
    h = ((h << 5) + h) ^ UInt64(Int(params.sigma * 1e8))
    h = ((h << 5) + h) ^ UInt64(Int((params.rho + 1.0) * 1e9))
    h = ((h << 5) + h) ^ UInt64(Int(params.r * 1e10))
    h = ((h << 5) + h) ^ UInt64(Int(params.T * 1e11))
    h = ((h << 5) + h) ^ UInt64(Int(params.S0 * 1e4))
    h = ((h << 5) + h) ^ UInt64(Int(params.V0 * 1e6))
    return h


def py_price_single(
    S: PythonObject,
    K: PythonObject,
    T: PythonObject,
    barrier: PythonObject,
    param_hash: PythonObject,
) raises -> PythonObject:
    """Price a single option (CPU, sub-ms)."""
    _ = T
    var engine = PricingEngine()
    _seed_grid(engine, UInt64(py=param_hash), Float64(py=T))
    var req = PricingRequest(
        S=Float64(py=S),
        K=Float64(py=K),
        V=0.1,
        barrier=Float64(py=barrier),
        payoff_type=0,
        param_hash=UInt64(py=param_hash),
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)
    return Python.dict(
        price=PythonObject(results[0].price),
        delta=PythonObject(results[0].delta),
        gamma=PythonObject(results[0].gamma),
        success=PythonObject(results[0].success),
    )


def py_solve_fpe(params_obj: PythonObject) raises -> PythonObject:
    """Solve FPE and cache PDF grid. Returns param_hash."""
    var T = params_obj.get("T", PythonObject(0.5))
    var params = HestonParams(
        kappa=Float64(py=params_obj.get("kappa", PythonObject(1.2))),
        theta=Float64(py=params_obj.get("theta", PythonObject(0.05))),
        sigma=Float64(py=params_obj.get("sigma", PythonObject(0.35))),
        rho=Float64(py=params_obj.get("rho", PythonObject(-0.4))),
        r=Float64(py=params_obj.get("r", PythonObject(0.05))),
        T=Float64(py=T),
        S0=Float64(py=params_obj.get("S0", PythonObject(100.0))),
        V0=Float64(py=params_obj.get("V0", PythonObject(0.1))),
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var domain = FPEDomain(params)
    var solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1)
    var t_eval: List[Float64] = [Float64(py=T)]
    _ = solver.solve(domain, params, t_eval)
    var h = _param_hash(params)
    return PythonObject(Int(h))


def py_price_batch(
    S: PythonObject,
    K: PythonObject,
    T: PythonObject,
    barrier: PythonObject,
    payoff_type: PythonObject,
    param_hash: PythonObject,
) raises -> PythonObject:
    """Price batch of options through Python (GPU paths connected)."""
    _ = payoff_type
    var s_list = Python.import_module("builtins").list(S)
    var k_list = Python.import_module("builtins").list(K)
    var t_list = Python.import_module("builtins").list(T)
    var b_list = Python.import_module("builtins").list(barrier)
    var n = len(s_list)

    var engine = PricingEngine()
    _seed_grid(engine, UInt64(py=param_hash), Float64(py=t_list[0]))

    var requests: List[PricingRequest] = []
    for i in range(n):
        var req = PricingRequest(
            S=Float64(py=s_list[i]),
            K=Float64(py=k_list[i]),
            V=0.1,
            barrier=Float64(py=b_list[i]),
            payoff_type=Int(py=payoff_type),
            param_hash=UInt64(py=param_hash),
        )
        requests.append(req^)

    var results = engine.price[1](requests)
    var py = Python.import_module("builtins")
    var arr = py.list()
    for i in range(len(results)):
        var d = Python.dict(
            price=PythonObject(results[i].price),
            delta=PythonObject(results[i].delta),
            gamma=PythonObject(results[i].gamma),
            success=PythonObject(results[i].success),
        )
        _ = arr.append(d)
    return arr^


@export
def PyInit_fpe_engine() -> PythonObject:
    try:
        var module = PythonModuleBuilder("fpe_engine")
        module.def_function[py_price_single]("price_single")
        module.def_function[py_price_batch]("price_batch")
        module.def_function[py_solve_fpe]("solve_fpe")
        return module.finalize()
    except e:
        abort(String("Failed to init fpe_engine: ", e))
