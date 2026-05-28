from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from server.compute_pipeline import ComputePipeline
from engines.fpe.heston_params import HestonParams
from std.memory import alloc, free, Layout


@fieldwise_init
struct FpeVecResult(TrivialRegisterPassable):
    var data: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


@fieldwise_init
struct FpeVec2Result(TrivialRegisterPassable):
    var s_data: UnsafePointer[Float64, MutExternalOrigin]
    var s_len: Int32
    var v_data: UnsafePointer[Float64, MutExternalOrigin]
    var v_len: Int32


@fieldwise_init
struct FpeGridPtsResult(TrivialRegisterPassable):
    var s_data: UnsafePointer[Float64, MutExternalOrigin]
    var s_len: Int32
    var v_data: UnsafePointer[Float64, MutExternalOrigin]
    var v_len: Int32
    var sw_data: UnsafePointer[Float64, MutExternalOrigin]
    var vw_data: UnsafePointer[Float64, MutExternalOrigin]


@fieldwise_init
struct FpeMatResult(TrivialRegisterPassable):
    var data: UnsafePointer[Float64, MutExternalOrigin]
    var n_rows: Int32
    var n_cols: Int32


@fieldwise_init
struct FpeGreeksResult(TrivialRegisterPassable):
    var delta: UnsafePointer[Float64, MutExternalOrigin]
    var gamma: UnsafePointer[Float64, MutExternalOrigin]
    var vega: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


@fieldwise_init
struct FpeOneshotResult(TrivialRegisterPassable):
    var price: UnsafePointer[Float64, MutExternalOrigin]
    var delta: UnsafePointer[Float64, MutExternalOrigin]
    var gamma: UnsafePointer[Float64, MutExternalOrigin]
    var vega: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


def _list_to_heap(var src: List[Float64]) -> UnsafePointer[Float64, MutExternalOrigin]:
    var n = len(src)
    if n == 0:
        return UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=0)
    var ptr = alloc[Float64](n)
    for i in range(n):
        ptr[i] = src[i]
    return ptr


def _mat_to_heap(
    data: List[List[Float64]]
) -> Tuple[UnsafePointer[Float64, MutExternalOrigin], Int, Int]:
    var n_rows = len(data)
    if n_rows == 0:
        return (
            UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=0),
            0,
            0,
        )
    var n_cols = len(data[0])
    var total = n_rows * n_cols
    var ptr = alloc[Float64](total)
    for i in range(n_rows):
        var row = data[i].copy()
        var base = i * n_cols
        for j in range(min(len(row), n_cols)):
            ptr[base + j] = row[j]
    return (ptr, n_rows, n_cols)


def _free_heap(ptr: UnsafePointer[Float64, MutExternalOrigin], n: Int):
    if ptr:
        free(ptr, Layout[Float64](count=n))


@export("fpe_compute_create", ABI="C")
def fpe_compute_create(
    kappa: Float64,
    theta: Float64,
    sigma: Float64,
    rho: Float64,
    r: Float64,
    T: Float64,
    S0: Float64,
    V0: Float64,
    n_s: Int32,
    n_v: Int32,
    barrier: Float64,
    option_type: Int32,
    num_insert: Int32,
) -> UnsafePointer[ComputePipeline, MutExternalOrigin]:
    var heston = HestonParams(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        r=r,
        T=T,
        S0=S0,
        V0=V0,
        S_min=0.0,
        S_max=S0 * 3.0,
        V_min=0.0,
        V_max=1.0,
    )
    if not heston.is_valid():
        return UnsafePointer[ComputePipeline, MutExternalOrigin](
            unsafe_from_address=0
        )
    var strikes = [100.0]
    var fp = FpeParams(
        heston=heston^,
        n_s=Int(n_s),
        n_v=Int(n_v),
        barrier=barrier,
        option_type=Int(option_type),
        strikes=strikes^,
    )
    var ptr = alloc[ComputePipeline](1)
    try:
        ptr.init_pointee_move(ComputePipeline(fp^, num_insert=Int(num_insert)))
        return ptr
    except:
        free(ptr, Layout[ComputePipeline](count=1))
        return UnsafePointer[ComputePipeline, MutExternalOrigin](
            unsafe_from_address=0
        )


@export("fpe_compute_destroy", ABI="C")
def fpe_compute_destroy(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
):
    if ptr:
        ptr.destroy_pointee()
        free(ptr, Layout[ComputePipeline](count=1))


@export("fpe_compute_knots", ABI="C")
def fpe_compute_knots(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) -> FpeVec2Result:
    var tup = ptr[].knots()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var s_len = len(s)
    var v_len = len(v)
    return FpeVec2Result(
        s_data=_list_to_heap(s^),
        s_len=Int32(s_len),
        v_data=_list_to_heap(v^),
        v_len=Int32(v_len),
    )


@export("fpe_compute_grid_points", ABI="C")
def fpe_compute_grid_points(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) -> FpeGridPtsResult:
    var tup = ptr[].grid_points()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var sw = tup[2].copy()
    var vw = tup[3].copy()
    var s_len = len(s)
    var v_len = len(v)
    return FpeGridPtsResult(
        s_data=_list_to_heap(s^),
        s_len=Int32(s_len),
        v_data=_list_to_heap(v^),
        v_len=Int32(v_len),
        sw_data=_list_to_heap(sw^),
        vw_data=_list_to_heap(vw^),
    )


@export("fpe_compute_initial_condition", ABI="C")
def fpe_compute_initial_condition(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeVecResult:
    var q0 = ptr[].initial_condition()
    var n = len(q0)
    return FpeVecResult(data=_list_to_heap(q0^), len=Int32(n))


@export("fpe_compute_solve", ABI="C")
def fpe_compute_solve(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeMatResult:
    var sol = ptr[].solve()
    var heap_tup = _mat_to_heap(sol)
    return FpeMatResult(
        data=heap_tup[0],
        n_rows=Int32(heap_tup[1]),
        n_cols=Int32(heap_tup[2]),
    )


@export("fpe_compute_pdf", ABI="C")
def fpe_compute_pdf(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeMatResult:
    var pdf_grid = ptr[].pdf()
    var heap_tup = _mat_to_heap(pdf_grid)
    return FpeMatResult(
        data=heap_tup[0],
        n_rows=Int32(heap_tup[1]),
        n_cols=Int32(heap_tup[2]),
    )


@export("fpe_compute_price", ABI="C")
def fpe_compute_price(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin],
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
) raises -> FpeVecResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var prices = ptr[].price_at(strikes)
    var n = len(prices)
    return FpeVecResult(data=_list_to_heap(prices^), len=Int32(n))


@export("fpe_compute_greeks", ABI="C")
def fpe_compute_greeks(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin],
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    rel_s: Float64,
    rel_v: Float64,
) raises -> FpeGreeksResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var g_tup = ptr[].greeks(strikes, rel_s=rel_s, rel_v=rel_v)
    var deltas = g_tup[0].copy()
    var gammas = g_tup[1].copy()
    var vegas = g_tup[2].copy()
    var d_len = len(deltas)
    return FpeGreeksResult(
        delta=_list_to_heap(deltas^),
        gamma=_list_to_heap(gammas^),
        vega=_list_to_heap(vegas^),
        len=Int32(d_len),
    )


@export("fpe_price_oneshot", ABI="C")
def fpe_price_oneshot(
    kappa: Float64,
    theta: Float64,
    sigma: Float64,
    rho: Float64,
    r: Float64,
    T: Float64,
    S0: Float64,
    V0: Float64,
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    barrier: Float64,
    option_type: Int32,
    n_s: Int32,
    n_v: Int32,
    num_insert: Int32,
) raises -> FpeOneshotResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var heston = HestonParams(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        r=r,
        T=T,
        S0=S0,
        V0=V0,
        S_min=0.0,
        S_max=S0 * 3.0,
        V_min=0.0,
        V_max=1.0,
    )
    var fp = FpeParams(
        heston=heston^,
        n_s=Int(n_s),
        n_v=Int(n_v),
        barrier=barrier,
        option_type=Int(option_type),
        strikes=strikes^,
    )
    var engine = PricingEngine(num_insert=Int(num_insert))
    var results = engine.price(fp)
    var count = len(results)
    var p_ptr = alloc[Float64](count)
    var d_ptr = alloc[Float64](count)
    var g_ptr = alloc[Float64](count)
    var v_ptr = alloc[Float64](count)
    for i in range(count):
        p_ptr[i] = results[i].price
        d_ptr[i] = results[i].delta
        g_ptr[i] = results[i].gamma
        v_ptr[i] = results[i].vega
    return FpeOneshotResult(
        price=p_ptr,
        delta=d_ptr,
        gamma=g_ptr,
        vega=v_ptr,
        len=Int32(count),
    )


@export("fpe_compute_free_vec", ABI="C")
def fpe_compute_free_vec(r: FpeVecResult):
    _free_heap(r.data, Int(r.len))


@export("fpe_compute_free_vec2", ABI="C")
def fpe_compute_free_vec2(r: FpeVec2Result):
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))


@export("fpe_compute_free_grid_pts", ABI="C")
def fpe_compute_free_grid_pts(r: FpeGridPtsResult):
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))


@export("fpe_compute_free_mat", ABI="C")
def fpe_compute_free_mat(r: FpeMatResult):
    if r.data:
        var total = Int(r.n_rows) * Int(r.n_cols)
        _free_heap(r.data, total)


@export("fpe_compute_free_greeks", ABI="C")
def fpe_compute_free_greeks(r: FpeGreeksResult):
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))


@export("fpe_compute_free_oneshot", ABI="C")
def fpe_compute_free_oneshot(r: FpeOneshotResult):
    _free_heap(r.price, Int(r.len))
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))
