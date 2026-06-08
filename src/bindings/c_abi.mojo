from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from server.compute_pipeline import ComputePipeline
from engines.fpe.heston_params import HestonParams


comptime PipelinePtr = UnsafePointer[ComputePipeline, MutExternalOrigin]
comptime F64Ptr = UnsafePointer[Float64, MutExternalOrigin]


def _null_f64() -> F64Ptr:
    return F64Ptr(unsafe_from_address=0)


def _is_null_ptr(ptr: PipelinePtr) -> Bool:
    return Int(ptr.__int__()) == 0


@fieldwise_init
struct FpeVecResult(TrivialRegisterPassable):
    var data: F64Ptr
    var len: Int32


@fieldwise_init
struct FpeVec2Result(TrivialRegisterPassable):
    var s_data: F64Ptr
    var s_len: Int32
    var v_data: F64Ptr
    var v_len: Int32


@fieldwise_init
struct FpeGridPtsResult(TrivialRegisterPassable):
    var s_data: F64Ptr
    var s_len: Int32
    var v_data: F64Ptr
    var v_len: Int32
    var sw_data: F64Ptr
    var vw_data: F64Ptr


@fieldwise_init
struct FpeMatResult(TrivialRegisterPassable):
    var data: F64Ptr
    var n_rows: Int32
    var n_cols: Int32


@fieldwise_init
struct FpeGreeksResult(TrivialRegisterPassable):
    var delta: F64Ptr
    var gamma: F64Ptr
    var vega: F64Ptr
    var len: Int32


@fieldwise_init
struct FpeOneshotResult(TrivialRegisterPassable):
    var price: F64Ptr
    var delta: F64Ptr
    var gamma: F64Ptr
    var vega: F64Ptr
    var len: Int32


def _list_to_heap(var src: List[Float64]) -> F64Ptr:
    var n = len(src)
    if n == 0:
        return _null_f64()
    var ptr = alloc[Float64](n)
    for i in range(n):
        ptr[i] = src[i]
    return ptr


def _mat_to_heap(
    data: List[List[Float64]]
) -> Tuple[F64Ptr, Int, Int]:
    var n_rows = len(data)
    if n_rows == 0:
        return (_null_f64(), 0, 0)
    var n_cols = len(data[0])
    if n_cols == 0:
        return (_null_f64(), 0, 0)
    if n_rows == 0 or n_cols == 0:
        return (_null_f64(), 0, 0)
    if n_rows > (1 << 62) // n_cols:
        return (_null_f64(), 0, 0)
    var total = n_rows * n_cols
    var ptr = alloc[Float64](total)
    for i in range(n_rows):
        var row = data[i].copy()
        var base = i * n_cols
        for j in range(min(len(row), n_cols)):
            ptr[base + j] = row[j]
    return (ptr, n_rows, n_cols)


def _free_heap(ptr: F64Ptr, n: Int):
    if n > 0:
        ptr.free()


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
    s_min: Float64,
    s_max: Float64,
) -> PipelinePtr:
    var actual_s_min = s_min if s_min >= 0.0 else 0.0
    var actual_s_max = s_max if s_max > 0.0 else S0 * 3.0
    var heston = HestonParams(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        r=r,
        T=T,
        S0=S0,
        V0=V0,
        S_min=actual_s_min,
        S_max=actual_s_max,
        V_min=0.0,
        V_max=1.0,
    )
    if not heston.is_valid():
        return PipelinePtr(unsafe_from_address=0)
    # Use S0 as dummy strike for pipeline validation; actual strikes passed at pricing time
    var strikes = [heston.S0]
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
        ptr.free()
        return PipelinePtr(unsafe_from_address=0)


@export("fpe_compute_destroy", ABI="C")
def fpe_compute_destroy(
    ptr: PipelinePtr,
):
    if not _is_null_ptr(ptr):
        ptr.destroy_pointee()
        ptr.free()


@export("fpe_compute_knots", ABI="C")
def fpe_compute_knots(
    ptr: PipelinePtr,
    result: UnsafePointer[FpeVec2Result, MutExternalOrigin],
):
    if _is_null_ptr(ptr):
        result[] = FpeVec2Result(
            s_data=_null_f64(),
            s_len=Int32(0),
            v_data=_null_f64(),
            v_len=Int32(0),
        )
        return
    var tup = ptr[].knots()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var s_len = len(s)
    var v_len = len(v)
    result[] = FpeVec2Result(
        s_data=_list_to_heap(s^),
        s_len=Int32(s_len),
        v_data=_list_to_heap(v^),
        v_len=Int32(v_len),
    )


@export("fpe_compute_grid_points", ABI="C")
def fpe_compute_grid_points(
    ptr: PipelinePtr,
    result: UnsafePointer[FpeGridPtsResult, MutExternalOrigin],
):
    if _is_null_ptr(ptr):
        result[] = FpeGridPtsResult(
            s_data=_null_f64(),
            s_len=Int32(0),
            v_data=_null_f64(),
            v_len=Int32(0),
            sw_data=_null_f64(),
            vw_data=_null_f64(),
        )
        return
    var tup = ptr[].grid_points()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var sw = tup[2].copy()
    var vw = tup[3].copy()
    var s_len = len(s)
    var v_len = len(v)
    result[] = FpeGridPtsResult(
        s_data=_list_to_heap(s^),
        s_len=Int32(s_len),
        v_data=_list_to_heap(v^),
        v_len=Int32(v_len),
        sw_data=_list_to_heap(sw^),
        vw_data=_list_to_heap(vw^),
    )


@export("fpe_compute_initial_condition", ABI="C")
def fpe_compute_initial_condition(
    ptr: PipelinePtr,
    result: UnsafePointer[FpeVecResult, MutExternalOrigin],
):
    try:
        var q0 = ptr[].initial_condition()
        var n = len(q0)
        result[] = FpeVecResult(data=_list_to_heap(q0^), len=Int32(n))
    except:
        result[] = FpeVecResult(data=_null_f64(), len=Int32(0))


@export("fpe_compute_solve", ABI="C")
def fpe_compute_solve(
    ptr: PipelinePtr,
    result: UnsafePointer[FpeMatResult, MutExternalOrigin],
):
    try:
        var sol = ptr[].solve()
        var heap_tup = _mat_to_heap(sol)
        result[] = FpeMatResult(
            data=heap_tup[0],
            n_rows=Int32(heap_tup[1]),
            n_cols=Int32(heap_tup[2]),
        )
    except:
        result[] = FpeMatResult(data=_null_f64(), n_rows=Int32(0), n_cols=Int32(0))


@export("fpe_compute_pdf", ABI="C")
def fpe_compute_pdf(
    ptr: PipelinePtr,
    result: UnsafePointer[FpeMatResult, MutExternalOrigin],
):
    try:
        var pdf_grid = ptr[].pdf()
        var heap_tup = _mat_to_heap(pdf_grid)
        result[] = FpeMatResult(
            data=heap_tup[0],
            n_rows=Int32(heap_tup[1]),
            n_cols=Int32(heap_tup[2]),
        )
    except:
        result[] = FpeMatResult(data=_null_f64(), n_rows=Int32(0), n_cols=Int32(0))


@export("fpe_compute_price", ABI="C")
def fpe_compute_price(
    ptr: PipelinePtr,
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    result: UnsafePointer[FpeVecResult, MutExternalOrigin],
):
    if n_K < 0:
        result[] = FpeVecResult(data=_null_f64(), len=Int32(0))
        return
    try:
        var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
        for i in range(Int(n_K)):
            strikes.append(K_ptr[i])
        var prices = ptr[].price_at(strikes)
        var n = len(prices)
        result[] = FpeVecResult(data=_list_to_heap(prices^), len=Int32(n))
    except:
        result[] = FpeVecResult(data=_null_f64(), len=Int32(0))


@export("fpe_compute_greeks", ABI="C")
def fpe_compute_greeks(
    ptr: PipelinePtr,
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    rel_s: Float64,
    rel_v: Float64,
    result: UnsafePointer[FpeGreeksResult, MutExternalOrigin],
):
    if n_K < 0:
        result[] = FpeGreeksResult(
            delta=_null_f64(),
            gamma=_null_f64(),
            vega=_null_f64(),
            len=Int32(0),
        )
        return
    try:
        var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
        for i in range(Int(n_K)):
            strikes.append(K_ptr[i])
        var g_tup = ptr[].greeks(strikes, rel_s=rel_s, rel_v=rel_v)
        var deltas = g_tup[0].copy()
        var gammas = g_tup[1].copy()
        var vegas = g_tup[2].copy()
        var d_len = len(deltas)
        result[] = FpeGreeksResult(
            delta=_list_to_heap(deltas^),
            gamma=_list_to_heap(gammas^),
            vega=_list_to_heap(vegas^),
            len=Int32(d_len),
        )
    except:
        result[] = FpeGreeksResult(
            delta=_null_f64(),
            gamma=_null_f64(),
            vega=_null_f64(),
            len=Int32(0),
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
    s_min: Float64,
    s_max: Float64,
    result: UnsafePointer[FpeOneshotResult, MutExternalOrigin],
):
    if n_K < 0:
        result[] = FpeOneshotResult(
            price=_null_f64(),
            delta=_null_f64(),
            gamma=_null_f64(),
            vega=_null_f64(),
            len=Int32(0),
        )
        return
    try:
        var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
        for i in range(Int(n_K)):
            strikes.append(K_ptr[i])
        var actual_s_min = s_min if s_min >= 0.0 else 0.0
        var actual_s_max = s_max if s_max > 0.0 else S0 * 3.0
        var heston = HestonParams(
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            r=r,
            T=T,
            S0=S0,
            V0=V0,
            S_min=actual_s_min,
            S_max=actual_s_max,
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
        result[] = FpeOneshotResult(
            price=p_ptr,
            delta=d_ptr,
            gamma=g_ptr,
            vega=v_ptr,
            len=Int32(count),
        )
    except:
        result[] = FpeOneshotResult(
            price=_null_f64(),
            delta=_null_f64(),
            gamma=_null_f64(),
            vega=_null_f64(),
            len=Int32(0),
        )


@export("fpe_compute_free_vec", ABI="C")
def fpe_compute_free_vec(ptr: UnsafePointer[FpeVecResult, MutExternalOrigin]):
    var r = ptr[]
    _free_heap(r.data, Int(r.len))


@export("fpe_compute_free_vec2", ABI="C")
def fpe_compute_free_vec2(ptr: UnsafePointer[FpeVec2Result, MutExternalOrigin]):
    var r = ptr[]
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))


@export("fpe_compute_free_grid_pts", ABI="C")
def fpe_compute_free_grid_pts(ptr: UnsafePointer[FpeGridPtsResult, MutExternalOrigin]):
    var r = ptr[]
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))
    _free_heap(r.sw_data, Int(r.s_len))
    _free_heap(r.vw_data, Int(r.v_len))


@export("fpe_compute_free_mat", ABI="C")
def fpe_compute_free_mat(ptr: UnsafePointer[FpeMatResult, MutExternalOrigin]):
    var r = ptr[]
    var n_rows = Int(r.n_rows)
    var n_cols = Int(r.n_cols)
    if n_rows == 0 or n_cols == 0:
        _free_heap(r.data, 0)
        return
    if n_rows > (1 << 62) // n_cols:
        _free_heap(r.data, 0)
        return
    var total = n_rows * n_cols
    _free_heap(r.data, total)


@export("fpe_compute_free_greeks", ABI="C")
def fpe_compute_free_greeks(ptr: UnsafePointer[FpeGreeksResult, MutExternalOrigin]):
    var r = ptr[]
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))


@export("fpe_compute_free_oneshot", ABI="C")
def fpe_compute_free_oneshot(ptr: UnsafePointer[FpeOneshotResult, MutExternalOrigin]):
    var r = ptr[]
    _free_heap(r.price, Int(r.len))
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))
