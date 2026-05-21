from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from engines.fpe.heston_params import HestonParams


def _option_type_from_string(type_str: String) -> Int:
    if type_str == "down_and_in_call":
        return 0
    elif type_str == "down_and_in_put":
        return 1
    elif type_str == "down_and_out_call":
        return 2
    elif type_str == "down_and_out_put":
        return 3
    elif type_str == "up_and_in_call":
        return 4
    elif type_str == "up_and_in_put":
        return 5
    elif type_str == "up_and_out_call":
        return 6
    elif type_str == "up_and_out_put":
        return 7
    elif type_str == "european_call":
        return 8
    elif type_str == "european_put":
        return 9
    else:
        return 8


def py_price(params_obj: PythonObject) raises -> PythonObject:
    var kappa = Float64(py=params_obj.get("kappa", PythonObject(1.2)))
    var theta = Float64(py=params_obj.get("theta", PythonObject(0.05)))
    var sigma = Float64(py=params_obj.get("sigma", PythonObject(0.35)))
    var rho = Float64(py=params_obj.get("rho", PythonObject(-0.4)))
    var r_rate = Float64(py=params_obj.get("r", PythonObject(0.05)))
    var T = Float64(py=params_obj.get("T", PythonObject(0.5)))
    var S0 = Float64(py=params_obj.get("S0", PythonObject(100.0)))
    var V0 = Float64(py=params_obj.get("V0", PythonObject(0.1)))
    var n_s = Int(py=params_obj.get("n_s", PythonObject(38)))
    var n_v = Int(py=params_obj.get("n_v", PythonObject(38)))
    var barrier = Float64(py=params_obj.get("barrier", PythonObject(0.0)))
    var rtol = Float64(py=params_obj.get("rtol", PythonObject(1e-4)))
    var atol = Float64(py=params_obj.get("atol", PythonObject(1e-6)))

    var option_type_int = 8
    var ot_obj = params_obj.get("option_type", PythonObject("european_call"))
    var py_str = Python.import_module("builtins").str
    if py_str == Python.type(ot_obj):
        option_type_int = _option_type_from_string(String(py=ot_obj))
    else:
        option_type_int = Int(py=ot_obj)

    var strikes: List[Float64] = []
    var K_obj = params_obj.get("K", PythonObject(100.0))
    var py_list = Python.import_module("builtins").list
    if py_list == Python.type(K_obj):
        for i in range(len(K_obj)):
            strikes.append(Float64(py=K_obj[i]))
    else:
        strikes.append(Float64(py=K_obj))

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
        heston=heston.copy(),
        n_s=n_s,
        n_v=n_v,
        barrier=barrier,
        option_type=option_type_int,
        strikes=strikes^,
    )

    var engine = PricingEngine(rtol=rtol, atol=atol)
    var results = engine.price(fpe_params)

    var py_prices = Python.list()
    var py_deltas = Python.list()
    var py_gammas = Python.list()
    var py_vegas = Python.list()
    var py_success = Python.list()
    for i in range(len(results)):
        _ = py_prices.append(PythonObject(results[i].price))
        _ = py_deltas.append(PythonObject(results[i].delta))
        _ = py_gammas.append(PythonObject(results[i].gamma))
        _ = py_vegas.append(PythonObject(results[i].vega))
        _ = py_success.append(PythonObject(results[i].success))

    return Python.dict(
        prices=py_prices,
        deltas=py_deltas,
        gammas=py_gammas,
        vegas=py_vegas,
        success=py_success,
    )


@export
def PyInit_fpe_engine() -> PythonObject:
    try:
        var module = PythonModuleBuilder("fpe_engine")
        module.def_function[py_price]("price")
        return module.finalize()
    except e:
        abort(String("Failed to init fpe_engine: ", e))
