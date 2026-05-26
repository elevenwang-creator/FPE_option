from std.python import Python, PythonObject
from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams

comptime MAX_STRIKES: Int = 1024

def _option_type_from_string(type_str: String) raises -> Int:
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
        raise Error("unknown option_type: " + type_str)

def option_type_from_py(ot_obj: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(ot_obj, builtins.str):
        return _option_type_from_string(String(py=ot_obj))
    elif builtins.isinstance(ot_obj, builtins.int):
        var val = Int(py=ot_obj)
        if val < 0 or val > 9:
            raise Error("option_type must be 0-9, got " + String(val))
        return val
    else:
        raise Error("option_type must be str or int")

def _get_float(kwargs: PythonObject, key: String, default: Float64) raises -> Float64:
    if kwargs.__contains__(key):
        return Float64(py=kwargs[key])
    return default

def _get_int(kwargs: PythonObject, key: String, default: Int) raises -> Int:
    if kwargs.__contains__(key):
        return Int(py=kwargs[key])
    return default

def fpe_params_from_kwargs(kwargs: PythonObject) raises -> FpeParams:
    var kappa = _get_float(kwargs, "kappa", 1.2)
    var theta = _get_float(kwargs, "theta", 0.05)
    var sigma = _get_float(kwargs, "sigma", 0.35)
    var rho = _get_float(kwargs, "rho", -0.4)
    var r_rate = _get_float(kwargs, "r", 0.05)
    var T = _get_float(kwargs, "T", 0.5)
    var S0 = _get_float(kwargs, "S0", 100.0)
    var V0 = _get_float(kwargs, "V0", 0.1)
    var n_s = _get_int(kwargs, "n_s", 38)
    var n_v = _get_int(kwargs, "n_v", 38)
    var barrier = _get_float(kwargs, "barrier", 0.0)

    if n_s < 4 or n_s > 256:
        raise Error("n_s must be in [4, 256], got " + String(n_s))
    if n_v < 4 or n_v > 256:
        raise Error("n_v must be in [4, 256], got " + String(n_v))

    var option_type_int = option_type_from_py(kwargs["option_type"])

    var builtins = Python.import_module("builtins")
    var K_obj = kwargs["K"] if kwargs.__contains__("K") else PythonObject(100.0)
    var strikes: List[Float64] = []
    if builtins.isinstance(K_obj, builtins.list):
        var k_len = Int(py=builtins.len(K_obj))
        if k_len > MAX_STRIKES:
            raise Error("K list too large, max " + String(MAX_STRIKES))
        for i in range(k_len):
            strikes.append(Float64(py=K_obj[i]))
    elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(
        K_obj, builtins.int
    ):
        strikes.append(Float64(py=K_obj))
    else:
        raise Error("K must be float, int, or list of floats")

    if len(strikes) == 0:
        raise Error("K must not be empty")

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
    if not heston.is_valid():
        raise Error("invalid Heston parameters")

    var fp = FpeParams(
        heston=heston^,
        n_s=n_s,
        n_v=n_v,
        barrier=barrier,
        option_type=option_type_int,
        strikes=strikes^,
    )
    if not fp.is_valid():
        raise Error("invalid FPE parameters (check barrier/option_type combo)")
    return fp^
