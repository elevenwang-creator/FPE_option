"""FPE Option Pricing Engine — Python binding.

Uses pre-compiled Mojo native module (_fpe_native.so).
Requires: pixi install && pixi run build

Usage:
    import fpe_engine
    result = fpe_engine.price(S0=60.0, K=[100.0], ...)
"""

import logging

_logger = logging.getLogger("fpe_engine")

try:
    from ._fpe_native import price as _native_price

    _NATIVE_AVAILABLE = True
except ImportError as e:
    _logger.warning("Mojo FPE engine not available: %s", e)
    _NATIVE_AVAILABLE = False
except Exception as e:
    _logger.error("Unexpected error loading Mojo FPE engine: %s", e)
    _NATIVE_AVAILABLE = False


def is_available() -> bool:
    return _NATIVE_AVAILABLE


_OPTION_TYPES = {
    "down_and_in_call": 0,
    "down_and_in_put": 1,
    "down_and_out_call": 2,
    "down_and_out_put": 3,
    "up_and_in_call": 4,
    "up_and_in_put": 5,
    "up_and_out_call": 6,
    "up_and_out_put": 7,
    "european_call": 8,
    "european_put": 9,
}

_MAX_STRIKES = 1024


def price(
    kappa: float = 1.2,
    theta: float = 0.05,
    sigma: float = 0.35,
    rho: float = -0.4,
    r: float = 0.1,
    T: float = 0.6,
    S0: float = 60.0,
    V0: float = 0.1,
    K: list[float] | float = 100.0,
    barrier: float = 0.0,
    option_type: str | int = "european_call",
    n_s: int = 38,
    n_v: int = 38,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install && pixi run build")

    if not (4 <= n_s <= 256):
        raise ValueError(f"n_s must be in [4, 256], got {n_s}")
    if not (4 <= n_v <= 256):
        raise ValueError(f"n_v must be in [4, 256], got {n_v}")

    if isinstance(K, (int, float)):
        K = [float(K)]
    else:
        K = [float(k) for k in K]

    if len(K) == 0:
        raise ValueError("K must not be empty")
    if len(K) > _MAX_STRIKES:
        raise ValueError(f"K list too large, max {_MAX_STRIKES}")

    if isinstance(option_type, str):
        if option_type not in _OPTION_TYPES:
            raise ValueError(
                f"unknown option_type '{option_type}', "
                f"must be one of: {list(_OPTION_TYPES.keys())}"
            )
        option_type_int = _OPTION_TYPES[option_type]
    else:
        option_type_int = int(option_type)
        if not (0 <= option_type_int <= 9):
            raise ValueError(f"option_type must be 0-9, got {option_type_int}")

    result = _native_price({
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
        "r": r,
        "T": T,
        "S0": S0,
        "V0": V0,
        "K": K,
        "barrier": barrier,
        "option_type": option_type_int,
        "n_s": n_s,
        "n_v": n_v,
        "rtol": rtol,
        "atol": atol,
    })

    return {
        "prices": list(result["prices"]),
        "deltas": list(result["deltas"]),
        "gammas": list(result["gammas"]),
        "vegas": list(result["vegas"]),
        "success": list(result["success"]),
    }
