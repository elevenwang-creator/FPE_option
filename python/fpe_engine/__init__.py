"""FPE Option Pricing Engine — Python binding.

Usage:
    import fpe_engine as fpe

    # One-shot (with Greeks)
    result = fpe.price(S0=60.0, K=[100.0], ...)
    # result.prices, result.deltas, result.gammas, result.vegas

    # Stepwise access to intermediates
    pipe = fpe.Compute(S0=60.0, V0=0.1, T=0.6, r=0.1)
    ks = pipe.knots
    gp = pipe.grid_points
    pdf = pipe.pdf
    prices = pipe.payoff_price(100.0)
    g = pipe.greeks([80.0, 100.0, 120.0])
"""

import logging

import numpy as np

_logger = logging.getLogger("fpe_engine")

try:
    from ._fpe_native import price as _native_price
    from ._fpe_native import Compute as NativeCompute
    _NATIVE_AVAILABLE = True
except ImportError as e:
    _logger.warning("Mojo FPE engine not available: %s", e)
    _NATIVE_AVAILABLE = False
except Exception as e:
    _logger.error("Unexpected error loading Mojo FPE engine: %s", e)
    _NATIVE_AVAILABLE = False

if _NATIVE_AVAILABLE:
    from .pricer import (
        Compute, FpeParams,
        PriceResult, KnotsResult, GridPointsResult,
        Basis1DResult, GreeksResult,
    )


def is_available() -> bool:
    return _NATIVE_AVAILABLE


_OPTION_TYPES = {
    "down_and_in_call": 0, "down_and_in_put": 1,
    "down_and_out_call": 2, "down_and_out_put": 3,
    "up_and_in_call": 4, "up_and_in_put": 5,
    "up_and_out_call": 6, "up_and_out_put": 7,
    "european_call": 8, "european_put": 9,
}


def price(
    kappa: float = 1.2, theta: float = 0.05, sigma: float = 0.35,
    rho: float = -0.4, r: float = 0.1, T: float = 0.6,
    S0: float = 60.0, V0: float = 0.1,
    K: list[float] | float = 100.0, barrier: float = 0.0,
    option_type: str | int = "european_call",
    n_s: int = 38, n_v: int = 38,
) -> PriceResult:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available")

    if isinstance(K, (int, float)):
        K = [float(K)]
    else:
        K = [float(k) for k in K]

    if isinstance(option_type, str):
        if option_type not in _OPTION_TYPES:
            raise ValueError(f"unknown option_type '{option_type}'")
        option_type_int = _OPTION_TYPES[option_type]
    else:
        option_type_int = int(option_type)

    kwargs = {
        "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
        "r": r, "T": T, "S0": S0, "V0": V0,
        "K": K, "barrier": barrier, "option_type": option_type_int,
        "n_s": n_s, "n_v": n_v,
    }
    result = _native_price(kwargs)
    return PriceResult(
        prices=np.array(result["prices"], dtype=np.float64),
        deltas=np.array(result["deltas"], dtype=np.float64),
        gammas=np.array(result["gammas"], dtype=np.float64),
        vegas=np.array(result["vegas"], dtype=np.float64),
    )
