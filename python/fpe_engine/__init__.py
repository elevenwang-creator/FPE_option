from typing import Any


try:
    import mojo.importer  # type: ignore[import-not-found]
    from fpe_engine import price_single as _price_single, solve_fpe as _solve_fpe_impl  # type: ignore[import-not-found]

    _MOJO_AVAILABLE = True
except ImportError:
    _MOJO_AVAILABLE = False
    _price_single = None
    _solve_fpe_impl = None


_price_single: Any
_solve_fpe_impl: Any


def is_available() -> bool:
    return _MOJO_AVAILABLE


def price_barrier_option(
    S: float, K: float, T: float, barrier: float, param_hash: int
) -> dict:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    if _price_single is None:
        raise RuntimeError("Mojo FPE engine symbol 'price_single' is unavailable")
    return _price_single(S, K, T, barrier, param_hash)


def solve_fpe(
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    T: float,
    S0: float,
    V0: float,
) -> int:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    if _solve_fpe_impl is None:
        raise RuntimeError("Mojo FPE engine symbol 'solve_fpe' is unavailable")
    return int(
        _solve_fpe_impl(
            {
                "kappa": kappa,
                "theta": theta,
                "sigma": sigma,
                "rho": rho,
                "r": r,
                "T": T,
                "S0": S0,
                "V0": V0,
            }
        )
    )

def price_batch(
    S: list[float], K: list[float], T: list[float], barrier: list[float], param_hash: int
) -> list[dict]:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    # Stubbing array-based dispatch for Python integration.
    # We would conventionally invoke `fpe_engine.price_batch` list proxy here.
    return []


def solve_fpe_batch(params_list: list[dict]) -> list[int]:
    raise NotImplementedError("solve_fpe_batch mapping requires fpe_engine module recompilation")

def calibrate_batch(quotes: list[dict], params_init: list[dict]) -> list[dict]:
    raise NotImplementedError("calibrate_batch mapping requires C++ NAIS extensions")

def nais_train(initial_params: dict, iters: int) -> list[float]:
    raise NotImplementedError("NAIS proxy mappings require full engine layout")

def nais_infer(S: float, V: float, T: float) -> dict:
    raise RuntimeError("Mojo NN context required")

def nais_vol_surface(S_grid: list[float], T_grid: list[float]) -> list[list[float]]:
    raise RuntimeError("Mojo NN context required")
