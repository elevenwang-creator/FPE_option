from typing import Any

try:
    import mojo.importer  # type: ignore[import-not-found]
    from fpe_option import price as _mojo_price  # type: ignore[import-not-found]
    from fpe_option import HestonParams as _HestonParams  # type: ignore[import-not-found]
    from fpe_option import OptionParams as _OptionParams  # type: ignore[import-not-found]
    from fpe_option import calibrate as _mojo_calibrate  # type: ignore[import-not-found]
    from fpe_option import nais_train as _mojo_nais_train  # type: ignore[import-not-found]
    from fpe_option import nais_vol_surface as _mojo_nais_vol_surface  # type: ignore[import-not-found]

    _MOJO_AVAILABLE = True
except ImportError:
    _MOJO_AVAILABLE = False


def is_available() -> bool:
    return _MOJO_AVAILABLE


def price(
    S: float,
    K: float,
    V: float,
    barrier: float,
    option_type: int = 1,
    *,
    kappa: float = 1.2,
    theta: float = 0.05,
    sigma: float = 0.35,
    rho: float = -0.4,
    r: float = 0.05,
    T: float = 0.5,
    S0: float = 100.0,
    V0: float = 0.1,
    S_min: float = 50.0,
    S_max: float = 150.0,
    V_min: float = 1e-4,
    V_max: float = 1.0,
) -> dict:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    heston = _HestonParams(
        kappa=kappa, theta=theta, sigma=sigma, rho=rho, r=r, T=T,
        S0=S0, V0=V0, S_min=S_min, S_max=S_max, V_min=V_min, V_max=V_max,
    )
    option = _OptionParams(S=S, K=K, V=V, barrier=barrier, option_type=option_type)
    result = _mojo_price(heston, option)
    return {
        "price": result.price,
        "delta": result.delta,
        "gamma": result.gamma,
        "vega": result.vega,
        "success": result.success,
    }


def calibrate(
    market_prices: list[float],
    strikes: list[float],
    expiries: list[float],
    params_init: dict | None = None,
) -> dict:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    if params_init is None:
        params_init = {
            "kappa": 1.2, "theta": 0.05, "sigma": 0.35, "rho": -0.4,
            "r": 0.05, "T": 0.5, "S0": 100.0, "V0": 0.1,
            "S_min": 50.0, "S_max": 150.0, "V_min": 1e-4, "V_max": 1.0,
        }
    init = _HestonParams(**params_init)
    fitted = _mojo_calibrate(market_prices, strikes, expiries, init)
    return {
        "kappa": fitted.kappa,
        "theta": fitted.theta,
        "sigma": fitted.sigma,
        "rho": fitted.rho,
        "r": fitted.r,
        "T": fitted.T,
    }


def nais_train(
    H: float = 0.07,
    eta: float = 1.9,
    rho: float = -0.9,
    r: float = 0.05,
    T: float = 0.5,
    S0: float = 100.0,
    V0: float = 0.04,
    epsilon_t: float = 0.04,
    M: int = 100000,
    N: int = 100,
    D: int = 2,
    iters: int = 1000,
    lr: float = 1e-3,
) -> dict:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    from fpe_option import RoughBergomiParams as _RoughBergomiParams  # type: ignore[import-not-found]
    bergomi = _RoughBergomiParams(
        H=H, eta=eta, rho=rho, r=r, T=T, S0=S0, V0=V0,
        epsilon_t=epsilon_t, M=M, N=N, D=D,
    )
    _mojo_nais_train(bergomi, iters=iters, lr=lr)
    return {"status": "trained"}


def nais_vol_surface(
    strikes: list[float],
    expiries: list[float],
    model: Any = None,
) -> list[list[float]]:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    if model is None:
        raise ValueError("Trained NAIS model required. Call nais_train() first.")
    return _mojo_nais_vol_surface(model, strikes, expiries)
