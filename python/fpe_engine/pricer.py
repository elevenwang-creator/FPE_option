from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy import sparse

from ._fpe_native import Compute as _NativeCompute


@dataclass
class PriceResult:
    prices: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas: np.ndarray


@dataclass
class KnotsResult:
    s: np.ndarray
    v: np.ndarray


@dataclass
class GridPointsResult:
    s: np.ndarray
    v: np.ndarray
    s_weights: np.ndarray
    v_weights: np.ndarray


@dataclass
class Basis1DResult:
    Bs: sparse.csr_matrix
    dBs: sparse.csr_matrix
    Bv: sparse.csr_matrix
    dBv: sparse.csr_matrix


@dataclass
class GreeksResult:
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray


def _csr_from_mojo(d: dict) -> sparse.csr_matrix:
    data = np.array(d["data"], dtype=np.float64)
    indices = np.array(d["indices"], dtype=np.int32)
    indptr = np.array(d["indptr"], dtype=np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=(d["nrows"], d["ncols"]))


_OPTION_TYPES = {
    "down_and_in_call": 0, "down_and_in_put": 1,
    "down_and_out_call": 2, "down_and_out_put": 3,
    "up_and_in_call": 4, "up_and_in_put": 5,
    "up_and_out_call": 6, "up_and_out_put": 7,
    "european_call": 8, "european_put": 9,
}


@dataclass
class FpeParams:
    kappa: float = 1.2
    theta: float = 0.05
    sigma: float = 0.35
    rho: float = -0.4
    r: float = 0.05
    T: float = 0.5
    S0: float = 100.0
    V0: float = 0.1
    n_s: int = 38
    n_v: int = 38
    option_type: str | int = "european_call"
    K: float | list | None = None
    barrier: float = 0.0

    def __post_init__(self):
        if isinstance(self.option_type, str):
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(f"unknown option_type '{self.option_type}'")
        else:
            if not (0 <= self.option_type <= 9):
                raise ValueError(f"option_type must be 0-9, got {self.option_type}")
        if not (4 <= self.n_s <= 256):
            raise ValueError(f"n_s must be in [4, 256], got {self.n_s}")
        if not (4 <= self.n_v <= 256):
            raise ValueError(f"n_v must be in [4, 256], got {self.n_v}")


def _normalize_K(K: float | int | list | np.ndarray) -> list[float]:
    if isinstance(K, np.ndarray):
        return K.tolist()
    if isinstance(K, (int, float)):
        return [float(K)]
    return [float(k) for k in K]


class Compute:
    """FPE computation context.

    Thin Python wrapper around Mojo ``PyComputePipeline``.
    Each property access triggers a Mojo call with Python-side caching.
    """

    def __init__(self, **kwargs):
        if "option_type" not in kwargs:
            kwargs["option_type"] = "european_put"
        self._pipe = _NativeCompute(**kwargs)
        self._knots_cache: KnotsResult | None = None
        self._grid_points_cache: GridPointsResult | None = None
        self._basis_1d_cache: Basis1DResult | None = None
        self._basis_2d_cache: sparse.csr_matrix | None = None
        self._ic_cache: np.ndarray | None = None
        self._solve_cache: list[np.ndarray] | None = None
        self._pdf_cache: np.ndarray | None = None

    @property
    def knots(self) -> KnotsResult:
        if self._knots_cache is None:
            raw = self._pipe.knots()
            self._knots_cache = KnotsResult(
                s=np.array(raw["s"], dtype=np.float64),
                v=np.array(raw["v"], dtype=np.float64),
            )
        return self._knots_cache

    @property
    def grid_points(self) -> GridPointsResult:
        if self._grid_points_cache is None:
            raw = self._pipe.grid_points()
            self._grid_points_cache = GridPointsResult(
                s=np.array(raw["s"], dtype=np.float64),
                v=np.array(raw["v"], dtype=np.float64),
                s_weights=np.array(raw["s_weights"], dtype=np.float64),
                v_weights=np.array(raw["v_weights"], dtype=np.float64),
            )
        return self._grid_points_cache

    @property
    def basis_1d(self) -> Basis1DResult:
        if self._basis_1d_cache is None:
            raw = self._pipe.basis_1d()
            self._basis_1d_cache = Basis1DResult(
                Bs=_csr_from_mojo(raw["Bs"]),
                dBs=_csr_from_mojo(raw["dBs"]),
                Bv=_csr_from_mojo(raw["Bv"]),
                dBv=_csr_from_mojo(raw["dBv"]),
            )
        return self._basis_1d_cache

    @property
    def basis_2d(self) -> sparse.csr_matrix:
        if self._basis_2d_cache is None:
            raw = self._pipe.basis_2d()
            self._basis_2d_cache = _csr_from_mojo(raw)
        return self._basis_2d_cache

    @property
    def initial_condition(self) -> np.ndarray:
        if self._ic_cache is None:
            self._ic_cache = np.array(self._pipe.initial_condition(), dtype=np.float64)
        return self._ic_cache

    @property
    def solve(self) -> list[np.ndarray]:
        if self._solve_cache is None:
            raw = self._pipe.solve()
            self._solve_cache = [np.array(t, dtype=np.float64) for t in raw]
        return self._solve_cache

    @property
    def pdf(self) -> np.ndarray:
        if self._pdf_cache is None:
            raw = self._pipe.pdf()
            self._pdf_cache = np.array(raw, dtype=np.float64)
        return self._pdf_cache

    def payoff_price(self, K: float | int | list | np.ndarray) -> np.ndarray:
        K_list = _normalize_K(K)
        raw = self._pipe.payoff_price(K_list)
        return np.array(raw, dtype=np.float64)

    def greeks(self, K: float | int | list | np.ndarray) -> GreeksResult:
        K_list = _normalize_K(K)
        raw = self._pipe.greeks(K_list)
        return GreeksResult(
            delta=np.array(raw["delta"], dtype=np.float64),
            gamma=np.array(raw["gamma"], dtype=np.float64),
            vega=np.array(raw["vega"], dtype=np.float64),
        )
