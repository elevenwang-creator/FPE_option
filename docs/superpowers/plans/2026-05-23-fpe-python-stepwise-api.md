# FPE Python Stepwise API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose each step of the FPE option pricing pipeline to Python via a stateful `FPEPricer` class with lazy-cached methods, using a session-handle pattern.

**Architecture:** Mojo holds `FPEState` in a global `Dict[Int, FPEState]`; Python `FPEPricer` wraps a `session_id` and calls flat Mojo `py_*` functions. Each method lazily computes prerequisites and caches results as Python attributes. CSR matrices are returned as (data, indices, indptr, shape) tuples and converted to `scipy.sparse.csr_matrix` on the Python side.

**Tech Stack:** Mojo (nightly, `def` not `fn`, `var` not `let`, `comptime` not `alias`), Python 3.14, NumPy, SciPy (sparse), pixi build system.

**Spec:** `docs/superpowers/specs/2026-05-23-fpe-python-stepwise-api-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/bindings/_fpe_state.mojo` | **Create** | FPEState struct + global session Dict + counter |
| `src/bindings/_fpe_native.mojo` | **Modify** | Add 9 new `py_*` functions + register in PyInit |
| `python/fpe_engine/pricer.py` | **Create** | FPEPricer class + result dataclasses |
| `python/fpe_engine/__init__.py` | **Modify** | Import + export FPEPricer |
| `tests/test_pricer_stepwise.py` | **Create** | Stepwise API integration tests |
| `scripts/build_cpp.sh` | **Modify** | Copy new pricer.py to site-packages |

---

## Task 1: Create `_fpe_state.mojo` — Session State Management

**Files:**
- Create: `src/bindings/_fpe_state.mojo`
- Test: Build check via `pixi run mojo build -I src --emit shared-lib -o /dev/null src/bindings/_fpe_native.mojo`

- [ ] **Step 1: Write the FPEState struct and session store**

```mojo
"""FPE session state for Python stepwise API.

Holds live Mojo objects (FPEDomain, FPECachedBasis, solver results)
indexed by session_id in a global Dict.
"""

from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.heston_params import HestonParams
from sparse.csr import CSRMatrix


struct FPEState(Movable):
    var domain: FPEDomain[3, 3]
    var cached: FPECachedBasis[3, 3]
    var heston: HestonParams
    var option_type: Int
    var barrier: Float64
    var rtol: Float64
    var atol: Float64
    var q0: Optional[List[Float64]]
    var solution: Optional[List[List[Float64]]]
    var pdf_grid: Optional[List[List[Float64]]]
    var M: Optional[CSRMatrix]
    var K_mat: Optional[CSRMatrix]


var _sessions: Dict[Int, FPEState] = {}
var _next_id: Int = 1


def fpe_state_create(
    heston: HestonParams,
    n_s: Int,
    n_v: Int,
    option_type: Int,
    barrier: Float64,
    rtol: Float64,
    atol: Float64,
    s_left_cond: String,
    s_right_cond: String,
) raises -> Int:
    var sid = _next_id
    _next_id = _next_id + 1
    var domain = FPEDomain[3, 3](
        heston,
        n_s=n_s,
        n_v=n_v,
        s_left_cond=s_left_cond,
        s_right_cond=s_right_cond,
    )
    var cached = domain.cached_basis()
    var state = FPEState(
        domain=domain^,
        cached=cached^,
        heston=heston.copy(),
        option_type=option_type,
        barrier=barrier,
        rtol=rtol,
        atol=atol,
        q0=None,
        solution=None,
        pdf_grid=None,
        M=None,
        K_mat=None,
    )
    _sessions[sid] = state^
    return sid


def fpe_state_get(sid: Int) raises -> ref FPEState:
    if not _sessions.contains(sid):
        raise Error("invalid session_id: " + String(sid))
    return _sessions[sid]


def fpe_state_get_mut(sid: Int) raises -> mut FPEState:
    if not _sessions.contains(sid):
        raise Error("invalid session_id: " + String(sid))
    return _sessions[sid]


def fpe_state_close(sid: Int) raises:
    if _sessions.contains(sid):
        _sessions.remove(sid)
```

- [ ] **Step 2: Build to verify compilation**

Run: `pixi run mojo build -I src --emit object -o /dev/null src/bindings/_fpe_state.mojo`

Expected: Clean compile (no errors). May have warnings about unused imports — add `from sparse.csr import CSRMatrix` at the top.

- [ ] **Step 3: Commit**

```bash
git add src/bindings/_fpe_state.mojo
git commit -m "feat: add FPEState session management for stepwise API"
```

---

## Task 2: Add `py_create_session` and `py_close` to `_fpe_native.mojo`

**Files:**
- Modify: `src/bindings/_fpe_native.mojo`
- Test: `pixi run build` + Python import check

- [ ] **Step 1: Add imports and session creation function**

Add at top of `_fpe_native.mojo` after existing imports:

```mojo
from std.math import exp
from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.initial_cond import initial_condition_from_cached
from engines.fpe.pdf import pdf_from_cached
from engines.fpe.solver import FPESolver
from server.pricer import Pricer, PDFGrid, _price_at
from server.payoffs import BarrierPayoff
from server.greeks import Greeks
from server.option_types import FpeParams
from sparse.csr import CSRMatrix
from bindings._fpe_state import (
    fpe_state_create, fpe_state_get, fpe_state_get_mut, fpe_state_close,
)
```

Add `py_create_session` function after the existing `_option_type_from_string`:

```mojo
def py_create_session(params_obj: PythonObject) raises -> PythonObject:
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

    if n_s < 4 or n_s > 256:
        raise Error("n_s must be in [4, 256], got " + String(n_s))
    if n_v < 4 or n_v > 256:
        raise Error("n_v must be in [4, 256], got " + String(n_v))

    var ot_obj = params_obj.get("option_type", PythonObject("european_call"))
    var builtins = Python.import_module("builtins")
    var option_type_int: Int
    if builtins.isinstance(ot_obj, builtins.str):
        option_type_int = _option_type_from_string(String(py=ot_obj))
    elif builtins.isinstance(ot_obj, builtins.int):
        option_type_int = Int(py=ot_obj)
    else:
        raise Error("option_type must be str or int")

    if option_type_int < 0 or option_type_int > 9:
        raise Error("option_type must be 0-9, got " + String(option_type_int))

    var fpe_params = FpeParams(
        heston=HestonParams(
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            r=r_rate, T=T, S0=S0, V0=V0,
            S_min=0.0, S_max=S0 * 3.0, V_min=0.0, V_max=1.0,
        ),
        n_s=n_s, n_v=n_v, barrier=barrier,
        option_type=option_type_int, strikes=[100.0],
    )

    if not fpe_params.is_valid():
        raise Error("invalid FPE parameters (check barrier/option_type combo)")

    var revised = fpe_params.revised_heston()
    var sid = fpe_state_create(
        revised, n_s, n_v, option_type_int, barrier, rtol, atol,
        fpe_params.s_left_cond(), fpe_params.s_right_cond(),
    )
    return PythonObject(sid)
```

Add `py_close`:

```mojo
def py_close(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    fpe_state_close(sid)
    return PythonObject(None)
```

- [ ] **Step 2: Register in PyInit**

Add to `PyInit__fpe_native` before `return module.finalize()`:

```mojo
module.def_function[py_create_session]("create_session")
module.def_function[py_close]("close")
```

- [ ] **Step 3: Build and test**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine._fpe_native import create_session, close; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/bindings/_fpe_native.mojo
git commit -m "feat: add py_create_session and py_close to native bindings"
```

---

## Task 3: Add `py_knots` and `py_basis_1d` to `_fpe_native.mojo`

**Files:**
- Modify: `src/bindings/_fpe_native.mojo`

- [ ] **Step 1: Add py_knots function**

```mojo
def _csr_to_python(mat: CSRMatrix) -> PythonObject:
    var py_data = Python.list()
    var py_indices = Python.list()
    var py_indptr = Python.list()
    for i in range(len(mat.data)):
        _ = py_data.append(PythonObject(mat.data[i]))
    for i in range(len(mat.indices)):
        _ = py_indices.append(PythonObject(mat.indices[i]))
    for i in range(len(mat.indptr)):
        _ = py_indptr.append(PythonObject(mat.indptr[i]))
    return Python.dict(
        data=py_data,
        indices=py_indices,
        indptr=py_indptr,
        nrows=PythonObject(mat.nrows),
        ncols=PythonObject(mat.ncols),
        nnz=PythonObject(mat.nnz()),
    )


def py_knots(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get(sid)
    var py_s = Python.list()
    var py_v = Python.list()
    for i in range(len(state.domain.s_knots)):
        _ = py_s.append(PythonObject(state.domain.s_knots[i]))
    for i in range(len(state.domain.v_knots)):
        _ = py_v.append(PythonObject(state.domain.v_knots[i]))
    return Python.dict(s=py_s, v=py_v)


def py_basis_1d(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get(sid)
    return Python.dict(
        Bs=_csr_to_python(state.cached.Bs),
        dBs=_csr_to_python(state.cached.dBs),
        Bv=_csr_to_python(state.cached.Bv),
        dBv=_csr_to_python(state.cached.dBv),
    )
```

- [ ] **Step 2: Register in PyInit**

```mojo
module.def_function[py_knots]("knots")
module.def_function[py_basis_1d]("basis_1d")
```

- [ ] **Step 3: Build and test**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine._fpe_native import create_session, knots, basis_1d, close; sid = create_session({'S0':100.0,'V0':0.1}); k = knots(sid); print('s_knots len:', len(k['s'])); b = basis_1d(sid); print('Bs rows:', b['Bs']['nrows']); close(sid); print('OK')"`

Expected: Non-zero knot/basis counts, then `OK`

- [ ] **Step 4: Commit**

```bash
git add src/bindings/_fpe_native.mojo
git commit -m "feat: add py_knots and py_basis_1d to native bindings"
```

---

## Task 4: Add `py_basis_2d` and `py_initial_condition` to `_fpe_native.mojo`

**Files:**
- Modify: `src/bindings/_fpe_native.mojo`

- [ ] **Step 1: Add py_basis_2d function**

The 2D basis is the Kronecker product `kron(Bs, Bv)`. We build it on demand from the cached 1D factors:

```mojo
from sparse.kron import kron


def py_basis_2d(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get(sid)
    var Bs_2d = kron(state.cached.Bs, state.cached.Bv)
    return _csr_to_python(Bs_2d)
```

- [ ] **Step 2: Add py_initial_condition function**

This computes q0 (and caches M in the session for later solver use):

```mojo
def py_initial_condition(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get_mut(sid)
    if state.q0 == None:
        var M = mass_from_cached(state.cached)
        var q0 = initial_condition_from_cached(state.cached, state.heston, M.copy())
        state.q0 = q0^
        state.M = M^
    var q0_val = state.q0.value()
    var py_q0 = Python.list()
    for i in range(len(q0_val)):
        _ = py_q0.append(PythonObject(q0_val[i]))
    return py_q0
```

- [ ] **Step 3: Register in PyInit**

```mojo
module.def_function[py_basis_2d]("basis_2d")
module.def_function[py_initial_condition]("initial_condition")
```

- [ ] **Step 4: Build and test**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine._fpe_native import create_session, initial_condition, close; sid = create_session({'S0':100.0,'V0':0.1}); q0 = initial_condition(sid); print('q0 len:', len(q0)); close(sid); print('OK')"`

Expected: `q0 len: <N>` (N = number of basis coefficients), then `OK`

- [ ] **Step 5: Commit**

```bash
git add src/bindings/_fpe_native.mojo
git commit -m "feat: add py_basis_2d and py_initial_condition to native bindings"
```

---

## Task 5: Add `py_solve` and `py_pdf` to `_fpe_native.mojo`

**Files:**
- Modify: `src/bindings/_fpe_native.mojo`

- [ ] **Step 1: Add py_solve function**

This runs the ODE solver and caches the solution. Note: `FPESolver.solve()` internally rebuilds cached basis, M, and K (see `solver.mojo:51-54`). The session's cached M/K are used only by `py_initial_condition` for its own caching, not by the solver. A future optimization could add a `solve_from_cached` method to `FPESolver` that accepts pre-built M/K, but that's out of scope here.

```mojo
def py_solve(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get_mut(sid)
    if state.solution == None:
        if state.q0 == None:
            var M = mass_from_cached(state.cached)
            var q0 = initial_condition_from_cached(state.cached, state.heston, M.copy())
            state.q0 = q0^
            state.M = M^
        if state.K_mat == None:
            var K_mat = stiffness_from_cached(state.cached, state.heston)
            state.K_mat = K_mat^
        var solver = FPESolver[1](
            rtol=state.rtol,
            atol=state.atol,
            max_step=state.heston.T / 5.0,
            first_step=1e-6,
        )
        var sol = solver.solve(state.domain, state.heston)
        state.solution = sol^
    var sol_val = state.solution.value()
    var py_sol = Python.list()
    for t in range(len(sol_val)):
        var py_t = Python.list()
        for i in range(len(sol_val[t])):
            _ = py_t.append(PythonObject(sol_val[t][i]))
        _ = py_sol.append(py_t)
    return py_sol
```

- [ ] **Step 2: Add py_pdf function**

```mojo
def py_pdf(sid_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get_mut(sid)
    if state.pdf_grid == None:
        var _ = py_solve(sid_obj)
    var state2 = fpe_state_get(sid)
    var sol_val = state2.solution.value()
    var q_T = sol_val[len(sol_val) - 1]
    var pdf = pdf_from_cached(state2.cached, q_T)
    state.pdf_grid = pdf^
    var pdf_val = state.pdf_grid.value()
    var py_pdf = Python.list()
    for i in range(len(pdf_val)):
        var py_row = Python.list()
        for j in range(len(pdf_val[i])):
            _ = py_row.append(PythonObject(pdf_val[i][j]))
        _ = py_pdf.append(py_row)
    return py_pdf
```

- [ ] **Step 3: Register in PyInit**

```mojo
module.def_function[py_solve]("solve")
module.def_function[py_pdf]("pdf")
```

- [ ] **Step 4: Build and test**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine._fpe_native import create_session, solve, close; sid = create_session({'S0':60.0,'V0':0.1,'T':0.6}); sol = solve(sid); print('timesteps:', len(sol)); close(sid); print('OK')"`

Expected: `timesteps: <N>` (typically 1-3 for Radau), then `OK`

- [ ] **Step 5: Commit**

```bash
git add src/bindings/_fpe_native.mojo
git commit -m "feat: add py_solve and py_pdf to native bindings"
```

---

## Task 6: Add `py_payoff_price` and `py_greeks` to `_fpe_native.mojo`

**Files:**
- Modify: `src/bindings/_fpe_native.mojo`

- [ ] **Step 1: Add py_payoff_price function**

This integrates payoff against the cached PDF grid:

```mojo
def py_payoff_price(sid_obj: PythonObject, K_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get(sid)

    if state.pdf_grid == None:
        var _ = py_pdf(sid_obj)
        state = fpe_state_get(sid)

    var strikes: List[Float64] = []
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(K_obj, builtins.list):
        var k_len = Int(py=builtins.len(K_obj))
        if k_len > MAX_STRIKES:
            raise Error("K list too large, max " + String(MAX_STRIKES))
        for i in range(k_len):
            strikes.append(Float64(py=K_obj[i]))
    elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
        strikes.append(Float64(py=K_obj))
    else:
        raise Error("K must be float, int, or list of floats")

    var pdf_val = state.pdf_grid.value()
    var payoff = BarrierPayoff(
        option_type=state.option_type,
        strikes=strikes^,
        barrier=state.barrier,
    )
    var grid = PDFGrid(
        pdf=pdf_val^,
        s_points=state.domain.s_points_phys.copy(),
        v_points=state.domain.v_points_phys.copy(),
        T=state.heston.T,
        ds_weights=state.domain.s_weights.copy(),
        dv_weights=state.domain.v_weights.copy(),
    )
    var prices = _price_at(grid, payoff)
    var discount = exp(-state.heston.r * state.heston.T)

    var py_prices = Python.list()
    for i in range(len(prices)):
        _ = py_prices.append(PythonObject(prices[i] * discount))
    return py_prices
```

- [ ] **Step 2: Add py_greeks function**

This uses bump-based finite differences (4 bumped pricing runs):

```mojo
def py_greeks(sid_obj: PythonObject, K_obj: PythonObject) raises -> PythonObject:
    var sid = Int(py=sid_obj)
    var state = fpe_state_get(sid)

    var strikes: List[Float64] = []
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(K_obj, builtins.list):
        var k_len = Int(py=builtins.len(K_obj))
        for i in range(k_len):
            strikes.append(Float64(py=K_obj[i]))
    elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
        strikes.append(Float64(py=K_obj))
    else:
        raise Error("K must be float, int, or list of floats")

        var base_params = FpeParams(
            heston=state.heston.copy(),
            n_s=state.cached.n_s,
            n_v=state.cached.n_v,
            barrier=state.barrier,
        option_type=state.option_type,
        strikes=strikes^,
    )

    var pricer = Pricer(rtol=state.rtol, atol=state.atol, num_insert=50)
    var p_base = pricer.price(base_params)

    var greeks_calc = Greeks()
    var g = greeks_calc.compute(pricer, base_params, p_base)

    var py_delta = Python.list()
    var py_gamma = Python.list()
    var py_vega = Python.list()
    for i in range(len(g[0])):
        _ = py_delta.append(PythonObject(g[0][i]))
        _ = py_gamma.append(PythonObject(g[1][i]))
        _ = py_vega.append(PythonObject(g[2][i]))

    return Python.dict(delta=py_delta, gamma=py_gamma, vega=py_vega)
```

- [ ] **Step 3: Register in PyInit**

```mojo
module.def_function[py_payoff_price]("payoff_price")
module.def_function[py_greeks]("greeks")
```

- [ ] **Step 4: Build and test**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine._fpe_native import create_session, payoff_price, greeks, close; sid = create_session({'S0':60.0,'V0':0.1,'T':0.6}); p = payoff_price(sid, [100.0]); print('price:', p); g = greeks(sid, [100.0]); print('delta:', g['delta']); close(sid); print('OK')"`

Expected: Reasonable price/delta values, then `OK`

- [ ] **Step 5: Commit**

```bash
git add src/bindings/_fpe_native.mojo
git commit -m "feat: add py_payoff_price and py_greeks to native bindings"
```

---

## Task 7: Create `python/fpe_engine/pricer.py` — FPEPricer class

**Files:**
- Create: `python/fpe_engine/pricer.py`

- [ ] **Step 1: Write the FPEPricer class with result dataclasses**

```python
"""FPE Stepwise Pricer — stateful API with lazy-cached pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy import sparse


@dataclass
class PriceResult:
    prices: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas: np.ndarray
    success: list[bool]


@dataclass
class KnotsResult:
    s: np.ndarray
    v: np.ndarray


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


def _normalize_K(K: float | int | list | np.ndarray) -> list[float]:
    if isinstance(K, np.ndarray):
        return K.tolist()
    if isinstance(K, (int, float)):
        return [float(K)]
    return [float(k) for k in K]


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


class FPEPricer:
    def __init__(
        self,
        kappa: float = 1.2,
        theta: float = 0.05,
        sigma: float = 0.35,
        rho: float = -0.4,
        r: float = 0.05,
        T: float = 0.5,
        S0: float = 100.0,
        V0: float = 0.1,
        n_s: int = 38,
        n_v: int = 38,
        option_type: str | int = "european_call",
        barrier: float = 0.0,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        from . import _native

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

        if not (4 <= n_s <= 256):
            raise ValueError(f"n_s must be in [4, 256], got {n_s}")
        if not (4 <= n_v <= 256):
            raise ValueError(f"n_v must be in [4, 256], got {n_v}")

        self._session_id = _native.create_session({
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "rho": rho,
            "r": r,
            "T": T,
            "S0": S0,
            "V0": V0,
            "n_s": n_s,
            "n_v": n_v,
            "option_type": option_type_int,
            "barrier": barrier,
            "rtol": rtol,
            "atol": atol,
        })
        self._native = _native
        self._closed = False
        self._knots_cache: KnotsResult | None = None
        self._basis_1d_cache: Basis1DResult | None = None
        self._basis_2d_cache: sparse.csr_matrix | None = None
        self._ic_cache: np.ndarray | None = None
        self._solve_cache: list[np.ndarray] | None = None
        self._pdf_cache: np.ndarray | None = None

    def knots(self) -> KnotsResult:
        if self._knots_cache is None:
            raw = self._native.knots(self._session_id)
            self._knots_cache = KnotsResult(
                s=np.array(raw["s"], dtype=np.float64),
                v=np.array(raw["v"], dtype=np.float64),
            )
        return self._knots_cache

    def basis_1d(self) -> Basis1DResult:
        if self._basis_1d_cache is None:
            raw = self._native.basis_1d(self._session_id)
            self._basis_1d_cache = Basis1DResult(
                Bs=_csr_from_mojo(raw["Bs"]),
                dBs=_csr_from_mojo(raw["dBs"]),
                Bv=_csr_from_mojo(raw["Bv"]),
                dBv=_csr_from_mojo(raw["dBv"]),
            )
        return self._basis_1d_cache

    def basis_2d(self) -> sparse.csr_matrix:
        if self._basis_2d_cache is None:
            raw = self._native.basis_2d(self._session_id)
            self._basis_2d_cache = _csr_from_mojo(raw)
        return self._basis_2d_cache

    def initial_condition(self) -> np.ndarray:
        if self._ic_cache is None:
            raw = self._native.initial_condition(self._session_id)
            self._ic_cache = np.array(raw, dtype=np.float64)
        return self._ic_cache

    def solve(self) -> list[np.ndarray]:
        if self._solve_cache is None:
            raw = self._native.solve(self._session_id)
            self._solve_cache = [np.array(t, dtype=np.float64) for t in raw]
        return self._solve_cache

    def pdf(self) -> np.ndarray:
        if self._pdf_cache is None:
            raw = self._native.pdf(self._session_id)
            self._pdf_cache = np.array(raw, dtype=np.float64)
        return self._pdf_cache

    def payoff_price(self, K: float | int | list | np.ndarray) -> np.ndarray:
        K_list = _normalize_K(K)
        raw = self._native.payoff_price(self._session_id, K_list)
        return np.array(raw, dtype=np.float64)

    def greeks(self, K: float | int | list | np.ndarray) -> GreeksResult:
        K_list = _normalize_K(K)
        raw = self._native.greeks(self._session_id, K_list)
        return GreeksResult(
            delta=np.array(raw["delta"], dtype=np.float64),
            gamma=np.array(raw["gamma"], dtype=np.float64),
            vega=np.array(raw["vega"], dtype=np.float64),
        )

    def price(self, K: float | int | list | np.ndarray) -> PriceResult:
        K_list = _normalize_K(K)
        result = self._native.price({
            "kappa": self._kappa, "theta": self._theta,
            "sigma": self._sigma, "rho": self._rho,
            "r": self._r, "T": self._T,
            "S0": self._S0, "V0": self._V0,
            "n_s": self._n_s, "n_v": self._n_v,
            "barrier": self._barrier,
            "option_type": self._option_type_int,
            "K": K_list, "rtol": self._rtol, "atol": self._atol,
        })
        return PriceResult(
            prices=np.array(result["prices"], dtype=np.float64),
            deltas=np.array(result["deltas"], dtype=np.float64),
            gammas=np.array(result["gammas"], dtype=np.float64),
            vegas=np.array(result["vegas"], dtype=np.float64),
            success=list(result["success"]),
        )

    def close(self) -> None:
        if not self._closed:
            try:
                self._native.close(self._session_id)
            except Exception:
                pass
            self._closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()
```

**Design note on `price()`:** The `price()` method calls the existing `py_price` function (full-pipeline path, backward-compatible). It re-runs the full pipeline independently from the stepwise methods — `price()` is the convenience shortcut, stepwise methods are for inspection.

The `__init__` method must also store the params for the `price()` call:

```python
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma
        self._rho = rho
        self._r = r
        self._T = T
        self._S0 = S0
        self._V0 = V0
        self._n_s = n_s
        self._n_v = n_v
        self._barrier = barrier
        self._option_type_int = option_type_int
        self._rtol = rtol
        self._atol = atol
```

- [ ] **Step 2: Commit**

```bash
git add python/fpe_engine/pricer.py
git commit -m "feat: add FPEPricer class with result dataclasses"
```

---

## Task 8: Update `python/fpe_engine/__init__.py` — Export FPEPricer

**Files:**
- Modify: `python/fpe_engine/__init__.py`

- [ ] **Step 1: Add conditional import**

After the existing `try/except` block that sets `_NATIVE_AVAILABLE`, add a conditional import:

```python
if _NATIVE_AVAILABLE:
    from .pricer import FPEPricer, PriceResult, KnotsResult, Basis1DResult, GreeksResult
```

This ensures `FPEPricer` is only importable when the native module is available (matching the design spec's `is_available()` check pattern).

- [ ] **Step 2: Test import**

Run: `pixi run python -c "from fpe_engine import FPEPricer, price; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add python/fpe_engine/__init__.py
git commit -m "feat: export FPEPricer from fpe_engine package"
```

---

## Task 9: Update `scripts/build_cpp.sh` — Install pricer.py

**Files:**
- Modify: `scripts/build_cpp.sh`

- [ ] **Step 1: Add pricer.py copy step**

After the line `cp python/fpe_engine/__init__.py "${SITE_PACKAGES}/fpe_engine/"`, add:

```bash
cp python/fpe_engine/pricer.py "${SITE_PACKAGES}/fpe_engine/"
```

- [ ] **Step 2: Build and test full flow**

Run: `pixi run build`

Then: `pixi run python -c "from fpe_engine import FPEPricer; p = FPEPricer(S0=60.0, V0=0.1, T=0.6); k = p.knots(); print('s_knots:', len(k.s)); p.close(); print('OK')"`

Expected: `s_knots: <N>`, then `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/build_cpp.sh
git commit -m "feat: install pricer.py in build script"
```

---

## Task 10: Write integration tests

**Files:**
- Create: `tests/test_pricer_stepwise.py`

- [ ] **Step 1: Write tests**

```python
"""Integration tests for FPEPricer stepwise API."""

import numpy as np
import pytest
from scipy import sparse

from fpe_engine import FPEPricer, price, KnotsResult, Basis1DResult, GreeksResult, PriceResult


@pytest.fixture
def pricer():
    p = FPEPricer(S0=60.0, V0=0.1, T=0.6, n_s=20, n_v=20)
    yield p
    p.close()


@pytest.fixture
def pricer_full():
    p = FPEPricer(S0=60.0, V0=0.1, T=0.6, r=0.1, n_s=38, n_v=38)
    yield p
    p.close()


class TestSessionLifecycle:
    def test_create_and_close(self):
        p = FPEPricer(S0=100.0)
        p.close()
        assert p._closed

    def test_double_close_safe(self):
        p = FPEPricer(S0=100.0)
        p.close()
        p.close()
        assert p._closed

    def test_context_manager(self):
        with FPEPricer(S0=100.0) as p:
            k = p.knots()
            assert len(k.s) > 0
        assert p._closed

    def test_invalid_n_s(self):
        with pytest.raises(ValueError, match="n_s"):
            FPEPricer(n_s=1)

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            FPEPricer(option_type="invalid")


class TestKnots:
    def test_returns_knots_result(self, pricer):
        k = pricer.knots()
        assert isinstance(k, KnotsResult)
        assert isinstance(k.s, np.ndarray)
        assert isinstance(k.v, np.ndarray)
        assert k.s.dtype == np.float64
        assert k.v.dtype == np.float64
        assert len(k.s) > 0
        assert len(k.v) > 0

    def test_cached(self, pricer):
        k1 = pricer.knots()
        k2 = pricer.knots()
        assert k1 is k2


class TestBasis1D:
    def test_returns_basis_1d_result(self, pricer):
        b = pricer.basis_1d()
        assert isinstance(b, Basis1DResult)
        assert isinstance(b.Bs, sparse.csr_matrix)
        assert isinstance(b.dBs, sparse.csr_matrix)
        assert isinstance(b.Bv, sparse.csr_matrix)
        assert isinstance(b.dBv, sparse.csr_matrix)
        assert b.Bs.shape[0] > 0
        assert b.Bv.shape[0] > 0

    def test_cached(self, pricer):
        b1 = pricer.basis_1d()
        b2 = pricer.basis_1d()
        assert b1 is b2


class TestBasis2D:
    def test_returns_csr_matrix(self, pricer):
        b = pricer.basis_2d()
        assert isinstance(b, sparse.csr_matrix)
        assert b.shape[0] > 0
        assert b.shape[1] > 0

    def test_cached(self, pricer):
        b1 = pricer.basis_2d()
        b2 = pricer.basis_2d()
        assert b1 is b2


class TestInitialCondition:
    def test_returns_ndarray(self, pricer):
        q0 = pricer.initial_condition()
        assert isinstance(q0, np.ndarray)
        assert q0.dtype == np.float64
        assert len(q0) > 0

    def test_cached(self, pricer):
        q1 = pricer.initial_condition()
        q2 = pricer.initial_condition()
        assert q1 is q2


class TestSolve:
    def test_returns_list_of_ndarray(self, pricer):
        sol = pricer.solve()
        assert isinstance(sol, list)
        assert len(sol) > 0
        for t in sol:
            assert isinstance(t, np.ndarray)
            assert t.dtype == np.float64


class TestPDF:
    def test_returns_2d_array(self, pricer):
        pdf = pricer.pdf()
        assert isinstance(pdf, np.ndarray)
        assert pdf.ndim == 2
        assert pdf.dtype == np.float64


class TestPayoffPrice:
    def test_single_strike(self, pricer):
        p = pricer.payoff_price(100.0)
        assert isinstance(p, np.ndarray)
        assert len(p) == 1
        assert p[0] >= 0.0

    def test_multiple_strikes(self, pricer):
        p = pricer.payoff_price([80.0, 100.0, 120.0])
        assert isinstance(p, np.ndarray)
        assert len(p) == 3

    def test_k_variation(self, pricer):
        p1 = pricer.payoff_price(80.0)
        p2 = pricer.payoff_price(120.0)
        assert p1[0] != p2[0]


class TestGreeks:
    def test_returns_greeks_result(self, pricer_full):
        g = pricer_full.greeks(100.0)
        assert isinstance(g, GreeksResult)
        assert isinstance(g.delta, np.ndarray)
        assert isinstance(g.gamma, np.ndarray)
        assert isinstance(g.vega, np.ndarray)
        assert len(g.delta) == 1


class TestPriceVsLegacy:
    def test_price_matches_legacy(self, pricer_full):
        K = [100.0]
        legacy = price(S0=60.0, V0=0.1, T=0.6, r=0.1, K=K)
        stepwise = pricer_full.price(K)
        assert isinstance(stepwise, PriceResult)
        np.testing.assert_allclose(
            stepwise.prices, legacy["prices"], rtol=1e-6,
        )


class TestLazyCaching:
    def test_knots_auto_cached_on_call(self):
        p = FPEPricer(S0=100.0)
        assert p._knots_cache is None
        k = p.knots()
        assert p._knots_cache is not None
        p.close()

    def test_solve_caches_result(self):
        p = FPEPricer(S0=100.0)
        sol = p.solve()
        assert p._solve_cache is not None
        assert p._ic_cache is not None
        p.close()

    def test_pdf_caches_result(self):
        p = FPEPricer(S0=100.0)
        pdf = p.pdf()
        assert p._pdf_cache is not None
        assert p._solve_cache is not None
        p.close()
```

- [ ] **Step 2: Run tests**

Run: `pixi run python -m pytest tests/test_pricer_stepwise.py -v --tb=short`

Expected: All tests pass (some may need parameter tuning for numerical precision).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pricer_stepwise.py
git commit -m "test: add stepwise API integration tests"
```

---

## Task 11: Fix compilation issues and refine

**Files:** Various, as discovered during build/test

- [ ] **Step 1: Full build cycle**

Run: `pixi run build`

Fix any compilation errors. Common issues:
- Mojo `ref` return from `fpe_state_get` may need `mut` for mutation of `q0`, `M`, `K_mat`, `solution`, `pdf_grid` fields
- `Optional` field mutation on a `ref` — may need to use `mut` reference or restructure
- `FPEState` may need `Copyable` conformance or explicit field mutation patterns

**Known risk:** Mojo's `ref` returns and `Dict` value mutation. If `fpe_state_get` returns `ref`, mutations to `state.q0` etc. should work. If not, we may need:
```mojo
def fpe_state_get_mut(sid: Int) -> mut FPEState:
```

- [ ] **Step 2: Full test cycle**

Run: `pixi run python -m pytest tests/test_pricer_stepwise.py -v`

Fix any runtime errors.

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: compilation and runtime fixes for stepwise API"
```

---

## Task 12: End-to-end validation

**Files:** None (validation only)

- [ ] **Step 1: Run full pipeline comparison**

Run:
```bash
pixi run python -c "
from fpe_engine import FPEPricer, price
import numpy as np

legacy = price(S0=60.0, V0=0.1, T=0.6, r=0.1, K=[80.0, 100.0, 120.0])
print('Legacy prices:', legacy['prices'])

with FPEPricer(S0=60.0, V0=0.1, T=0.6, r=0.1) as p:
    print('Knots s:', len(p.knots().s), 'v:', len(p.knots().v))
    print('Basis Bs:', p.basis_1d().Bs.shape)
    print('q0 len:', len(p.initial_condition()))
    print('PDF shape:', p.pdf().shape)
    step_prices = p.payoff_price([80.0, 100.0, 120.0])
    print('Stepwise prices:', step_prices)
    np.testing.assert_allclose(step_prices, legacy['prices'], rtol=1e-6)
    print('MATCH: stepwise prices match legacy!')
"
```

Expected: `MATCH: stepwise prices match legacy!`

- [ ] **Step 2: Run all tests**

Run: `pixi run python -m pytest tests/test_pricer_stepwise.py -v`

Expected: All pass.

- [ ] **Step 3: Final commit (if any fixes)**

```bash
git add -A
git commit -m "chore: final validation fixes for stepwise API"
```
