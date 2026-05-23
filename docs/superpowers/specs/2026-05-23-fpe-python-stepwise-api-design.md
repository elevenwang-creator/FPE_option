# FPE Python Stepwise API Design

## Summary

Expose each step of the FPE option pricing pipeline to Python end-users via a stateful `FPEPricer` class with lazy-cached methods. Users can inspect intermediate results (knots, basis, initial condition, PDF, etc.) and call payoff pricing or greeks independently.

## Motivation

The current Python binding (`_fpe_native.mojo`) exposes only `price()` as a monolithic function. Users cannot inspect intermediate results like knot vectors, B-spline basis matrices, initial conditions, or the PDF grid. Researchers and quants need step-level access for debugging, validation, and custom post-processing.

## Architecture

### Session-handle pattern

- **Mojo side**: Flat functions that take a `session_id: Int` to look up live Mojo state (FPEDomain, FPECachedBasis, solver results) stored in a Mojo-side `Dict[Int, FPEState]`.
- **Python side**: `FPEPricer` class wraps the session handle, provides lazy-cached methods, and converts Mojo outputs to NumPy/SciPy types.

### Data flow

```
FPEPricer.__init__(params)
    → Mojo: create session, build FPEDomain + FPECachedBasis
    → return session_id

knots()        → Mojo: py_knots(session_id)           → KnotsResult
basis_1d()     → Mojo: py_basis_1d(session_id)        → Basis1DResult
basis_2d()     → Mojo: py_basis_2d(session_id)        → scipy.sparse.csr_matrix
initial_condition() → Mojo: py_initial_condition(session_id) → np.ndarray
solve()        → Mojo: py_solve(session_id)            → list[np.ndarray]
pdf()          → Mojo: py_pdf(session_id)              → np.ndarray (2D)
payoff_price(K) → Mojo: py_payoff_price(session_id, K) → np.ndarray
greeks(K)      → Mojo: py_greeks(session_id, K)        → GreeksResult
price(K)       → full pipeline                        → PriceResult
close()        → Mojo: py_close(session_id)            → frees state
```

## Python API

### Result objects

```python
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
    Bs: scipy.sparse.csr_matrix
    dBs: scipy.sparse.csr_matrix
    Bv: scipy.sparse.csr_matrix
    dBv: scipy.sparse.csr_matrix

@dataclass
class GreeksResult:
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
```

### FPEPricer class

```python
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
    ): ...

    def knots(self) -> KnotsResult: ...
    def basis_1d(self) -> Basis1DResult: ...
    def basis_2d(self) -> scipy.sparse.csr_matrix: ...
    def initial_condition(self) -> np.ndarray: ...
    def solve(self) -> list[np.ndarray]: ...
    def pdf(self) -> np.ndarray: ...
    def payoff_price(self, K: float | list[float]) -> np.ndarray: ...
    def greeks(self, K: float | list[float]) -> GreeksResult: ...
    def price(self, K: float | list[float]) -> PriceResult: ...
    def close(self) -> None: ...

    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
```

### Backward compatibility

The existing `fpe_engine.price()` function remains unchanged. It returns `dict` with keys: `prices`, `deltas`, `gammas`, `vegas`, `success`.

### Lazy caching

Each method auto-invokes preceding pipeline steps if not yet computed:

| Method | Auto-invokes prerequisite |
|---|---|
| `knots()` | (none — built in constructor) |
| `basis_1d()` | `knots()` |
| `basis_2d()` | `basis_1d()` |
| `initial_condition()` | `basis_1d()` + computes mass matrix |
| `solve()` | `initial_condition()` + builds stiffness matrix |
| `pdf()` | `solve()` |
| `payoff_price(K)` | `pdf()` |
| `greeks(K)` | 4 bumped `payoff_price` calls + finite differences |
| `price(K)` | Full pipeline (separate code path, equivalent to current `fpe_engine.price()`) |

Cached results are stored as Python attributes. Re-calling returns the cached value without Mojo round-trip, except for `payoff_price(K)` and `greeks(K)` which re-run when `K` changes.

## Mojo-side changes

### New file: `src/bindings/_fpe_state.mojo`

Holds the `FPEState` struct and a global `Dict[Int, FPEState]` for session management:

```mojo
struct FPEState:
    var domain: FPEDomain[3, 3]
    var cached: FPECachedBasis[3, 3]
    var q0: Optional[List[Float64]]
    var solution: Optional[List[List[Float64]]]
    var pdf_grid: Optional[List[List[Float64]]]
    var heston: HestonParams
    var option_type: Int
    var barrier: Float64
    var rtol: Float64
    var atol: Float64
```

Global session store with thread-safe counter:

```mojo
var _sessions: Dict[Int, FPEState] = {}
var _next_id: Int = 1
```

### New functions in `src/bindings/_fpe_native.mojo`

Each function takes `session_id` as first `PythonObject` arg:

| Mojo function | Python name | Description |
|---|---|---|
| `py_create_session(params)` | `_create_session` | Build domain + cached basis, return session_id |
| `py_knots(session_id)` | `_knots` | Return s_knots, v_knots as Python lists |
| `py_basis_1d(session_id)` | `_basis_1d` | Return Bs, dBs, Bv, dBv as COO data (data, rows, cols, shape) |
| `py_basis_2d(session_id)` | `_basis_2d` | Return tensor product Bs⊗Bv as COO data |
| `py_initial_condition(session_id)` | `_initial_condition` | Compute and return q0 coefficient vector |
| `py_solve(session_id)` | `_solve` | Run ODE solver, return solution at each timestep |
| `py_pdf(session_id)` | `_pdf` | Compute PDF grid from solution |
| `py_payoff_price(session_id, K)` | `_payoff_price` | Integrate payoff against PDF |
| `py_greeks(session_id, K)` | `_greeks` | Bump-based greeks |
| `py_close(session_id)` | `_close` | Remove session from dict |

### CSR → Python conversion strategy

CSRMatrix fields (`data`, `indices`, `indptr`) are returned as Python lists. The Python wrapper converts to `scipy.sparse.csr_matrix` using the `(data, indices, indptr)` constructor, which is zero-copy for the underlying arrays after numpy conversion.

## Python-side implementation

### New file: `python/fpe_engine/pricer.py`

Contains the `FPEPricer` class with:
- Constructor calls `_native._create_session(params_dict)` → stores `self._session_id`
- Each method calls the corresponding `_native._function(self._session_id, ...)`
- Converts return values: lists → `np.array`, CSR triples → `scipy.sparse.csr_matrix`
- `close()` calls `_native._close(self._session_id)`
- `__del__` calls `close()` as safety net
- Context manager protocol

### Modified file: `python/fpe_engine/__init__.py`

- Import `FPEPricer` from `.pricer`
- Export it at module level: `from .pricer import FPEPricer`
- Keep existing `price()` function unchanged for backward compatibility

## Error handling

- Invalid parameters: raise `ValueError` (Python-side validation before Mojo call)
- Mojo-side errors: propagate as `RuntimeError` (Mojo `Error` → Python exception)
- Invalid session_id: Mojo raises `Error("invalid session_id")`, Python wraps as `RuntimeError`
- Missing native module: same fallback as current code (warning + `is_available()` check)

## Testing

- Unit tests for each step: verify output shapes and types
- Integration test: compare `FPEPricer.price(K)` result with existing `fpe_engine.price()` result
- Session lifecycle test: create, use, close; verify double-close is safe
- K variation test: call `payoff_price(K1)` then `payoff_price(K2)`, verify different results

## File changes summary

| File | Change |
|---|---|
| `src/bindings/_fpe_state.mojo` | **New** — FPEState struct + session dict |
| `src/bindings/_fpe_native.mojo` | **Modified** — add 9 new `py_*` functions + register them in `PyInit__fpe_native` |
| `python/fpe_engine/pricer.py` | **New** — FPEPricer class + result dataclasses |
| `python/fpe_engine/__init__.py` | **Modified** — import + export FPEPricer |
| `tests/test_pricer_stepwise.py` | **New** — stepwise API tests |
