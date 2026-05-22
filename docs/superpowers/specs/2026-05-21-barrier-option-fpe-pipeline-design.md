# Barrier Option FPE Pricing Pipeline Redesign

Date: 2026-05-21

## Problem

The current server layer has fragmented types, a two-step cache-based pricing flow, and only 4 option types (BarrierUpAndOut, BarrierDownAndIn, EuropeanCall, EuropeanPut). The `OptionParams` struct doesn't carry Heston model parameters, knot counts, or barrier-aware boundary conditions. `PricingEngine` requires manual `store_pdf()` seeding before pricing. The Python/C++ API exposes this two-step complexity.

## Design Decisions

1. **FpeParams replaces OptionParams** — wraps HestonParams + n_s, n_v, barrier, option_type
2. **10 option types** — 8 standard barrier options + EuropeanCall + EuropeanPut
3. **Int enum internally, String API boundary** — Python accepts `"up_and_out_call"`, C++ accepts `int32_t`
4. **Barrier as S boundary** — S_min/S_max revised so barrier is the grid edge with Dirichlet BC
5. **Single BarrierPayoff struct** — dispatches on option_type, stores K/barrier internally
6. **No PDFCache** — PricingEngine solves FPE directly per call
7. **Flat dict Python API** — single dict with all fields, Mojo transforms to FpeParams
8. **Remove old API** — `py_price_single`, `py_price_batch`, `py_solve_fpe`, `fpe_init`, `fpe_destroy`, `fpe_price_single`, `fpe_price_batch` all removed

## FpeParams Struct

```mojo
@fieldwise_init
struct FpeParams(Copyable, Movable, Writable):
    var heston: HestonParams # kappa, theta, sigma, rho, r, T, S0, V0, S_min, S_max, V_min, V_max
    var n_s: Int              # S-direction knot count
    var n_v: Int              # V-direction knot count
    var barrier: Float64      # barrier level (0.0 = no barrier)
    var option_type: Int      # 0-9 enum
    var strikes: List[Float64] # strike prices (1 or more)

    def revised_heston(self) -> HestonParams:
        # Returns HestonParams with S_min/S_max revised based on barrier
        # Down options (0-3): S_min = barrier (barrier < S0)
        # Up options (4-7): S_max = barrier (barrier > S0)
        # Vanilla (8-9): unchanged from heston.S_min/S_max

    def s_left_cond(self) -> String:
        # Down options: "dirichlet" (barrier at S_min)
        # Up options: "neumann"
        # Vanilla: "dirichlet" (current default)

    def s_right_cond(self) -> String:
        # Down options: "neumann"
        # Up options: "dirichlet" (barrier at S_max)
        # Vanilla: "neumann" (current default)

    def is_valid(self) -> Bool:
        # Validate all fields including option_type in [0, 9]
        # For Down options (0-3): barrier > 0 and barrier < heston.S0
        # For Up options (4-7): barrier > 0 and barrier > heston.S0
        # For vanilla (8-9): barrier == 0.0
```

### option_type Enum

| Int | Type | S range adjustment | Boundary conditions |
|---|---|---|---|
| 0 | DownAndIn Call | S_min = barrier | left=dirichlet, right=neumann |
| 1 | DownAndIn Put | S_min = barrier | left=dirichlet, right=neumann |
| 2 | DownAndOut Call | S_min = barrier | left=dirichlet, right=neumann |
| 3 | DownAndOut Put | S_min = barrier | left=dirichlet, right=neumann |
| 4 | UpAndIn Call | S_max = barrier | left=neumann, right=dirichlet |
| 5 | UpAndIn Put | S_max = barrier | left=neumann, right=dirichlet |
| 6 | UpAndOut Call | S_max = barrier | left=neumann, right=dirichlet |
| 7 | UpAndOut Put | S_max = barrier | left=neumann, right=dirichlet |
| 8 | European Call | unchanged | left=dirichlet, right=neumann |
| 9 | European Put | unchanged | left=dirichlet, right=neumann |

### String-to-Int Mapping (Python API)

```python
OPTION_TYPE_MAP = {
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
```

## BarrierPayoff Struct

Replaces the 4 existing payoff structs with a single dispatching struct:

```mojo
struct BarrierPayoff(Payoff):
    var option_type: Int
    var strikes: List[Float64]  # 1 or more strike prices
    var barrier: Float64

    def evaluate(self, S: Float64) -> List[Float64]:
        # Returns payoff values for ALL strikes at once given S
        # No per-strike loop outside — this is the inner kernel
        #
        # Barrier activation (same for all strikes):
        # 0 DownAndIn Call:   active when S <= barrier
        # 1 DownAndIn Put:    active when S <= barrier
        # 2 DownAndOut Call:  active when S > barrier
        # 3 DownAndOut Put:   active when S > barrier
        # 4 UpAndIn Call:     active when S >= barrier
        # 5 UpAndIn Put:      active when S >= barrier
        # 6 UpAndOut Call:    active when S < barrier
        # 7 UpAndOut Put:     active when S < barrier
        # 8 European Call:    always active
        # 9 European Put:     always active
        #
        # Per strike i:
        #   Call (even): max(S - strikes[i], 0) when active, else 0.0
        #   Put  (odd):  max(strikes[i] - S, 0) when active, else 0.0
```

The `Payoff` trait updates `evaluate` signature from `evaluate(S, K, barrier) -> Float64` to `evaluate(S) -> List[Float64]` since strikes and barrier are now struct fields. Integration loops over grid points once, accumulating all strike prices simultaneously.

### Greeks Integration

The `Greeks` struct currently types the payoff parameter as `EuropeanCall`. It must be updated:
- All method signatures change `payoff: EuropeanCall` → `payoff: BarrierPayoff`
- `_price_at()` removes `K` and `barrier` params (now in payoff struct)
- `compute_delta/gamma/vega/theta` return `List[Float64]` (one per strike) instead of `Float64`
- Finite differences: perturb S (or V), re-integrate all strikes, return delta/gamma per strike

```mojo
# Before:
def compute_delta(self, grid, interp, S, V, K, barrier, payoff: EuropeanCall) -> Float64

# After:
def compute_delta(self, grid, interp, S, V, payoff: BarrierPayoff) -> List[Float64]
```

## PricingEngine Pipeline

Current flow:
```
PricingEngine.price() → PDFCache lookup → Pricer[B] → integrate payoff
```

New flow:
```
PricingEngine.price(fpe_params) → FPESolver → pdf → for each K: integrate BarrierPayoff → List[PricingResult]
```

> **Note on batch/GPU paths**: The current `Pricer[B]` with GPU batch dispatch is removed in this redesign. Batch pricing will be re-added in a future iteration when the single-option pipeline is validated. The `FPESolver[B]` dispatch (B=1 CPU, B>1 GPU) remains in the engine layer.

Steps:
1. Validate `FpeParams`
2. Compute `revised_heston()` — barrier-adjusted S_min/S_max
3. Build `FPEDomain` with `n_s`, `n_v` + revised HestonParams + barrier-informed boundary conditions (`fpe_params.s_left_cond()`, `fpe_params.s_right_cond()`)
4. Solve FPE via `FPESolver[1]` → get pdf at T (done once)
5. Build `PDFGrid` from solution
6. Create `BarrierPayoff(fpe_params.option_type, fpe_params.strikes, barrier)` — one instance for all strikes
7. Integrate payoff over grid: for each (S,V) point, `payoff.evaluate(S)` returns `List[Float64]` of all strike payoffs; accumulate prices for all strikes in a single pass
8. Compute Greeks via finite differences: perturb S/V, re-integrate all strikes, return per-strike Greeks
9. Return `List[PricingResult]` (one per strike)

### FPEDomain Changes

`FPEDomain.__init__` currently hard-codes `left_cond="dirichlet", right_cond="neumann"` for the S basis. It needs to accept boundary conditions from FpeParams:

```mojo
def __init__(
    out self,
    params: HestonParams,
    n_s: Int = 38,
    n_v: Int = 38,
    num_insert: Int = 251,
    s_left_cond: String = "dirichlet",
    s_right_cond: String = "neumann",
):
    # ...
    var r_s = RecombinationBasis[Self.degree_s](
        b_s^, left_cond=s_left_cond, right_cond=s_right_cond
    )
```

## Python API

Single entry point: `fpe_engine.price(params_dict)` where `params_dict` is a flat dict:

```python
# Single strike
result = fpe_engine.price({
    "kappa": 1.2, "theta": 0.05, "sigma": 0.35, "rho": -0.4,
    "r": 0.05, "T": 0.5, "S0": 100.0, "V0": 0.1,
    "K": 100.0,
    "barrier": 120.0, "option_type": "up_and_out_call",
    "n_s": 38, "n_v": 38, "rtol": 1e-4, "atol": 1e-6,
})
# result = {"prices": [4.52], "deltas": [0.31], "gammas": [0.02], " vegas": [0.15], "success": True}

# Multiple strikes (numpy array or list)
result = fpe_engine.price({
    "kappa": 1.2, "theta": 0.05, "sigma": 0.35, "rho": -0.4,
    "r": 0.05, "T": 0.5, "S0": 100.0, "V0": 0.1,
    "K": [95.0, 100.0, 105.0],    # list or numpy array
    "barrier": 120.0, "option_type": "up_and_out_call",
    "n_s": 38, "n_v": 38, "rtol": 1e-4, "atol": 1e-6,
})
# result = {"prices": [5.12, 4.52, 3.91], "deltas": [0.35, 0.31, 0.27], ...}
```

Mojo implementation: `py_price(params_obj: PythonObject) -> PythonObject` — extracts fields from dict, maps `option_type` string to Int, constructs `FpeParams`, calls `PricingEngine.price()`.

## C++ API

Single entry point replacing all old functions:

```c
int32_t fpe_price(
    // Heston params
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    // Option params
    const double* K, int32_t n_strikes,  // strike array, 1 or more
    double barrier,
    int32_t option_type,  // 0-9
    int32_t n_s, int32_t n_v,
    // Solver params
    double rtol, double atol,
    // Output (arrays of size n_strikes)
    double* out_prices, double* out_deltas, double* out_gammas, double* out_vegas
);
```

## Files Changed

### Modified
- `src/server/option_types.mojo` — Replace `OptionParams` with `FpeParams`, remove `RoughBergomiParams`, `NAISModel`
- `src/server/payoffs.mojo` — Replace 4 structs with `BarrierPayoff`, update `Payoff` trait
- `src/server/pricing_engine.mojo` — Simplified pipeline, no cache, direct FPE solve
- `src/server/pricer.mojo` — **Deleted** (merged into pricing_engine.mojo)
- `src/server/greeks.mojo` — Update to use `BarrierPayoff` instead of `EuropeanCall`
- `src/server/__init__.mojo` — Update re-exports
- `src/engines/fpe/domain.mojo` — Accept boundary conditions as params
- `src/engines/fpe/heston_params.mojo` — No changes (used as-is)
- `src/bindings/python_module.mojo` — Single `py_price` function
- `src/bindings/c_abi.mojo` — Single `fpe_price` function
- `src/bindings/__init__.mojo` — No changes
- `cpp/include/fpe_engine.h` — Updated header
- `python/examples/backtest.py` — Rewrite using new API
- `cpp/examples/live_trading.cpp` — Rewrite using new API

### Deleted
- `src/server/pdf_cache.mojo` — Removed (no cache needed); `PDFGrid` struct relocated to `src/server/pricing_engine.mojo`
- Old payoff structs (replaced by BarrierPayoff)

### Removed code
- `PricingRequest` struct
- `Pricer[B]` struct
- `PDFCache` / `PDFGrid` (PDFGrid retained internally in pricing_engine as local variable)
- `py_price_single`, `py_price_batch`, `py_solve_fpe` Python functions
- `fpe_init`, `fpe_destroy`, `fpe_price_single`, `fpe_price_batch` C functions
- `_seed_grid` helper (duplicated in python_module.mojo and c_abi.mojo)
- `RoughBergomiParams`, `NAISModel`, `FBSDEParams` import (NAIS-related, not FPE)

### Retained
- `PDFGrid` struct — relocated from `pdf_cache.mojo` into `pricing_engine.mojo` as a local data container (no caching, no Python serialization methods)
- `Interpolator` — retained for potential future use in Greeks computation
- `server/interpolator.mojo` — kept but not used in initial implementation

## Data Flow

```
Python dict / C args
    │
    ▼
py_price() / fpe_price()
    │
    ▼
FpeParams(heston=HestonParams(...), n_s, n_v, barrier, option_type, strikes)
│
├──► revised_heston() → HestonParams with barrier-adjusted S range
│
├──► FPEDomain(revised_heston, n_s, n_v, s_left_cond(), s_right_cond())
│        │
│        └──► RecombinationBasis with barrier-informed Dirichlet/Neumann
│
├──► FPESolver[1].solve(domain, revised_heston) → pdf (solved once)
│
├──► BarrierPayoff(option_type, strikes, barrier)
│        evaluate(S) → List[Float64] (all strike payoffs at once)
│
├──► Single integration pass: accumulate all strike prices simultaneously
│
├──► Greeks: finite differences on all strikes at once
│
└──► List[PricingResult] (one per strike)
```
