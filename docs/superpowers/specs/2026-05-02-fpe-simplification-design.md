# FPE Option Engine Simplification Design

> **Approach**: Facade Pattern (Approach A)
> **Goal**: Reduce from 168 files / 18,540 lines to ~115 files / ~13,200 lines
> **Principle**: "Less is more" — single-call API like Python packages, dead code removal, GPU consolidation, deduplication

---

## 1. New Facade API

### 1.1 Top-level module: `src/fpe_option.mojo`

This is the **only file most users need to import**. It re-exports the public API:

```mojo
from fpe_option import (
    price, price_batch, calibrate,
    nais_train, nais_vol_surface,
    HestonParams, OptionParams, PricingResult,
    RoughBergomiParams, NAISModel,
)
```

### 1.2 Core types

```mojo
# src/server/option_types.mojo

@fieldwise_init
struct OptionParams(Copyable, Movable):
    var S: Float64
    var K: Float64
    var V: Float64          # variance — required by 2D FPE (S,V) PDE
    var barrier: Float64
    var option_type: Int    # 0=up_and_out_call, 1=call, 2=down_and_in_put, 3=put

@fieldwise_init
struct PricingResult(Copyable, Movable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool

@fieldwise_init
struct RoughBergomiParams(Copyable, Movable, Hashable):
    var H: Float64          # Hurst parameter (0 < H < 0.5)
    var eta: Float64         # vol of vol
    var rho: Float64         # correlation (fixes existing "pho" typo in FBSDEParams)
    var r: Float64           # risk-free rate
    var T: Float64           # maturity
    var S0: Float64          # initial spot (Xi[0] in FBSDEParams)
    var V0: Float64          # initial variance (Xi[1] if present)
    var epsilon_t: Float64   # initial forward variance
    var M: Int               # number of Monte Carlo paths
    var N: Int               # number of time steps
    var D: Int               # dimension (2 for S+V)

@fieldwise_init
struct NAISModel(Copyable, Movable):
    var net: NaisNet
    var params: RoughBergomiParams
```

**Design notes:**
- `OptionParams.option_type` uses `Int` (not `String`) to match existing `Pricer._payoff_value()` int-based dispatch (0-3)
- `OptionParams.V` is required — the FPE is a 2D PDE in (S,V), variance is essential for pricing
- `RoughBergomiParams` extends `FBSDEParams` with `S0`/`V0` aliases for `Xi`, adds `Hashable` trait, fixes `pho` → `rho` typo
- `RoughBergomiParams` includes `M`, `N`, `D` training dimensions required by `FBSDELoss` and `VarianceProcess`
- The `price()` facade internally computes `param_hash` from `HestonParams` (which already has `Hashable` trait) — no need for user to provide it

### 1.3 Facade functions

```mojo
# src/fpe_option.mojo

def price(
    heston: HestonParams,
    option: OptionParams,
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> PricingResult:
    """Single option pricing. Full pipeline: domain -> M/K -> q0 -> RADAU5 -> PDF -> integrate.
    Sensible defaults, no boilerplate."""

def price_batch(
    heston: HestonParams,
    options: List[OptionParams],
) raises -> List[PricingResult]:
    """Batch pricing with auto GPU/CPU dispatch."""

def calibrate(
    market_prices: List[Float64],
    strikes: List[Float64],
    expiries: List[Float64],
    init: HestonParams,
    max_iter: Int = 50,
    tol: Float64 = 1e-6,
) raises -> HestonParams:
    """Calibrate Heston params to market prices via Levenberg-Marquardt."""

def nais_train(
    bergomi: RoughBergomiParams,
    iters: Int = 1000,
    lr: Float64 = 1e-3,
) raises -> NAISModel:
    """Train NAIS-Net on rough Bergomi FBSDE. Returns trained model."""

def nais_vol_surface(
    model: NAISModel,
    strikes: List[Float64],
    expiries: List[Float64],
) raises -> List[List[Float64]]:
    """Generate implied vol surface from trained NAIS model."""
```

### 1.4 Usage examples

```mojo
# Mode 1: Single pricing
from fpe_option import price, HestonParams, OptionParams

var heston = HestonParams(kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
    r=0.05, T=0.5, S0=100.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0)
var option = OptionParams(S=100.0, K=100.0, V=0.1, barrier=120.0, option_type=0)  # 0=up_and_out_call
var result = price(heston, option)
print(result.price, result.delta, result.gamma, result.vega)

# Mode 2: Batch pricing
var results = price_batch(heston, options)

# Mode 3: Calibration
var fitted = calibrate(market_prices, strikes, expiries, heston)

# Mode 4: NAIS
var model = nais_train(RoughBergomiParams(H=0.07, eta=1.9, rho=-0.9, r=0.05, T=0.5, S0=100.0, V0=0.04, epsilon_t=0.04, M=100000, N=100, D=2))
var surface = nais_vol_surface(model, strikes, expiries)
```

### 1.5 Python API

```python
# python/fpe_engine/__init__.py
from fpe_option import price, HestonParams, OptionParams

result = price(
    HestonParams(kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=0.5, S0=100.0, V0=0.1),
    OptionParams(S=100.0, K=100.0, V=0.1, barrier=120.0, option_type=0)  # 0=up_and_out_call
)
# -> PricingResult(price=4.32, delta=0.52, gamma=0.03, vega=0.18, success=True)
```

---

## 2. Dead Code Deletion

### 2.1 Files to delete entirely

| File | Lines | Reason |
|------|-------|--------|
| `src/sparse/coo.mojo` | 221 | Only used by 2 test files (`test_sparse.mojo`, `test_sparse_coo_diag.mojo`). Delete both tests along with this file — COO is not used in production code (all modules produce CSR directly). |
| `src/numerics/ode/rk45.mojo` | 172 | Only used by 2 test files (`test_ode.mojo`, `test_rk45.mojo`). Delete both tests along with this file — FPE uses Radau exclusively (stiff system). |
| ~~`src/sparse/csc.mojo`~~ | ~~65~~ | ~~KEEP — actively used by `SparseLU.factorize()` and `Radau5._update_real_data()`/`_update_complex_data()`~~ |
| `src/server/vol_surface.mojo` | 20 | Stub with no real implementation |
| Root `dump_matrices.mojo` | 52 | Duplicates `examples/dump_matrices.mojo` |
| Root `_test_bench.mojo` | 11 | Scratch file |
| Root `_test_max.mojo` | 4 | Scratch file |
| Root `_test_par.mojo` | 17 | Scratch file |
| Root `test_simd.mojo` | varies | Scratch file |
| Root `test_time.mojo` | varies | Scratch file |
| Root `test_csr_debug.mojo` | varies | Scratch file |
| Root `test_gl_grid.mojo` | varies | Scratch file |
| Root `test_basis_debug.mojo` | varies | Scratch file |
| Root `test_radau.mojo` | varies | Scratch file |

### 2.1.1 Test files to delete alongside production files

| File | Reason |
|------|--------|
| `tests/test_sparse.mojo` | Tests `COOMatrix` — deleted along with `coo.mojo` |
| `tests/test_sparse_coo_diag.mojo` | Tests `COOMatrix` — deleted along with `coo.mojo` |
| `tests/test_ode.mojo` | Tests `RungeKutta45` — deleted along with `rk45.mojo` |
| `tests/test_rk45.mojo` | Tests `RungeKutta45` — deleted along with `rk45.mojo` |

### 2.2 Partial deletions (trim dead code within files)

| File | What to remove | Reason |
|------|----------------|--------|
| `src/server/payoffs.mojo` | Delete `PayoffRegistry` struct (~8 lines). Keep `Payoff` trait + 4 payoff structs. | `PayoffRegistry` is defined but never imported or used anywhere. `Pricer` uses int-based dispatch. |
| ~~`src/numerics/nn/autograd.mojo`~~ | ~~KEEP — `Tape`, `Variable`, `TapeEntry` are actively used by NAIS training (`nais_net.mojo` forward_tracked, `fbsde.mojo` FBSDELoss, `trainer.mojo` _collect_param_indices). Only `GradientTape` was not the sole consumer.~~ | ~~Originally claimed unused — incorrect. `Tape` provides reverse-mode AD used by NAIS FBSDE loss computation.~~ |
| `src/numerics/nn/__init__.mojo` | Remove re-exports of ~~`Variable`, `TapeEntry`, `Tape`~~ (kept — actively used). No changes needed. |
| `src/sparse/__init__.mojo` | Remove re-exports of `COOMatrix`. ~~Remove `CSCMatrix`~~ (kept — used by SparseLU and Radau). |

### 2.3 Update `__init__.mojo` files

Every `__init__.mojo` that re-exports deleted symbols must be updated to remove those re-exports.

---

## 3. GPU Boilerplate Consolidation

### 3.1 Problem

13 GPU files each repeat the same 3-line pattern:

```mojo
comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
```

### 3.2 Solution

Add convenience aliases to `src/gpu_utils/dtype.mojo`:

```mojo
# Convenience aliases — use these instead of repeating ternary
comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
```

### 3.3 Files to update (replace 3-line pattern with 1-line import)

| File | Current | Replace with |
|------|---------|-------------|
| `src/sparse/gpu_kernels.mojo` | 3-line ternary | `from gpu_utils.dtype import GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT` |
| `src/numerics/linalg_gpu.mojo` | same | same |
| `src/numerics/bspline/knots_gpu.mojo` | same | same |
| `src/numerics/ode/radau_gpu.mojo` | same | same |
| `src/engines/fpe/domain_gpu.mojo` | same | same |
| `src/engines/fpe/galerkin_gpu.mojo` | same | same |
| `src/engines/fpe/initial_cond_gpu.mojo` | same | same |
| `src/engines/fpe/pdf_gpu.mojo` | same | same |
| `src/engines/fpe/gpu/executor.mojo` | same | same |
| `src/engines/nais/gpu_forward_kernels.mojo` | same | same |
| `src/engines/nais/gpu_train_kernels.mojo` | same | same |
| `src/engines/calibrator/objective_gpu.mojo` | same | same |
| `src/server/gpu_pricing_kernels.mojo` | same | same |

---

## 4. Deduplication

### 4.1 Redefined utility functions

| Location | Issue | Fix |
|----------|-------|-----|
| `engines/nais/inferencer.mojo:24` | `_abs_f64(x)` redefines `numerics.utils.abs_f64` | Delete local `_abs_f64`, use `from numerics.utils import abs_f64` |
| `engines/nais/nais_net.mojo:23` | `_linear(W, b, x)` duplicates `StableLinear.forward` | Delete `_linear`, use `StableLinear` directly |
| `engines/nais/nais_net.mojo:45` | `_sin_vec(x)` — simple `sin()` map, only used in `NaisNet` | Keep but add `@always_inline` (acceptable internal helper) |

### 4.2 Duplicated Brownian path generation

| Location | Issue | Fix |
|----------|-------|-----|
| `engines/nais/trainer.mojo:12` | `_generate_brownian_paths(M, N, D)` | Move to `engines/nais/utils.mojo` |
| `engines/nais/gpu_trainer.mojo` | Same function, verbatim copy | Import from `engines/nais/utils.mojo` |

Create `src/engines/nais/utils.mojo` with `_generate_brownian_paths`, `_flatten_mat`, `_flatten_vec`, `_flatten_net_params`.

### 4.3 Duplicated trapezoidal weights

| Location | Issue | Fix |
|----------|-------|-----|
| `server/pricer.mojo:240` | `_compute_trap_weights(points)` | Use `PDFGrid.precompute_weights` logic (already in pdf_cache.mojo) or extract to shared utility |
| `server/pdf_cache.mojo` | `precompute_weights()` method | Keep as canonical, call from pricer |

### 4.4 Duplicated payoff integration in Greeks

| Location | Issue | Fix |
|----------|-------|-----|
| `server/greeks.mojo` | `_price_at()` re-implements payoff integration without SIMD | Call `Pricer._integrate_payoff_fast()` instead of re-implementing |

### 4.5 C ABI placeholder

| Location | Issue | Fix |
|----------|-------|-----|
| `bindings/c_abi.mojo` | `_uniform_pdf` is a placeholder, not real FPE | Make it call real FPE solver like `python_module.mojo` does, or delegate to `price()` facade |

---

## 5. Test Cleanup

### 5.1 Radau test consolidation (8 files → 1)

| File | Lines | Action |
|------|-------|--------|
| `tests/test_radau_simple.mojo` | 294 | **KEEP** — rename to `tests/test_radau.mojo` |
| `tests/test_radau_debug.mojo` | 280 | DELETE — development debug traces |
| `tests/test_radau_debug_trace.mojo` | 429 | DELETE — development debug traces |
| `tests/test_radau_fortran_compare.mojo` | 330 | DELETE — one-time comparison, no longer needed |
| `tests/test_radau_mass_debug.mojo` | 318 | DELETE — development debug |
| `tests/test_radau_optimized.mojo` | 313 | DELETE — superseded by main test |
| `tests/test_radau_step_debug.mojo` | 267 | DELETE — development debug |
| `tests/test_radau_hairer_tol.mojo` | 233 | DELETE — tolerance sensitivity, covered by main test |

Merge key assertions from `test_radau_hairer_tol.mojo` (tolerance validation) and `test_radau_fortran_compare.mojo` (reference comparison) into the single `test_radau.mojo`.

### 5.2 Benchmark consolidation (19 files → 4)

| File | Action |
|------|--------|
| `benchmarks/bench_sparse_ops.mojo` | **KEEP** |
| `benchmarks/bench_radau.mojo` (or `bench_radau_optimized.mojo`) | **KEEP** — rename to `bench_radau.mojo` |
| `benchmarks/bench_fpe_solve.mojo` | **KEEP** |
| `benchmarks/bench_single_pricing.mojo` | **KEEP** |
| All other 15 benchmark files | DELETE |

### 5.3 Python reference files

Move root-level `.py` files to `docs/python_reference/`:

| File | Action |
|------|--------|
| `FPE_Solver_Final_Version.py` | Move to `docs/python_reference/` |
| `NAIS_rBM.py` | Move to `docs/python_reference/` |
| All comparison/debug `.py` scripts | DELETE (e.g., `compare_fpe.py`, `debug_compare.py`, `full_compare.py`, `onestep_compare.py`, `quick_compare.py`, `step_by_step.py`, `verify_schur.py`, etc.) |

---

## 6. `__init__.mojo` Updates

Every `__init__.mojo` file needs updating to reflect the changes:

- `src/sparse/__init__.mojo`: Remove `COOMatrix` re-export. **KEEP** `CSCMatrix` (used by SparseLU + Radau).
- `src/numerics/nn/__init__.mojo`: **No changes** — `Tape`, `Variable`, `TapeEntry` are all actively used by NAIS training.
- `src/numerics/ode/__init__.mojo`: Remove `RungeKutta45` re-export (file + tests deleted).
- `src/server/__init__.mojo`: Remove `VolSurfaceGenerator` re-export; add `OptionParams`, `PricingResult`
- `src/engines/nais/__init__.mojo`: Add `RoughBergomiParams` re-export; add `utils` module
- `src/__init__.mojo` (top-level): Add `fpe_option` module re-exports

---

## 7. Mojo v0.26.3 Compliance

All new and modified code must follow:

| Rule | Convention |
|------|-----------|
| Function keyword | `def` only, never `fn` |
| Compile-time constants | `comptime X = ...`, never `alias X = ...` |
| Constructor signature | `def __init__(out self, ...)` |
| Mutable method | `def method(mut self, ...)` |
| Struct parameter access | `Self.ParamName` inside struct body |
| Auto-init decorator | `@fieldwise_init` (not `@value`) |
| String protocol | `Writable` trait, `write_to()` (not `Stringable`/`__str__`) |
| List construction | `[1, 2, 3]` literal (not `List[Int](1, 2, 3)`) |
| Copy/transfer | `.copy()` or `^` for non-ImplicitlyCopyable types |
| Error handling | Explicit `raises` on all functions that can fail |
| Imports | `from std.X import ...` (not `from X import ...`) |

---

## 8. Impact Summary

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Total `.mojo` files | 168 | ~115 | -53 |
| Total `.mojo` lines | 18,540 | ~13,200 | -5,340 |
| Test files | 57 | ~44 | -13 |
| Benchmark files | 19 | 4 | -15 |
| Root scratch files | 7 | 0 | -7 |
| GPU boilerplate copies | 13 | 0 (centralized) | -13 |
| User API steps to price | 7 manual steps | 1 function call | -6 |
| Python API stubs | 5 `NotImplementedError` | 2 real functions + 1 facade | Cleaner |

---

## 9. Dependency Graph (Post-Simplification)

```
src/fpe_option.mojo  ← User entry point
├── server/option_types.mojo  (OptionParams, PricingResult, RoughBergomiParams, NAISModel)
├── engines/fpe/
│   ├── heston_params.mojo
│   ├── domain.mojo
│   ├── galerkin.mojo
│   ├── initial_cond.mojo
│   ├── solver.mojo
│   ├── pdf.mojo
│   └── gpu/executor.mojo
├── engines/nais/
│   ├── nais_net.mojo
│   ├── fbsde.mojo
│   ├── trainer.mojo
│   ├── gpu_trainer.mojo
│   ├── inferencer.mojo
│   ├── volterra.mojo
│   ├── variance.mojo
│   └── utils.mojo  (shared helpers, NEW)
├── engines/calibrator/
│   ├── calibrator.mojo
│   └── objective.mojo
├── server/
│   ├── pricer.mojo
│   ├── pdf_cache.mojo
│   ├── interpolator.mojo
│   ├── greeks.mojo
│   └── payoffs.mojo  (trimmed)
├── numerics/
│   ├── utils.mojo
│   ├── linalg.mojo
│   ├── sparse_lu.mojo
│   ├── bspline/
│ ├── ode/ (Radau only, RK45 deleted)
│ ├── optim/
│ └── nn/ (GradientTape + Tape — both kept, Tape used by NAIS training)
├── sparse/ (CSR + CSC + Diag + ops — COO deleted)
├── gpu_utils/  (with GPU_DTYPE/GPU_MAX_N/GPU_VEC_LAYOUT aliases)
└── bindings/  (python_module.mojo, c_abi.mojo)
```

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ~~Deleting CSC/COO~~ | ~~CSC kept — actively used by SparseLU and Radau. COO deleted with its 2 test files (no production consumers).~~ |
| Removing RK45 may be needed for non-stiff problems | FPE is stiff by nature; RK45 can be re-added from git history if needed. Tests deleted alongside. |
| GPU alias consolidation may break compilation | Test `pixi run mojo build` after each change |
| ~~Tape removal~~ | ~~Tape/Variable/TapeEntry kept — actively used by NAIS training (forward_tracked, FBSDELoss, _collect_param_indices).~~ |
| Facade hides useful configurability | Internal modules remain importable for advanced users |
