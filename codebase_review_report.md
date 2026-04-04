# FPE Option Pricing Engine — Codebase Review Report

> **Date**: 2026-04-01  
> **Scope**: Full review of all source, test, benchmark, binding, and supporting files against [CODING_PLAN.md](file:///Users/knight/Agent/FPE_option/CODING_PLAN.md)  
> **Mojo Version**: v0.26.2 | **Environment**: pixi

---

## Executive Summary

The project has **all planned source files created** across all 5 phases, with **functional implementations** for the core pipeline (sparse math → B-spline → ODE → FPE solver → pricing server). The code compiles, tests are structured, and the end-to-end chain from Heston parameters to option pricing works. However, **several critical planned features are implemented as stubs, approximations, or simplified substitutes**, and key performance optimizations are absent.

| Metric | Plan | Actual | Assessment |
|---|---|---|---|
| Source files | ~36 | 51 (incl. `__init__`) | ✅ All modules present |
| Test files | ~10 | 10 | ✅ Complete set |
| Benchmark files | ~6 | 1 | ⚠️ Only [bench_pricing.mojo](file:///Users/knight/Agent/FPE_option/benchmarks/bench_pricing.mojo) |
| Example files | 4 Mojo | 0 Mojo | ❌ Missing entirely |
| Total source LoC | ~4,500 est. | ~3,800 | ⚠️ 84% — stubs reduce count |
| SIMD/vectorize usage | Extensive | **None** | ❌ Critical gap |
| GPU kernels functional | Yes | Stubs only | ⚠️ Expected for macOS dev |

---

## Phase-by-Phase Analysis

### Phase 1: Foundation ✅ Mostly Complete

#### 1.1 Project Bootstrap ✅

| Item | Status | Notes |
|---|---|---|
| [pixi.toml](file:///Users/knight/Agent/FPE_option/pixi.toml) | ✅ | Correct channels, Mojo v0.26.2 pinned |
| [mojoproject.toml](file:///Users/knight/Agent/FPE_option/mojoproject.toml) | ⚠️ | Missing `edition` field and `src` path config (plan: `src/` as source root) |
| [src/__init__.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/__init__.mojo) | ✅ | Top-level package exists (checked via sparse `__init__`) |
| MAX Kernels integration test | ✅ | [test_max_integration.mojo](file:///Users/knight/Agent/FPE_option/tests/test_max_integration.mojo) — verifies `vectorize`, `parallelize`, `Layout`, `LayoutTensor`, `has_accelerator()`, math functions |

#### 1.2 Sparse Math Core ✅ Functional, ⚠️ Missing Optimizations

| File | Plan LoC | Actual LoC | Status | Missing Items |
|---|---|---|---|---|
| [csr.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/csr.mojo) | ~200 | 80 | ⚠️ | No `spmv_into`, `get`, `transpose`, `to_gpu`. **No SIMD `vectorize[]` in `spmv`** — uses scalar loop. `nnz` is method not field. |
| [coo.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/coo.mojo) | ~100 | 98 | ✅ | Method named `append` instead of `add`. Missing `from_dense` static method. Sort uses insertion sort instead of `std.algorithm`. |
| [diag.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/diag.mojo) | ~60 | 37 | ⚠️ | Missing `inverse()` method. `matvec` renamed to `diag_vec_mul`. **No SIMD `vectorize`** — scalar loop. |
| [ops.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/ops.mojo) | ~250 | 99 | ⚠️ | Missing `add()` and `scale()` sparse ops (reimplemented via dense conversion in [galerkin.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo)). `spmm` takes `List[List[...]]` instead of a `DenseMatrix` struct. |
| [gpu_kernels.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/gpu_kernels.mojo) | ~120 | 60 | ✅ | Both `spmv_kernel` and `batch_spmv_kernel` implemented with `LayoutTensor`. Correctly guarded by GPU availability. |
| [__init__.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/__init__.mojo) | — | 6 | ✅ | Proper re-exports |
| [test_sparse.mojo](file:///Users/knight/Agent/FPE_option/tests/test_sparse.mojo) | 10 tests | 6 tests | ⚠️ | Missing: `test_csr_from_dense_roundtrip`, `test_csr_spmv_simd`, `test_kron_identity`, `test_diag_inverse` |

> [!WARNING]
> **Critical Gap**: The `spmv` implementation uses a scalar `for` loop instead of the planned `vectorize[]` + SIMD accumulation. This is the hottest inner loop in the entire FPE pipeline and directly impacts all performance targets (G1: ≥30× vs scipy).

---

### Phase 2: B-Spline + ODE + Optimizer ✅ Functional, ⚠️ Deviations

#### 2.1 B-Spline Module ✅

| File | Plan LoC | Actual LoC | Status | Notes |
|---|---|---|---|---|
| [knots.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/bspline/knots.mojo) | ~80 | 97 | ✅ | `GenerateKnots` struct with `uniform` (via `linspace`) and `non-uniform` (parabolic). Missing `chebyshev` and `from_data` methods. |
| [basis.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/bspline/basis.mojo) | ~250 | 105 | ⚠️ | Good `de_boor_cox` with `comptime for` unrolling. Missing `evaluate_batch` (SIMD batch eval), `evaluate_deriv` (only has `first_derivative_all`), `collocation_matrix`, `greville_abscissae`. |
| [recombination.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/bspline/recombination.mojo) | ~150 | 82 | ✅ | Good implementation. Supports Dirichlet/Neumann combinations. Uses `spgemm` for matrix composition. |
| [tensor_product.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/bspline/tensor_product.mojo) | ~200 | 33 | ⚠️ | Has `eval_tensor`, `partial_s`, `partial_v`. Missing `mass_matrix()` and `stiffness_components()` — these are implemented externally in [galerkin.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo). |
| [test_bspline.mojo](file:///Users/knight/Agent/FPE_option/tests/test_bspline.mojo) | 6 tests | 5 tests | ⚠️ | Missing `test_basis_vs_reference`. Has unique tests for degree-1 basis functions. |

> [!NOTE]  
> The B-spline basis uses `comptime for` to unroll the De Boor-Cox recursion — this is the correct approach from the plan. However, inner-loop SIMD vectorization is absent.

#### 2.2 ODE Integrator ⚠️ Significant Deviation

| File | Plan | Actual | Status | Notes |
|---|---|---|---|---|
| [types.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/types.mojo) | `ODESystem` trait + `ODESolution` | ✅ | ✅ | Clean trait with `rhs()` + `dim()`. `ODESolution` uses `@fieldwise_init`. |
| [rk45.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/rk45.mojo) | ~200 | 193 | ✅ | Full Dormand-Prince implementation with `comptime` Butcher tableau. Adaptive step size with safety factor. |
| [radau.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/radau.mojo) | ~300 (RadauIIA) | 237 | ⚠️ | **`RadauIIA` is a stub** — delegates to `BackwardEuler` which uses backward Euler + Richardson extrapolation, NOT the planned 3-stage implicit Radau IIA. Missing Newton iteration for implicit stages. |
| [test_ode.mojo](file:///Users/knight/Agent/FPE_option/tests/test_ode.mojo) | 4 tests | 3 tests | ⚠️ | Missing `test_radau_vs_reference`. Has `ExpDecay`, `LinearGrowth`, `StiffDecay` systems. |

> [!IMPORTANT]
> **RadauIIA is NOT implemented**. The `RadauIIA` struct simply wraps `BackwardEuler` (1st-order implicit + Richardson extrapolation). The plan called for a proper 3-stage Radau IIA method with `comptime` Butcher tableau and Newton iteration. This affects the accuracy and stability of the FPE ODE integration (Phase 3).

#### 2.3 Optimizer Module ⚠️ Different Approach

| File | Plan | Actual | Status | Notes |
|---|---|---|---|---|
| [osqp.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/optim/osqp.mojo) | ADMM-based QP (~250 LoC) | 82 | ⚠️ | **Not ADMM**. Uses `ProjectedGradient` (projected gradient descent for NNLS). `OSQP` struct delegates to `ProjectedGradient`. No sparse matrix support — works on dense `List[List[Float64]]`. |
| [lm.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/optim/lm.mojo) | ~200 | 172 | ✅ | Full Levenberg-Marquardt with LU solve, trust-region λ update. Uses traits `ResidualCallable` and `JacobianCallable`. |
| [test_optim.mojo](file:///Users/knight/Agent/FPE_option/tests/test_optim.mojo) | 5 tests | 3 tests | ⚠️ | Missing `test_osqp_vs_reference`, `test_lm_nonlinear_lsq`. |

---

### Phase 3: FPE Engine ✅ End-to-End Pipeline Works, ⚠️ Mode 2/3 Stubs

#### 3.1 FPE Core ✅

| File | Actual LoC | Status | Notes |
|---|---|---|---|
| [heston_params.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/heston_params.mojo) | 36 | ✅ | Clean struct with `feller_condition()`, `is_valid()`. `HestonParamsBatch[B]` uses `List` instead of `InlineArray`. Extra domain fields (`S_min`, `S_max`, `V_min`, `V_max`). |
| [domain.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/domain.mojo) | 107 | ✅ | Builds `TensorProductBasis`, quadrature weights, coordinate mapping. Hardcoded `degree=3`. |
| [galerkin.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo) | 156 | ✅ | Full `GalerkinAssembler` with mass and stiffness matrices. Implements Heston PDE operator (drift_s, drift_v, diff_ss, diff_vv, diff_sv). **But**: heavy use of dense conversions (`_transpose`, `_scale`, `_add` via `to_dense`/`from_dense` round-trips). |
| [initial_cond.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/initial_cond.mojo) | 88 | ✅ | Projects bivariate Gaussian onto basis using NNLS. Non-negativity enforced. |
| [solver.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/solver.mojo) | 217 | ✅ | Full `FPESolver` with `comptime` dispatch: `B==1` → CPU, `B>1 + GPU` → stub, `B>1 + no GPU` → stub. Dense LU solve for `M⁻¹K`. |
| [pdf.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/pdf.mojo) | 33 | ✅ | `PDFComputer` reconstructs PDF from coefficients via `Φ @ q(t)`. |

> [!TIP]
> The Galerkin assembler does extensive dense-sparse-dense conversions. Operations like `_transpose`, `_scale`, `_add` convert to dense, operate, then convert back to CSR. This is correct but will be very slow for larger grids. The planned `sparse.ops.add` and `sparse.ops.scale` would avoid these conversions.

#### 3.2 Mode 1 — CPU Single Pricing ✅

| File | Status | Notes |
|---|---|---|
| [pdf_cache.mojo](file:///Users/knight/Agent/FPE_option/src/server/pdf_cache.mojo) | ✅ | `PDFCache` with `Dict[UInt64, PDFGrid]`, `store/get/contains`. Missing `save_to_disk`, `load_from_disk`, `precompute`. |
| [interpolator.mojo](file:///Users/knight/Agent/FPE_option/src/server/interpolator.mojo) | ⚠️ | **Bilinear** (plan: bicubic). No SIMD. Has `interpolate_batch`. |
| [payoffs.mojo](file:///Users/knight/Agent/FPE_option/src/server/payoffs.mojo) | ⚠️ | 4 of 6 payoff types: `BarrierUpAndOut`, `BarrierDownAndIn`, `EuropeanCall`, `EuropeanPut`. Missing `BarrierUpAndIn`, `BarrierDownAndOut`. Missing `PayoffRegistry`. Simplified API (scalar S, not `Span`). |
| [greeks.mojo](file:///Users/knight/Agent/FPE_option/src/server/greeks.mojo) | ⚠️ | Delta and Gamma implemented. **Missing Vega and Theta**. Hardcoded to `EuropeanCall` payoff (not generic over `Payoff` trait). |
| [pricer.mojo](file:///Users/knight/Agent/FPE_option/src/server/pricer.mojo) | ✅ | `Pricer[B]` with `comptime` dispatch. Numeric integration of payoff over PDF grid. |
| [pricing_engine.mojo](file:///Users/knight/Agent/FPE_option/src/server/pricing_engine.mojo) | ✅ | Top-level orchestrator with cache lookup. |

#### 3.3 Mode 2 — GPU Batch Pricing ⚠️ Stub

| Item | Status |
|---|---|
| `src/server/gpu_pricing_kernels.mojo` | ❌ **File does not exist** |
| GPU batch path in [pricer.mojo](file:///Users/knight/Agent/FPE_option/src/server/pricer.mojo) | Delegates to CPU single path |

#### 3.4 Mode 3 — Calibration ✅ Functional

| File | Status | Notes |
|---|---|---|
| [objective.mojo](file:///Users/knight/Agent/FPE_option/src/engines/calibrator/objective.mojo) | ✅ | Computes residuals by solving FPE per option, integrating call price. |
| [calibrator.mojo](file:///Users/knight/Agent/FPE_option/src/engines/calibrator/calibrator.mojo) | ✅ | Full LM calibration loop with central-difference Jacobian. Parameter bounds enforced. **Duplicates LU solve code** (3rd copy). |
| [test_calibrator.mojo](file:///Users/knight/Agent/FPE_option/tests/test_calibrator.mojo) | ✅ | Synthetic market data test. Sets `max_iter=0` (identity test only). |

> [!WARNING]
> The calibrator test uses `max_iter=0`, meaning it **never actually runs calibration iterations**. It only checks that the initial parameter (1% perturbed) is within 10% of the true parameter. This does not validate the LM optimization loop.

---

### Phase 4: NAIS Engine ⚠️ Scaffolding Only

| File | Actual LoC | Status | Notes |
|---|---|---|---|
| [stable_linear.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/nn/stable_linear.mojo) | 111 | ✅ | Weight-constrained layer with `W = -(R^T R + εI)`. Pure `List[List[Float64]]` — **no MAX `matmul`**. |
| [autograd.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/nn/autograd.mojo) | 35 | ⚠️ | `GradientTape` uses **finite-difference** instead of reverse-mode autodiff. No `Tape`, `Variable`, `backward()`. |
| [adam.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/nn/adam.mojo) | 64 | ✅ | Standard Adam optimizer with bias correction. Works on flat `List[Float64]`. |
| [nais_net.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/nais_net.mojo) | 119 | ✅ | Full NAIS-Net architecture: `Linear → [StableLinear + skip + sin] × 3 → Linear × 2`. Returns [(u, phi)](file:///Users/knight/Agent/FPE_option/tests/reference/generate_reference.py#19-23). |
| [volterra.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/volterra.mojo) | 75 | ✅ | Fractional BM via direct convolution (hybrid scheme). **No MAX `rfft`/`irfft`** — uses O(N²) direct convolution. |
| [variance.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/variance.mojo) | 36 | ✅ | Rough Bergomi variance process. |
| [fbsde.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/fbsde.mojo) | 86 | ✅ | Forward-backward SDE loss with terminal condition. Matches [NAIS_rBM.py](file:///Users/knight/Agent/FPE_option/NAIS_rBM.py) structure. |
| [trainer.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/trainer.mojo) | 27 | ❌ | **Pure stub**: returns geometric sequence `1.0 * 0.8^n`. Does not call `net.forward()`, `fbsde.compute()`, or `adam.step()`. |
| [inferencer.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/inferencer.mojo) | 46 | ✅ | Online inference [(t,S,V) → (price,delta)](file:///Users/knight/Agent/FPE_option/tests/reference/generate_reference.py#19-23) and `vol_surface()` sweep. |

> [!IMPORTANT]
> The NAIS training loop is a **dummy stub** — it does not train. The autograd module uses finite-difference approximation instead of reverse-mode autodiff. Until these are implemented, the NAIS engine cannot learn from data. This blocks Quality Gate **G6**.

---

### Phase 5: Bindings + Production Polish ⚠️ Partial

#### 5.1 Python Bindings ✅

| File | Status | Notes |
|---|---|---|
| [python_module.mojo](file:///Users/knight/Agent/FPE_option/src/bindings/python_module.mojo) | ⚠️ | Exports `py_price_single` and `py_solve_fpe`. Missing: `py_price_batch`, `py_calibrate_batch`, `py_solve_fpe_batch`, `py_nais_train`, `py_nais_infer`, `py_nais_vol_surface`. |
| [python/fpe_engine/__init__.py](file:///Users/knight/Agent/FPE_option/python/fpe_engine/__init__.py) | ✅ | Type-hinted wrapper with [is_available()](file:///Users/knight/Agent/FPE_option/python/fpe_engine/__init__.py#19-21) guard. Exports [price_barrier_option](file:///Users/knight/Agent/FPE_option/python/fpe_engine/__init__.py#23-31) and [solve_fpe](file:///Users/knight/Agent/FPE_option/python/fpe_engine/__init__.py#33-61). |

#### 5.2 C ABI ⚠️

| File | Status | Notes |
|---|---|---|
| [c_abi.mojo](file:///Users/knight/Agent/FPE_option/src/bindings/c_abi.mojo) | ⚠️ | `fpe_init`, `fpe_destroy`, `fpe_price_single`. Missing: `fpe_price_batch`, `fpe_calibrate`, `fpe_precompute_pdf`, `fpe_load_cache`. |
| [fpe_engine.h](file:///Users/knight/Agent/FPE_option/cpp/include/fpe_engine.h) | ✅ | Matches exported C functions. |
| [live_trading.cpp](file:///Users/knight/Agent/FPE_option/cpp/examples/live_trading.cpp) | ✅ | Working example consumer. |

#### 5.3 Benchmarks & Examples ❌ Mostly Missing

| Planned | Status |
|---|---|
| `benchmarks/bench_sparse_ops.mojo` | ❌ Missing |
| `benchmarks/bench_bspline.mojo` | ❌ Missing |
| `benchmarks/bench_fpe_solve.mojo` | ❌ Missing |
| `benchmarks/bench_single_pricing.mojo` | ⚠️ Exists as [bench_pricing.mojo](file:///Users/knight/Agent/FPE_option/benchmarks/bench_pricing.mojo) (single file, 32 lines, no timing) |
| `benchmarks/bench_gpu_batch_pricing.mojo` | ❌ Missing |
| `benchmarks/bench_nais_inference.mojo` | ❌ Missing |
| `examples/single_price.mojo` | ❌ Missing |
| `examples/batch_price.mojo` | ❌ Missing |
| `examples/calibrate.mojo` | ❌ Missing |
| `examples/nais_train_infer.mojo` | ❌ Missing |
| [python/examples/backtest.py](file:///Users/knight/Agent/FPE_option/python/examples/backtest.py) | ✅ Present |

#### Reference Data ✅

| File | Status |
|---|---|
| [generate_reference.py](file:///Users/knight/Agent/FPE_option/tests/reference/generate_reference.py) | ✅ Comprehensive: generates knots, B-spline, FPE matrices, initial cond, ODE solution, PDF grid, barrier price, NAIS processes. 369 lines. |

---

## Cross-Cutting Concerns

### Code Quality Issues

| Issue | Severity | Details |
|---|---|---|
| **Duplicated LU solve** | 🟡 Medium | `_lu_solve` appears in 3 files: [radau.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/radau.mojo), [lm.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/optim/lm.mojo), [solver.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/solver.mojo), [calibrator.mojo](file:///Users/knight/Agent/FPE_option/tests/test_calibrator.mojo). Should be extracted to a shared `numerics/linalg/lu.mojo` module. |
| **Duplicated `_abs`, `_max`, `_min`, `_zeros`, `_copy_vec`** | 🟡 Medium | Utility functions duplicated across 6+ files. Need a shared `numerics/utils.mojo`. |
| **Dense round-trips in Galerkin** | 🟡 Medium | `_transpose`, `_scale`, `_add` all do `to_dense → operate → from_dense`. This is O(n²) for sparse matrices that should be O(nnz). |
| **Missing sparse `add`/`scale`** | 🟡 Medium | Plan specified these in [sparse/ops.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/ops.mojo) but they're missing, forcing dense fallbacks. |
| **`vol_surface.mojo` missing** | 🟡 Medium | Planned `src/server/vol_surface.mojo` (`VolSurfaceGenerator`) does not exist. The vol surface functionality is partially in [inferencer.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/inferencer.mojo). |
| **`gpu_pricing_kernels.mojo` missing** | 🟡 Medium | Planned GPU pricing kernels file does not exist. |
| **`test_gpu_batch.mojo` missing** | 🟡 Medium | No GPU-specific tests. |

### Performance Optimization Status

| Optimization | Plan | Status |
|---|---|---|
| `vectorize[]` SIMD loops | Extensive (spmv, basis eval, payoff) | ❌ **Not used anywhere** |
| `parallelize[]` CPU threading | Batch pricing, ODE parallel | ❌ Not used (test only) |
| `comptime for` unrolling | Butcher tableau, basis recursion | ✅ Used in [basis.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/bspline/basis.mojo), [rk45.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/rk45.mojo) |
| `comptime` dispatch (`B==1` / GPU / CPU) | Solver, pricer | ✅ Architecture in place |
| MAX `matmul` / `gemv` | StableLinear, Galerkin | ❌ Manual loops instead |
| MAX `rfft` / `irfft` | Volterra FFT convolution | ❌ Direct O(N²) convolution |
| `Span` arguments for zero-copy | spmv, ODE rhs | ❌ Uses `List` copies everywhere |
| `InlineArray` for batch params | `HestonParamsBatch[B]` | ❌ Uses `List` instead |

---

## Quality Gate Readiness

| Gate | Target | Ready? | Blockers |
|---|---|---|---|
| **G1: Sparse** | All sparse tests pass; spmv ≥30× scipy | ⚠️ | Tests pass (6/10 planned). **No SIMD** → performance target unlikely. Missing 4 test cases. |
| **G2: Numerics** | B-spline, ODE, optim tests pass; match Python refs to 1e-10 | ⚠️ | Core tests pass. No reference comparison tests. RadauIIA is a stub (BackwardEuler wrapper). |
| **G3: FPE Mode 1** | Single pricing <1ms cached; price matches ref to 1e-6 | ⚠️ | Pipeline works end-to-end. No timing validation. No reference comparison. Bilinear instead of bicubic. |
| **G4: FPE Mode 2** | 1000 options <10ms GPU | ❌ | GPU pricing kernels missing. Batch path stubs to CPU. |
| **G5: FPE Mode 3** | Calibration converges | ⚠️ | Calibrator code exists but test uses `max_iter=0`. Needs validation. |
| **G6: NAIS** | Training loss converges; matches TF | ❌ | Trainer is a dummy stub. Autograd is finite-difference. |
| **G7: Production** | Python + C bindings; all benchmarks | ⚠️ | Bindings exist but limited exports. Benchmarks mostly missing. |

---

## File Inventory

### Source Files: 51 (all [__init__.mojo](file:///Users/knight/Agent/FPE_option/src/server/__init__.mojo) included)

```
src/
├── bindings/        (__init__, c_abi, python_module)        = 3 files
├── engines/
│   ├── calibrator/  (__init__, calibrator, objective)       = 3 files
│   ├── fpe/         (__init__, domain, galerkin, heston_params,
│   │                 initial_cond, pdf, solver)             = 7 files
│   └── nais/        (__init__, fbsde, inferencer, nais_net,
│                     trainer, variance, volterra)           = 7 files
├── numerics/
│   ├── bspline/     (__init__, basis, knots, recombination,
│   │                 tensor_product)                        = 5 files
│   ├── nn/          (__init__, adam, autograd, stable_linear) = 4 files
│   ├── ode/         (__init__, radau, rk45, types)          = 4 files
│   └── optim/       (__init__, lm, osqp)                    = 3 files
├── server/          (__init__, greeks, interpolator, payoffs,
│                     pdf_cache, pricer, pricing_engine)     = 7 files
└── sparse/          (__init__, coo, csr, diag, gpu_kernels,
                      ops)                                   = 6 files
```

### Missing (Planned but Not Created)

| Planned File | Category |
|---|---|
| `src/server/vol_surface.mojo` | Server |
| `src/server/gpu_pricing_kernels.mojo` | GPU |
| `benchmarks/bench_sparse_ops.mojo` | Benchmark |
| `benchmarks/bench_bspline.mojo` | Benchmark |
| `benchmarks/bench_fpe_solve.mojo` | Benchmark |
| `benchmarks/bench_gpu_batch_pricing.mojo` | Benchmark |
| `benchmarks/bench_nais_inference.mojo` | Benchmark |
| `examples/single_price.mojo` | Example |
| `examples/batch_price.mojo` | Example |
| `examples/calibrate.mojo` | Example |
| `examples/nais_train_infer.mojo` | Example |
| `tests/test_gpu_batch.mojo` | Test |

---

## Priority Recommendations

### 🔴 High Priority (Correctness)

1. **Implement RadauIIA properly** — Replace `BackwardEuler` wrapper with actual 3-stage implicit Radau IIA. The FPE ODE is stiff; backward Euler + Richardson is first-order accurate.
2. **Fix calibrator test** — Set `max_iter > 0` and verify convergence. Current test is vacuous.
3. **Implement NAIS `Trainer`** — Connect forward pass → loss → autograd → Adam. Currently a dummy.
4. **Implement reverse-mode autograd** — Finite-difference does not scale for neural network training.

### 🟡 Medium Priority (Performance)

5. **Add SIMD `vectorize[]` to `spmv`** — This is the plan's signature optimization and the innermost hot loop.
6. **Implement sparse `add`/`scale` in [ops.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/ops.mojo)** — Remove dense round-trips from Galerkin assembly.
7. **Use `Span` instead of `List` copies** — Eliminate allocation overhead in ODE inner loops.
8. **Upgrade interpolator** — Bilinear → Bicubic for accuracy.

### 🟢 Low Priority (Completeness)

9. **Add missing benchmark files** — 5 of 6 planned benchmarks missing.
10. **Add Mojo examples** — All 4 planned example files missing.
11. **Extract shared utilities** — `_lu_solve`, `_abs`, `_zeros`, etc. into shared modules.
12. **Add missing payoff types** — `BarrierUpAndIn`, `BarrierDownAndOut`, `PayoffRegistry`.
13. **Add missing Greeks** — Vega, Theta.
14. **Expand Python/C bindings** — Add batch and NAIS exports.
15. **Create `vol_surface.mojo`** and `gpu_pricing_kernels.mojo`.

---

## Summary Statistics

| Category | Files | Lines of Code |
|---|---|---|
| Sparse module | 6 | ~380 |
| Numerics (bspline, ode, optim, nn) | 13 | ~1,290 |
| FPE Engine | 7 | ~637 |
| NAIS Engine | 7 | ~389 |
| Calibrator | 3 | ~307 |
| Server (pricing) | 7 | ~445 |
| Bindings | 3 | ~178 |
| **Total Source** | **46** | **~3,626** |
| Tests | 10 | ~870 |
| Benchmarks | 1 | 32 |
| Reference scripts | 1 | 369 |
| Python bindings | 2 | 110 |
| C/C++ | 2 | 68 |
| **Grand Total** | **62** | **~5,075** |

---

*Report generated from full manual codebase review against CODING_PLAN.md v3*
