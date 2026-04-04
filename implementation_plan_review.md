# FPE Option Pricing Engine — Implementation Plan Review Report

> **Date**: 2026-04-01  
> **Reviewed Against**: [IMPLEMENTATION_PLAN.md](file:///Users/knight/Agent/FPE_option/IMPLEMENTATION_PLAN.md) v3 (unified parametric design)  
> **Scope**: Full codebase audit — architecture, layers 0–5, runtime modes, performance, testing

---

## Executive Summary

The codebase successfully **scaffolds the entire layered architecture** described in IMPLEMENTATION_PLAN.md v3. All 5 layers are present and the end-to-end FPE pipeline compiles and runs. The **unified parametric design** (`comptime batch_size` dispatch) is architecturally in place. However, the implementation is a **functional skeleton** rather than a production-grade engine — the code compiles and produces results but deviates significantly from the plan's performance-critical specifications in three fundamental areas:

1. **MAX AI Kernels (Layer 0) are not utilized** — zero calls to `kernels.linalg.matmul`, `kernels.nn.rfft`, `qr_factorization`, or any MAX compute kernels
2. **SIMD/vectorize/parallelize optimizations are absent** — all inner loops are scalar
3. **Key numerical methods are substituted** — RadauIIA → BackwardEuler, ADMM → ProjectedGradient, reverse-mode autodiff → finite-difference

| Plan Aspect | Status | Fidelity |
|---|---|---|
| Directory structure (§14) | ✅ All files present | 95% — 2 server files missing |
| Layer 0: MAX Kernels integration | ⚠️ Imports tested, **never used in production code** | 10% |
| Layer 1: Sparse Math | ✅ Functional, missing SIMD/GPU dispatch | 60% |
| Layer 2: Domain Numerics | ⚠️ Functional, major method substitutions | 50% |
| Layer 3: Engines (FPE/NAIS/Calibrator) | ⚠️ Pipeline works, stubs in GPU/NAIS | 55% |
| Layer 4: Pricing Server | ✅ Functional CPU path | 65% |
| Layer 5: Bindings | ⚠️ Partial exports | 40% |
| Unified parametric design (`comptime`) | ✅ Architecture correct, GPU stubs | 70% |
| Performance targets | ❌ No optimizations implemented | 5% |
| Testing strategy | ⚠️ Tests exist, no reference comparisons | 50% |

---

## 1. Project Overview Compliance (§1)

### Source Reconstruction

The plan specifies reconstruction of three Python codebases:

| Component | Source | Reconstruction Status |
|---|---|---|
| **FPE Solver** | [FPE_Solver_Final_Version.py](file:///Users/knight/Agent/FPE_option/FPE_Solver_Final_Version.py) (1,153 lines) | ✅ Core pipeline works: B-spline → Galerkin → ODE → PDF → pricing |
| **NAIS-Net** | [NAIS_rBM.py](file:///Users/knight/Agent/FPE_option/NAIS_rBM.py) (448 lines) | ⚠️ Architecture scaffolded, trainer is a stub, no actual training |
| **Pricing Server** | [BarrierOptionPricing.ipynb](file:///Users/knight/Agent/FPE_option/BarrierOptionPricing.ipynb) + new | ✅ CPU single pricing works with PDF cache |
| **Bindings** | New | ⚠️ Python + C ABI present but limited exports |

### Key Constraints Assessment

| Constraint | Plan Target | Current Status |
|---|---|---|
| Sub-ms single pricing | <1ms per option (cached PDF) | ⚠️ Pipeline works but no timing validation, no SIMD optimization |
| GPU batch pricing | <10ms per 1000 options | ❌ GPU pricing kernel missing, stubs to CPU |
| GPU batch calibration | N parameter sets simultaneously | ❌ CPU-only, stubs to single-thread |
| GPU portability | NVIDIA, AMD, Apple Silicon | ⚠️ `has_accelerator()` guards present, GPU kernels not functional |
| Full Mojo-native | No Python/scipy in production path | ✅ No Python dependencies in Mojo code |
| Dual bindings | Python + C++ | ✅ Both present (limited exports) |
| Extensible payoffs | Barrier + European | ✅ `Payoff` trait with 4 implementations |

---

## 2. Unified Compute Model Assessment (§2)

> [!IMPORTANT]
> The **architectural pattern** of `comptime batch_size` dispatch is correctly implemented throughout the codebase. This is the plan's central innovation. However, the **GPU paths are all stubs** that delegate back to CPU.

### Parametric Pattern Implementation

The plan's core pattern:
```mojo
struct FPESolver[batch_size: Int]:
    comptime if batch_size == 1:  # CPU path
    else:
        comptime if has_accelerator():  # GPU path
        else:  # CPU fallback
```

#### Verification across codebase:

| Struct | Plan Pattern | Implementation |
|---|---|---|
| [FPESolver[B]](file:///Users/knight/Agent/FPE_option/src/engines/fpe/solver.mojo#L128-L156) | `comptime` 3-way dispatch | ✅ Architecture correct. GPU/parallel stubs delegate to `_integrate_cpu` |
| [Pricer[B]](file:///Users/knight/Agent/FPE_option/src/server/pricer.mojo#L28-L43) | `comptime` 3-way dispatch | ✅ Architecture correct. GPU/CPU-parallel stubs delegate to `_price_single` |
| [GalerkinAssembler[B]](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo#L87) | Batch-aware M, K assembly | ⚠️ `B` parameter present but **not used** — always assembles single |
| [InitialCondition[B]](file:///Users/knight/Agent/FPE_option/src/engines/fpe/initial_cond.mojo#L70) | Batch-aware delta projection | ⚠️ `B` parameter present but **not used** — always computes single |
| [PDFComputer[B]](file:///Users/knight/Agent/FPE_option/src/engines/fpe/pdf.mojo#L19) | Batch-aware reshape | ⚠️ `B` parameter present but **not used** |
| [Greeks[B]](file:///Users/knight/Agent/FPE_option/src/server/greeks.mojo#L7) | Batch-aware finite diffs | ⚠️ `B` parameter present but **not used** |

> [!NOTE]
> The `[B]` parameter is plumbed through all structs as the plan prescribes, establishing the correct type-level infrastructure. The batch dimension just hasn't been operationalized yet — batch paths all resolve to single-element computation.

### Three Runtime Modes

| Mode | Plan | Implementation Status |
|---|---|---|
| **Mode 1: CPU Single** | `FPESolver[1].solve()` → cached PDF → `Pricer[1]` → sub-ms | ✅ **Working end-to-end**. Missing SIMD optimizations for performance target. |
| **Mode 2: GPU Batch** | `FPESolver[1]` + `Pricer[N]` GPU kernel (1 thread/option) | ❌ `gpu_pricing_kernels.mojo` **does not exist**. `Pricer._price_gpu_batch` stubs to CPU. |
| **Mode 3: GPU Calibration** | `Calibrator[B]` using `FPESolver[B]` in LM loop | ⚠️ Calibrator LM loop works on CPU. No GPU. `FPESolver[B]` stubs to single. |

---

## 3. Layer-by-Layer Analysis

### Layer 0: MAX AI Kernels (§4)

> [!WARNING]
> **Critical finding**: The plan positions MAX AI Kernels as the performance foundation ("Layer 0: Pre-built, optimized — No code to write"). The test file [test_max_integration.mojo](file:///Users/knight/Agent/FPE_option/tests/test_max_integration.mojo) verifies that MAX imports compile. However, **no production code calls any MAX kernel**.

| MAX Component | Plan Usage | Actual Usage |
|---|---|---|
| `kernels.linalg.matmul` | NN forward/backward, Jacobian | ❌ Manual loops in [stable_linear.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/nn/stable_linear.mojo), [nais_net.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/nais_net.mojo) |
| `kernels.linalg.gemv` | Sparse-dense mixed ops, PDF computation | ❌ Not used |
| `kernels.linalg.bmm` | Batch pricing: N options × matrix | ❌ Not used |
| `kernels.linalg.qr_factorization` | LM solver: QR for least-squares step | ❌ LU solve instead (custom, duplicated 4×) |
| `kernels.linalg.transpose` | Matrix operations throughout | ❌ Custom dense `_transpose` via `to_dense()` |
| `kernels.nn.rfft` / `irfft` | Volterra FFT convolution | ❌ O(N²) direct convolution in [volterra.mojo](file:///Users/knight/Agent/FPE_option/src/engines/nais/volterra.mojo) |
| `kernels.nn.activations` | NAIS-Net (sin, relu) | ❌ `std.math.sin` called directly |
| `layout.Layout` / `LayoutTensor` | All GPU tensor layout | ✅ Used in [gpu_kernels.mojo](file:///Users/knight/Agent/FPE_option/src/sparse/gpu_kernels.mojo) (sparse kernels) |
| `std.algorithm.vectorize` | SIMD loops everywhere | ✅ **Imported in test only**, never used in production |
| `std.algorithm.parallelize` | Multi-core CPU pricing | ✅ **Imported in test only**, never used in production |
| `std.random.philox` | GPU-parallel Monte Carlo | ❌ Not used |

**Impact**: The plan projects 50–2000× speedups from MAX Kernels (§1 table). Without using them, the codebase runs at essentially Python-equivalent algorithmic speed (with Mojo compilation overhead gains only).

### Layer 1: Sparse Math Library (§5)

The plan states: *"The only Layer 1 code we write. MAX Kernels focus on dense operations — sparse is our gap."*

#### 5.1 Sparse Matrix Formats

| Spec | Status | Details |
|---|---|---|
| `CSRMatrix[dtype]` fields: `data`, `indices`, `indptr`, `nrows`, `ncols`, `nnz` | ⚠️ | `nnz` is a method (`def nnz() -> Int`) not a stored field. Plan specifies `var nnz: Int`. |
| `CSRMatrix.spmv(x: Span[...])` | ⚠️ | Takes `List[Scalar[dtype]]` not `Span`. **No SIMD vectorization**. Plan: "SIMD-vectorized row dot products." |
| `CSRMatrix.to_gpu(ctx: DeviceContext) -> GPUCSRMatrix` | ❌ | Not implemented. No `GPUCSRMatrix` struct. |
| `COOMatrix.to_csr()` sort | ⚠️ | Uses insertion sort. Plan: "Uses `std.algorithm` for parallel sort." |
| `DiagMatrix` | ⚠️ | Missing `inverse()`. `matvec` renamed `diag_vec_mul`. No `vectorize`. |

#### 5.2 Sparse Operations

| Operation | Plan | Status |
|---|---|---|
| `spmv(A, x)` + SIMD | Plan: "SIMD row dot products via `vectorize`" | ❌ Scalar loop |
| `kron(A, B)` | CSR × CSR → CSR via COO | ✅ Implemented correctly |
| `spgemm(A, B)` | Row-by-row accumulation | ✅ Implemented with hash-map per row |
| `spmm(A, D)` | Sparse × dense | ✅ Implemented |
| `add(A, B)` | Merge sorted index lists | ❌ Missing (reimplemented via dense in [galerkin.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo)) |
| `scale(α, A)` | Scale data values | ❌ Missing (reimplemented via dense in [galerkin.mojo](file:///Users/knight/Agent/FPE_option/src/engines/fpe/galerkin.mojo)) |

#### 5.3 GPU Sparse Kernels

| Kernel | Status | Notes |
|---|---|---|
| [spmv_kernel](file:///Users/knight/Agent/FPE_option/src/sparse/gpu_kernels.mojo#L6-L29) | ✅ | Matches plan exactly: one thread per row, `LayoutTensor` args |
| [batch_spmv_kernel](file:///Users/knight/Agent/FPE_option/src/sparse/gpu_kernels.mojo#L32-L59) | ✅ | `global_idx.x` = row, `global_idx.y` = batch. Correct grid layout. |

> [!TIP]
> The GPU sparse kernels are the most plan-faithful code in the project. They match the IMPLEMENTATION_PLAN pseudocode almost verbatim, using `LayoutTensor`, `global_idx`, and `rebind`.

### Layer 2: Domain Numerics (§6)

#### 6.1 B-Spline Module

| Component | Plan Spec | Status | Plan Optimization | Actual |
|---|---|---|---|---|
| `GenerateKnots` | SIMD `vectorize`, `comptime` Chebyshev | ⚠️ | Scalar loops | Scalar loops, no Chebyshev |
| `BSplineBasis[degree]` | `comptime for` unrolled + SIMD batch eval | ✅/⚠️ | `comptime for` ✅, SIMD batch ❌ | `comptime for` present, no `vectorize` |
| `RecombinationBasis` | Pre-computed `comptime` recombination | ⚠️ | Not `comptime` | Runtime computation |
| `TensorProductBasis` | Fused Kronecker via `kron()` | ✅ | Uses sparse `kron` | Correct |

#### 6.2 ODE Integrator Module

The plan specifies two solvers:

| Solver | Plan | Implementation | Fidelity |
|---|---|---|---|
| `RungeKutta45` | Dormand-Prince at `comptime` | [rk45.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/rk45.mojo) | ✅ **Excellent** — full adaptive DP5(4) with `comptime` Butcher tableau, safety factor, error norm. 193 lines. |
| `RadauIIA` | 3-stage implicit Radau IIA, Newton iteration, `comptime` Butcher tableau | [radau.mojo](file:///Users/knight/Agent/FPE_option/src/numerics/ode/radau.mojo) | ❌ **Not implemented** — `RadauIIA` delegates to `BackwardEuler` (1st-order implicit + Richardson extrapolation) |

> [!CAUTION]
> The plan states: *"RadauIIA: Stiff FPE system. LU solve per step."* and *"ODE RHS uses sparse `spmv` for `M⁻¹Kq`."* The actual implementation:
> 1. Uses `BackwardEuler` not RadauIIA (order 1 vs order 5)
> 2. Computes `M⁻¹K` as a **dense matrix** upfront, not sparse `spmv` per step
> 3. The `FPESystem` ODE rhs uses **dense matvec** not sparse `spmv`
>
> This is the single biggest numerical fidelity gap. RadauIIA with sparse spmv is critical for accuracy on stiff FPE systems and is specified throughout the plan.

**Plan's FPESystem pseudocode vs actual**:

```diff
 # PLAN (§6.2):
 struct FPESystem(ODESystem):
-    var M_inv_K: CSRMatrix[DType.float64]   # pre-computed -M⁻¹K
-    def rhs(self, t, y, mut dydt):
-        self.M_inv_K.spmv_into(y, dydt)     # sparse spmv
 
 # ACTUAL (solver.mojo):
 struct FPELinearSystem(ODESystem):
+    var A: List[List[Float64]]              # dense M⁻¹K
+    def rhs(self, t, y, mut dydt):
+        for i in range(n):
+            for j in range(n):              # dense matvec O(n²)
+                acc += self.A[i][j] * y[j]
```

#### 6.3 Convex Optimizer

| Component | Plan | Implementation | Notes |
|---|---|---|---|
| `OSQP` | ADMM-based QP solver | ❌ `OSQP` wraps `ProjectedGradient` | Plan mentions ADMM; plan also mentions `ProjectedGradient` as "simpler alternative" — **the simpler path was chosen** |
| `LevenbergMarquardt` | Uses **MAX `qr_factorization`** | ❌ Custom LU solve | Plan: "uses MAX Kernels qr_factorization". Actual: hand-written LU with partial pivoting |

#### 6.4 Neural Network Runtime

| Component | Plan: "Builds ON TOP of MAX Kernels" | Actual |
|---|---|---|
| `StableLinear` | **`kernels.linalg.matmul`** for W×x | Manual `for i in for j` loop |
| `AutoDiff` | Custom reverse-mode tape: `Tape`, `Variable`, `backward()` | ❌ `GradientTape` with **finite-difference** only |
| `Adam` | `kernels.linalg.gemv` for param update | Manual loop |
| Activations | `kernels.nn.activations` | `std.math.sin` directly |
| FFT conv | `kernels.nn.rfft` + `irfft` | Direct O(N²) convolution |

### Layer 3: Engines (§7)

#### 7.1 FPE Engine

| Component | Plan Responsibility | Implemented? | Notes |
|---|---|---|---|
| `HestonParams` | Validated params with Feller check | ✅ | `feller_condition()`, `is_valid()`. Extra domain bounds. |
| `HestonParamsBatch[B]` | `InlineArray[HestonParams, B]` | ⚠️ | Uses `List` not `InlineArray` — loses comptime size guarantee |
| `FPEDomain` | Grid, knots, B-spline basis | ✅ | `build_basis()`, quadrature weights, coordinate mapping |
| `GalerkinAssembler[B]` | Batch-aware M, K assembly | ⚠️ | Works correctly for single. Heavy dense conversions (plan: sparse assembly via COO → CSR) |
| `InitialCondition[B]` | Batch-aware delta projection + OSQP | ✅ | Bivariate Gaussian + NNLS projection. Non-negativity enforced. |
| `FPESolver[B]` | Unified solver with `comptime` dispatch | ✅ | Correct 3-way dispatch. Dense `M⁻¹K` computation (plan: sparse). GPU/parallel stubs. |
| `PDFComputer[B]` | `pdf = Φ @ q(t)` | ✅ | Uses `spmv` for reconstruction — **one of few places sparse math is used correctly** |

#### 7.2 NAIS Engine

| Component | Plan Spec | Status |
|---|---|---|
| `NaisNet` | `Linear → [StableLinear + skip + sin] × L → Linear`. Uses MAX `matmul` + `activations` | ⚠️ Architecture correct. No MAX kernel calls. |
| `VolterraProcess[B]` | Fractional BM via hybrid scheme. Uses MAX `rfft`/`irfft` | ⚠️ Correct algorithm, O(N²) direct convolution instead of FFT |
| `VarianceProcess[B]` | Rough Bergomi | ✅ Correct formula |
| `FBSDELoss[B]` | Forward-backward SDE loss | ✅ Correct structure matching [NAIS_rBM.py](file:///Users/knight/Agent/FPE_option/NAIS_rBM.py) |
| `Trainer[B]` | GPU training loop: forward → loss → backward → Adam | ❌ **Dummy stub** — returns `0.8^n` without calling network |
| `Inferencer[B]` | [(t, S, V) → (price, φ, Du)](file:///Users/knight/Agent/FPE_option/cpp/examples/live_trading.cpp#5-25) | ✅ Forward pass + vol surface generation |

#### 7.3 Calibrator

| Component | Plan | Status |
|---|---|---|
| `ObjectiveFunction[B]` | [(params, market) → (residuals, jacobian)](file:///Users/knight/Agent/FPE_option/cpp/examples/live_trading.cpp#5-25) | ✅ Computes residuals via FPE solve per option |
| `Calibrator[B]` | LM loop: solve FPE[B] → price → loss → update | ✅ Full LM with central-difference Jacobian. CPU only. |
| `BatchCalibrator` | Orchestrate B parallel calibrations using `grouped_matmul`, `bmm` | ❌ Not implemented. No batch GPU calibration. |

### Layer 4: Pricing Server (§8)

| Component | Plan Spec | Status |
|---|---|---|
| `PricingEngine` | Unified `price[B]()`, `calibrate[B]()` | ⚠️ `price[B]()` present. No `calibrate[B]()`. |
| `Pricer[B]` | Unified: B=1→CPU SIMD, B=N→GPU parallel | ✅ Architecture. GPU stub. No SIMD. |
| `PDFCache` | Dict + disk serialize + `precompute()` | ⚠️ Dict lookup only. No disk I/O, no precompute. |
| `BicubicInterpolator` (plan) / `Interpolator` (actual) | SIMD bicubic via `vectorize[]` | ⚠️ **Bilinear** interpolation, no SIMD |
| `PayoffRegistry` (plan) / ad-hoc (actual) | Extensible registry with `register[P: Payoff]()` | ❌ No registry. Payoff types dispatched via `Int` enum in pricer. |
| `Greeks[B]` | δ, γ, Vega, Θ — batch-aware | ⚠️ Only δ, γ. No Vega, Θ. Hardcoded to `EuropeanCall`. |
| `VolSurfaceGenerator` | NAIS-Net implied vol output | ❌ File does not exist (`vol_surface.mojo` missing) |
| `gpu_pricing_kernels.mojo` | `payoff_integration_kernel`, `greeks_kernel` | ❌ File does not exist |

### Layer 5: Bindings (§9)

#### 9.1 Python Extension Module

Plan specifies 8 exported functions:

| Function | Status |
|---|---|
| `py_price_single` / `"price_single"` | ✅ Exported |
| `py_solve_fpe` / `"solve_fpe"` | ✅ Exported |
| `py_price_batch` / `"price_batch"` | ❌ Missing |
| `py_calibrate_batch` / `"calibrate_batch"` | ❌ Missing |
| `py_solve_fpe_batch` / `"solve_fpe_batch"` | ❌ Missing |
| `py_nais_train` / `"nais_train"` | ❌ Missing |
| `py_nais_infer` / `"nais_infer"` | ❌ Missing |
| `py_nais_vol_surface` / `"nais_vol_surface"` | ❌ Missing |

**Exported: 2 of 8** (25%)

#### 9.2 C ABI Shared Library

Plan specifies 7 exported functions:

| Function | Status |
|---|---|
| `fpe_init` | ✅ (no `config_path` param) |
| `fpe_destroy` | ✅ |
| `fpe_price_single` | ✅ (missing `vega` output) |
| `fpe_price_batch` | ❌ Missing |
| `fpe_calibrate` | ❌ Missing |
| `fpe_precompute_pdf` | ❌ Missing |
| `fpe_load_cache` | ❌ Missing |

**Exported: 3 of 7** (43%)

---

## 4. Project Structure Compliance (§14)

### Files Present vs Plan

| Directory | Plan Files | Present | Missing |
|---|---|---|---|
| `src/sparse/` | 6 | 6 (✅) | — |
| `src/numerics/bspline/` | 5 | 5 (✅) | — |
| `src/numerics/ode/` | 4 | 4 (✅) | — |
| `src/numerics/optim/` | 3 | 3 (✅) | — |
| `src/numerics/nn/` | 4 | 4 (✅) | — |
| `src/engines/fpe/` | 7 | 7 (✅) | — |
| `src/engines/nais/` | 7 | 7 (✅) | — |
| `src/engines/calibrator/` | 3 | 3 (✅) | — |
| `src/server/` | 8 | 7 (⚠️) | `vol_surface.mojo` |
| `src/bindings/` | 3 | 3 (✅) | — |
| `tests/` | 10 + ref/ | 11 (✅) | `test_nn.mojo` → `test_nais_engine.mojo` covers it; extra `test_max_integration.mojo`, `test_calibrator.mojo`, `test_bindings.mojo` |
| `benchmarks/` | 6 | 1 (❌) | 5 missing |
| `examples/` | 4 Mojo | 0 (❌) | All missing |
| `python/` | 2 + examples | 2 (⚠️) | `research.ipynb` missing |
| `cpp/` | 2 | 2 (✅) | — |

**Source file presence: 50/51 planned (98%)**  
**Total including tests/bench/examples: 55/72 planned (76%)**

---

## 5. Performance Budget Assessment (§11)

### Mode 1: CPU Single Pricing (<1ms)

| Step | Plan Budget | Code Path | Optimization Status |
|---|---|---|---|
| Cache lookup | <1μs | `PDFCache.get()` → `Dict` O(1) | ✅ Correct approach |
| PDF interpolation | <50μs | `Interpolator.interpolate()` | ❌ **Bilinear** (plan: bicubic). No SIMD. |
| Payoff integration | <100μs | `Pricer._integrate_payoff()` | ❌ No pre-computed quad weights. No SIMD dot. |
| Greeks (finite diff) | <200μs | `Greeks.compute_delta()`, `compute_gamma()` | ❌ Only 2 of 5 Greeks. No SIMD. |
| **Total** | **<400μs** | Full path exists | ❌ **Likely >400μs** — no optimizations |

### Mode 2: GPU Batch Pricing

| Metric | Plan | Actual |
|---|---|---|
| GPU payoff integration kernel | 1 thread/option | **File does not exist** |
| GPU Greeks kernel | N × 4 finite diffs | **File does not exist** |
| 1000 options <10ms | Target | ❌ CPU scalar fallback |

### Mode 3: GPU Batch Calibration

| Metric | Plan | Actual |
|---|---|---|
| B parallel FPE solves (GPU) | Thread-block per param set | ❌ Serial CPU |
| MAX `grouped_matmul` for Jacobian | Plan: "MAX Kernels used" | ❌ Manual LU solve |
| <500ms per iteration (B=64) | Target | ❌ No GPU, likely >>500ms |

### Offline FPE Solve Speedup

| Step | Python | Mojo+MAX Target | Actual Mojo | Status |
|---|---|---|---|---|
| B-spline basis | ~5s | ~50ms (100×) | No SIMD → ~1s est. | ⚠️ ~5× |
| M/K assembly | ~3s | ~30ms (100×) | Dense conversions → ~2s est. | ⚠️ ~1.5× |
| Initial cond (OSQP) | ~10s | ~200ms (50×) | Projected gradient → ~5s est. | ⚠️ ~2× |
| ODE (Radau) | ~30s | ~500ms (60×) | BackwardEuler/dense → ~10s est. | ⚠️ ~3× |
| **Total** | **~48s** | **~800ms (60×)** | **~18s est.** | ⚠️ **~2.5×** |

> [!WARNING]
> The plan targets **60× overall speedup**. Current implementation likely achieves only **~2–3×** due to missing SIMD, dense instead of sparse operations, and degraded numerical methods. The bulk of the planned speedup comes from MAX Kernels + SIMD which are not utilized.

---

## 6. Testing Strategy Assessment (§13)

### Reference Data Generation

| Item | Status | Notes |
|---|---|---|
| [generate_reference.py](file:///Users/knight/Agent/FPE_option/tests/reference/generate_reference.py) | ✅ Comprehensive | 369 lines. Generates: knots, B-spline, FPE matrices, initial cond, ODE solution, PDF grid, barrier price, NAIS processes. |
| `.npz` reference files | ❌ Not generated | `tests/reference/data/` directory empty or not present |
| Reference comparison in Mojo tests | ❌ No test loads `.npz` | Tests use hardcoded known values, not Python references |

### Unit Test Coverage

| Test File | Tests | Plan Validation | Status |
|---|---|---|---|
| [test_sparse.mojo](file:///Users/knight/Agent/FPE_option/tests/test_sparse.mojo) | 6 | spmv, COO→CSR, diag, kron, spgemm, spmm | ⚠️ Missing SIMD spmv test, kron identity, from_dense roundtrip |
| [test_bspline.mojo](file:///Users/knight/Agent/FPE_option/tests/test_bspline.mojo) | 5 | degree-1 values, partition of unity, knots, recombination, tensor product | ⚠️ Missing: degree-3 tests, reference comparison ("Output matches Python to 1e-10") |
| [test_ode.mojo](file:///Users/knight/Agent/FPE_option/tests/test_ode.mojo) | 3 | exp decay, linear growth, stiff decay | ⚠️ Missing: "match scipy" validation, Van der Pol μ=1000 |
| [test_optim.mojo](file:///Users/knight/Agent/FPE_option/tests/test_optim.mojo) | 3 | NNLS, negative target, LM linear fit | ⚠️ Missing: "Match CVXPY output" |
| [test_max_integration.mojo](file:///Users/knight/Agent/FPE_option/tests/test_max_integration.mojo) | 8 | vectorize, parallelize, math, random, layout, GPU | ✅ Matches plan: "Verify MAX imports work" |
| [test_fpe_engine.mojo](file:///Users/knight/Agent/FPE_option/tests/test_fpe_engine.mojo) | 1 | Small grid end-to-end | ⚠️ Missing: "Matrix entries match Python 1e-8" |
| [test_nais_engine.mojo](file:///Users/knight/Agent/FPE_option/tests/test_nais_engine.mojo) | 4 | StableLinear shape, NaisNet forward, Volterra, trainer | ⚠️ Missing: "Forward/backward match TF", "Forward pass matches TF" |
| [test_pricing_server.mojo](file:///Users/knight/Agent/FPE_option/tests/test_pricing_server.mojo) | 4 | interpolation, payoffs, integration | ✅ Good functional tests |
| [test_calibrator.mojo](file:///Users/knight/Agent/FPE_option/tests/test_calibrator.mojo) | 1 | Synthetic market convergence | ❌ `max_iter=0` — **does not test calibration** |
| [test_bindings.mojo](file:///Users/knight/Agent/FPE_option/tests/test_bindings.mojo) | 2 | Cache miss, cached PDF pricing | ✅ |
| **test_gpu_batch.mojo** | — | Plan: GPU batch matches CPU | ❌ **File does not exist** |

### Benchmark Tests

Plan specifies (§13):
```mojo
from std.benchmark import Bench, BenchConfig, BenchId, Bencher
```

Only benchmark: [bench_pricing.mojo](file:///Users/knight/Agent/FPE_option/benchmarks/bench_pricing.mojo) — a simple loop of 10,000 pricing calls. **Does not use `std.benchmark`**, has no timing output.

---

## 7. Risk Analysis Review (§12)

| Risk | Plan Mitigation | Current Status |
|---|---|---|
| MAX Kernels API stability | "Pin to v26.2; wrap behind thin adapters" | ⚠️ Pinned to v26.2 ✅, but no adapters (MAX not used) |
| Sparse matrix correctness | "Bit-exact vs scipy at every operation" | ⚠️ Correctness tests exist, no scipy comparison |
| ODE solver numerical stability | "Same Butcher tableaux as scipy; step-by-step validation" | ❌ RadauIIA not implemented. BackwardEuler is order-1. |
| GPU batch SpMV performance | "Start with simple row-per-thread" | ✅ Kernel implemented correctly |
| Autograd complexity | "Start with manual backprop for NAIS-Net" | ❌ Finite-difference instead of manual backprop |
| Cross-platform GPU | "Test NVIDIA + Apple Silicon" | ⚠️ `has_accelerator()` guards present, not tested |
| MAX Kernels rfft precision | "Validate FFT output vs scipy" | ❌ rfft not used |
| Batch pricing GPU memory | "Stream batches if >10K" | ❌ No GPU pricing |

---

## 8. Code Quality Observations

### Positive Patterns

1. **Consistent parametric architecture** — the `[B: Int]` parameter flows cleanly through all engine structs
2. **`comptime if` dispatch** correctly structured in solver and pricer
3. **`comptime` Butcher tableau** in RK45 — good use of Mojo's compile-time features
4. **`comptime for` in B-spline** De Boor-Cox recursion — matches plan exactly
5. **Clean trait design** — `ODESystem`, `Payoff`, `ResidualCallable`, `JacobianCallable`
6. **Proper `@fieldwise_init`** usage throughout
7. **`@export` C ABI** functions with correct `UnsafePointer` signatures

### Structural Issues

| Issue | Files Affected | Plan Violation |
|---|---|---|
| **LU solve duplicated 4 times** | `radau.mojo`, `lm.mojo`, `solver.mojo`, `calibrator.mojo` | Plan: use MAX `qr_factorization` |
| **`_abs`, `_max`, `_min`, `_zeros`, `_copy_vec` duplicated** | 7+ files | Should be shared utilities |
| **Dense round-trips in Galerkin** | `galerkin.mojo` (`_transpose`, `_scale`, `_add`) | Plan: "Sparse assembly via COO → CSR" |
| **`FPELinearSystem` dense matvec** | `solver.mojo` | Plan: "ODE RHS uses sparse `spmv`" |
| **`HestonParamsBatch[B]` uses `List`** | `heston_params.mojo` | Plan: "InlineArray[HestonParams, B]" for comptime size |

---

## 9. Gap Prioritization

### Tier 1: Architectural Correctness (blocks all quality gates)

| # | Gap | Plan Reference | Impact |
|---|---|---|---|
| 1 | **Implement RadauIIA** (3-stage implicit, Newton iteration, comptime Butcher) | §6.2, §7.1, §10 Phase 2 | FPE ODE accuracy. BackwardEuler is order-1 vs order-5. |
| 2 | **Use sparse `spmv` in ODE RHS** instead of dense `M⁻¹K` matrix | §6.2 `FPESystem` pseudocode | O(nnz) vs O(n²). Enables scaling to larger grids. |
| 3 | **Implement NAIS Trainer** (connect forward → loss → autograd → Adam) | §7.2, §10 Phase 4 | Blocks NAIS engine entirely. |
| 4 | **Implement reverse-mode autodiff** (Tape/Variable/backward) | §6.4 NN Runtime | Finite-difference doesn't scale for NN training. |

### Tier 2: MAX Kernels Integration (blocks performance targets)

| # | Gap | Plan Reference | Speedup Unlocked |
|---|---|---|---|
| 5 | **Use `kernels.linalg.matmul`** in StableLinear, NaisNet | §4, §6.4 | 50–200× for NN forward pass |
| 6 | **Use `kernels.nn.rfft`/`irfft`** in Volterra | §4, §7.2 | 100–500× for FFT convolution |
| 7 | **Use `qr_factorization`** in LM solver | §4, §6.3 | Replaces 4× duplicated LU solve |
| 8 | **Add `vectorize[]` to `spmv`** | §5.2 | 30–80× for sparse matvec (hottest loop) |
| 9 | **Add `vectorize[]` to B-spline eval** | §6.1 | 50–100× for basis evaluation |
| 10 | **Add `parallelize[]` to batch paths** | §2 | Multi-core CPU pricing/assembly |

### Tier 3: Feature Completeness (blocks production release)

| # | Gap | Plan Reference |
|---|---|---|
| 11 | Create `gpu_pricing_kernels.mojo` (payoff + Greeks GPU kernels) | §8, §14 |
| 12 | Create `vol_surface.mojo` (VolSurfaceGenerator) | §8, §14 |
| 13 | Add missing Python exports (6 of 8 functions) | §9.1 |
| 14 | Add missing C ABI exports (4 of 7 functions) | §9.2 |
| 15 | Create 5 benchmark files with `std.benchmark` | §13, §14 |
| 16 | Create 4 Mojo example files | §14 |
| 17 | Implement bicubic interpolation (currently bilinear) | §8 Mode 1 budget |
| 18 | Add Vega + Theta Greeks | §8 "∂price/∂σ, ∂price/∂T" |
| 19 | Add `PayoffRegistry` with `register[P: Payoff]()` | §8 |
| 20 | Implement `PDFCache.save_to_disk`/`load_from_disk`/`precompute` | §8 |
| 21 | Add sparse `add()` and `scale()` to `ops.mojo` | §5.2 |
| 22 | Change `HestonParamsBatch[B]` to use `InlineArray` | §7.1 |
| 23 | Generate `.npz` reference data and add comparison tests | §13 |
| 24 | Fix calibrator test (`max_iter=0` → actual iteration test) | §13 |

---

## 10. Summary Metrics

### Lines of Code by Layer

| Layer | Plan Role | Files | LoC | Assessment |
|---|---|---|---|---|
| Layer 0 (MAX Kernels) | Pre-built, optimized | 0 (external) | 0 | ❌ Available but not called |
| Layer 1 (Sparse) | Custom sparse math | 6 | ~380 | ⚠️ Functional, no SIMD |
| Layer 2 (Numerics) | B-spline, ODE, optim, NN | 13 | ~1,290 | ⚠️ Substituted methods |
| Layer 3 (Engines) | FPE, NAIS, Calibrator | 17 | ~1,333 | ⚠️ Pipeline works, stubs |
| Layer 4 (Server) | Pricing server | 7 | ~445 | ⚠️ CPU-only, missing files |
| Layer 5 (Bindings) | Python + C | 3 | ~178 | ⚠️ Partial exports |
| **Total Source** | | **46** | **~3,626** | |
| Tests | | 11 | ~935 | |
| Benchmarks | | 1 | 32 | |
| Support (Python/C++) | | 5 | 547 | |
| **Grand Total** | | **63** | **~5,140** | |

### Quality Gate Status Against Plan Phases

| Phase | Plan Weeks | Validation Criteria | Status |
|---|---|---|---|
| **Phase 1: Foundation** | 1–3 | `mojo build` succeeds; MAX imports verified; sparse tests match scipy | ⚠️ Builds ✅, MAX imports ✅, sparse tests pass (no scipy comparison) |
| **Phase 2: B-Spline + ODE** | 4–7 | "Output matches Python to 1e-10"; ODE matches scipy; OSQP matches CVXPY | ⚠️ Functional but no reference comparisons. RadauIIA not implemented. |
| **Phase 3: FPE + 3 Modes** | 8–12 | Mode 1 sub-ms; Mode 2 <10ms GPU; Mode 3 converges | ⚠️ Mode 1 works (not timed). Mode 2/3 CPU stubs. |
| **Phase 4: NAIS** | 13–17 | Forward/backward match TF; loss converges; inference matches Python | ❌ Trainer stub. No TF comparison. |
| **Phase 5: Bindings + Production** | 18–22 | Python + C bindings work; all benchmarks meet targets | ⚠️ Basic bindings work. Benchmarks missing. |

### Architectural Fidelity Score

| Dimension | Score | Rationale |
|---|---|---|
| **Structure** | 9/10 | Directory layout, file organization, module hierarchy nearly exact |
| **Type Design** | 8/10 | Parametric `[B: Int]`, traits, `comptime` dispatch all correct |
| **Algorithm Correctness** | 5/10 | Core pipeline works but RadauIIA, ADMM, autograd all substituted |
| **MAX Integration** | 1/10 | Imports verified, never called in production code |
| **Performance Optimization** | 1/10 | Zero `vectorize`, `parallelize`, or MAX kernels in hot paths |
| **GPU Readiness** | 3/10 | Sparse GPU kernels good; pricing/calibration GPU paths are stubs |
| **Test Coverage** | 5/10 | Tests exist for all modules; no reference data comparison |
| **Production Completeness** | 4/10 | Bindings partial, benchmarks missing, examples missing |
| **Overall** | **4.5/10** | Solid skeleton needing numerical methods, optimization, and GPU integration |

---

*Report generated from full codebase review against IMPLEMENTATION_PLAN.md v3 — 2026-04-01*
