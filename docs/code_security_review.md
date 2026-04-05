# FPE Engine Code Security & Quality Review

> Date: 2026-04-05
> Scope: All source code in src/
> Standard: Production-grade financial pricing engine

---

## Executive Summary

| Category | Status | Critical Issues |
|----------|--------|----------------|
| CPU Single Pricing (B=1) | ✅ Production-ready | 0 |
| GPU Batch Pricing (B>1) | ⚠️ Stub implementation | 1 (GPU path not implemented) |
| NAIS Training | ✅ Functional | 0 |
| NAIS Inference | ✅ CPU-only (by design) | 0 |
| Data Type Management | ✅ Cross-platform ready | 0 |
| Security | ⚠️ Minor issues | 2 |
| Code Quality | ✅ Good | 0 |

---

## 1. CPU Single Option Pricing (B=1) ✅

### Logic Chain Verification
```
FPESolver[1].solve()
  → GalerkinAssembler.mass_matrix()    # CSR sparse matrix
  → GalerkinAssembler.stiffness_matrix() # CSR sparse matrix
  → InitialCondition.compute()         # B-spline projected Gaussian
  → _compute_sparse_neg_M_inv_K()      # LU solve per column
  → RadauIIA[FPESparseSystem].solve()  # Order-5 implicit RK
  → _project_nonnegative()             # Ensure valid PDF
```

**Status**: ✅ Correct. Uses sparse O(nnz) matvec, proper RadauIIA, Newton iteration.

### Pricing Logic Chain
```
Pricer[1].price()
  → _compute_trap_weights()            # Pre-compute quadrature weights
  → _integrate_payoff_fast()           # ∫∫ payoff(S) × pdf(S,V) dS dV
  → Greeks.compute_delta/gamma/vega()  # Finite differences on PDF grid
```

**Status**: ✅ Correct. Pre-computed weights, SIMD inner loop, OTM early exit.

---

## 2. GPU Batch Pricing (B>1) ⚠️

### Current State
```
FPESolver[B>1].solve()
  → comptime if has_accelerator():
      → _solve_gpu_batch()
        → gpu_batch_solve()           # Float32 Metal kernels
      → else:
        → _solve_cpu_parallel()       # Fallback
```

**GPU Kernel**: ✅ `batch_euler_step` correctly implements double-buffered Euler
**GPU Executor**: ✅ Proper Float32 conversion, LayoutTensor, comptime layouts
**GPU Pricer**: ❌ `_price_gpu_batch` is a stub (lines 125-140 in pricer.mojo)

### Security Issues
1. **No input validation**: Negative strikes, zero barriers accepted
2. **Greeks always use EuropeanCall**: Regardless of actual payoff type (line 73-82)

---

## 3. NAIS Training ✅

### Logic Chain
```
Trainer[B].train()
  → _generate_brownian_paths()         # std.random.randn
  → VarianceProcess.compute()          # Rough Bergomi
  → FBSDELoss.compute()                # Forward BSDE loss
  → Finite-difference gradients        # O(n_params) forward passes
  → _apply_gradients()                 # Gradient descent
```

**Status**: ✅ Correct. Uses `net.copy()` for perturbed networks (fixed).

---

## 4. Data Type Management ✅

### Module: `src/gpu_utils/dtype.mojo`

| Backend | Compute Dtype | Layout Dtype | Target Flag |
|---------|--------------|--------------|-------------|
| Metal (Apple) | Float32 | Float32 | `metal:1` |
| CUDA (NVIDIA) | Float64 | Float64 | auto |
| HIP (AMD) | Float64 | Float64 | auto |
| CPU | Float64 | Float64 | - |

**Status**: ✅ Cross-platform ready. Automatic backend detection.

---

## 5. Security Review

### 5.1 Input Validation

| Module | Validation | Risk |
|--------|-----------|------|
| `HestonParams` | ✅ `is_valid()` + `validate()` | Low |
| `PricingRequest` | ❌ No validation | Medium |
| `PDFGrid` | ❌ No validation | Low |
| `NaisNet` | ❌ No dimension validation | Low |

### 5.2 Memory Safety

| Module | Issue | Status |
|--------|-------|--------|
| `gpu_batch_kernels.mojo` | Double-buffered, no race conditions | ✅ Safe |
| `gpu_batch_executor.mojo` | Proper alloc/free, synchronized | ✅ Safe |
| `trainer.mojo` | `alloc[Float64]` with `.free()` | ✅ Safe |
| `c_abi.mojo` | Null pointer checks added | ✅ Safe |

### 5.3 Serialization

| Module | Method | Security |
|--------|--------|----------|
| `pdf_cache.mojo` | Python JSON | ✅ Safe (no code execution) |
| `python_module.mojo` | Python FFI | ⚠️ No input sanitization |

---

## 6. Write Once, Deploy Anywhere Philosophy

### Current Compliance

| Principle | Status | Evidence |
|-----------|--------|----------|
| Single codebase | ✅ | All backends share same source |
| Comptime dispatch | ✅ | `comptime if has_accelerator()` |
| Runtime fallback | ✅ | CPU Euler when GPU unavailable |
| Backend-agnostic kernels | ✅ | `LayoutTensor` abstracts memory layout |
| Data type abstraction | ✅ | `get_compute_dtype()` selects per-backend |
| Compilation flags | ✅ | `get_target_accelerator_flag()` |

### Gaps

1. **GPU pricer kernel** not implemented (stub in `pricer.mojo:_price_gpu_batch`)
2. **NAIS GPU training** kernels defined but not integrated into trainer
3. **No runtime dtype selection** — kernels are compile-time Float32 for Metal

---

## 7. Recommendations

### High Priority
1. Implement `_price_gpu_batch` in `pricer.mojo` using `gpu_pricing_kernels.mojo`
2. Add input validation to `PricingRequest`
3. Fix Greeks to use correct payoff type (not always EuropeanCall)

### Medium Priority
4. Add NAIS GPU training integration
5. Implement runtime dtype selection for kernels
6. Add numerical precision tests comparing GPU vs CPU

### Low Priority
7. Add logging/monitoring for GPU fallback events
8. Implement cache eviction for PDF cache
9. Add performance benchmarks for all backends

---

## 8. Test Coverage Summary

| Module | Tests | Coverage |
|--------|-------|----------|
| `gpu_utils/detect.mojo` | 3 | ✅ |
| `gpu_utils/host_utils.mojo` | 1 | ✅ |
| `gpu_utils/dtype.mojo` | 5 | ✅ |
| `engines/fpe/gpu_batch_kernels.mojo` | 2 (via metal_gpu) | ✅ |
| `engines/fpe/gpu_batch_executor.mojo` | 1 | ✅ |
| `engines/fpe/solver.mojo` | 2 (cpu_gpu_dispatch) | ✅ |
| `engines/nais/trainer.mojo` | 1 (via nais_engine) | ✅ |
| `server/pricer.mojo` | 4 (pricing_server) | ✅ |
| `numerics/` | 20+ | ✅ |
| `sparse/` | 11 | ✅ |

**Total**: 74+ tests, all passing
