# FPE Option Engine Simplification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the FPE Option Engine from 168 files/18,540 lines to ~115 files/~13,200 lines via Facade Pattern API, dead code deletion, GPU consolidation, and deduplication.

**Architecture:** Add a facade module `src/fpe_option.mojo` that re-exports a single-call API (`price`, `price_batch`, `calibrate`, `nais_train`, `nais_vol_surface`). Delete dead code (COO sparse format, RK45 ODE solver, scratch files, debug tests, excess benchmarks). Consolidate GPU dtype boilerplate into shared aliases. Deduplicate utility functions across NAIS modules.

**Tech Stack:** Mojo v0.26.3 (Nightly), Pixi for environment/dependency management, `pixi run mojo build` / `pixi run mojo test` for verification.

**Spec:** `docs/superpowers/specs/2026-05-02-fpe-simplification-design.md`

---

## File Structure

### Files to CREATE

| File | Responsibility |
|------|---------------|
| `src/server/option_types.mojo` | `OptionParams`, `RoughBergomiParams`, `NAISModel` structs |
| `src/fpe_option.mojo` | Facade module — re-exports public API + facade functions |
| `src/engines/nais/utils.mojo` | Shared NAIS helpers (`_generate_brownian_paths`, `_flatten_mat`, `_flatten_vec`, `_flatten_net_params`, `_unflatten_mat`, `_unflatten_vec`) |
| `docs/python_reference/` | Directory for relocated Python reference files |
| `tests/test_facade.mojo` | Tests for the facade API |

### Files to DELETE (production)

| File | Reason |
|------|--------|
| `src/sparse/coo.mojo` | COO format unused in production — only tested, never called |
| `src/numerics/ode/rk45.mojo` | RK45 unused — FPE uses Radau exclusively (stiff system) |
| `src/server/vol_surface.mojo` | Stub with no real implementation |

### Files to DELETE (root scratch)

| File | Reason |
|------|--------|
| `dump_matrices.mojo` | Duplicates `examples/dump_matrices.mojo` |
| `_test_bench.mojo` | Scratch file |
| `_test_max.mojo` | Scratch file |
| `_test_par.mojo` | Scratch file |
| `test_simd.mojo` | Scratch file |
| `test_time.mojo` | Scratch file |
| `test_csr_debug.mojo` | Scratch file |
| `test_gl_grid.mojo` | Scratch file |
| `test_basis_debug.mojo` | Scratch file |
| `test_radau.mojo` | Scratch file |

### Files to DELETE (tests alongside production)

| File | Reason |
|------|--------|
| `tests/test_sparse.mojo` | Tests `COOMatrix` — deleted with `coo.mojo` |
| `tests/test_sparse_coo_diag.mojo` | Tests `COOMatrix` — deleted with `coo.mojo` |
| `tests/test_ode.mojo` | Tests `RungeKutta45` — deleted with `rk45.mojo` |
| `tests/test_rk45.mojo` | Tests `RungeKutta45` — deleted with `rk45.mojo` |

### Files to DELETE (Radau debug tests — 7 files)

| File | Reason |
|------|--------|
| `tests/test_radau_debug.mojo` | Development debug traces |
| `tests/test_radau_debug_trace.mojo` | Development debug traces |
| `tests/test_radau_fortran_compare.mojo` | One-time comparison, no longer needed |
| `tests/test_radau_mass_debug.mojo` | Development debug |
| `tests/test_radau_optimized.mojo` | Superseded by main test |
| `tests/test_radau_step_debug.mojo` | Development debug |
| `tests/test_radau_hairer_tol.mojo` | Tolerance sensitivity, covered by main test |

### Files to DELETE (benchmarks — 14 files)

All except: `bench_sparse_ops.mojo`, `bench_radau.mojo`/`bench_radau_optimized.mojo`, `bench_fpe_solve.mojo`, `bench_single_pricing.mojo`

### Files to MODIFY

| File | Change |
|------|-------|
| `src/sparse/__init__.mojo` | Remove `COOMatrix` re-export; keep `CSCMatrix` |
| `src/numerics/ode/__init__.mojo` | Remove `RungeKutta45` re-export |
| `src/server/__init__.mojo` | Remove `VolSurfaceGenerator`; add `OptionParams`, `PricingResult` from `option_types` |
| `src/engines/nais/__init__.mojo` | Add `utils` module; add `RoughBergomiParams` re-export |
| `src/server/payoffs.mojo` | Delete `PayoffRegistry` struct (lines 77-82) |
| `src/gpu_utils/dtype.mojo` | Add `GPU_DTYPE`, `GPU_MAX_N`, `GPU_VEC_LAYOUT` convenience aliases |
| `src/gpu_utils/__init__.mojo` | Add re-exports of `GPU_DTYPE`, `GPU_MAX_N`, `GPU_VEC_LAYOUT` |
| 13 GPU `*_gpu.mojo` files | Replace 3-line ternary with `from gpu_utils.dtype import GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT` |
| `src/engines/nais/nais_net.mojo` | Delete `_linear` (line 23-42), use `StableLinear` directly; add `@always_inline` to `_sin_vec` |
| `src/engines/nais/inferencer.mojo` | Delete `_abs_f64` (line 24-27), use `from numerics.utils import abs_f64` |
| `src/engines/nais/trainer.mojo` | Delete `_generate_brownian_paths`, `_flatten_mat`, `_flatten_vec`, `_flatten_net_params`, `_unflatten_mat`, `_unflatten_vec`; import from `engines.nais.utils` |
| `src/engines/nais/gpu_trainer.mojo` | Delete `_generate_brownian_paths`, `_flatten_net_params`, `_unflatten_net_params`; import from `engines.nais.utils` |
| `src/server/greeks.mojo` | Replace `_price_at()` with call to `Pricer._integrate_payoff_fast()` |
| `src/server/pricer.mojo` | Replace `_compute_trap_weights` with `grid.ds_weights`/`grid.dv_weights` when available |
| `src/bindings/c_abi.mojo` | Replace `_uniform_pdf` placeholder with real FPE solver call |
| `python/fpe_engine/__init__.py` | Wire to real `price()`, `calibrate()`, `nais_train()` facades |
| `tests/test_radau_simple.mojo` | Rename to `tests/test_radau.mojo`; merge key assertions from deleted tests |

---

## Task 1: Create `src/server/option_types.mojo`

**Files:**
- Create: `src/server/option_types.mojo`
- Reference: `src/server/pricer.mojo:22-48` (existing `PricingRequest`/`PricingResult`)
- Reference: `src/engines/nais/fbsde.mojo:10-21` (existing `FBSDEParams`)
- Reference: `src/engines/nais/nais_net.mojo:59-77` (existing `NaisNet`)

- [ ] **Step 1: Write the new file**

```mojo
# src/server/option_types.mojo

from engines.nais.nais_net import NaisNet


@fieldwise_init
struct OptionParams(Copyable, Movable):
    var S: Float64
    var K: Float64
    var V: Float64
    var barrier: Float64
    var option_type: Int

    def is_valid(self) -> Bool:
        return (
            self.S > 0.0
            and self.K > 0.0
            and self.V >= 0.0
            and (self.barrier == 0.0 or self.barrier > self.S)
            and self.option_type >= 0
            and self.option_type <= 3
        )


@fieldwise_init
struct PricingResult(Copyable, Movable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool


@fieldwise_init
struct RoughBergomiParams(Copyable, Movable, Hashable):
    var H: Float64
    var eta: Float64
    var rho: Float64
    var r: Float64
    var T: Float64
    var S0: Float64
    var V0: Float64
    var epsilon_t: Float64
    var M: Int
    var N: Int
    var D: Int

    def to_fbsde_params(self) -> FBSDEParams:
        return FBSDEParams(
            Xi=[self.S0, self.V0],
            T=self.T,
            M=self.M,
            N=self.N,
            D=self.D,
            H=self.H,
            eta=self.eta,
            pho=self.rho,
            r=self.r,
            epsilon_t=self.epsilon_t,
        )


@fieldwise_init
struct NAISModel(Copyable, Movable):
    var net: NaisNet
    var params: RoughBergomiParams
```

**Note:** `PricingResult` is currently defined in `src/server/pricer.mojo:42-48`. We will later make `pricer.mojo` import it from here instead. `FBSDEParams` import requires `from engines.nais.fbsde import FBSDEParams` — we'll add that when we wire the `to_fbsde_params` method.

- [ ] **Step 2: Verify it compiles**

Run: `pixi run mojo build src/server/option_types.mojo`

Expected: Compilation succeeds (may need import fixes — the `to_fbsde_params` method needs `from engines.nais.fbsde import FBSDEParams`)

- [ ] **Step 3: Add the FBSDEParams import and fix**

```mojo
from engines.nais.fbsde import FBSDEParams
from engines.nais.nais_net import NaisNet
```

Run: `pixi run mojo build src/server/option_types.mojo`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/server/option_types.mojo
git commit -m "feat: add OptionParams, PricingResult, RoughBergomiParams, NAISModel structs"
```

---

## Task 2: Create `src/engines/nais/utils.mojo` (shared helpers)

**Files:**
- Create: `src/engines/nais/utils.mojo`
- Source: `src/engines/nais/trainer.mojo:12-83` (extract helpers)
- Source: `src/engines/nais/gpu_trainer.mojo:21-113` (duplicate helpers)

- [ ] **Step 1: Write the new file**

Extract these functions from `trainer.mojo` into `utils.mojo`:
- `_generate_brownian_paths` (line 12-29 in trainer.mojo)
- `_flatten_mat` (line 31-35)
- `_flatten_vec` (line 38-41)
- `_flatten_net_params` (line 44-74)
- `_unflatten_mat` (line 77+)
- `_unflatten_vec` (from trainer.mojo)

```mojo
# src/engines/nais/utils.mojo

from engines.nais.nais_net import NaisNet
from std.memory import alloc
from std.random import randn


def _generate_brownian_paths(M: Int, N: Int, D: Int) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    var total = M * (N + 1) * D
    var buf = alloc[Float64](total)
    randn(buf, total)

    var idx = 0
    for _ in range(M):
        var path: List[List[Float64]] = []
        for _ in range(N + 1):
            var step: List[Float64] = []
            for _ in range(D):
                step.append(buf[idx])
                idx += 1
            path.append(step^)
        out.append(path^)
    buf.free()
    return out^


def _flatten_mat(mut p: List[Float64], W: List[List[Float64]]):
    for i in range(len(W)):
        for j in range(len(W[i])):
            p.append(W[i][j])


def _flatten_vec(mut p: List[Float64], b: List[Float64]):
    for j in range(len(b)):
        p.append(b[j])


def _flatten_net_params(net: NaisNet) -> List[Float64]:
    var p: List[Float64] = []

    _flatten_mat(p, net.layer1)
    _flatten_vec(p, net.layer1_b)

    _flatten_mat(p, net.layer2.W)
    _flatten_vec(p, net.layer2.b)
    _flatten_mat(p, net.layer3.W)
    _flatten_vec(p, net.layer3.b)
    _flatten_mat(p, net.layer4.W)
    _flatten_vec(p, net.layer4.b)

    _flatten_mat(p, net.layer2_input)
    _flatten_vec(p, net.layer2_input_b)
    _flatten_mat(p, net.layer3_input)
    _flatten_vec(p, net.layer3_input_b)
    _flatten_mat(p, net.layer4_input)
    _flatten_vec(p, net.layer4_input_b)

    _flatten_mat(p, net.layer5)
    _flatten_vec(p, net.layer5_b)
    _flatten_mat(p, net.layer6)
    _flatten_vec(p, net.layer6_b)

    return p^


def _unflatten_mat(p: List[Float64], idx: Int, mut W: List[List[Float64]]) -> Int:
    var pos = idx
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = p[pos]
            pos += 1
    return pos


def _unflatten_vec(p: List[Float64], idx: Int, mut b: List[Float64]) -> Int:
    var pos = idx
    for j in range(len(b)):
        b[j] = p[pos]
        pos += 1
    return pos
```

**Note:** Copy the `_unflatten_mat` and `_unflatten_vec` implementations verbatim from `trainer.mojo:77+`.

- [ ] **Step 2: Verify it compiles**

Run: `pixi run mojo build src/engines/nais/utils.mojo`

Expected: PASS

- [ ] **Step 3: Update `trainer.mojo` — delete local copies, import from utils**

In `src/engines/nais/trainer.mojo`:
- Delete `_generate_brownian_paths` (lines 12-29)
- Delete `_flatten_mat` (lines 31-35)
- Delete `_flatten_vec` (lines 38-41)
- Delete `_flatten_net_params` (lines 44-74)
- Delete `_unflatten_mat` (lines 77+)
- Delete `_unflatten_vec` (find it in the file)
- Add import: `from engines.nais.utils import _generate_brownian_paths, _flatten_mat, _flatten_vec, _flatten_net_params, _unflatten_mat, _unflatten_vec`
- Remove unused imports: `from std.memory import alloc` (if only used by deleted function), `from std.random import randn` (if only used by deleted function)

Run: `pixi run mojo build src/engines/nais/trainer.mojo`

Expected: PASS

- [ ] **Step 4: Update `gpu_trainer.mojo` — delete local copies, import from utils**

In `src/engines/nais/gpu_trainer.mojo`:
- Delete `_generate_brownian_paths` (lines 21-39)
- Delete `_flatten_net_params` (lines 42+) — note this is a verbatim duplicate of trainer's version but written differently (inline loops vs helper calls). Replace with the canonical version from utils.
- Delete `_unflatten_net_params` if present
- Add import: `from engines.nais.utils import _generate_brownian_paths, _flatten_net_params, _unflatten_mat, _unflatten_vec`
- Remove unused imports: `from std.memory import alloc` (if only used by deleted function)

Run: `pixi run mojo build src/engines/nais/gpu_trainer.mojo`

Expected: PASS

- [ ] **Step 5: Update `src/engines/nais/__init__.mojo`**

Add: `from engines.nais.utils import _generate_brownian_paths, _flatten_net_params, _unflatten_mat, _unflatten_vec`

**Note:** Only export what's needed by external consumers. The helper functions prefixed with `_` are conventionally internal, so only add them to `__init__.mojo` if other modules outside `engines/nais/` need them. Check with:

```bash
grep -r "from engines.nais.trainer import _generate_brownian_paths\|from engines.nais.gpu_trainer import _generate_brownian_paths" src/ tests/
```

If no external consumers, skip adding to `__init__.mojo`.

- [ ] **Step 6: Run existing NAIS tests**

Run: `pixi run mojo test tests/test_nais_tracked_forward.mojo tests/test_fbsde_tracked.mojo tests/test_brownian_paths.mojo`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/engines/nais/utils.mojo src/engines/nais/trainer.mojo src/engines/nais/gpu_trainer.mojo src/engines/nais/__init__.mojo
git commit -m "refactor: extract shared NAIS helpers into utils.mojo, deduplicate from trainer + gpu_trainer"
```

---

## Task 3: GPU dtype consolidation

**Files:**
- Modify: `src/gpu_utils/dtype.mojo` (add convenience aliases)
- Modify: `src/gpu_utils/__init__.mojo` (add re-exports)
- Modify: 13 GPU `*_gpu.mojo` files (replace 3-line pattern with import)

- [ ] **Step 1: Add convenience aliases to `dtype.mojo`**

Append after line 51 (after `CPU_VEC_LAYOUT`):

```mojo
comptime GPU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
```

Run: `pixi run mojo build src/gpu_utils/dtype.mojo`

Expected: PASS

- [ ] **Step 2: Update `src/gpu_utils/__init__.mojo`**

Add to the existing imports:

```mojo
from gpu_utils.dtype import GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT
```

Run: `pixi run mojo build src/gpu_utils/__init__.mojo`

Expected: PASS

- [ ] **Step 3: Update each GPU file (one at a time, verify after each)**

For each of the 13 files, replace the 3-line ternary pattern:

**Before (example from `sparse/gpu_kernels.mojo`):**
```mojo
comptime SPARSE_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime SPARSE_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime SPARSE_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
```

**After:**
```mojo
from gpu_utils.dtype import GPU_DTYPE, GPU_MAX_N, GPU_VEC_LAYOUT

comptime SPARSE_DTYPE = GPU_DTYPE
comptime SPARSE_MAX_N = GPU_MAX_N
comptime SPARSE_VEC_LAYOUT = GPU_VEC_LAYOUT
```

**Important:** Each file uses a *different* local name (e.g., `SPARSE_DTYPE`, `PRICER_DTYPE`, `GPU_RADAU_DTYPE`). The local `comptime` alias stays — just change the RHS from the ternary to the shared constant. Also remove the `from gpu_utils.dtype import METAL_DTYPE, METAL_MAX_N, ...` and `from std.sys import has_apple_gpu_accelerator` imports if they're only used by the deleted ternary lines.

**Files to update (in order of dependency — leaf modules first):**

1. `src/sparse/gpu_kernels.mojo` — `SPARSE_DTYPE` / `SPARSE_MAX_N` / `SPARSE_VEC_LAYOUT`
2. `src/numerics/linalg_gpu.mojo` — `GPU_LA_DTYPE` / (check for MAX_N/VEC_LAYOUT)
3. `src/numerics/bspline/knots_gpu.mojo` — `GPU_KNOT_DTYPE` / ...
4. `src/numerics/ode/radau_gpu.mojo` — `GPU_RADAU_DTYPE` / ...
5. `src/engines/fpe/domain_gpu.mojo` — `GPU_DOM_DTYPE` / ...
6. `src/engines/fpe/galerkin_gpu.mojo` — `GPU_GAL_DTYPE` / ...
7. `src/engines/fpe/initial_cond_gpu.mojo` — `GPU_IC_DTYPE` / ...
8. `src/engines/fpe/pdf_gpu.mojo` — `GPU_PDF_DTYPE` / ...
9. `src/engines/fpe/gpu/executor.mojo` — `GPU_DTYPE` / ...
10. `src/engines/nais/gpu_forward_kernels.mojo` — `FORWARD_DTYPE` / ...
11. `src/engines/nais/gpu_train_kernels.mojo` — `GPU_DTYPE` / ...
12. `src/engines/calibrator/objective_gpu.mojo` — `GPU_OBJ_DTYPE` / ...
13. `src/server/gpu_pricing_kernels.mojo` — `PRICER_DTYPE` / ...

**After each file:** Run `pixi run mojo build <file>` to verify.

**Note:** Some files may only define `*_DTYPE` without `MAX_N` or `VEC_LAYOUT`. Read each file first to determine exactly which lines to replace. Only import the constants that are actually used.

- [ ] **Step 4: Run full build**

Run: `pixi run mojo build src/`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: consolidate GPU dtype boilerplate into shared gpu_utils aliases"
```

---

## Task 4: Delete dead code (production files)

**Files:**
- Delete: `src/sparse/coo.mojo`
- Delete: `src/numerics/ode/rk45.mojo`
- Delete: `src/server/vol_surface.mojo`
- Modify: `src/sparse/__init__.mojo` (remove `COOMatrix`)
- Modify: `src/numerics/ode/__init__.mojo` (remove `RungeKutta45`)
- Modify: `src/server/__init__.mojo` (remove `VolSurfaceGenerator`)

- [ ] **Step 1: Delete `src/sparse/coo.mojo`**

```bash
rm src/sparse/coo.mojo
```

- [ ] **Step 2: Update `src/sparse/__init__.mojo`**

**Before:**
```mojo
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.diag import DiagMatrix
from sparse.ops import add, scale, sparse_transpose, spgemm, kron
```

**After:**
```mojo
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.diag import DiagMatrix
from sparse.ops import add, scale, sparse_transpose, spgemm, kron
```

No changes needed — `COOMatrix` was never re-exported from `__init__.mojo`. Verify:

```bash
grep -n "COOMatrix\|coo" src/sparse/__init__.mojo
```

Expected: No matches.

- [ ] **Step 3: Delete `src/numerics/ode/rk45.mojo`**

```bash
rm src/numerics/ode/rk45.mojo
```

- [ ] **Step 4: Update `src/numerics/ode/__init__.mojo`**

**Before:**
```mojo
# FPE Engine — numerics.ode

from numerics.ode.types import ODESystem, ODESolution
from numerics.ode.rk45 import RungeKutta45
from numerics.ode.radau import RadauSparseLinearSolver
```

**After:**
```mojo
# FPE Engine — numerics.ode

from numerics.ode.types import ODESystem, ODESolution
from numerics.ode.radau import RadauSparseLinearSolver
```

- [ ] **Step 5: Delete `src/server/vol_surface.mojo`**

```bash
rm src/server/vol_surface.mojo
```

- [ ] **Step 6: Refactor `pricer.mojo` to use `PricingResult` from `option_types`**

`PricingResult` is now defined in both `src/server/pricer.mojo` (lines 42-48) and our new `src/server/option_types.mojo`. To avoid duplicate definition, remove it from `pricer.mojo`:

- Delete `PricingResult` struct definition (lines 42-48 of `pricer.mojo`)
- Add import: `from server.option_types import PricingResult`

The structs have identical fields and traits, so this is a safe re-parenting.

- [ ] **Step 7: Update `src/server/__init__.mojo`**

**Before:**
```mojo
# FPE Engine — server

from server.pdf_cache import PDFGrid, PDFCache
from server.interpolator import Interpolator
from server.payoffs import Payoff, BarrierUpAndOut, BarrierDownAndIn, EuropeanCall, EuropeanPut
from server.greeks import Greeks
from server.pricer import PricingRequest, PricingResult, Pricer
from server.pricing_engine import PricingEngine
```

**After:**
```mojo
# FPE Engine — server

from server.pdf_cache import PDFGrid, PDFCache
from server.interpolator import Interpolator
from server.payoffs import Payoff, BarrierUpAndOut, BarrierDownAndIn, EuropeanCall, EuropeanPut
from server.greeks import Greeks
from server.pricer import PricingRequest, Pricer
from server.pricing_engine import PricingEngine
from server.option_types import OptionParams, PricingResult, RoughBergomiParams, NAISModel
```

**Note:** `PricingResult` now comes from `option_types.mojo` (the canonical definition). `pricer.mojo` imports it from there too.

- [ ] **Step 8: Verify build**

Run: `pixi run mojo build src/`

Expected: PASS (now that `PricingResult` is deduplicated)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: delete COO sparse format, RK45 ODE solver, vol_surface stub"
```

---

## Task 5: Delete root scratch files

**Files:**
- Delete: 10 root `.mojo` scratch files

- [ ] **Step 1: Delete all root scratch files**

```bash
rm dump_matrices.mojo _test_bench.mojo _test_max.mojo _test_par.mojo test_simd.mojo test_time.mojo test_csr_debug.mojo test_gl_grid.mojo test_basis_debug.mojo test_radau.mojo
```

- [ ] **Step 2: Verify nothing imports these**

```bash
grep -rn "from.*dump_matrices\|from.*_test_bench\|from.*_test_max\|from.*_test_par\|from.*test_simd\|from.*test_time\|from.*test_csr_debug\|from.*test_gl_grid\|from.*test_basis_debug\|from.*test_radau" src/ tests/ benchmarks/
```

Expected: No matches.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: delete root-level scratch/test .mojo files"
```

---

## Task 6: Delete test and benchmark files

**Files:**
- Delete: 4 test files (COO/RK45 tests)
- Delete: 7 Radau debug test files
- Delete: 14 benchmark files
- Rename: `tests/test_radau_simple.mojo` → `tests/test_radau.mojo`
- Modify: `tests/test_radau.mojo` (merge key assertions from deleted tests)

- [ ] **Step 1: Delete COO and RK45 test files**

```bash
rm tests/test_sparse.mojo tests/test_sparse_coo_diag.mojo tests/test_ode.mojo tests/test_rk45.mojo
```

- [ ] **Step 2: Delete Radau debug test files**

```bash
rm tests/test_radau_debug.mojo tests/test_radau_debug_trace.mojo tests/test_radau_fortran_compare.mojo tests/test_radau_mass_debug.mojo tests/test_radau_optimized.mojo tests/test_radau_step_debug.mojo tests/test_radau_hairer_tol.mojo
```

- [ ] **Step 3: Rename and merge Radau test**

```bash
mv tests/test_radau_simple.mojo tests/test_radau.mojo
```

Read the deleted test files from git to extract key assertions for merging:

```bash
git show HEAD:tests/test_radau_hairer_tol.mojo
git show HEAD:tests/test_radau_fortran_compare.mojo
```

**Specific assertions to merge:**
1. From `test_radau_hairer_tol.mojo`: Add a `test_hairer_tolerance()` function that verifies Radau5 solution accuracy against Hairer-Wanner reference values with multiple tolerance levels (rtol=1e-4, 1e-6, 1e-8). The key assertion pattern is: `assert abs(sol - reference) < rtol * scale_factor`
2. From `test_radau_fortran_compare.mojo`: Add a `test_fortran_reference()` function that compares Mojo Radau5 output against known Fortran RADAU5 reference solutions (typically the AREN routine test problem with known solution at t=1.0). The key assertion is comparing the final state vector element-by-element with tolerance 1e-8.

**If the deleted files are not available from git** (already cleaned), this step is optional — the existing `test_radau_simple.mojo` already covers the core Radau5 functionality.

- [ ] **Step 4: Delete excess benchmark files**

Keep only:
- `benchmarks/bench_sparse_ops.mojo`
- `benchmarks/bench_radau.mojo` (or `bench_radau_optimized.mojo` — rename whichever exists)
- `benchmarks/bench_fpe_solve.mojo`
- `benchmarks/bench_single_pricing.mojo`

```bash
rm benchmarks/bench_bspline.mojo benchmarks/bench_empty.mojo benchmarks/bench_gpu_batch_pricing.mojo benchmarks/bench_lu_n1000.mojo benchmarks/bench_n1000.mojo benchmarks/bench_nais_inference.mojo benchmarks/bench_ndarray.mojo benchmarks/bench_pricing.mojo benchmarks/bench_radau_fair.mojo benchmarks/bench_radau_large.mojo benchmarks/bench_radau_single_step.mojo benchmarks/bench_radau_very_large.mojo benchmarks/bench_sparse_vs_dense.mojo benchmarks/test_gpu_val.mojo
```

If `bench_radau.mojo` doesn't exist but `bench_radau_optimized.mojo` does:
```bash
mv benchmarks/bench_radau_optimized.mojo benchmarks/bench_radau.mojo
```

- [ ] **Step 5: Verify remaining tests still pass**

Run: `pixi run mojo test tests/test_radau.mojo`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: consolidate tests (8 Radau→1) and benchmarks (18→4), delete COO/RK45 tests"
```

---

## Task 7: Create facade module `src/fpe_option.mojo`

**Files:**
- Create: `src/fpe_option.mojo`
- Reference: `src/server/pricing_engine.mojo` (existing `PricingEngine`)
- Reference: `src/engines/fpe/heston_params.mojo` (existing `HestonParams`)
- Reference: `src/engines/fpe/solver.mojo` (existing `FPESolver`)
- Reference: `src/engines/nais/trainer.mojo` (existing `Trainer`)
- Reference: `src/engines/nais/inferencer.mojo` (existing `Inferencer`)
- Reference: `src/engines/calibrator/calibrator.mojo` (existing `Calibrator`)

**Note:** `src/__init__.mojo` does not exist in this repo. The facade module `src/fpe_option.mojo` serves as the top-level entry point. No separate `__init__.mojo` is needed.

**Note:** `PricingResult` was already moved from `pricer.mojo` to `option_types.mojo` in Task 4. This task just uses the canonical import.

- [ ] **Step 1: Read and verify API signatures**

**⚠️ The code below is provisional.** You MUST read the actual method signatures in Step 4 and adjust before running the build. Key unknowns: `HestonParams.__hash__()`, `FPEDomain[3,3]` constructor, `Trainer.train()` signature, `Inferencer.vol_surface()` method, `PricingEngine.price[]` batch size parameter.

```mojo
# src/fpe_option.mojo

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.solver import FPESolver
from engines.fpe.pdf import PDFComputer
from server.option_types import OptionParams, PricingResult, RoughBergomiParams, NAISModel
from server.pricer import PricingRequest, Pricer
from server.pricing_engine import PricingEngine
from server.pdf_cache import PDFGrid
from engines.nais.nais_net import NaisNet
from engines.nais.fbsde import FBSDEParams
from engines.nais.trainer import Trainer
from engines.nais.inferencer import Inferencer
from engines.calibrator.calibrator import Calibrator


def _solve_and_cache(
    heston: HestonParams,
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> Tuple[PricingEngine, UInt64]:
    var param_hash = heston.__hash__()
    var engine = PricingEngine()
    var domain = FPEDomain[3, 3](heston, n_s=n_s, n_v=n_v)
    var solver = FPESolver[1](rtol=rtol, atol=atol, max_step=0.1)
    var t_eval: List[Float64] = [0.0, heston.T]
    var sol = solver.solve(domain, heston, t_eval)

    var pdf: List[List[Float64]] = []
    for i in range(len(domain.s_points)):
        var row: List[Float64] = []
        for j in range(len(domain.v_points)):
            row.append(sol[len(sol) - 1][i * len(domain.v_points) + j])
        pdf.append(row^)

    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=domain.s_points.copy(),
        v_points=domain.v_points.copy(),
        T=heston.T,
        ds_weights=ds^,
        dv_weights=dv^,
    )
    grid.precompute_weights()
    engine.store_pdf(param_hash, grid^)
    return (engine^, param_hash)


def price(
    heston: HestonParams,
    option: OptionParams,
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> PricingResult:
    var engine, param_hash = _solve_and_cache(heston, n_s, n_v, rtol, atol)
    var req = PricingRequest(
        S=option.S,
        K=option.K,
        V=option.V,
        barrier=option.barrier,
        payoff_type=option.option_type,
        param_hash=param_hash,
    )
    var results = engine.price[1]([req^])
    if len(results) > 0:
        return results[0]
    return PricingResult(price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False)


def price_batch(
    heston: HestonParams,
    options: List[OptionParams],
) raises -> List[PricingResult]:
    var engine, param_hash = _solve_and_cache(heston)
    var requests: List[PricingRequest] = []
    for opt in options:
        requests.append(PricingRequest(
            S=opt.S, K=opt.K, V=opt.V,
            barrier=opt.barrier, payoff_type=opt.option_type,
            param_hash=param_hash,
        ))
    return engine.price[16](requests^)


def calibrate(
    market_prices: List[Float64],
    strikes: List[Float64],
    expiries: List[Float64],
    init: HestonParams,
    max_iter: Int = 50,
    tol: Float64 = 1e-6,
) raises -> HestonParams:
    var calibrator = Calibrator(max_iter=max_iter, tol=tol)
    return calibrator.calibrate(market_prices, strikes, expiries, init)


def nais_train(
    bergomi: RoughBergomiParams,
    iters: Int = 1000,
    lr: Float64 = 1e-3,
) raises -> NAISModel:
    var fbsde_params = bergomi.to_fbsde_params()
    var trainer = Trainer()
    var net = trainer.train(fbsde_params, iters=iters, lr=lr)
    return NAISModel(net=net^, params=bergomi)


def nais_vol_surface(
    model: NAISModel,
    strikes: List[Float64],
    expiries: List[Float64],
) raises -> List[List[Float64]]:
    var inferencer = Inferencer(model.net)
    return inferencer.vol_surface(strikes, expiries)
```

**Important caveats:**
- `HestonParams.__hash__()` — verify this exists (it has `Hashable` trait). If `Hashable` doesn't provide `__hash__()` automatically, use the `_param_hash` function from `python_module.mojo:44-56`.
- `FPEDomain[3, 3]` — the `3, 3` are B-spline degree parameters. Verify these are the correct defaults.
- `engine.price[1]` — `1` is the batch size for CPU single pricing. `engine.price[16]` — `16` is the batch size for GPU dispatch.
- `Trainer.train()` — verify the exact method signature by reading `trainer.mojo`.
- `Inferencer.vol_surface()` — verify this method exists by reading `inferencer.mojo`.

- [ ] **Step 4: Read and verify API signatures**

Read the following files to confirm exact method signatures before finalizing the facade:

```bash
# Check Trainer.train signature
grep -n "def train" src/engines/nais/trainer.mojo

# Check Inferencer methods
grep -n "def " src/engines/nais/inferencer.mojo

# Check Calibrator.calibrate signature
grep -n "def calibrate" src/engines/calibrator/calibrator.mojo

# Check FPEDomain constructor
grep -n "def __init__" src/engines/fpe/domain.mojo

# Check FPESolver constructor
grep -n "def __init__" src/engines/fpe/solver.mojo
```

Adjust facade code based on actual signatures found.

- [ ] **Step 5: Verify facade compiles**

Run: `pixi run mojo build src/fpe_option.mojo`

Expected: PASS (may require signature adjustments from Step 4)

- [ ] **Step 6: Commit**

```bash
git add src/fpe_option.mojo src/server/pricer.mojo
git commit -m "feat: add facade module fpe_option.mojo with single-call API"
```

---

## Task 8: Write facade test

**Files:**
- Create: `tests/test_facade.mojo`

- [ ] **Step 1: Write the test**

```mojo
# tests/test_facade.mojo

from fpe_option import price, HestonParams, OptionParams, PricingResult


def test_price_basic() raises:
    var heston = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )
    var option = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=0.0, option_type=1,
    )
    var result = price(heston, option)
    assert result.success, "Pricing should succeed"
    assert result.price > 0.0, "Call option price should be positive"
    assert result.delta > 0.0, "Call delta should be positive"
    assert result.vega > 0.0, "Call vega should be positive"


def test_price_barrier() raises:
    var heston = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )
    var option = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=120.0, option_type=0,
    )
    var result = price(heston, option)
    assert result.success, "Barrier pricing should succeed"
    assert result.price > 0.0, "Barrier call price should be positive"
    assert result.price < 20.0, "Barrier call should be cheaper than vanilla"


def test_option_params_validation():
    var valid = OptionParams(S=100.0, K=100.0, V=0.1, barrier=0.0, option_type=1)
    assert valid.is_valid(), "Valid params should pass"

    var invalid = OptionParams(S=-1.0, K=100.0, V=0.1, barrier=0.0, option_type=1)
    assert not invalid.is_valid(), "Negative S should fail validation"


def main() raises:
    test_option_params_validation()
    test_price_basic()
    test_price_barrier()
    print("All facade tests passed!")
```

- [ ] **Step 2: Run test**

Run: `pixi run mojo test tests/test_facade.mojo`

Expected: PASS (may need adjustments if API signatures differ)

- [ ] **Step 3: Commit**

```bash
git add tests/test_facade.mojo
git commit -m "test: add facade API tests for price, barrier pricing, param validation"
```

---

## Task 9: Partial deletions and deduplication

**Files:**
- Modify: `src/server/payoffs.mojo` (delete `PayoffRegistry`)
- Modify: `src/engines/nais/nais_net.mojo` (delete `_linear`, add `@always_inline` to `_sin_vec`)
- Modify: `src/engines/nais/inferencer.mojo` (delete `_abs_f64`)
- Modify: `src/server/greeks.mojo` (replace `_price_at` with call to pricer)
- Modify: `src/server/pricer.mojo` (use `grid.ds_weights`/`grid.dv_weights` when available)

- [ ] **Step 1: Delete `PayoffRegistry` from `payoffs.mojo`**

Delete lines 77-82 (the `PayoffRegistry` struct).

Run: `pixi run mojo build src/server/payoffs.mojo`

- [ ] **Step 2: Replace `_linear` in `nais_net.mojo` with `StableLinear`**

**Important:** `_linear` is called 6 times in the `forward()` method (lines 108, 110, 114, 118, 122, 123). We must replace each call.

The challenge: `_linear(W, b, x)` takes raw `List[List[Float64]]` / `List[Float64]`, but `StableLinear` has a `.forward()` method. We need to check the `StableLinear` interface.

Read `src/numerics/nn/stable_linear.mojo` to understand `StableLinear.forward()` signature.

**If `StableLinear` can't directly replace `_linear`** (because the raw weight layers like `layer1`, `layer5`, `layer6` are `List[List[Float64]]`, not `StableLinear`), then:
- Option A: Keep `_linear` but add `@always_inline` (it's only used in `NaisNet`)
- Option B: Wrap raw layers into `StableLinear` at init time

**Recommendation:** Read `stable_linear.mojo` first. If `StableLinear` only wraps `W` + `b` and has a compatible `.forward()`, convert `layer1`/`layer5`/`layer6` to `StableLinear`. If not, keep `_linear` with `@always_inline`.

- [ ] **Step 3: Delete `_abs_f64` from `inferencer.mojo`**

In `src/engines/nais/inferencer.mojo`:
- Delete lines 24-27 (`_abs_f64` function)
- Add import: `from numerics.utils import abs_f64`
- Replace all calls to `_abs_f64(...)` with `abs_f64(...)` throughout the file

Run: `pixi run mojo build src/engines/nais/inferencer.mojo`

- [ ] **Step 4: Add `@always_inline` to `_sin_vec` in `nais_net.mojo`**

```mojo
@always_inline
def _sin_vec(x: List[Float64]) -> List[Float64]:
```

- [ ] **Step 5: Fix `_price_at` in `greeks.mojo`**

Currently `greeks.mojo:26-53` re-implements payoff integration without the SIMD optimization that `pricer.mojo`'s `_integrate_payoff_fast` has. However, `_integrate_payoff_fast` is a method on `Pricer[B]`, which requires a `PricingRequest` and weights.

**Approach:** Instead of calling `Pricer._integrate_payoff_fast` (which would create a circular dependency `greeks → pricer → greeks`), extract the integration logic into a shared function:

Create a free function `_integrate_pdf_with_weights` in `pdf_cache.mojo` or a new `server/integration.mojo`:

```mojo
@always_inline
def integrate_payoff_on_grid(
    grid: PDFGrid,
    payoff_fn: fn(Float64, Float64, Float64) -> Float64,
    K: Float64,
    barrier: Float64,
) -> Float64:
    var price = 0.0
    for i in range(len(grid.s_points)):
        var S = grid.s_points[i]
        var payoff_val = payoff_fn(S, K, barrier)
        if payoff_val == 0.0:
            continue
        for j in range(len(grid.v_points)):
            price += grid.pdf[i][j] * payoff_val * grid.ds_weights[i] * grid.dv_weights[j]
    return price
```

Then both `Pricer._integrate_payoff_fast` and `Greeks._price_at` call this shared function.

**However**, this is complex refactoring. The simpler approach: just have `_price_at` use `grid.ds_weights`/`grid.dv_weights` instead of recomputing them. This already eliminates the duplicate weight computation:

```mojo
def _price_at(self, grid: PDFGrid, interp: Interpolator, S: Float64, V: Float64, K: Float64, barrier: Float64, payoff: EuropeanCall) -> Float64:
    _ = self
    _ = interp
    _ = S
    _ = V
    var price = 0.0
    for i in range(len(grid.s_points)):
        for j in range(len(grid.v_points)):
            var s_val = grid.s_points[i]
            var payoff_val = payoff.evaluate(s_val, K, barrier)
            price += grid.pdf[i][j] * payoff_val * grid.ds_weights[i] * grid.dv_weights[j]
    return price
```

This eliminates the weight recomputation (which was the main duplication concern) without introducing circular dependencies.

Run: `pixi run mojo build src/server/greeks.mojo`

- [ ] **Step 6: Replace `_compute_trap_weights` in `pricer.mojo` with grid weights**

In `src/server/pricer.mojo`, `_compute_trap_weights` (line 240-254) duplicates the logic in `PDFGrid.precompute_weights` (line 15-32 of `pdf_cache.mojo`).

**Change:** In `_price_single`, `_price_cpu_parallel`, and `_price_gpu_batch`, use `grid.ds_weights`/`grid.dv_weights` if they're already computed (non-empty), and fall back to `_compute_trap_weights` only if empty.

This is already done in `_price_gpu_batch` (lines 167-172). Apply the same pattern to `_price_single` and `_price_cpu_parallel`:

```mojo
var ds_weights = grid.ds_weights if len(grid.ds_weights) > 0 else self._compute_trap_weights(grid.s_points)
var dv_weights = grid.dv_weights if len(grid.dv_weights) > 0 else self._compute_trap_weights(grid.v_points)
```

Run: `pixi run mojo build src/server/pricer.mojo`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: deduplicate helpers, delete PayoffRegistry, use grid weights in greeks/pricer"
```

---

## Task 10: Fix C ABI placeholder

**Files:**
- Modify: `src/bindings/c_abi.mojo`

- [ ] **Step 1: Replace `_uniform_pdf` with real FPE solver**

Current `c_abi.mojo:6-14` uses a uniform PDF placeholder. The real pattern is in `python_module.mojo:13-41` (`_seed_grid`).

Replace `_uniform_pdf` with a function that calls the real FPE solver, similar to `python_module._seed_grid`:

```mojo
def _solve_pdf(n_s: Int, n_v: Int, params: HestonParams) raises -> List[List[Float64]]:
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v)
    var solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1)
    var t_eval: List[Float64] = [0.0, params.T]
    var sol = solver.solve(domain, params, t_eval)

    var pdf: List[List[Float64]] = []
    for i in range(n_s):
        var row: List[Float64] = []
        for j in range(n_v):
            row.append(sol[len(sol) - 1][i * n_v + j])
        pdf.append(row^)
    return pdf^
```

Then update `_seed_grid` to use `_solve_pdf` instead of `_uniform_pdf`.

Add required imports:
```mojo
from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver
```

- [ ] **Step 2: Verify build**

Run: `pixi run mojo build src/bindings/c_abi.mojo`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/bindings/c_abi.mojo
git commit -m "fix: replace C ABI uniform PDF placeholder with real FPE solver"
```

---

## Task 11: Update Python API

**Files:**
- Modify: `python/fpe_engine/__init__.py`

- [ ] **Step 1: Rewrite Python API to use facade**

Replace the 5 `NotImplementedError` stubs and the low-level `price_barrier_option` with clean facade wrappers:

```python
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
    S: float, K: float, V: float, barrier: float, option_type: int = 1,
    kappa: float = 1.2, theta: float = 0.05, sigma: float = 0.35,
    rho: float = -0.4, r: float = 0.05, T: float = 0.5,
    S0: float = 100.0, V0: float = 0.1,
    S_min: float = 50.0, S_max: float = 150.0,
    V_min: float = 1e-4, V_max: float = 1.0,
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
        "kappa": fitted.kappa, "theta": fitted.theta,
        "sigma": fitted.sigma, "rho": fitted.rho,
        "r": fitted.r, "T": fitted.T,
    }


def nais_train(
    H: float = 0.07, eta: float = 1.9, rho: float = -0.9,
    r: float = 0.05, T: float = 0.5, S0: float = 100.0,
    V0: float = 0.04, epsilon_t: float = 0.04,
    M: int = 100000, N: int = 100, D: int = 2,
    iters: int = 1000, lr: float = 1e-3,
) -> dict:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    from fpe_option import RoughBergomiParams as _RoughBergomiParams
    bergomi = _RoughBergomiParams(
        H=H, eta=eta, rho=rho, r=r, T=T, S0=S0, V0=V0,
        epsilon_t=epsilon_t, M=M, N=N, D=D,
    )
    model = _mojo_nais_train(bergomi, iters=iters, lr=lr)
    return {"status": "trained"}


def nais_vol_surface(
    strikes: list[float], expiries: list[float], model: Any = None,
) -> list[list[float]]:
    if not _MOJO_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available. Run: pixi install")
    if model is None:
        raise ValueError("Trained NAIS model required. Call nais_train() first.")
    return _mojo_nais_vol_surface(model, strikes, expiries)
```

- [ ] **Step 2: Commit**

```bash
git add python/fpe_engine/__init__.py
git commit -m "feat: rewrite Python API to use facade module, replace NotImplementedError stubs"
```

---

## Task 12: Move Python reference files

**Files:**
- Move: `FPE_Solver_Final_Version.py`, `NAIS_rBM.py` → `docs/python_reference/`
- Delete: All comparison/debug `.py` scripts

- [ ] **Step 1: Create directory and move reference files**

```bash
mkdir -p docs/python_reference
mv FPE_Solver_Final_Version.py docs/python_reference/
mv NAIS_rBM.py docs/python_reference/
```

- [ ] **Step 2: Delete comparison/debug Python scripts**

```bash
rm analysis_report.py check_pdf_drift.py clean_duplicates.py compare_both_q0.py compare_fpe.py compute_schur.py debug_compare.py debug_python_ref.py full_compare.py generate_colab_nb.py onestep_compare.py quick_compare.py run_python_ref.py run_scipy_ref.py scipy_compare.py step_by_step.py tmp_fix.py verify_schur.py
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: move Python reference files to docs/, delete debug/comparison scripts"
```

---

## Task 13: Final integration and verification

**Files:**
- All modified files

- [ ] **Step 1: Full build**

Run: `pixi run mojo build src/`

Expected: PASS

- [ ] **Step 2: Run all remaining tests**

Run: `pixi run mojo test tests/`

Expected: All PASS (some tests may need import path fixes)

- [ ] **Step 3: Run facade test**

Run: `pixi run mojo test tests/test_facade.mojo`

Expected: PASS

- [ ] **Step 4: Verify file counts**

```bash
echo "Production .mojo files:" && find src/ -name "*.mojo" | wc -l
echo "Test .mojo files:" && find tests/ -name "*.mojo" | wc -l
echo "Benchmark .mojo files:" && find benchmarks/ -name "*.mojo" | wc -l
echo "Root .mojo files:" && find . -maxdepth 1 -name "*.mojo" | wc -l
echo "Total .mojo lines:" && find src/ tests/ benchmarks/ -name "*.mojo" -exec cat {} \; | wc -l
```

Expected: ~115 production files, ~44 test files, 4 benchmarks, 0 root scratch files, ~13,200 total lines

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: final integration verification after FPE simplification"
```
