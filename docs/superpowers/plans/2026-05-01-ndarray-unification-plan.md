# NDArray Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all `List[Float64]` usage in numerics/sparse/engines modules. Fuse `NDArray[Float64]` as the sole data container, keeping original function/struct names (no `_nd` suffix). Delete all `_list` and `_nd` duplicate versions.

**Architecture:** Bottom-up migration in 7 layers (L0a→L1→L2→L3→L4→L5→L6→L0b). Each layer compiles and passes tests before proceeding. Naming: keep original names (e.g., `ODESystem`, not `ODESystemND`), make them NDArray-native, delete duplicates.

**Tech Stack:** Mojo v0.26.3 (Nightly), pixi environment, NDArray[ElementType] from `src/numerics/utils/ndarray.mojo`

**Spec:** `docs/superpowers/specs/2026-05-01-ndarray-unification-design.md`

---

## File Structure

### Files Modified Per Layer

| Layer | File | Action |
|-------|------|--------|
| L0a | `src/numerics/utils/__init__.mojo` | Remove _list exports |
| L1 | `src/sparse/csr.mojo` | Add spmv_into ND, rename spmv_nd→spmv, delete List spmv/spmv_into/spmv_inplace_fixed, delete List to_dense/from_dense |
| L1 | `src/sparse/ops.mojo` | Rename diag_scale_nd→diag_scale, spmm_nd→spmm; delete List versions |
| L1 | `src/sparse/diag.mojo` | Delete List constructor, delete diag_vec_mul |
| L1 | `src/sparse/csc.mojo` | No structural changes (already generic) |
| L1 | `src/sparse/coo.mojo` | No structural changes |
| L1 | Consumers: initial_cond.mojo, solver.mojo, bspline files | Update imports/calls |
| L2 | `src/numerics/linalg.mojo` | Rename _nd→primary, delete List versions + local copy_mat/copy_vec |
| L2 | `src/numerics/sparse_lu.mojo` | Rename solve_nd→solve, delete List solve |
| L2 | `src/numerics/__init__.mojo` | Update exports |
| L2 | Consumers: lm.mojo, radau.mojo | Update imports/calls |
| L3 | `src/numerics/ode/types.mojo` | Delete ODESystem/ODESolution (List), rename ND→primary |
| L3 | `src/numerics/ode/rk45.mojo` | Delete List RK45, rename ND→primary |
| L3 | `src/numerics/ode/radau.mojo` | Delete List Radau+LinearODESystem, rename ND→primary |
| L3 | `src/numerics/ode/__init__.mojo` | Update exports |
| L3 | Consumers: solver.mojo | Update imports/calls |
| L4 | `src/engines/fpe/initial_cond.mojo` | Rewrite _delta_approx_flat, _normalize_nonnegative, compute to NDArray-native |
| L4 | `src/engines/fpe/solver.mojo` | Delete List structs/methods, rename ND→primary |
| L4 | `src/engines/fpe/objective.mojo` | Fix bug + rewrite to NDArray |
| L4 | `src/engines/fpe/pdf.mojo` | Rewrite to NDArray |
| L5 | `src/numerics/optim/lm.mojo` | Delete List traits/solve, rename ND→primary |
| L5 | `src/numerics/optim/osqp.mojo` | Delete List methods, rename ND→primary, rewrite ProjectedGradient/OSQP |
| L5 | `src/numerics/optim/calibrator.mojo` | Rewrite all to NDArray |
| L5 | `src/numerics/optim/__init__.mojo` | Update exports |
| L5 | `src/numerics/nn/stable_linear.mojo` | Delete List StableLinear, rename ND→primary, delete _matmul_vec |
| L5 | `src/numerics/nn/__init__.mojo` | Update exports |
| L6 | `src/engines/nais/nais_net.mojo` | Rewrite all to NDArray (helpers + struct fields + forward + tracked) |
| L6 | `src/engines/nais/autograd.mojo` | Rewrite Tape I/O boundary to NDArray |
| L6 | `src/engines/nais/trainer.mojo` | Update ~50+ NaisNet field accesses `[i][j]`→`[i,j]`, linspace_list→linspace |
| L6 | `src/engines/nais/gpu_trainer.mojo` | Update ~36 NaisNet field accesses, remove linspace_list import |
| L6 | `src/engines/nais/fbsde.mojo` | Update FBSDEParams.Xi + method signatures for NDArray NaisNet.forward |
| L6 | `src/engines/nais/inferencer.mojo` | Update NaisNet.forward return type handling |
| L6 | `src/engines/nais/volterra.mojo` | Rewrite 3D arrays to NDArray, replace zeros_3d_list/zeros_list |
| L6 | `src/engines/nais/variance.mojo` | Update for NDArray return from volterra.generate |
| DEFER | `src/engines/fpe/domain.mojo` | 7x linspace_list calls + 8 List[Float64] fields — defer (cascades into B-spline subsystem). Keep `linspace_list` definition for this file. |
| L0b | `src/numerics/utils/constructors.mojo` | Delete _list function definitions (except `linspace_list` — deferred for domain.mojo) |
| L0b | `src/numerics/utils/copy.mojo` | Delete _list function definitions |
| L0b | Remaining consumers | Remove _list imports (except domain.mojo) |

---

## Key Technical Constraints

- **Mojo indentation is semantic** — `else:` at wrong indent pairs with wrong `if:`. After editing any Mojo file, verify indentation with byte-level check: `python3 -c "with open('FILE') as f: lines=f.readlines(); [print(f'{i+1}: {repr(lines[i])}') for i in range(L1-1,L2)]"`
- **No `mojo check`** — only `pixi run mojo run -I src tests/test_foo.mojo`
- **`IntTuple.__getitem__` returns `IntTuple`** — use `Int(s[i])`
- **`NDArray.shape()` returns `IntTuple`** — use `Int(D.shape()[dim])`
- **No zero-length NDArray** — error cases use `NDArray[Float64](1)` placeholder
- **`CSRMatrix[dtype: DType = DType.float64]`** — bare `CSRMatrix` works but generic/trait contexts need `CSRMatrix[DType.float64]`
- **`@fieldwise_init` structs** require ALL fields as keyword args
- **`Vector` = `NDArray[Float64]`** (comptime alias in utils/__init__.mojo)
- **`copy_from_fixed(src: Self)`** for NDArray→NDArray copy
- **`memset_zero(ptr, count)`** works with element count (not byte count)
- **`UnsafePointer` has no `.offset()`** — use `p + n` pointer arithmetic

---

## Task 1: L0a — Remove _list exports from __init__.mojo

**Files:**
- Modify: `src/numerics/utils/__init__.mojo:13,15`

- [ ] **Step 1: Remove _list exports from utils/__init__.mojo**

Edit `src/numerics/utils/__init__.mojo`:
- Remove `zeros_list, zeros_mat_list, zeros_3d_list, linspace_list` from line 13
- Remove `copy_vec_list, copy_mat_list, swap_rows_list` from line 15
- Keep the primary NDArray exports on lines 12, 14

- [ ] **Step 2: Verify existing consumers still compile**

Consumers that import _list functions directly (not via __init__.mojo) will still work. Run:
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
```
Expected: 18 tests PASS

- [ ] **Step 3: Commit**
```bash
git add src/numerics/utils/__init__.mojo
git commit -m "refactor: remove _list function exports from utils/__init__.mojo"
```

---

## Task 2: L1 — CSR sparse methods migration

**Files:**
- Modify: `src/sparse/csr.mojo:74-106,108-159,309-353,164-190`
- Test: `tests/test_sparse.mojo`, `tests/test_sparse_coo_diag.mojo`

- [ ] **Step 2.1: Add spmv_into (NDArray, no zero-init)**

Add new method `spmv_into(self, x: NDArray[Float64], mut y: NDArray[Float64])` after line 190 (after `spmv_nd`). Implementation: iterate rows, compute dot product, write to `y[i]` — do NOT call `y.zero_out()` first. This matches the semantics of the List `spmv_into` (accumulate into y).

```mojo
def spmv_into(self, x: NDArray[Float64], mut y: NDArray[Float64]):
    var n = Int(self.row_ptr.shape()[0]) - 1
    for i in range(n):
        var s = 0.0
        for k in range(Int(self.row_ptr[i]), Int(self.row_ptr[i + 1])):
            s += self.data[k] * x[Int(self.col_ind[k])]
        y[i] = s
```

- [ ] **Step 2.2: Rename spmv_nd → spmv**

Replace `spmv_nd` name with `spmv` at line 164. Keep the `y.zero_out()` call — this is the API contract: `spmv` gives fresh output.

- [ ] **Step 2.3: Delete List-based spmv, spmv_into, spmv_inplace_fixed**

Delete lines 74-159 (List `spmv`, List `spmv_into`, List `spmv_inplace_fixed`).

- [ ] **Step 2.4: Rename to_dense_nd → to_dense, from_dense_nd → from_dense**

Rename at lines 355 and 362. Delete the old List-based `to_dense` (lines 309-320) and `from_dense` (lines 322-353).

- [ ] **Step 2.5: Verify indentation**

Run byte-level indentation check on csr.mojo after edits.

- [ ] **Step 2.6: Run sparse tests**
```bash
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_sparse_coo_diag.mojo
```
Expected: 15 tests PASS

- [ ] **Step 2.7: Commit**
```bash
git add src/sparse/csr.mojo
git commit -m "refactor: migrate CSR spmv/to_dense/from_dense to NDArray-primary, delete List versions"
```

---

## Task 3: L1 — Sparse ops migration

**Files:**
- Modify: `src/sparse/ops.mojo:171-198,234-265`

- [ ] **Step 3.1: Rename diag_scale_nd → diag_scale**

At line 186, rename `diag_scale_nd` to `diag_scale`. Delete the List-based `diag_scale` at lines 171-183.

- [ ] **Step 3.2: Rename spmm_nd → spmm**

At line 252, rename `spmm_nd` to `spmm`. Delete the List-based `spmm` at lines 234-249.

- [ ] **Step 3.3: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_sparse_coo_diag.mojo
```

- [ ] **Step 3.4: Commit**
```bash
git add src/sparse/ops.mojo
git commit -m "refactor: migrate diag_scale/spmm to NDArray-primary, delete List versions"
```

---

## Task 4: L1 — DiagMatrix + consumer updates

**Files:**
- Modify: `src/sparse/diag.mojo:18-28`
- Modify: Consumers that call List-based sparse methods (search with grep)

- [ ] **Step 4.1: Delete List constructor and diag_vec_mul from diag.mojo**

Delete `__init__(out self, var values: List[Float64])` at lines 18-22.
Delete `diag_vec_mul(self, x: List[Float64]) -> List[Float64]` at lines 24-28.

- [ ] **Step 4.2: Update consumer imports/calls**

Search for any remaining calls to List-based sparse methods across the codebase:
```bash
rg "spmv\(|diag_scale\(|spmm\(|to_dense\(\)|from_dense\(" --include="*.mojo" src/
```
Update any calls that pass List arguments to pass NDArray instead. Key consumers:
- `src/engines/fpe/initial_cond.mojo` — uses `DiagMatrix(var values: List[Float64])`, `.spmv(b: List)`, `.to_dense()`
- `src/engines/fpe/solver.mojo` — uses `diag_scale(A, row_scale: List, col_scale: List)`
- BSpline files if they use List-based sparse methods

- [ ] **Step 4.3: Run all sparse + related tests**
```bash
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_sparse_coo_diag.mojo
pixi run mojo run -I src tests/test_linalg.mojo
```

- [ ] **Step 4.4: Commit**
```bash
git add src/sparse/diag.mojo src/engines/fpe/initial_cond.mojo src/engines/fpe/solver.mojo
git commit -m "refactor: delete List-based DiagMatrix, update consumers to NDArray"
```

---

## Task 5: L2 — Linalg migration

**Files:**
- Modify: `src/numerics/linalg.mojo:17-158,161-261`
- Modify: `src/numerics/__init__.mojo:7-9`

- [ ] **Step 5.1: Delete List-based functions from linalg.mojo**

Delete these functions (lines 17-158):
- `lu_solve` (List version, lines 17-64)
- `compute_jacobian` (List version, lines 67-99)
- `dense_matvec` (List version, lines 102-111)
- `sparse_matvec` (List version, lines 114-141)
- `copy_mat` (local List duplicate, lines 144-151)
- `copy_vec` (local List duplicate, lines 154-158)

- [ ] **Step 5.2: Rename _nd functions to primary names**

- `lu_solve_nd` → `lu_solve` (lines 161-191)
- `compute_jacobian_nd` → `compute_jacobian` (lines 194-221)
- `dense_matvec_nd` → `dense_matvec` (lines 224-232)
- `sparse_matvec_nd` → `sparse_matvec` (lines 235-261)

Also remove the `import copy_vec_util / copy_mat_util` aliases if they were only needed to avoid collision with the local List-based `copy_vec`/`copy_mat`. After deletion, the imports can use the original names.

- [ ] **Step 5.3: Update numerics/__init__.mojo exports**

Edit `src/numerics/__init__.mojo`:
- Line 7: Remove `zeros_list, copy_vec_list, copy_mat_list` from exports
- Line 8: Remove `lu_solve, dense_matvec, compute_jacobian` (List-based re-exports)
- Line 9: Rename `lu_solve_nd → lu_solve`, `dense_matvec_nd → dense_matvec`, `sparse_matvec_nd → sparse_matvec`, `compute_jacobian_nd → compute_jacobian`
- Or merge lines 8-9 into one clean export line

- [ ] **Step 5.4: Update consumers**

Search for callers of `_nd` linalg functions:
```bash
rg "lu_solve_nd|dense_matvec_nd|sparse_matvec_nd|compute_jacobian_nd" --include="*.mojo" src/
```
Update to use primary names (e.g., `lu_solve` instead of `lu_solve_nd`). Key consumers:
- `src/numerics/optim/lm.mojo` — calls `lu_solve_nd`
- `src/numerics/ode/radau.mojo` — may call compute_jacobian_nd

- [ ] **Step 5.5: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_linalg.mojo
```
Expected: 7 tests PASS

- [ ] **Step 5.6: Commit**
```bash
git add src/numerics/linalg.mojo src/numerics/__init__.mojo src/numerics/optim/lm.mojo
git commit -m "refactor: migrate linalg to NDArray-primary, delete List versions"
```

---

## Task 6: L2 — SparseLU migration

**Files:**
- Modify: `src/numerics/sparse_lu.mojo:237-258`

- [ ] **Step 6.1: Delete List solve, rename solve_nd → solve**

Delete `solve(mut self, b: List[Float64]) -> List[Float64]` at lines 237-250.
Rename `solve_nd` → `solve` at line 252.

- [ ] **Step 6.2: Update consumers**

Search for `solve_nd` calls on SparseLU:
```bash
rg "SparseLU.*solve_nd|\.solve_nd\(" --include="*.mojo" src/
```
Update to use `solve`. Also search for `SparseLU().solve(` with List arguments and update.

- [ ] **Step 6.3: Run tests**
```bash
pixi run mojo run -I src tests/test_linalg.mojo
```

- [ ] **Step 6.4: Commit**
```bash
git add src/numerics/sparse_lu.mojo
git commit -m "refactor: migrate SparseLU.solve to NDArray-primary, delete List version"
```

---

## Task 7: L3 — ODE types migration

**Files:**
- Modify: `src/numerics/ode/types.mojo:4-19,22-37`

- [ ] **Step 7.1: Delete List-based ODESystem and ODESolution**

Delete `trait ODESystem` (lines 4-11) and `struct ODESolution` (lines 14-19).

- [ ] **Step 7.2: Rename NDArray versions to primary names**

- `ODESystemND` → `ODESystem` (lines 22-29)
- `ODESolutionND` → `ODESolution` (lines 32-37)

- [ ] **Step 7.3: Update __init__.mojo exports**

Edit `src/numerics/ode/__init__.mojo`:
- Line 3: Remove `ODESystemND, ODESolutionND` (now just `ODESystem, ODESolution`)
- Keep: `from .types import ODESystem, ODESolution`

- [ ] **Step 7.4: Update all consumers of ODESystem/ODESolution**

Search for `ODESystemND|ODESolutionND|ODESystem[^N]` usage across codebase:
```bash
rg "ODESystemND|ODESolutionND" --include="*.mojo" src/
```
Update imports to use `ODESystem`/`ODESolution` (the NDArray versions). Key consumers:
- `src/numerics/ode/rk45.mojo`
- `src/numerics/ode/radau.mojo`
- `src/engines/fpe/solver.mojo`

- [ ] **Step 7.5: Run available ODE tests**
```bash
pixi run mojo run -I src tests/test_rk45.mojo
pixi run mojo run -I src tests/test_radau_simple.mojo
```

- [ ] **Step 7.6: Commit**
```bash
git add src/numerics/ode/types.mojo src/numerics/ode/__init__.mojo
git commit -m "refactor: migrate ODESystem/ODESolution to NDArray-primary, delete List versions"
```

---

## Task 8: L3 — RK45 migration

**Files:**
- Modify: `src/numerics/ode/rk45.mojo:18-171,174-362`

- [ ] **Step 8.1: Delete List-based RungeKutta45**

Delete `struct RungeKutta45[System: ODESystem]` (lines 18-171).

- [ ] **Step 8.2: Rename RungeKutta45ND → RungeKutta45**

At line 174, rename `RungeKutta45ND` → `RungeKutta45`. Also rename the method `solve_nd` → `solve`.

Rename `_build_rk_solution_nd` → `_build_rk_solution` (lines 343-362).

- [ ] **Step 8.3: Update __init__.mojo**

Edit `src/numerics/ode/__init__.mojo` line 4:
- Remove `RungeKutta45ND` (now just `RungeKutta45`)

- [ ] **Step 8.4: Update consumers**

Search for `RungeKutta45ND` usage:
```bash
rg "RungeKutta45ND" --include="*.mojo" src/ tests/
```
Update to `RungeKutta45`.

- [ ] **Step 8.5: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_rk45.mojo
```

- [ ] **Step 8.6: Commit**
```bash
git add src/numerics/ode/rk45.mojo src/numerics/ode/__init__.mojo
git commit -m "refactor: migrate RungeKutta45 to NDArray-primary, delete List version"
```

---

## Task 9: L3 — Radau migration

**Files:**
- Modify: `src/numerics/ode/radau.mojo:96-109,112-927,930-1697`

**WARNING**: This is the largest single edit (~800 lines deleted). The List-based `RadauSparseLinearSolver` spans lines 112-906. Proceed carefully.

- [ ] **Step 9.1: Delete List-based LinearODESystem + RadauSparseLinearSolver**

Delete `trait LinearODESystem` (lines 96-101).
Delete `struct RadauSparseLinearSolver[System: LinearODESystem]` (lines 112-906).
This includes internal List-based helpers like `_build_solution`, `_build_csr_to_csc_map`, `_build_newton_rhs`, etc.

- [ ] **Step 9.2: Rename NDArray versions to primary names**

- `LinearODESystemND` → `LinearODESystem` (lines 104-109)
- `_build_solution_nd` → `_build_solution` (lines 908-927)
- `RadauSparseLinearSolverND` → `RadauSparseLinearSolver` (lines 930-1697)
- Inside the struct: rename `solve_nd` → `solve`

- [ ] **Step 9.3: Update __init__.mojo**

Edit `src/numerics/ode/__init__.mojo` line 5:
- Remove `RadauSparseLinearSolverND, LinearODESystemND`
- Keep: `RadauSparseLinearSolver, LinearODESystem`

- [ ] **Step 9.4: Update consumers**

Search for `RadauSparseLinearSolverND|LinearODESystemND` usage:
```bash
rg "RadauSparseLinearSolverND|LinearODESystemND" --include="*.mojo" src/
```
Key consumers:
- `src/engines/fpe/solver.mojo` — uses `RadauSparseLinearSolverND[FPESparseLinearSystemND]` and `LinearODESystemND`

- [ ] **Step 9.5: Verify indentation with byte-level check**

This is a massive edit — MUST verify indentation afterward.

- [ ] **Step 9.6: Run tests**
```bash
pixi run mojo run -I src tests/test_radau_simple.mojo
```

- [ ] **Step 9.7: Commit**
```bash
git add src/numerics/ode/radau.mojo src/numerics/ode/__init__.mojo src/engines/fpe/solver.mojo
git commit -m "refactor: migrate Radau/LinearODESystem to NDArray-primary, delete List versions"
```

---

## Task 10: L3 — Full ODE layer test gate

**Files:** Test only

- [ ] **Step 10.1: Run all ODE tests**
```bash
pixi run mojo run -I src tests/test_rk45.mojo
pixi run mojo run -I src tests/test_radau_simple.mojo
pixi run mojo run -I src tests/test_linalg.mojo
pixi run mojo run -I src tests/test_sparse.mojo
```
Expected: All PASS

- [ ] **Step 10.2: Commit if any fixups needed**

---

## Task 11: L4 — initial_cond NDArray migration

**Files:**
- Modify: `src/engines/fpe/initial_cond.mojo:17-150`

- [ ] **Step 11.1: Rewrite _normalize_nonnegative for NDArray**

Replace `def _normalize_nonnegative(mut x: List[Float64])` (lines 17-20) with:
```mojo
def _normalize_nonnegative(mut x: NDArray[Float64]):
    var n = Int(x.shape()[0])
    var total = 0.0
    for i in range(n):
        if x[i] < 0.0:
            x[i] = 0.0
        total += x[i]
    if total > 0.0:
        for i in range(n):
            x[i] = x[i] / total
```

- [ ] **Step 11.2: Rewrite _delta_approx_flat → _delta_approx returning NDArray**

Replace `def _delta_approx_flat(domain, params, sigma0) -> List[Float64]` (lines 23-68) with NDArray-native version:
- Use `zeros(n_s * n_v)` instead of `zeros_list(n_s * n_v)`
- Use `NDArray[Float64]` local variables instead of `List[Float64]`
- Use `spmv` (NDArray version) instead of List `spmv`
- Use `DiagMatrix(n)` constructor instead of `DiagMatrix(var values: List[Float64])`
- Use `solve_nnls_sparse` (NDArray version) instead of List version
- Call `_normalize_nonnegative` (now NDArray-native)
- Return `NDArray[Float64]`

- [ ] **Step 11.3: Rewrite InitialCondition.compute to be NDArray-native**

Replace `def compute(self, domain, params, sigma0) raises -> List[Float64]` (lines 75-137) with NDArray-native implementation:
- Return type: `NDArray[Float64]`
- Call `_delta_approx` (NDArray version)
- Use `sparse_transpose`/`spgemm` (CSR→CSR, unchanged)
- Use `DiagMatrix(n)` + `.to_csr()` 
- Use `spmv` (NDArray version)
- Use `solve_nnls_sparse` (NDArray version)
- Call `_normalize_nonnegative` (NDArray version)

- [ ] **Step 11.4: Delete compute_nd wrapper**

Delete `def compute_nd(self, domain, params, sigma0) raises -> NDArray[Float64]` (lines 139-150) — no longer needed.

- [ ] **Step 11.5: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
```

- [ ] **Step 11.6: Commit**
```bash
git add src/engines/fpe/initial_cond.mojo
git commit -m "refactor: rewrite InitialCondition.compute to NDArray-native, delete wrapper"
```

---

## Task 12: L4 — solver.mojo migration

**Files:**
- Modify: `src/engines/fpe/solver.mojo:40-259,261-389`

- [ ] **Step 12.1: Delete List-based FPE system structs**

Delete:
- `FPESparseSystem(ODESystem)` (lines 40-53) — List-based
- `FPESparseLinearSystem(LinearODESystem)` (lines 56-68) — List-based
- `FPEDenseSystem(ODESystem)` (lines 71-89) — List-based

- [ ] **Step 12.2: Rename NDArray structs to primary names**

- `FPESparseSystemND` → `FPESparseSystem` (lines 92-105)
- `FPESparseLinearSystemND` → `FPESparseLinearSystem` (lines 108-124)
- `FPEDenseSystemND` → `FPEDenseSystem` (lines 127-146)
- Update trait references: `ODESystemND` → `ODESystem`, `LinearODESystemND` → `LinearODESystem` (already done in L3)

- [ ] **Step 12.3: Delete List-based FPESolver methods**

Delete:
- `solve` (List version, lines 156-173)
- `_integrate_cpu_sparse` (List version, lines 175-223)
- `_solve_gpu_batch` (List version, lines 225-239)
- `_solve_cpu_parallel` (List version, lines 241-259)
- `_compute_sparse_neg_M_inv_K` (lines 367-389) if it uses List internally

- [ ] **Step 12.4: Rename NDArray FPESolver methods to primary names**

- `solve_nd` → `solve` (lines 261-278)
- `_integrate_cpu_sparse_nd` → `_integrate_cpu_sparse` (lines 280-327)
- `_solve_gpu_batch_nd` → `_solve_gpu_batch` (lines 329-346)
- `_solve_cpu_parallel_nd` → `_solve_cpu_parallel` (lines 348-365)

- [ ] **Step 12.5: Update method signatures to use NDArray-native compute**

In `_integrate_cpu_sparse` (renamed from _nd):
- `InitialCondition.compute` now returns `NDArray[Float64]` directly (no wrapper)
- `diag_scale` now takes NDArray args (done in L1)
- `RadauSparseLinearSolver` now takes `NDArray[Float64]` y0 (done in L3)
- Remove any remaining `zeros_list` / `copy_vec_list` calls — use `zeros` / `copy_vec`

- [ ] **Step 12.6: Remove copy_mat_list import if verified unused**

Check if `copy_mat_list` import at solver.mojo:19 is actually used in any remaining code. If not, delete the import.

- [ ] **Step 12.7: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_zeros_3d.mojo
pixi run mojo run -I src tests/test_ndarray.mojo
```

- [ ] **Step 12.8: Commit**
```bash
git add src/engines/fpe/solver.mojo
git commit -m "refactor: migrate FPESolver to NDArray-primary, delete List versions"
```

---

## Task 13: L4 — objective.mojo bug fix + NDArray migration

**Files:**
- Modify: `src/engines/calibrator/objective.mojo:26-50,75-101` (Note: path is `src/engines/calibrator/objective.mojo`)

- [ ] **Step 13.1: Fix pre-existing bug — map_s_to_physical / map_v_to_physical**

At line 34: Replace `domain.map_s_to_physical(domain.s_points[i])` with `domain.s_points_phys[i]`
At lines 45-47: Replace `domain.map_v_to_physical(domain.v_points_phys[j])` with `domain.v_points_phys[j]`

Verify the correct attribute names on FPEDomain by reading `src/engines/fpe/domain.mojo`.

- [ ] **Step 13.2: Rewrite _integrate_call_price to use NDArray**

Replace `def _integrate_call_price(domain, pdf: List[List[Float64]], strike) -> Float64` with NDArray-native version:
- `pdf` parameter: `NDArray[Float64]` (2D)
- Use `Int(pdf.shape()[0])` / `Int(pdf.shape()[1])` for dimensions
- Access `pdf[i, j]` instead of `pdf[i][j]`

- [ ] **Step 13.3: Rewrite ObjectiveFunction.compute to return NDArray**

Replace `def compute(self, params) raises -> List[Float64]` with NDArray-native version:
- `FPESolver.solve` now returns `NDArray[Float64]`
- `PDFComputer.compute` now returns `NDArray[Float64]` (will be done in Task 14)
- Return `NDArray[Float64]` instead of `List[Float64]`

- [ ] **Step 13.4: Verify indentation + attempt compile**

Note: objective.mojo may not have a standalone test. Try:
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
```

- [ ] **Step 13.5: Commit**
```bash
git add src/engines/calibrator/objective.mojo
git commit -m "fix: map_s_to_physical bug, migrate objective.mojo to NDArray"
```

---

## Task 14: L4 — pdf.mojo migration

**Files:**
- Modify: `src/engines/fpe/pdf.mojo:4-32`

- [ ] **Step 14.1: Rewrite _reshape_to_grid and PDFComputer.compute**

Replace List-based functions with NDArray-native:
- `_reshape_to_grid(flat: NDArray[Float64], n_s: Int, n_v: Int) -> NDArray[Float64]` — returns 2D NDArray
- `PDFComputer.compute(self, domain, q_t: NDArray[Float64]) -> NDArray[Float64]` — returns 2D NDArray

Use `NDArray[Float64]` with 2D shape `(n_s, n_v)` instead of `List[List[Float64]]`.

- [ ] **Step 14.2: Run tests + commit**
```bash
git add src/engines/fpe/pdf.mojo
git commit -m "refactor: migrate PDFComputer to NDArray"
```

---

## Task 15: L5 — LM traits + solve migration

**Files:**
- Modify: `src/numerics/optim/lm.mojo:5-18,32-183`
- Modify: `src/numerics/optim/__init__.mojo`

- [ ] **Step 15.1: Delete List-based traits**

Delete `trait ResidualCallable` (lines 5-6) and `trait JacobianCallable` (lines 9-10).

- [ ] **Step 15.2: Rename NDArray traits to primary names**

- `ResidualCallableND` → `ResidualCallable` (lines 13-14)
- `JacobianCallableND` → `JacobianCallable` (lines 17-18)

- [ ] **Step 15.3: Delete List-based solve, rename solve_nd → solve**

Delete `LevenbergMarquardt.solve` (List version, lines 32-109).
Rename `solve_nd` → `solve` (lines 111-183).

- [ ] **Step 15.4: Update __init__.mojo exports**

Edit `src/numerics/optim/__init__.mojo` line 2:
- Remove `ResidualCallableND, JacobianCallableND`
- Add `ResidualCallable, JacobianCallable`

- [ ] **Step 15.5: Update consumers**

Search for `ResidualCallableND|JacobianCallableND|solve_nd` usage:
```bash
rg "ResidualCallableND|JacobianCallableND|\.solve_nd\(" --include="*.mojo" src/
```
Update to primary names.

- [ ] **Step 15.6: Run tests + commit**
```bash
pixi run mojo run -I src tests/test_optim.mojo
git add src/numerics/optim/lm.mojo src/numerics/optim/__init__.mojo
git commit -m "refactor: migrate LM traits/solve to NDArray-primary, delete List versions"
```

---

## Task 16: L5 — OSQP migration

**Files:**
- Modify: `src/numerics/optim/osqp.mojo:58-359`

- [ ] **Step 16.1: Delete List-based solve_nnls_sparse, rename _nd → primary**

Delete `solve_nnls_sparse` (List version, lines 58-156).
Rename `solve_nnls_sparse_nd` → `solve_nnls_sparse` (lines 158-254).

- [ ] **Step 16.2: Rewrite solve_nnls_dense to NDArray**

Replace `def solve_nnls_dense(self, A: List[List[Float64]], b: List[Float64]) -> List[Float64]` (lines 256-265) with NDArray-native version:
- `A: NDArray[Float64]` (2D), `b: NDArray[Float64]` (1D) → `NDArray[Float64]`
- Convert A to CSR internally if needed, or implement dense NNLS directly
- Same algorithm, NDArray I/O

- [ ] **Step 16.3: Rewrite ProjectedGradient to NDArray**

Replace `struct ProjectedGradient` (lines 289-359):
- `def solve(self, A: NDArray[Float64], b: NDArray[Float64]) -> NDArray[Float64]`
- Internal `AtA: NDArray[Float64]` and `Atb: NDArray[Float64]` computed from NDArray inputs
- Use NDArray element-wise ops instead of List indexing

- [ ] **Step 16.4: Rewrite OSQP.solve_nnls to NDArray**

Replace `def solve_nnls(self, A: List[List[Float64]], b: List[Float64]) -> List[Float64]` (lines 283-286) with:
```mojo
def solve_nnls(self, A: NDArray[Float64], b: NDArray[Float64]) -> NDArray[Float64]:
    return self.solver.solve_nnls_sparse(...)
```
(Or delegate to solve_nnls_dense if A is dense)

- [ ] **Step 16.5: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_optim.mojo
```

- [ ] **Step 16.6: Commit**
```bash
git add src/numerics/optim/osqp.mojo
git commit -m "refactor: migrate OSQP to NDArray-primary, rewrite ProjectedGradient"
```

---

## Task 17: L5 — StableLinear migration

**Files:**
- Modify: `src/numerics/nn/stable_linear.mojo:14-134,137-223`
- Modify: `src/numerics/nn/__init__.mojo`

- [ ] **Step 17.1: Delete List-based StableLinear + helpers**

Delete:
- `_matmul_vec` (lines 14-29) — dead code
- `struct StableLinear` (List version, lines 32-122)
- `make_stable_linear` (List factory, lines 125-134)

- [ ] **Step 17.2: Rename NDArray versions to primary names**

- `StableLinearND` → `StableLinear` (lines 137-211)
- `make_stable_linear_nd` → `make_stable_linear` (lines 214-223)

- [ ] **Step 17.3: Update nn/__init__.mojo**

Edit `src/numerics/nn/__init__.mojo` line 1:
- Remove `StableLinear, make_stable_linear, StableLinearND, make_stable_linear_nd`
- Keep: `StableLinear, make_stable_linear`

- [ ] **Step 17.4: Update consumers**

Search for `StableLinearND|make_stable_linear_nd` usage:
```bash
rg "StableLinearND|make_stable_linear_nd" --include="*.mojo" src/
```
Update to primary names. Key consumers:
- `src/engines/nais/nais_net.mojo` — will be updated in L6

- [ ] **Step 17.5: Run tests + commit**
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
git add src/numerics/nn/stable_linear.mojo src/numerics/nn/__init__.mojo
git commit -m "refactor: migrate StableLinear to NDArray-primary, delete List version"
```

---

## Task 18: L5 — Calibrator migration

**Files:**
- Modify: `src/engines/calibrator/calibrator.mojo:8-150`

- [ ] **Step 18.1: Rewrite _params_to_vec / _vec_to_params for NDArray**

Replace:
- `def _params_to_vec(p: HestonParams) -> List[Float64]` → `-> NDArray[Float64]`
- `def _vec_to_params(x: NDArray[Float64], base: HestonParams) -> HestonParams` — use `x[i]` indexing

- [ ] **Step 18.2: Rewrite CalibratorResidual / CalibratorJacobian for NDArray**

Replace:
- `CalibratorResidual[B](ResidualCallable)` — now implements NDArray `ResidualCallable`
  - `def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]`
  - Uses `_vec_to_params` (NDArray), `ObjectiveFunction.compute` (NDArray), `abs_f64`/`max_f64`
- `CalibratorJacobian[B](JacobianCallable)` — now implements NDArray `JacobianCallable`
  - `def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]`
  - Uses finite-difference Jacobian with NDArray

- [ ] **Step 18.3: Rewrite calibrate to NDArray**

Replace `def calibrate(self, market_prices: List[Float64], ...)` with:
- `market_prices: NDArray[Float64]`, `strikes: NDArray[Float64]`, `expiries: NDArray[Float64]`
- Uses `LevenbergMarquardt.solve` (NDArray version)
- Uses `_params_to_vec`/`_vec_to_params` (NDArray)

- [ ] **Step 18.4: Rewrite calibrate_batch to NDArray**

Update `calibrate_batch` similarly if it exists.

- [ ] **Step 18.5: Verify indentation + attempt compile + commit**
```bash
git add src/engines/calibrator/calibrator.mojo
git commit -m "refactor: migrate Calibrator to NDArray-primary"
```

---

## Task 19: L5 — Full optim layer test gate

- [ ] **Step 19.1: Run all optim tests**
```bash
pixi run mojo run -I src tests/test_optim.mojo
pixi run mojo run -I src tests/test_adam.mojo
```

- [ ] **Step 19.2: Run full test suite**
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_sparse_coo_diag.mojo
pixi run mojo run -I src tests/test_linalg.mojo
pixi run mojo run -I src tests/test_rk45.mojo
pixi run mojo run -I src tests/test_radau_simple.mojo
pixi run mojo run -I src tests/test_autograd_tape.mojo
pixi run mojo run -I src tests/test_brownian_paths.mojo
pixi run mojo run -I src tests/test_zeros_3d.mojo
```
Expected: All passing tests still pass

- [ ] **Step 19.3: Commit fixups if needed**

---

## Task 20: L6 — NaisNet NDArray migration

**Files:**
- Modify: `src/engines/nais/nais_net.mojo:12-387`

**WARNING**: This is a large rewrite. NaisNet is entirely List-based with no NDArray version.

- [ ] **Step 20.1: Rewrite helper functions**

- `_make_weights(in_dim, out_dim, scale) -> NDArray[Float64]` — return 2D NDArray using `zeros_mat(in_dim, out_dim)` then fill with random init
- `_linear(W: NDArray[Float64], b: NDArray[Float64], x: NDArray[Float64]) -> NDArray[Float64]` — mat-vec using `W.row(i)` dot `x` + `b[i]`
- `_sin_vec(x: NDArray[Float64]) -> NDArray[Float64]` — element-wise sin
- `_add_vec(a: NDArray[Float64], b: NDArray[Float64]) -> NDArray[Float64]` — element-wise add

- [ ] **Step 20.2: Rewrite NaisNet struct fields**

Replace all `List[List[Float64]]` fields with `NDArray[Float64]`:
- `layer1: NDArray[Float64]`, `layer1_b: NDArray[Float64]`
- `layer2: StableLinear` (now NDArray-native from L5)
- `layer2_input: NDArray[Float64]`, `layer2_input_b: NDArray[Float64]`
- Similarly for layers 3-6

- [ ] **Step 20.3: Rewrite NaisNet.forward**

Update `forward(self, t: Float64, x: NDArray[Float64]) -> Tuple[Float64, NDArray[Float64]]`:
- Use `_linear` (NDArray), `_sin_vec` (NDArray), `_add_vec` (NDArray)
- Use `StableLinear.forward` (NDArray, from L5)

- [ ] **Step 20.4: Rewrite NaisNet.forward_tracked**

Update `forward_tracked(self, t, x: NDArray[Float64], mut tape: Tape, param_indices)`:
- Use NDArray-based Tape (Task 21)
- Rewrite `_linear_tracked_with_indices`, `_linear_tracked_record_weights`, `_stable_linear_forward_tracked` for NDArray

- [ ] **Step 20.5: Rewrite _count_params**

Update to count parameters from NDArray shapes instead of List lengths.

- [ ] **Step 20.6: Verify indentation + attempt compile**

- [ ] **Step 20.7: Commit**
```bash
git add src/engines/nais/nais_net.mojo
git commit -m "refactor: rewrite NaisNet to NDArray-native, delete List version"
```

---

## Task 21: L6 — Autograd Tape NDArray migration

**Files:**
- Modify: `src/engines/nais/autograd.mojo:5-149`

- [ ] **Step 21.1: Rewrite Tape internal storage**

Replace:
- `var values: List[Float64]` → `var values: NDArray[Float64]` (pre-allocated, resize by creating new NDArray)
- `var adjoints: List[Float64]` → `var adjoints: NDArray[Float64]`
- `var op_types: List[Int]` → `var op_types: NDArray[Int]`
- `var inputs_a: List[Int]` → `var inputs_a: NDArray[Int]`
- `var inputs_b: List[Int]` → `var inputs_b: NDArray[Int]`
- `var partials: List[Float64]` → `var partials: NDArray[Float64]`

**Challenge**: NDArray doesn't support dynamic append like List. Strategy: use a capacity-doubling approach:
- Allocate NDArray with initial capacity (e.g., 256)
- Track `var len: Int` for actual usage
- When full, allocate new NDArray of 2x capacity, copy data

Alternatively, keep internal storage as List but provide NDArray I/O at the boundary. This is simpler and avoids the dynamic allocation problem.

**Recommended approach**: Keep Tape internals as List for now (it's an internal implementation detail, not a public API). Only change the I/O boundary: `gradients_for` returns `NDArray[Float64]` instead of `List[Float64]`, and `record_linear` takes NDArray index arrays.

This keeps the autograd working while meeting the "no List in public APIs" goal.

- [ ] **Step 21.2: Update Tape I/O boundary**

- `gradients_for(self, param_indices: List[Int]) -> NDArray[Float64]` — convert result to NDArray before returning
- `record_linear(self, W_idx: List[Int], b_idx: List[Int], x_idx: List[Int]) -> List[Int]` — keep internal as List for now

- [ ] **Step 21.3: Verify indentation + run tests**
```bash
pixi run mojo run -I src tests/test_autograd_tape.mojo
```

- [ ] **Step 21.4: Commit**
```bash
git add src/engines/nais/autograd.mojo
git commit -m "refactor: migrate Tape I/O to NDArray, keep internals as List"
```

---

## Task 22: L6 — NaisNet consumer updates

**Files:**
- Modify: `src/engines/nais/trainer.mojo` — update NaisNet usage
- Modify: `src/engines/nais/inferencer.mojo` — update NaisNet usage

- [ ] **Step 22.1: Update trainer.mojo for NDArray NaisNet**

Search for List-based calls in trainer.mojo:
```bash
rg "List\[Float64\]|zeros_list|linspace_list|_list" --include="*.mojo" src/engines/nais/trainer.mojo
```
Update:
- `NaisNet.forward` now takes/returns NDArray
- `Tape.gradients_for` now returns NDArray
- Replace `linspace_list` → `linspace`
- Replace `zeros_list` → `zeros`

- [ ] **Step 22.2: Update inferencer.mojo**

Similar updates for inference code.

- [ ] **Step 22.3: Run tests + commit**
```bash
pixi run mojo run -I src tests/test_autograd_tape.mojo
git add src/engines/nais/trainer.mojo src/engines/nais/inferencer.mojo
git commit -m "refactor: update trainer/inferencer for NDArray-native NaisNet"
```

---

## Task 23: L0b — Delete _list function definitions

**Files:**
- Modify: `src/numerics/utils/constructors.mojo:35-74`
- Modify: `src/numerics/utils/copy.mojo:22-44`

- [ ] **Step 23.1: Verify no remaining _list consumers**

Search entire codebase:
```bash
rg "zeros_list|zeros_mat_list|zeros_3d_list|linspace_list|copy_vec_list|copy_mat_list|swap_rows_list" --include="*.mojo" src/
```
Expected: Zero matches (all consumers migrated in L1-L6).

- [ ] **Step 23.2: Delete _list definitions from constructors.mojo**

Delete:
- `zeros_list` (lines 35-39)
- `zeros_mat_list` (lines 42-49)
- `zeros_3d_list` (lines 52-62)
- `linspace_list` (lines 65-74)

- [ ] **Step 23.3: Delete _list definitions from copy.mojo**

Delete:
- `copy_vec_list` (lines 22-26)
- `copy_mat_list` (lines 29-33)
- `swap_rows_list` (lines 36-44)

- [ ] **Step 23.4: Run all tests**
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_linalg.mojo
pixi run mojo run -I src tests/test_optim.mojo
pixi run mojo run -I src tests/test_rk45.mojo
pixi run mojo run -I src tests/test_autograd_tape.mojo
```

- [ ] **Step 23.5: Commit**
```bash
git add src/numerics/utils/constructors.mojo src/numerics/utils/copy.mojo
git commit -m "refactor: delete _list function definitions, unification complete"
```

---

## Task 24: Full test suite verification

- [ ] **Step 24.1: Run ALL test files**
```bash
pixi run mojo run -I src tests/test_ndarray.mojo
pixi run mojo run -I src tests/test_sparse.mojo
pixi run mojo run -I src tests/test_sparse_coo_diag.mojo
pixi run mojo run -I src tests/test_linalg.mojo
pixi run mojo run -I src tests/test_zeros_3d.mojo
pixi run mojo run -I src tests/test_optim.mojo
pixi run mojo run -I src tests/test_adam.mojo
pixi run mojo run -I src tests/test_rk45.mojo
pixi run mojo run -I src tests/test_radau_simple.mojo
pixi run mojo run -I src tests/test_autograd_tape.mojo
pixi run mojo run -I src tests/test_brownian_paths.mojo
```

- [ ] **Step 24.2: Verify no List[Float64] remains in numerics/sparse/engines**
```bash
rg "List\[Float64\]" --include="*.mojo" src/numerics/ src/sparse/ src/engines/
```
Expected: Zero matches (or only in non-migrated files documented as exceptions).

- [ ] **Step 24.3: Verify no _nd or _list suffixed functions remain**
```bash
rg "_nd\b|_list\b" --include="*.mojo" src/numerics/ src/sparse/ src/engines/
```
Expected: Zero matches.

- [ ] **Step 24.4: Final commit**
```bash
git add -A
git commit -m "refactor: NDArray unification complete — all List[Float64] eliminated from numerics/sparse/engines"
```

---

## Summary

| Task | Layer | Key Action | Estimated Complexity |
|------|-------|-----------|---------------------|
| 1 | L0a | Remove _list exports | Low |
| 2 | L1 | CSR spmv/to_dense/from_dense | Medium |
| 3 | L1 | Sparse ops diag_scale/spmm | Low |
| 4 | L1 | DiagMatrix + consumer updates | Medium |
| 5 | L2 | Linalg functions | Medium |
| 6 | L2 | SparseLU.solve | Low |
| 7 | L3 | ODE types | Low |
| 8 | L3 | RK45 | Low |
| 9 | L3 | Radau (largest delete ~800 lines) | High |
| 10 | L3 | Test gate | Low |
| 11 | L4 | initial_cond rewrite | High |
| 12 | L4 | solver.mojo migration | High |
| 13 | L4 | objective.mojo bug fix + migration | High |
| 14 | L4 | pdf.mojo migration | Medium |
| 15 | L5 | LM traits + solve | Low |
| 16 | L5 | OSQP + ProjectedGradient rewrite | High |
| 17 | L5 | StableLinear | Low |
| 18 | L5 | Calibrator rewrite | High |
| 19 | L5 | Test gate | Low |
| 20 | L6 | NaisNet rewrite (entire struct) | High |
| 21 | L6 | Autograd Tape I/O boundary | Medium |
| 22 | L6 | Trainer/inferencer updates | Medium |
| 23 | L0b | Delete _list definitions | Low |
| 24 | Final | Full test suite + verification | Low |
