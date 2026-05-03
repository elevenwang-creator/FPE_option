# NDArray Unification Design Spec

**Date**: 2026-05-01
**Status**: Approved
**Goal**: Eliminate all `List[Float64]` usage in numerics/sparse/engines modules. Fuse `NDArray[Float64]` as the sole data container, keeping original function/struct names (no `_nd` suffix). Delete all `_list` and `_nd` duplicate versions.

## Background

Phase 1-3 introduced `NDArray[ElementType]` alongside existing List-based code. This created parallel APIs: `zeros`/`zeros_list`, `ODESystem`/`ODESystemND`, `solve`/`solve_nd`, etc. The unification removes this duplication — every function/struct has one canonical name, NDArray-native.

## Naming Convention

| Before Migration | After Migration |
|------------------|-----------------|
| `zeros` (NDArray) + `zeros_list` (List) | `zeros` (NDArray only) |
| `copy_vec` (NDArray) + `copy_vec_list` (List) | `copy_vec` (NDArray only) |
| `ODESystem` (List) + `ODESystemND` (NDArray) | `ODESystem` (NDArray) |
| `ODESolution` (List) + `ODESolutionND` (NDArray) | `ODESolution` (NDArray) |
| `RungeKutta45` (List) + `RungeKutta45ND` (NDArray) | `RungeKutta45` (NDArray) |
| `RadauSparseLinearSolver` + `RadauSparseLinearSolverND` | `RadauSparseLinearSolver` (NDArray) |
| `LinearODESystem` + `LinearODESystemND` | `LinearODESystem` (NDArray) |
| `FPESparseSystem` + `FPESparseSystemND` | `FPESparseSystem` (NDArray) |
| `FPEDenseSystem` + `FPEDenseSystemND` | `FPEDenseSystem` (NDArray) |
| `FPESparseLinearSystem` + `FPESparseLinearSystemND` | `FPESparseLinearSystem` (NDArray) |
| `solve` (List) + `solve_nd` (NDArray) | `solve` (NDArray) |
| `spmv` (List) + `spmv_nd` (NDArray) | `spmv` (NDArray) |
| `diag_scale` (List) + `diag_scale_nd` (NDArray) | `diag_scale` (NDArray) |
| `lu_solve` (List) + `lu_solve_nd` (NDArray) | `lu_solve` (NDArray) |
| `StableLinear` (List) + `StableLinearND` (NDArray) | `StableLinear` (NDArray) |
| `ResidualCallable` + `ResidualCallableND` | `ResidualCallable` (NDArray) |
| `JacobianCallable` + `JacobianCallableND` | `JacobianCallable` (NDArray) |
| `LevenbergMarquardt.solve` + `solve_nd` | `LevenbergMarquardt.solve` (NDArray) |
| `solve_nnls_sparse` + `solve_nnls_sparse_nd` | `solve_nnls_sparse` (NDArray) |
| `compute` (List) + `compute_nd` (NDArray wrapper) | `compute` (NDArray-native) |
| `Vector` alias | Remains as `NDArray[Float64]` alias |

## Migration Approach: Bottom-Up, 7 Layers

Each layer must compile and pass tests before the next begins.

### L0: Utils (constructors, copy) — Definition Cleanup (Final Step)

**Files**: `src/numerics/utils/constructors.mojo`, `src/numerics/utils/copy.mojo`, `src/numerics/utils/__init__.mojo`

**NOTE**: `_list` functions have active consumers across L1–L6. This layer is split into two phases:
- **L0a (early, before L1)**: Remove `_list` exports from `__init__.mojo` so no NEW consumers can import them. Existing consumers still import directly from the file.
- **L0b (final, after L6)**: Delete `_list` function definitions once all consumers are migrated.

**L0a Actions**:
- Remove `zeros_list`, `zeros_mat_list`, `zeros_3d_list`, `linspace_list` exports from `__init__.mojo`
- Remove `copy_vec_list`, `copy_mat_list`, `swap_rows_list` exports from `__init__.mojo`

**L0b Actions** (run after L6 complete):
- Delete `zeros_list`, `zeros_mat_list`, `zeros_3d_list`, `linspace_list` definitions from constructors.mojo
- Delete `copy_vec_list`, `copy_mat_list`, `swap_rows_list` definitions from copy.mojo
- Remove `copy_mat_list` import from solver.mojo (verify it's unused before deleting)
- Remove any remaining `_list` imports from consumer files

**Active consumers of _list functions** (migrated in L1–L6):
- `zeros_list`: rk45.mojo, stable_linear.mojo, solver.mojo, linalg.mojo, nais_net.mojo, volterra.mojo, calibrator.mojo
- `zeros_mat_list`: stable_linear.mojo, nais_net.mojo, volterra.mojo
- `zeros_3d_list`: volterra.mojo
- `linspace_list`: trainer.mojo, gpu_trainer.mojo, domain.mojo
- `copy_vec_list`: rk45.mojo

**Test gate**: test_ndarray.mojo (18 tests)

### L1: Sparse (CSR, CSC, COO, diag, ops)

**Files**: `src/sparse/csr.mojo`, `src/sparse/csc.mojo`, `src/sparse/coo.mojo`, `src/sparse/diag.mojo`, `src/sparse/ops.mojo`

**Actions**:
- Add `spmv_into(self, x: NDArray[Float64], mut y: NDArray[Float64])` — in-place, **no zero-init** (accumulates into y, matching List `spmv_into` semantics)
- Rename `spmv_nd` → `spmv` — this version **zero-inits y** before writing (API contract: caller gets fresh output)
- **API contract**: `spmv(x, mut y)` zero-inits y; `spmv_into(x, mut y)` does NOT zero-init y (accumulates). Both take NDArray.
- Delete List `spmv`, List `spmv_into`, List `spmv_inplace_fixed`
- Keep `spmv_fixed` (Vector in-place, used internally by Radau), `spmv_triple_fixed` (Vector, used by Radau collocation), `spmv_addassign_fixed` (Vector, used by Radau Newton) — these are internal Vector-based methods not exposed at the API boundary. They will be kept as-is for now and can be migrated to NDArray in a future cleanup pass.
- Rename `diag_scale_nd` → `diag_scale`; delete List `diag_scale`
- Rename `spmm_nd` → `spmm`; delete List `spmm`
- Rename `to_dense_nd` → `to_dense`; delete List `to_dense`
- Rename `from_dense_nd` → `from_dense`; delete List `from_dense`
- COOMatrix: `to_csr`/`to_csc` already return CSRMatrix/CSCMatrix (NDArray storage), no change needed
- DiagMatrix: delete List-based constructor; keep NDArray-based constructor as primary
- Update all consumers: bspline files, initial_cond.mojo, solver.mojo, osqp.mojo

**Items**: ~6 functions removed, 1 added

**Test gate**: test_sparse.mojo (10 tests), test_sparse_coo_diag.mojo (5 tests)

### L2: Linalg + SparseLU

**Files**: `src/numerics/linalg.mojo`, `src/numerics/sparse_lu.mojo`, `src/numerics/__init__.mojo`

**Actions**:
- Rename `lu_solve_nd` → `lu_solve`; delete List `lu_solve`
- Rename `dense_matvec_nd` → `dense_matvec`; delete List `dense_matvec`
- Rename `sparse_matvec_nd` → `sparse_matvec`; delete List `sparse_matvec`
- Rename `compute_jacobian_nd` → `compute_jacobian`; delete List `compute_jacobian`
- Delete `sparse_matvec` (dead code, no consumers)
- Rename `SparseLU.solve_nd` → `SparseLU.solve`; delete List `solve`
- Update `__init__.mojo` exports (remove `_nd` suffixes from re-exports)
- Update all consumers (lm.mojo, radau.mojo)

**Items**: ~6 functions removed

**Test gate**: test_linalg.mojo (7 tests)

### L3: ODE (types, RK45, Radau)

**Files**: `src/numerics/ode/types.mojo`, `src/numerics/ode/rk45.mojo`, `src/numerics/ode/radau.mojo`, `src/numerics/ode/__init__.mojo`

**Actions**:
- Delete `ODESystem` trait (List) and `ODESolution` struct (List)
- Rename `ODESystemND` → `ODESystem`, `ODESolutionND` → `ODESolution`
- Delete `LinearODESystem` trait (List); rename `LinearODESystemND` → `LinearODESystem`
- Delete `RungeKutta45` (List version); rename `RungeKutta45ND` → `RungeKutta45`
- Delete `RadauSparseLinearSolver` (List version); rename `RadauSparseLinearSolverND` → `RadauSparseLinearSolver`
- Delete `_build_solution` (List helper); rename `_build_solution_nd` → `_build_solution`
- Delete `_build_rk_solution` if exists; rename `_build_rk_solution_nd` → `_build_rk_solution`
- Update `__init__.mojo` exports

**Items**: ~6 types/functions removed

**Test gate**: test_rk45.mojo, test_radau_simple.mojo

### L4: FPE (initial_cond, solver, objective, pdf)

**Files**: `src/engines/fpe/initial_cond.mojo`, `src/engines/fpe/solver.mojo`, `src/engines/fpe/objective.mojo`, `src/engines/fpe/pdf.mojo`

**Actions**:

initial_cond.mojo:
- Rewrite `_delta_approx_flat` → `_delta_approx` returning NDArray
- Rewrite `_normalize_nonnegative` for NDArray in-place
- Rewrite `InitialCondition.compute` to be NDArray-native (eliminate wrapper `compute_nd`)
- Delete `compute_nd` (no longer needed as wrapper)

solver.mojo:
- Delete `FPESparseSystem` (List); rename `FPESparseSystemND` → `FPESparseSystem`
- Delete `FPEDenseSystem` (List); rename `FPEDenseSystemND` → `FPEDenseSystem`
- Delete `FPESparseLinearSystem` (List); rename `FPESparseLinearSystemND` → `FPESparseLinearSystem`
- Delete `_integrate_cpu_sparse`/`_solve_gpu_batch`/`_solve_cpu_parallel` (List versions)
- Rename `_integrate_cpu_sparse_nd` → `_integrate_cpu_sparse`, `_solve_gpu_batch_nd` → `_solve_gpu_batch`, `_solve_cpu_parallel_nd` → `_solve_cpu_parallel`
- Rename `solve_nd` → `solve`; delete List `solve`
- Update `InitialCondition.compute` calls to pass/return NDArray

objective.mojo (pre-existing bug fix):
- Fix `domain.map_s_to_physical(...)` → `domain.s_points_phys[...]`
- Fix `domain.map_v_to_physical(...)` → `domain.v_points_phys[...]`
- Rewrite `_integrate_call_price` to use NDArray inputs
- Rewrite `ObjectiveFunction.compute` to return `NDArray[Float64]`

pdf.mojo:
- Rewrite `PDFComputer.compute` to return `NDArray[Float64]` (2D)

**Items**: ~10 items removed, 5 gaps filled, 1 bug fixed

**Test gate**: test_zeros_3d.mojo, integration test

### L5: Optim (LM, OSQP, Calibrator)

**Files**: `src/numerics/optim/lm.mojo`, `src/numerics/optim/osqp.mojo`, `src/numerics/optim/calibrator.mojo`, `src/numerics/optim/__init__.mojo`

**Actions**:

lm.mojo:
- Delete `ResidualCallable` (List) / `JacobianCallable` (List); rename `ResidualCallableND` → `ResidualCallable`, `JacobianCallableND` → `JacobianCallable`
- Delete `LevenbergMarquardt.solve` (List); rename `solve_nd` → `solve`

osqp.mojo:
- Delete `OSQPSolver.solve_nnls_sparse` (List); rename `solve_nnls_sparse_nd` → `solve_nnls_sparse`
- Rewrite `solve_nnls_dense` to take/return NDArray (replace List params with NDArray, same algorithm)
- Rewrite `ProjectedGradient` struct to use NDArray internally and in `solve` signature
- Rewrite `OSQP.solve_nnls` to take/return NDArray

calibrator.mojo:
- Rewrite `CalibratorResidual` to implement `ResidualCallable` (NDArray) directly
- Rewrite `CalibratorJacobian` to implement `JacobianCallable` (NDArray) directly
- Rewrite `_params_to_vec`/`_vec_to_params` to return/take NDArray
- Rewrite `calibrate` to use NDArray throughout; delete List version

**Items**: ~8 items removed, 4 gaps filled

**Test gate**: test_optim.mojo, test_adam.mojo

### L6: NAIS Net + Autograd

**Files**: `src/engines/nais/nais_net.mojo`, `src/engines/nais/autograd.mojo`

**Actions**:

nais_net.mojo:
- Delete `_matmul_vec` (dead code in stable_linear.mojo, not linalg.mojo)
- Rewrite `NaisNet` struct: all `List[List[Float64]]` fields → `NDArray[Float64]`, `List[Float64]` → `NDArray[Float64]`
- Replace `StableLinear` (List) usage with `StableLinear` (NDArray) — already renamed in L5
- Rewrite `_make_weights` to return NDArray (2D)
- Rewrite `_linear` using NDArray mat-vec
- Rewrite `_sin_vec`/`_add_vec` as NDArray element-wise ops
- Rewrite tracked operations (`_linear_tracked_with_indices`, `_linear_tracked_record_weights`, `_stable_linear_forward_tracked`) for NDArray

autograd.mojo:
- Rewrite `Tape` internal storage: `values`/`adjoints` as `NDArray[Float64]`, `indices` as `NDArray[Int]`
- Rewrite `record_value`/`add`/`mul`/`sin`/`linear`/`backward`/`gradients_for` for NDArray

**Items**: ~10 items rewritten, 7 gaps filled

**Test gate**: test_autograd_tape.mojo, integration test

## Pre-Existing Bug Fixes

| Bug | Location | Fix | Layer |
|-----|----------|-----|-------|
| `map_s_to_physical` missing | objective.mojo:34 | → `domain.s_points_phys[i]` | L4 |
| `map_v_to_physical` missing | objective.mojo:37 | → `domain.v_points_phys[j]` | L4 |
| test_sparse_lu failure | test_sparse_lu.mojo | Fix during L2 | L2 |
| test_ode failure | test_ode.mojo | Fix during L3 | L3 |
| test_bspline failure (tensor_product_shapes) | test_bspline.mojo | Fix during L1 if sparse-related | L1 |
| test_calibrator failure | test_calibrator.mojo | Fix during L5 | L5 |

## Testing Protocol

1. After each file edit: `pixi run mojo run -I src tests/test_<module>.mojo`
2. After each layer: run ALL test files
3. Indentation verification: byte-level check via `python3 -c` script after editing any Mojo file
4. No `mojo check` available — use `mojo run` with test scripts
5. After ALL layers: full test suite must pass

## Key Technical Constraints

- **Mojo indentation is semantic** — wrong indent pairs with wrong `if:`
- **`IntTuple.__getitem__` returns `IntTuple`** — must use `Int(s[i])`
- **`NDArray.shape()` returns `IntTuple`** — use `Int(D.shape()[dim])`
- **No zero-length NDArray** — error cases use `NDArray[Float64](1)` placeholder
- **`CSRMatrix[dtype: DType = DType.float64]`** — bare `CSRMatrix` works but generic/trait contexts need explicit `CSRMatrix[DType.float64]`
- **`UnsafePointer` has no `.offset()`** — use `p + n` pointer arithmetic
- **`@fieldwise_init` structs** require ALL fields as keyword args
- **`memset_zero(ptr, count)`** works with element count (not byte count)

## Success Criteria

- Zero `List[Float64]` usage in `src/numerics/`, `src/sparse/`, `src/engines/`
- All `_list` and `_nd` suffixed functions/types deleted
- All original names (`ODESystem`, `solve`, `spmv`, etc.) use NDArray
- All existing tests pass (excluding truly unfixable pre-existing bugs, documented)
- No semantic regressions (same numerical results)
