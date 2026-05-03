# NDArray Numerics Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert all numerics/ consumers from List[Float64]-based API to NDArray[Float64]-based API, eliminating `_list` suffix functions and enabling SIMD/vectorization throughout.

**Architecture:** Incremental conversion using parallel traits. Each module gets NDArray-native functions/traits alongside existing List versions. Consumers are switched one at a time. Once all consumers of a List function are migrated, the List version is removed.

**Tech Stack:** Mojo v0.26.3 (Nightly), NDArray[Float64] from `numerics.utils.ndarray`, Vector = NDArray[Float64] alias, pixi for env management.

---

## Key Design Decisions

1. **Parallel trait strategy**: Create `ODESystemND` alongside `ODESystem` (List-based). Migrate consumers incrementally. Remove old trait once all migrated. Same for `ResidualCallableND`/`JacobianCallableND` alongside `ResidualCallable`/`JacobianCallable`.
2. **`ODESolutionND`**: New struct with `NDArray[Float64]` fields for `t` and `y`. Error cases use `NDArray[Float64](1)` as placeholder (since NDArray requires n>0). Consumers check `.success` before accessing data.
3. **Pre-allocated buffers**: For rk45, use `copy_from_fixed()` instead of ownership transfer (`^`) to keep pre-allocated buffers valid across loop iterations.
4. **NDArray-to-NDArray copy**: Always use `copy_from_fixed()`, never `copy_from()` (which takes List).
5. **No zero-length NDArray**: All NDArray constructors require n > 0. Error-case ODESolution uses size-1 placeholder.

---

## Dependency Graph

```
ode/types.mojo (ODESystemND + ODESolutionND) <-- NEW parallel traits
    ├── rk45.mojo (new RungeKutta45ND uses ODESystemND + ODESolutionND)
    ├── radau.mojo (new solve_nd uses ODESolutionND)
    └── solver.mojo (FPESparseSystemND/FPEDenseSystemND implement ODESystemND)

sparse_lu.mojo (add solve_nd())
    └── linalg.mojo (compute_jacobian_nd calls lu.solve_nd)

linalg.mojo (add _nd functions)
    ├── solver.mojo (uses lu_solve_nd)
    └── lm.mojo (uses lu_solve_nd)

lm.mojo (add ResidualCallableND/JacobianCallableND + solve_nd)
    └── calibrator.mojo (implements ND traits)

osqp.mojo (add solve_nnls_sparse_nd)
    └── initial_cond.mojo (stays on List path for now)

stable_linear.mojo (add StableLinearND with NDArray fields)
    └── nais_net.mojo (uses StableLinearND)
```

## Conversion Order

1. **ode/types.mojo** — add `ODESystemND` trait + `ODESolutionND` struct
2. **sparse_lu.mojo** — add `solve_nd()`
3. **linalg.mojo** — add NDArray-native `_nd` functions
4. **rk45.mojo** — add `RungeKutta45ND` using `ODESystemND` + `ODESolutionND`
5. **radau.mojo** — add `solve_nd` using `ODESolutionND`
6. **solver.mojo** — add NDArray-native ODE system implementations
7. **lm.mojo** — add `ResidualCallableND`/`JacobianCallableND` + `solve_nd`
8. **osqp.mojo** — add `solve_nnls_sparse_nd` + `ProjectedGradientND`
9. **stable_linear.mojo** — add `StableLinearND` + `make_stable_linear_nd`
10. **Integration test** — verify all tests pass

---

### Task 1: Add parallel ODE types (ODESystemND + ODESolutionND)

**Files:**
- Modify: `src/numerics/ode/types.mojo`
- Modify: `src/numerics/ode/__init__.mojo`

**Why first:** Both rk45 and radau need NDArray-based ODE types. Using parallel traits avoids breaking 15+ test/benchmark files that implement the old `ODESystem`.

- [ ] **Step 1: Add ODESystemND trait and ODESolutionND struct to types.mojo**

Append to `src/numerics/ode/types.mojo` (keep existing `ODESystem` and `ODESolution` untouched):

```mojo
from numerics.utils.ndarray import NDArray

trait ODESystemND:
    """NDArray-based interface for ODE right-hand side: dy/dt = f(t, y)."""

    def rhs(self, t: Float64, y: NDArray[Float64], mut dydt: NDArray[Float64]) raises:
        ...

    def dim(self) -> Int:
        ...


@fieldwise_init
struct ODESolutionND(Copyable, Movable):
    var t: NDArray[Float64]
    var y: NDArray[Float64]
    var success: Bool
    var message: String
```

Add `from numerics.utils.ndarray import NDArray` import at top of file.

- [ ] **Step 2: Update `__init__.mojo` re-exports**

Add to `src/numerics/ode/__init__.mojo`:
```mojo
from numerics.ode.types import ODESystemND, ODESolutionND
```

- [ ] **Step 3: Verify existing tests still pass (no breakage)**

Run: `pixi run mojo run -I src tests/test_ndarray.mojo 2>&1 | tail -3`
Expected: 18 pass (no dependency on ode/types)

---

### Task 2: Add sparse_lu.solve_nd

**Files:**
- Modify: `src/numerics/sparse_lu.mojo:236-249`

**Why:** The only List touchpoint in an otherwise fully Vector-native file. Adding `solve_nd()` eliminates the List-to-Vector-to-List bridge for NDArray callers.

- [ ] **Step 1: Add solve_nd method after existing solve()**

```mojo
def solve_nd(mut self, b: NDArray[Float64]) -> NDArray[Float64]:
    var n = self.n
    var x = Vector(n)
    x.copy_from_fixed(b)
    var work = Vector(n)
    self.solve_inplace(x, work)
    return x^
```

Keep existing `solve(b: List[Float64]) -> List[Float64]` for backward compat.

- [ ] **Step 2: Verify compilation**

Run: `pixi run mojo check src/numerics/sparse_lu.mojo -I src 2>&1 | grep error || echo OK`

---

### Task 3: Add NDArray-native linalg functions

**Files:**
- Modify: `src/numerics/linalg.mojo`
- Modify: `src/numerics/__init__.mojo`
- Modify: `tests/test_linalg.mojo`

**Why:** linalg is the critical hub. Adding `_nd` versions unlocks NDArray-native consumers without breaking existing List-based ones.

**Design decisions:**
- Add `_nd` suffix functions alongside existing List-based functions (no renaming of existing functions)
- 2D matrices become `NDArray[Float64](nrows, ncols)` with `A[i, j]` indexing
- Local `copy_mat`/`copy_vec` are only used internally. Keep them for List functions. Use `numerics.utils.copy_vec`/`copy_mat` for NDArray versions.

- [ ] **Step 1: Add NDArray-native functions to linalg.mojo**

Add import: `from numerics.utils import abs_f64, zeros, zeros_mat, copy_vec, copy_mat`

Key signatures:
```mojo
def lu_solve_nd(A: NDArray[Float64], b: NDArray[Float64]) -> NDArray[Float64]:
def dense_matvec_nd(A: NDArray[Float64], x: NDArray[Float64]) -> NDArray[Float64]:
def sparse_matvec_nd(A: CSRMatrix, x: NDArray[Float64]) -> NDArray[Float64]:
def compute_jacobian_nd(M: CSRMatrix, K: CSRMatrix) raises -> NDArray[Float64]:
```

`lu_solve_nd`: Use `copy_mat(A)` and `copy_vec(b)` from `numerics.utils`. `A[i, j]` indexing. `A.swap_rows(i, pivot)` (already exists on NDArray). Return `x^`.

`compute_jacobian_nd`: Use `SparseLU.solve_nd(rhs_vec)` instead of `lu.solve(rhs)`. Build `neg_K` as `zeros_mat(n, n)`. Build `rhs` column as `Vector(n)`. Result `J` as `zeros_mat(n, n)` with `J[i, col] = x[i]`.

`sparse_matvec_nd`: Same SIMD logic but `zeros(n)` for output. `x[A.indices[p]]` (NDArray supports Int indexing).

`dense_matvec_nd`: `A[i, j]` indexing, `zeros(n)` for output.

- [ ] **Step 2: Update numerics/__init__.mojo re-exports**

Add: `lu_solve_nd`, `dense_matvec_nd`, `sparse_matvec_nd`, `compute_jacobian_nd`

- [ ] **Step 3: Add NDArray linalg tests**

Add `test_lu_solve_nd`, `test_dense_matvec_nd`, `test_sparse_matvec_nd` to `tests/test_linalg.mojo`.

Run: `pixi run mojo run -I src tests/test_linalg.mojo 2>&1 | tail -3`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/numerics/linalg.mojo src/numerics/__init__.mojo tests/test_linalg.mojo
git commit -m "feat: add NDArray-native linalg functions (lu_solve_nd, dense_matvec_nd, sparse_matvec_nd, compute_jacobian_nd)"
```

---

### Task 4: Add RK45ND using ODESystemND + ODESolutionND

**Files:**
- Modify: `src/numerics/ode/rk45.mojo`

**Why:** RK45 is entirely List-based with massive per-step allocation (8 `zeros_list` calls per step). Pre-allocated Vectors eliminate heap allocation in the hot loop.

**Design decisions:**
- Add `RungeKutta45ND[System: ODESystemND]` alongside existing `RungeKutta45[System: ODESystem]`
- `solve_nd()` takes `y0: NDArray[Float64]`, returns `ODESolutionND`
- Pre-allocate k1..k7, ytmp, y5 as `Vector(n)` OUTSIDE the while loop
- **CRITICAL**: Use `y.copy_from_fixed(y5)` instead of `y = y5^` -- ownership transfer would consume y5, breaking the pre-allocated buffer for next iteration
- `t_values` accumulates as `List[Float64]`, `y_values` accumulates as `List[NDArray[Float64]]` (copies via `copy_vec(y)`)
- At return: convert Lists to `NDArray[Float64]`

- [ ] **Step 1: Add RungeKutta45ND struct and solve_nd method**

Import: `from numerics.utils import zeros, copy_vec, abs_f64, max_f64, min_f64`
Import: `from numerics.ode.types import ODESolutionND, ODESystemND`

Key differences from List version:
- `y0: NDArray[Float64]` instead of `y0: List[Float64]`
- `var y = copy_vec(y0)` (NDArray copy)
- `k1..k7, ytmp, y5` = `Vector(n)` each, allocated once before loop
- `y.copy_from_fixed(y5)` instead of `y = y5^` -- preserves y5 buffer
- `y_values.append(copy_vec(y))` -- proper NDArray copy
- Error case: `return ODESolutionND(NDArray[Float64](1), NDArray[Float64](1, 1), False, "...")`

- [ ] **Step 2: Build ODESolutionND from accumulated data**

```mojo
var nt = len(t_values)
var t_arr = NDArray[Float64](nt)
for i in range(nt):
    t_arr[i] = t_values[i]

var y_arr = NDArray[Float64](nt, n)
for idx in range(nt):
    for i in range(n):
        y_arr[idx, i] = y_values[idx][i]

return ODESolutionND(t_arr^, y_arr^, True, "RK45 integration successful")
```

- [ ] **Step 3: Update `ode/__init__.mojo` to export RungeKutta45ND**

- [ ] **Step 4: Verify compilation**

Run: `pixi run mojo check src/numerics/ode/rk45.mojo -I src 2>&1 | grep error || echo OK`

---

### Task 5: Add radau solve_nd using ODESolutionND

**Files:**
- Modify: `src/numerics/ode/radau.mojo`

**Why:** Radau is already Vector-native internally. Adding `solve_nd()` with NDArray boundary avoids the List-to-Vector-to-List bridge.

**Key changes:**
- Add `LinearODESystemND` trait alongside `LinearODESystem`
- Add `RadauSparseLinearSolverND[System: LinearODESystemND]` alongside existing struct
- `solve_nd()`: `y0: NDArray[Float64]`, return `ODESolutionND`
- `y.copy_from_fixed(y0)` instead of `y.copy_from(y0)` (copy_from takes List, copy_from_fixed takes NDArray)
- `K.spmv_nd(y0, k0_vec)` instead of `K.spmv(y0)` (line 176)
- All `ODESolution(...)` returns become `ODESolutionND(...)` with NDArray-converted t/y

- [ ] **Step 1: Add LinearODESystemND trait**

```mojo
trait LinearODESystemND:
    def get_M(self) -> CSRMatrix: ...
    def get_K(self) -> CSRMatrix: ...
```

(This is identical to `LinearODESystem` since get_M/get_K return CSRMatrix, not List. The parallel trait is needed for the generic parameter on `RadauSparseLinearSolverND`.)

- [ ] **Step 2: Add RadauSparseLinearSolverND struct**

Copy `RadauSparseLinearSolver` with `System: LinearODESystemND` parameter. Change `solve` to `solve_nd` with:
- `y0: NDArray[Float64]` parameter
- `y.copy_from_fixed(y0)` (not `copy_from`)
- `var k0_vec = Vector(n); K.spmv_nd(y0, k0_vec)` (not `K.spmv(y0)`)
- All `return ODESolution(...)` become `return _build_solution_nd(t_values, y_values, n, success, message)`

- [ ] **Step 3: Add _build_solution_nd helper inside radau.mojo**

```mojo
def _build_solution_nd(
    t_values: List[Float64],
    y_values: List[NDArray[Float64]],
    n: Int,
    success: Bool,
    message: String,
) -> ODESolutionND:
    var nt = len(t_values)
    if nt == 0:
        return ODESolutionND(NDArray[Float64](1), NDArray[Float64](1, n), success, message)
    var t_arr = NDArray[Float64](nt)
    for i in range(nt):
        t_arr[i] = t_values[i]
    var y_arr = NDArray[Float64](nt, n)
    for idx in range(nt):
        for i in range(n):
            y_arr[idx, i] = y_values[idx][i]
    return ODESolutionND(t_arr^, y_arr^, success, message)
```

**CRITICAL**: `y_values` type changes from `List[List[Float64]]` to `List[NDArray[Float64]]`. Each `y_values.append(...)` must use `y_values.append(copy_vec(y))` instead of `y_values.append(y.to_list())`.

- [ ] **Step 4: Update `ode/__init__.mojo` to export new types**

- [ ] **Step 5: Verify compilation**

Run: `pixi run mojo check src/numerics/ode/radau.mojo -I src 2>&1 | grep error || echo OK`

---

### Task 6: Add NDArray-native ODE system implementations in solver.mojo

**Files:**
- Modify: `src/engines/fpe/solver.mojo`

**Why:** solver.mojo implements `ODESystem` (List-based). Add `ODESystemND` implementations so the NDArray-based ODE solvers can be used.

**Key changes:**
- Add `FPESparseSystemND(ODESystemND)` -- `rhs` takes `NDArray[Float64]`, uses `spmv_nd` instead of `spmv_into`
- Add `FPEDenseSystemND(ODESystemND)` -- `rhs` takes `NDArray[Float64]`, `A` stored as `NDArray[Float64]` (2D)
- Add `FPESparseLinearSystemND(LinearODESystemND)` -- same as FPESparseLinearSystem (get_M/get_K unchanged)
- Replace `_csr_to_dense_float` calls with `M.to_dense_nd()` (already exists on CSRMatrix)
- Replace `diag_scale` with `diag_scale_nd` (import from `sparse.ops`)
- Replace `lu_solve` with `lu_solve_nd` in `_compute_sparse_neg_M_inv_K_nd`
- Replace `zeros_list` with `zeros`, `copy_vec_list` with `copy_vec`

**CRITICAL: FPESolver return type**
- `FPESolver.solve()` currently returns `List[List[Float64]]`
- After calling `RadauSparseLinearSolverND.solve_nd()`, gets `ODESolutionND` with `sol.y` as 2D NDArray
- Extract rows from 2D NDArray using `sol.y.row(idx)` (returns 1D view)
- For unscale loop: `var q_unscaled = zeros(n); for i in range(n): q_unscaled[i] = Dinv[i] * sol.y.row(idx)[i]`
- Add `_integrate_cpu_sparse_nd` returning `NDArray[Float64]` (2D: n_times x n_vars)

- [ ] **Step 1: Add NDArray-native ODE system structs**

```mojo
struct FPESparseSystemND(ODESystemND):
    var neg_M_inv_K: CSRMatrix
    def rhs(self, t: Float64, y: NDArray[Float64], mut dydt: NDArray[Float64]) raises:
        _ = t
        self.neg_M_inv_K.spmv_nd(y, dydt)  # NOT spmv_into
    def dim(self) -> Int:
        return self.neg_M_inv_K.nrows

struct FPESparseLinearSystemND(LinearODESystemND):
    var M: CSRMatrix
    var K: CSRMatrix
    def __init__(out self, var M: CSRMatrix, var K: CSRMatrix):
        self.M = M^
        self.K = K^
    def get_M(self) -> CSRMatrix: return self.M.copy()
    def get_K(self) -> CSRMatrix: return self.K.copy()
```

- [ ] **Step 2: Add _integrate_cpu_sparse_nd method**

- `D`/`Dinv` as `Vector(n)` (from `zeros(n)`)
- `diag_scale_nd(M, Dinv, Dinv)` instead of `diag_scale`
- Import: `from sparse.ops import diag_scale_nd`
- Use `RadauSparseLinearSolverND[FPESparseLinearSystemND]`
- Convert `ODESolutionND.y` to `List[List[Float64]]` at return (for API compat)

- [ ] **Step 3: Replace `_csr_to_dense_float` with `M.to_dense_nd()`**

The existing `CSRMatrix.to_dense_nd()` returns `NDArray[Float64](nrows, ncols)`. Use it directly in `_compute_sparse_neg_M_inv_K_nd` with `lu_solve_nd`.

- [ ] **Step 4: Verify compilation**

Run: `pixi run mojo check src/engines/fpe/solver.mojo -I src 2>&1 | grep error || echo OK`

---

### Task 7: Add NDArray-native LM traits + solve_nd

**Files:**
- Modify: `src/numerics/optim/lm.mojo`
- Modify: `src/numerics/optim/__init__.mojo`
- Modify: `src/engines/calibrator/calibrator.mojo`
- Modify: `tests/test_optim.mojo`

**Why:** LM traits use List[Float64]. Add NDArray-native parallel traits.

- [ ] **Step 1: Add NDArray-native traits and solve_nd to lm.mojo**

```mojo
trait ResidualCallableND:
    def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]: ...

trait JacobianCallableND:
    def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]: ...
```

Note: `JacobianCallableND.__call__` returns `NDArray[Float64]` (2D: mxn matrix) instead of `List[List[Float64]]`.

Add `solve_nd` method to `LevenbergMarquardt`:
- `x0: NDArray[Float64]`, return `NDArray[Float64]`
- Uses `lu_solve_nd(JtJ, neg_Jtr)` instead of `lu_solve(JtJ, neg_Jtr)`
- JtJ as `zeros_mat(n, n)`, neg_Jtr as `zeros(n)`
- All `J[k][i]` become `J[k, i]` (2D NDArray indexing)
- `x.copy_from_fixed(x_new)` instead of `x = x_new^`

- [ ] **Step 2: Update optim/__init__.mojo re-exports**

Add: `ResidualCallableND`, `JacobianCallableND`

- [ ] **Step 3: Add CalibratorResidualND/CalibratorJacobianND to calibrator.mojo**

Keep old implementations. Add new NDArray-native ones.

- [ ] **Step 4: Add NDArray-native test to test_optim.mojo**

Add `test_lm_nd` with NDArray-native `ResidualCallableND`/`JacobianCallableND` implementations.

Run: `pixi run mojo run -I src tests/test_optim.mojo 2>&1 | tail -3`
Expected: all tests pass

---

### Task 8: Add NDArray-native OSQP functions

**Files:**
- Modify: `src/numerics/optim/osqp.mojo`
- Modify: `src/numerics/optim/__init__.mojo`

**Why:** `solve_nnls_sparse` has a List-to-Vector-to-List bridge. Adding `solve_nnls_sparse_nd` eliminates it.

**Note:** `initial_cond.mojo` still uses `solve_nnls_sparse` (List). It stays on the List path for now. Will be converted in a future phase.

- [ ] **Step 1: Add solve_nnls_sparse_nd to OSQPSolver**

```mojo
def solve_nnls_sparse_nd(
    self, M: CSRMatrix[DType.float64], b: NDArray[Float64]
) -> NDArray[Float64]:
```

- Direct `Mt.spmv_nd(b, q)` -- no bridge needed
- Return `z` directly (Vector = NDArray[Float64])
- Fallback: `return Vector(n)` (zeroed Vector)

- [ ] **Step 2: Add solve_nnls_dense_nd to OSQPSolver**

```mojo
def solve_nnls_dense_nd(
    self, A: NDArray[Float64], b: NDArray[Float64]
) -> NDArray[Float64]:
    var M = CSRMatrix[DType.float64].from_dense_nd(A)
    return self.solve_nnls_sparse_nd(M, b)
```

- [ ] **Step 3: Add ProjectedGradientND struct**

Rewrite with Vector pre-allocation: `c`, `grad`, `AtA` (2D), `Atb` as NDArray.

- [ ] **Step 4: Add solve_nnls_nd to OSQP wrapper**

```mojo
def solve_nnls_nd(self, A: NDArray[Float64], b: NDArray[Float64]) -> NDArray[Float64]:
    return self._solver.solve_nnls_dense_nd(A, b)
```

- [ ] **Step 5: Update optim/__init__.mojo**

- [ ] **Step 6: Verify with test_optim**

Run: `pixi run mojo run -I src tests/test_optim.mojo 2>&1 | tail -3`
Expected: all tests pass

---

### Task 9: Add NDArray-native StableLinear

**Files:**
- Modify: `src/numerics/nn/stable_linear.mojo`
- Modify: `src/numerics/nn/__init__.mojo`

**Why:** StableLinear stores W/b as List fields. Add NDArray-based `StableLinearND` alongside.

- [ ] **Step 1: Add StableLinearND struct**

```mojo
struct StableLinearND(Copyable, Movable):
    var W: NDArray[Float64]  # 2D: (in_features, out_features)
    var b: NDArray[Float64]  # 1D: (out_features,)
    var epsilon: Float64
```

All methods use `W[i, j]` indexing, `zeros_mat`/`zeros` from `numerics.utils`.

- [ ] **Step 2: Add make_stable_linear_nd factory**

```mojo
def make_stable_linear_nd(
    in_features: Int, out_features: Int, epsilon: Float64 = 0.01
) -> StableLinearND:
```

Uses `zeros_mat(in_features, out_features)` and `zeros(out_features)`.

- [ ] **Step 3: Update nn/__init__.mojo re-exports**

Add: `StableLinearND`, `make_stable_linear_nd`

- [ ] **Step 4: Update nais_net.mojo to use StableLinearND**

Add `StableLinearND` variant alongside existing List-based version.

- [ ] **Step 5: Verify compilation**

Run: `pixi run mojo check src/numerics/nn/stable_linear.mojo -I src 2>&1 | grep error || echo OK`

---

### Task 10: Integration test

**Files:**
- Verify all test files

- [ ] **Step 1: Run full test suite**

```bash
for t in test_ndarray test_sparse test_sparse_coo_diag test_linalg test_optim test_adam test_autograd_tape; do
    echo -n "$t: "; pixi run mojo run -I src tests/${t}.mojo 2>&1 | grep "Summary" | head -1
done
```

Expected: all pass

- [ ] **Step 2: Check compilation of all source files**

```bash
for f in src/numerics/ode/types.mojo src/numerics/sparse_lu.mojo src/numerics/linalg.mojo src/numerics/ode/rk45.mojo src/numerics/ode/radau.mojo src/engines/fpe/solver.mojo src/numerics/optim/lm.mojo src/numerics/optim/osqp.mojo src/numerics/nn/stable_linear.mojo; do
    echo -n "$f: "; pixi run mojo check $f -I src 2>&1 | grep error || echo OK
done
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: add NDArray-native parallel APIs across numerics/ (ODESystemND, linalg_nd, RK45ND, RadauND, LM-ND, OSQP-ND, StableLinearND)"
```

---

## Important Notes

1. **Parallel strategy**: Both List and NDArray versions coexist. List versions keep their original names. NDArray versions get `ND` suffix on types and `_nd` suffix on functions.
2. **ODESystemND/ODESolutionND** are NEW types. Existing `ODESystem`/`ODESolution` are untouched. All 15+ test/benchmark files continue to work unchanged.
3. **Pre-existing test failures** (`test_sparse_lu`, `test_ode`, `test_calibrator`, `test_bspline`) are NOT caused by NDArray changes. Do not fix them in this plan.
4. **`spmv_nd` already exists** on CSRMatrix -- use it instead of `spmv(List)`.
5. **`from_dense_nd` already exists** on CSRMatrix -- use it instead of `from_dense(List[List])`.
6. **`diag_scale_nd` already exists** in ops.mojo -- use it instead of `diag_scale`.
7. **`IntTuple.__getitem__` returns IntTuple** -- always use `Int(s[i])` when extracting shape dimensions.
8. **NDArray 2D indexing**: `A[i, j]` not `A[i][j]`. All nested List access must be converted.
9. **NDArray copy**: Use `copy_from_fixed()` for NDArray-to-NDArray. Use `copy_from()` for List-to-NDArray.
10. **No zero-length NDArray**: Error-case `ODESolutionND` uses `NDArray[Float64](1)` as placeholder.
11. **Remaining List consumers after this plan**: `initial_cond.mojo` (uses `solve_nnls_sparse`), `pdf.mojo` (uses `spmv(List)`), all test/benchmark files using old `ODESystem`. These will be migrated in a future phase when old traits are deprecated.
