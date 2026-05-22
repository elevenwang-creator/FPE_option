# Pipeline Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate redundant B-spline basis computation, duplicate matrix assembly, and reduce ODE solver LU cost to bring pipeline time from ~184s toward ~60s.

**Architecture:** Two-pronged approach: (1) Cache B-spline basis + integration weights in `FPEDomain`, refactor `GalerkinAssembler` to single-pass assembly, overload `FPESolver.solve()` to accept pre-built matrices, refactor `InitialCondition`/`PDFComputer` to reuse cached basis. (2) Widen RADAU5 LU reuse band and split the 2n×2n complex system into two n×n real solves using the already-factored real LU as preconditioner. Keep CSC format throughout (no CSR↔CSC transform bottlenecks).

**Tech Stack:** Mojo v0.26.3 (Nightly), pixi, existing sparse CSR/CSC infrastructure, existing `SparseLU`

---

## Profiling Baseline

Current wall time (~184s for n=2025, 64 RADAU5 steps):
- Step 3 (Galerkin assembly): ~12.9s — `domain.build_basis()` called 5+ times, each rebuilds `BSplineBasis` → `RecombinationBasis` (with spgemm) → `TensorProductBasis` from scratch
- Step 4 (InitialCondition): ~3s — another `build_basis()` + `integ_weights()` + mass matrix assembly (duplicate of step 3)
- Step 5 (RADAU5 ODE solve): ~165.7s — of which ~143s is LU factorization, dominated by the 2n×2n complex system
- Step 6 (PDFComputer): ~1s — two more `build_basis()` calls

**Redundancy count:**
| Computation | Times Called | Should Be |
|---|---|---|
| `build_basis()` | 5+ | 1 |
| `integ_weights()` | 3+ | 1 |
| `eval_tensor()` (two_basis) | 5+ | 1 |
| M assembly | 2 (caller + solver) | 1 |
| K assembly | 2 (caller + solver) | 1 |
| `BSplineBasis.__init__` + knot copy | 10+ | 2 |
| `RecombinationBasis.__init__` + spgemm R matrix | 10+ | 2 |
| Complex system LU factorize (2n×2n) | ~20-30 | 0 (replace with n×n solves) |

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/engines/fpe/domain.mojo` | Domain with cached basis/weights | **Modify** — add lazy cache fields |
| `src/engines/fpe/galerkin.mojo` | Galerkin assembler (M and K) | **Modify** — single-pass `assemble_all()`, use cached basis |
| `src/engines/fpe/initial_cond.mojo` | Initial condition q0 | **Modify** — accept optional M, use cached basis |
| `src/engines/fpe/pdf.mojo` | PDF evaluation | **Modify** — use cached basis |
| `src/engines/fpe/solver.mojo` | FPE solver | **Modify** — overload `solve(M, K, q0, t_eval)` |
| `src/numerics/bspline/tensor_product.mojo` | TensorProductBasis | **Modify** — cache `eval_tensor`, `partial_s`, `partial_v` results |
| `src/numerics/bspline/recombination.mojo` | RecombinationBasis | **Modify** — cache `recombination_matrix()` |
| `src/numerics/ode/radau.mojo` | RADAU5 solver | **Modify** — widen LU band, split complex system |
| `src/numerics/ode/types.mojo` | ODE traits | **Modify** — add `get_M_ref`/`get_K_ref` to trait |
| `src/engines/fpe/fpe_facade.mojo` | Top-level facade | **Modify** — call new single-pass API |
| `examples/single_price.mojo` | Integration test | **Modify** — use new API |
| `examples/test_perf.mojo` | Performance benchmark | **Create** — before/after timing comparison |
| `generate_radau.py` | Radau code generator | **Modify** — add split complex system logic |

---

## Task 1: Cache Basis and Weights in FPEDomain

**Files:**
- Modify: `src/engines/fpe/domain.mojo:138-236`
- Test: `examples/single_price.mojo`

The `FPEDomain` struct currently rebuilds `TensorProductBasis` from scratch on every `build_basis()` call. We add lazy-cached fields so the basis is built once and reused.

**Key constraint:** `TensorProductBasis[degree_s, degree_v]` is a generic type — cannot be stored as `Optional` directly in Mojo. Instead, use a `_basis_built: Bool` flag + owned field. But Mojo structs require all fields initialized in `__init__`. Since `TensorProductBasis` has no default constructor, we need a different approach: **store the cached basis components separately** (BSplineBasis, RecombinationBasis, RecombinationMatrix) and reconstruct `TensorProductBasis` on demand, or **use a builder pattern that produces a `FPECachedDomain` wrapper**.

Chosen approach: **Wrapper struct `FPECachedBasis`** that holds the pre-built basis and weights, constructed once from `FPEDomain`. This avoids modifying `FPEDomain`'s `__init__` signature.

- [ ] **Step 1: Create `FPECachedBasis` struct in `domain.mojo`**

```mojo
struct FPECachedBasis[degree_s: Int, degree_v: Int](Movable):
    var basis: TensorProductBasis[Self.degree_s, Self.degree_v]
    var weights: CSRMatrix
    var two_basis: CSRMatrix
    var s_partial: CSRMatrix
    var v_partial: CSRMatrix
    var two_basis_T: CSRMatrix

    def __init__(out self, domain: FPEDomain[Self.degree_s, Self.degree_v]):
        self.basis = domain.build_basis()
        self.weights = domain.integ_weights()
        self.two_basis = self.basis.eval_tensor(domain.s_points, domain.v_points)
        self.s_partial = self.basis.partial_s(domain.s_points, domain.v_points)
        self.v_partial = self.basis.partial_v(domain.s_points, domain.v_points)
        self.two_basis_T = self.two_basis.transpose()
```

- [ ] **Step 2: Add `cached_basis()` method to `FPEDomain`**

```mojo
def cached_basis(self) -> FPECachedBasis[Self.degree_s, Self.degree_v]:
    return FPECachedBasis[Self.degree_s, Self.degree_v](self)
```

- [ ] **Step 3: Verify compilation**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Compiles (old API still works)

- [ ] **Step 4: Commit**
```bash
git add src/engines/fpe/domain.mojo
git commit -m "feat: add FPECachedBasis for lazy-cached basis/weights"
```

---

## Task 2: Refactor GalerkinAssembler to Single-Pass Assembly

**Files:**
- Modify: `src/engines/fpe/galerkin.mojo:33-123`
- Test: `examples/single_price.mojo`

Currently `mass_matrix()` and `stiffness_matrix()` each call `domain.build_basis()` independently, recomputing the basis, two_basis, and weights twice. Refactor to use `FPECachedBasis` and add `assemble_all()` that computes both M and K in one pass.

- [ ] **Step 1: Add `assemble_all()` method to `GalerkinAssembler`**

```mojo
def assemble_all(
    self, domain: FPEDomain[Self.B], params: HestonParams
) -> Tuple[CSRMatrix, CSRMatrix, FPECachedBasis[Self.B]]:
    var cached = FPECachedBasis[Self.B](domain)
    var M = self._mass_from_cached(cached)
    var K = self._stiffness_from_cached(cached, domain, params)
    return (M^, K^, cached^)
```

- [ ] **Step 2: Implement `_mass_from_cached()`**

```mojo
def _mass_from_cached(self, cached: FPECachedBasis[Self.B]) -> CSRMatrix:
    var two_basis_Tw = diag_col_scale(cached.two_basis_T, cached.weights)
    return two_basis_Tw @ cached.two_basis
```

- [ ] **Step 3: Implement `_stiffness_from_cached()`**

```mojo
def _stiffness_from_cached(
    self,
    cached: FPECachedBasis[Self.B],
    domain: FPEDomain[Self.B],
    params: HestonParams,
) -> CSRMatrix:
    # Use cached.two_basis, cached.s_partial, cached.v_partial, cached.weights
    # Same math as current stiffness_matrix(), just no redundant build_basis() calls
    # ... (full implementation with all the k1..k8 coefficients and diag operations)
```

- [ ] **Step 4: Refactor existing `mass_matrix()` and `stiffness_matrix()` to use cached path**

```mojo
def mass_matrix(self, domain: FPEDomain[Self.B]) -> CSRMatrix:
    var cached = FPECachedBasis[Self.B](domain)
    return self._mass_from_cached(cached^)

def stiffness_matrix(self, domain: FPEDomain[Self.B], params: HestonParams) -> CSRMatrix:
    var cached = FPECachedBasis[Self.B](domain)
    return self._stiffness_from_cached(cached^, domain, params)
```

- [ ] **Step 5: Verify compilation and functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same output as before (PDF integral ~0.63)

- [ ] **Step 6: Commit**
```bash
git add src/engines/fpe/galerkin.mojo src/engines/fpe/domain.mojo
git commit -m "feat: single-pass Galerkin assembly via FPECachedBasis"
```

---

## Task 3: Overload FPESolver.solve() to Accept Pre-Built Matrices

**Files:**
- Modify: `src/engines/fpe/solver.mojo:93-110`
- Modify: `src/numerics/ode/types.mojo:80-85`
- Test: `examples/single_price.mojo`

Currently `FPESolver.solve(domain, params, t_eval)` re-assembles M and K internally (lines 99-101), duplicating the caller's assembly work. Add an overload that accepts pre-built M, K, q0.

Also, the `LinearODESystem` trait's `get_M()`/`get_K()` return copies (`.copy()` at solver.mojo:61,63), which is wasteful. Change the trait to return `ref` views.

- [ ] **Step 1: Add `get_M_ref`/`get_K_ref` to `LinearODESystem` trait**

In `types.mojo`:
```mojo
trait LinearODESystem:
    def get_M_ref(ref self) -> ref CSRMatrix:
        ...

    def get_K_ref(ref self) -> ref CSRMatrix:
        ...
```

- [ ] **Step 2: Implement in `FPESparseLinearSystem`**

```mojo
def get_M_ref(ref self) -> ref CSRMatrix:
    return self.M

def get_K_ref(ref self) -> ref CSRMatrix:
    return self.K
```

- [ ] **Step 3: Update `RadauSparseLinearSolver.solve()` to use `get_M_ref`/`get_K_ref`**

In `radau.mojo:113-114`, change:
```mojo
var M_ref = system.get_M_ref()
var K_ref = system.get_K_ref()
```
And pass `M_ref`/`K_ref` by reference to all internal methods instead of copying.

**Note:** This requires careful refactoring — the `M`/`K` variables are used throughout the solver's hot loop. The `_build_real_system`, `_build_complex_system`, `_update_real_data_fast`, `_update_complex_data_fast`, and `spmv` calls all need `M`/`K`. Since these are read-only in the solver, `ref` access is safe. However, `FixedSizeVector.spmv` already takes `CSRMatrix` by `read` convention, so no change needed there.

- [ ] **Step 4: Add `solve_with_matrices()` overload to `FPESolver`**

```mojo
def solve_with_matrices(
    self,
    var M: CSRMatrix,
    var K: CSRMatrix,
    q0: List[Float64],
    t_eval: List[Float64],
) raises -> List[List[Float64]]:
    comptime if Self.B == 1:
        return self._integrate_cpu_sparse(M^, K^, q0, t_eval)
    else:
        comptime if has_accelerator():
            return self._solve_gpu_batch(M^, K^, q0, t_eval)
        else:
            return self._solve_cpu_parallel(M^, K^, q0, t_eval)
```

- [ ] **Step 5: Update old `solve()` to use `solve_with_matrices()` internally but keep backward compat**

- [ ] **Step 6: Verify compilation and functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 7: Commit**
```bash
git add src/engines/fpe/solver.mojo src/numerics/ode/types.mojo src/numerics/ode/radau.mojo
git commit -m "feat: overload FPESolver.solve() to accept pre-built M, K, q0"
```

---

## Task 4: Refactor InitialCondition to Reuse Cached Basis and M

**Files:**
- Modify: `src/engines/fpe/initial_cond.mojo:69-139`
- Test: `examples/single_price.mojo`

Currently `InitialCondition.compute()` independently calls `domain.build_basis()` and assembles the mass matrix (the `galerkin_matrix` at line 87). If the caller already has M and a cached basis, these are pure waste.

- [ ] **Step 1: Add `compute_with_cached()` overload**

```mojo
def compute_with_cached(
    self,
    cached: FPECachedBasis,
    M: CSRMatrix,
    domain: FPEDomain,
    params: HestonParams,
    sigma0: Float64 = 0.1,
) raises -> List[Float64]:
    # Skip build_basis() — use cached.two_basis, cached.weights
    # Skip galerkin_matrix assembly — use M directly
    var delta_flat = _delta_approx_flat(domain, params, sigma0)
    var w_delta = cached.weights.spmv_new(delta_flat)
    var galerkin_projection = cached.two_basis_T.spmv_new(w_delta)

    # D-scaling from M diagonal
    var Dinv_diag: List[Float64] = []
    for i in range(M.nrows):
        var diag_val = 0.0
        for p in range(M.indptr[i], M.indptr[i + 1]):
            if M.indices[p] == i:
                diag_val = M.data[p]
                break
        if diag_val > 0.0:
            Dinv_diag.append(1.0 / sqrt(diag_val))
        else:
            Dinv_diag.append(1.0)

    var Dinv = DiagMatrix(Dinv_diag.copy()).to_csr()
    var galerkin_scaled = (Dinv @ M) @ Dinv
    var galerkin_proj_scaled = Dinv.spmv_new(galerkin_projection)

    var ones = List[Float64]()
    for _ in range(cached.two_basis.nrows):
        ones.append(1.0)
    var m = cached.two_basis_T.spmv_new(cached.weights.spmv_new(ones))

    # OSQP NNLS solve (unchanged)
    var osqp = OSQPSolver(...)
    var c_result = osqp.solve_nnls_sparse(galerkin_scaled, galerkin_proj_scaled)
    var result = Dinv.spmv_new(c_result)
    _normalize_nonnegative(result)
    var m_dot_r = 0.0
    for i in range(len(result)):
        m_dot_r += m[i] * result[i]
    if m_dot_r > 0.0:
        for i in range(len(result)):
            result[i] = result[i] / m_dot_r
    return result^
```

- [ ] **Step 2: Keep old `compute()` for backward compatibility** (calls `compute_with_cached` internally)

- [ ] **Step 3: Verify compilation and functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same q0 output

- [ ] **Step 4: Commit**
```bash
git add src/engines/fpe/initial_cond.mojo
git commit -m "feat: InitialCondition.compute_with_cached() reuses M and basis"
```

---

## Task 5: Refactor PDFComputer to Reuse Cached Basis

**Files:**
- Modify: `src/engines/fpe/pdf.mojo:21-38`
- Test: `examples/single_price.mojo`

`PDFComputer.compute()` calls `domain.build_basis()` + `basis.eval_tensor()` every time. For t=0 and t=T evaluations in the pipeline, this means 2 full basis rebuilds.

- [ ] **Step 1: Add `compute_with_cached()` overload**

```mojo
def compute_with_cached(
    self, cached: FPECachedBasis, q_t: List[Float64]
) -> List[List[Float64]]:
    var pdf_flat_scalar = cached.two_basis.spmv_new(q_t)
    var pdf_flat: List[Float64] = []
    for i in range(len(pdf_flat_scalar)):
        pdf_flat.append(pdf_flat_scalar[i])
    return _reshape_to_grid(pdf_flat, len(cached.two_basis.nrows) ...)
```

**Note:** Need to pass `n_s`/`n_v` or store in `FPECachedBasis`. Add `n_s: Int` and `n_v: Int` fields to `FPECachedBasis`.

- [ ] **Step 2: Store quad point counts in `FPECachedBasis`**

Add to `FPECachedBasis.__init__`:
```mojo
self.n_s = len(domain.s_points)
self.n_v = len(domain.v_points)
```

- [ ] **Step 3: Keep old `compute()` for backward compat**

- [ ] **Step 4: Verify compilation and functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 5: Commit**
```bash
git add src/engines/fpe/pdf.mojo src/engines/fpe/domain.mojo
git commit -m "feat: PDFComputer.compute_with_cached() reuses basis"
```

---

## Task 6: Wire Up Pipeline — Update single_price.mojo to Use Cached Path

**Files:**
- Modify: `examples/single_price.mojo:158-250`

Update the pipeline to use `assemble_all()` + `compute_with_cached()` + `solve_with_matrices()` + `compute_with_cached()` for PDF.

- [ ] **Step 1: Replace step 3 (assembly)**

```mojo
var assembler = GalerkinAssembler[1]()
var (M, K, cached) = assembler.assemble_all(domain, params)
```

- [ ] **Step 2: Replace step 4 (initial condition)**

```mojo
var q0 = InitialCondition[1]().compute_with_cached(cached, M, domain, params, sigma0=0.1)
```

- [ ] **Step 3: Replace step 5 (ODE solve)**

```mojo
var q_t = fpe_solver.solve_with_matrices(M^, K^, q0, t_eval)
```

- [ ] **Step 4: Replace step 6 (PDF)**

```mojo
var pdf_grid_init = pdf_computer.compute_with_cached(cached, q_init)
var pdf_grid = pdf_computer.compute_with_cached(cached, q_final)
```

- [ ] **Step 5: Run and verify functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same numerical output, step 3 time should drop from ~12.9s to ~5s (single basis build + single eval_tensor)

- [ ] **Step 6: Commit**
```bash
git add examples/single_price.mojo
git commit -m "feat: pipeline uses cached basis/weights throughout"
```

---

## Task 7: Cache RecombinationMatrix in RecombinationBasis

**Files:**
- Modify: `src/numerics/bspline/recombination.mojo:14-75`

`RecombinationBasis.eval_all()` and `first_derivative_all()` each call `self.recombination_matrix()`, which builds the same R matrix twice (once for eval, once for derivative). The R matrix depends only on `num_basis`, `left_cond`, `right_cond` — all immutable after construction. Cache it.

- [ ] **Step 1: Add cached `_R` and `_R_built` fields to `RecombinationBasis`**

**Challenge:** `RecombinationBasis` is `Copyable, Movable`. `CSRMatrix` is `Movable` but not `Copyable`. We can't have an uninitialized `CSRMatrix` field (no default constructor). 

**Solution:** Use `Optional[CSRMatrix]` — but `CSRMatrix` may not satisfy `Optional` constraints. Alternative: make `_R` a `List[Optional[CSRMatrix]]` with length 1, or restructure to compute R in `__init__` and store it.

**Chosen approach:** Compute `R` eagerly in `__init__` and store as a field. Since `RecombinationBasis.__init__` already takes `var basis`, we can compute R there.

```mojo
struct RecombinationBasis[degree: Int](Movable):
    var basis: BSplineBasis[Self.degree]
    var left_cond: String
    var right_cond: String
    var _R: CSRMatrix

    def __init__(
        out self,
        var basis: BSplineBasis[Self.degree],
        left_cond: String = "dirichlet",
        right_cond: String = "neumann",
    ):
        self.basis = basis^
        self.left_cond = left_cond
        self.right_cond = right_cond
        self._R = self._build_recombination_matrix()
```

Rename old `recombination_matrix()` to `_build_recombination_matrix()` (private). Add public `recombination_matrix()` that returns `self._R` (no copy — use `read` convention).

- [ ] **Step 2: Update `eval_all()` and `first_derivative_all()` to use `self._R`**

```mojo
def eval_all(self, points: List[Float64]) -> CSRMatrix:
    var B = self.basis.eval_all(points)
    return B @ self._R

def first_derivative_all(self, points: List[Float64]) -> CSRMatrix:
    var dB = self.basis.first_derivative_all(points)
    return dB @ self._R
```

- [ ] **Step 3: Remove `Copyable` conformance from `RecombinationBasis`** (CSRMatrix is Movable, not Copyable — already the case)

- [ ] **Step 4: Verify compilation**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same output, slightly faster

- [ ] **Step 5: Commit**
```bash
git add src/numerics/bspline/recombination.mojo
git commit -m "perf: cache recombination matrix in RecombinationBasis.__init__"
```

---

## Task 8: Single-Pass eval in BSplineBasis.eval_all()

**Files:**
- Modify: `src/numerics/bspline/basis.mojo:71-124`

`eval_all()` currently calls `de_boor_cox()` twice per (point, basis) pair — once in the counting pass and once in the fill pass. Refactor to a single-pass approach: eval into a dense row, then count and compact.

- [ ] **Step 1: Refactor `eval_all()` to single-pass**

```mojo
def eval_all(self, points: List[Float64]) -> CSRMatrix:
    var n_pts = len(points)
    var n_basis = self.num_basis

    # First pass: evaluate and count nnz per row
    var row_data = List[List[Float64]]()  # dense rows
    var row_counts = List[Int]()
    var total_nnz = 0

    for row in range(n_pts):
        var x = points[row]
        var dense_row = List[Float64]()
        var cnt = 0
        for col in range(n_basis):
            var value = self.de_boor_cox(x, col)
            dense_row.append(value)
            if value > 1e-12 or value < -1e-12:
                cnt += 1
        row_data.append(dense_row^)
        row_counts.append(cnt)
        total_nnz += cnt

    var result = CSRMatrix(n_pts, n_basis, total_nnz)
    # ... fill from row_data (no second de_boor_cox call)
```

**Note:** This trades memory (dense row storage) for compute (eliminating duplicate de_boor_cox). For n_pts=200 and n_basis=45, this is ~18KB per row — acceptable. The dominant cost is de_boor_cox evaluation, so eliminating 50% of evaluations is the win.

- [ ] **Step 2: Apply same refactoring to `first_derivative_all()`**

- [ ] **Step 3: Verify compilation and functional correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 4: Commit**
```bash
git add src/numerics/bspline/basis.mojo
git commit -m "perf: single-pass B-spline eval eliminates duplicate de_boor_cox"
```

---

## Task 9: Widen RADAU5 LU Reuse Band

**Files:**
- Modify: `generate_radau.py` — update `quot1`/`quot2` constants
- Regenerate: `src/numerics/ode/radau.mojo`

Currently `quot1=0.8, quot2=1.5` — any step size change outside [0.8, 1.5]× the previous LU step size triggers full re-factorization. Widen to `quot1=0.5, quot2=2.5` with a theta decay check: if `theta < 0.01` (Jacobian still fresh), allow even wider reuse.

- [ ] **Step 1: Update `generate_radau.py` constants**

In the `solve()` method template:
```python
# Before:
quot1 = 0.8
quot2 = 1.5

# After:
quot1 = 0.5
quot2 = 2.5
```

And add theta-based widening:
```python
# In need_lu check:
if h_lu == 0.0:
    need_lu = True
else:
    h_ratio = abs(h / h_lu)
    if h_ratio < quot1 or h_ratio > quot2:
        need_lu = True
    elif theta < 0.01:
        # Jacobian is still fresh — allow wider band
        if h_ratio < 0.3 or h_ratio > 4.0:
            need_lu = True
        else:
            need_lu = False
```

- [ ] **Step 2: Regenerate `radau.mojo`**
Run: `python generate_radau.py`

- [ ] **Step 3: Verify compilation and correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same numerical output, fewer "LU refact" prints (if any), fewer total LU factorizations

- [ ] **Step 4: Commit**
```bash
git add generate_radau.py src/numerics/ode/radau.mojo
git commit -m "perf: widen RADAU5 LU reuse band with theta decay check"
```

---

## Task 10: Split Complex System into Two Real Systems

**Files:**
- Modify: `generate_radau.py` — add split complex solve logic
- Modify: `src/numerics/ode/radau.mojo` — remove `lu_complex`, add split solve

This is the largest single optimization. Instead of factorizing the 2n×2n complex system `[ALPH*M+h*K, +BETA*M; -BETA*M, ALPH*M+h*K]`, we solve `(A + iB)(x + iy) = f + ig` using the Schur complement approach:

1. The real part `A = ALPH*M + h*K` is already `E1/U1`-scaled — we have its LU factored (`lu_real`).
2. Solve `A*u = g` (real LU solve) → get `u = A^{-1}*g`
3. Solve `A*v = f - B*u` (real LU solve) → get `x = v`
4. `y = u - B*x` where we solve `A*w = B*x` → get `y = w`

Wait — this isn't quite right. Let me derive carefully.

The 2n×2n system is:
```
[A  B] [x]   [f]
[-B A] [y] = [g]
```

where `A = ALPH*M + h*K` and `B = BETA*M`.

From the second block: `-B*x + A*y = g` → `A*y = g + B*x`

Substitute into first block: `A*x + B*y = f` → `A*x + B*(A^{-1}*(g + B*x)) = f`
→ `A*x + B*A^{-1}*g + B*A^{-1}*B*x = f`
→ `(A + B*A^{-1}*B)*x = f - B*A^{-1}*g`

This is the Schur complement: `S = A + B*A^{-1}*B`. Since `A` is already LU-factored, `B*A^{-1}*B` requires:
1. Solve `A*U = B` column-by-column (n LU solves of size n) — too expensive

**Better approach:** Since `B = BETA*M` and `A = ALPH*M + h*K`, both are sparse and `B` has the same sparsity pattern as `M`. Use GMRES preconditioned by the real LU.

**Simplest effective approach:** Use the real LU as preconditioner for iterative solve of the complex system. The block structure means GMRES converges in 2-3 iterations.

- [ ] **Step 1: Add GMRES solver to `radau.mojo`**

Implement a simple restarted GMRES(30) that uses `lu_real.solve_inplace()` as preconditioner:

```mojo
def _solve_complex_preconditioned(
    self,
    rhs_complex: FixedSizeVector,  # 2n: [rhs_cx, rhs_cx2]
    mut dF2_dF3: FixedSizeVector,   # output: [dF2, dF3]
    lu_real: SparseLU,
    E1: CSRMatrix,                  # A = ALPH*M + h*K (CSR)
    BETA_M: CSRMatrix,              # B = BETA*M (CSR, precomputed once)
    work_n: FixedSizeVector,        # scratch size n
    work_2n: FixedSizeVector,       # scratch size 2n
    n: Int,
    max_iter: Int = 5,
):
    # GMRES with real-LU preconditioner
    # ...
```

The matrix-vector product for the 2n system:
```
[A  B] [x]   = [A*x + B*y]
[-B A] [y]     [-B*x + A*y]
```

Preconditioner: block-diagonal `diag(A^{-1}, A^{-1})` — apply `lu_real.solve_inplace()` on each n-block.

- [ ] **Step 2: Precompute `BETA_M = scale(BETA, M)` once at solver init**

Add to `solve()`:
```mojo
var BETA_M = scale(BETA, M)
```

- [ ] **Step 3: Replace `lu_complex.factorize(E2_csc_cached)` with `_solve_complex_preconditioned()`**

In the Newton iteration loop, replace:
```mojo
# OLD:
lu_complex.solve_inplace(dF2_dF3, work_2n)

# NEW:
self._solve_complex_preconditioned(
    rhs_complex, dF2_dF3, lu_real, E1_cached, BETA_M,
    work_n, work_2n, n
)
```

- [ ] **Step 4: Remove `lu_complex`, `E2_cached`, `E2_csc_cached`, `E2_csr_to_csc`**

These are no longer needed. Free the `E2_csr_to_csc` pointer earlier. Remove the `_build_complex_system()` and `_update_complex_data_fast()` methods.

- [ ] **Step 5: Update `generate_radau.py`** with the new GMRES logic and regenerate `radau.mojo`

- [ ] **Step 6: Verify compilation and numerical correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`
Expected: Same err_norm trajectory, fewer total factorization calls, faster ODE solve

- [ ] **Step 7: Commit**
```bash
git add generate_radau.py src/numerics/ode/radau.mojo
git commit -m "perf: replace 2n complex LU with GMRES preconditioned by real LU"
```

---

## Task 11: Eliminate Unnecessary Copies

**Files:**
- Modify: `src/engines/fpe/solver.mojo:60-64` — `get_M()`/`get_K()` return `.copy()`
- Modify: `src/engines/fpe/domain.mojo:226` — `knots.copy()` in `build_basis()`
- Modify: `src/engines/fpe/galerkin.mojo:56-57` — `s_points_phys.copy()`, `v_points_phys.copy()`
- Modify: `src/engines/fpe/initial_cond.mojo:106` — `Dinv_diag.copy()`

With the `FPECachedBasis` approach, `build_basis()` is called only once, so `knots.copy()` becomes less critical. But `get_M()`/`get_K()` still do `.copy()` which is wasteful for the `LinearODESystem` trait.

- [ ] **Step 1: After Task 3's `get_M_ref`/`get_K_ref` is in place, update all callers in `radau.mojo`**

- [ ] **Step 2: In `galerkin.mojo`, pass `domain.s_points_phys` by `read` reference instead of `.copy()`**

This requires changing the method signatures that consume these lists. Since `_stiffness_from_cached` builds `s_diag` etc. from `s_points_phys`, we can iterate over `domain.s_points_phys` directly without copying.

- [ ] **Step 3: Verify compilation and correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 4: Commit**
```bash
git add src/engines/fpe/solver.mojo src/engines/fpe/galerkin.mojo
git commit -m "perf: eliminate unnecessary .copy() in hot paths"
```

---

## Task 12: Deduplicate _build_*/_update_* Merge Logic

**Files:**
- Modify: `generate_radau.py` — extract shared merge logic
- Regenerate: `src/numerics/ode/radau.mojo`

The user noted that `_build_real_system`, `_build_complex_system`, `_update_real_data_fast`, `_update_complex_data_fast` all share duplicated sorted-merge logic for iterating over M and K entries. Extract this into a helper.

- [ ] **Step 1: Extract `_merge_mk_row()` helper in `generate_radau.py`**

A code generator function that emits the merge loop for a single row, parameterized by:
- `alpha_M`: coefficient for M entries (U1 for real, ALPH for complex top-left)
- `h_K`: coefficient for K entries (h for all)
- `write_B_block`: whether to also emit the +BETA*M block (complex only)

- [ ] **Step 2: Rewrite `_build_real_system` and `_build_complex_system` using the helper**

- [ ] **Step 3: Similarly extract `_merge_mk_update_row()` for the `_update_*_fast` methods**

These differ from `_build_*` in that they also write to CSC via the `E1_csr_to_csc` map.

- [ ] **Step 4: Regenerate `radau.mojo`**
Run: `python generate_radau.py`

- [ ] **Step 5: Verify compilation and correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 6: Commit**
```bash
git add generate_radau.py src/numerics/ode/radau.mojo
git commit -m "refactor: deduplicate merge logic in _build_*/_update_*"
```

---

## Task 13: Keep CSC Format Throughout (No CSR↔CSC Transform Bottleneck)

**Files:**
- Modify: `src/numerics/ode/radau.mojo` — ensure all internal matrices stay in CSC where possible

The user requested: "cancel CSR type keep CSC, prevent transform bottleneck". Currently the solver builds `E1_cached` and `E2_cached` as CSR, then converts to CSC via `to_csc()` for LU factorization. The `_update_*_fast` methods update both CSR and CSC data arrays in lockstep.

After Task 10 removes the complex system, `E2_cached`/`E2_csc_cached`/`E2_csr_to_csc` are eliminated entirely. For `E1`, we should consider building it in CSC directly, or at minimum, only maintaining the CSC copy (not the CSR one).

- [ ] **Step 1: Build E1 in CSC directly instead of CSR→CSC conversion**

Replace `_build_real_system()` (returns CSR) with `_build_real_system_csc()` (returns CSC). Use the same two-pass counting sort approach as `CSRMatrix.to_csc()`, but produce CSC directly.

- [ ] **Step 2: Update `_update_real_data_fast()` to update CSC directly**

Since we no longer maintain `E1_cached` (CSR), the update writes directly to `E1_csc`:
```mojo
def _update_real_data_fast_csc(
    self, M, K, h, n, mut E1_csc, M_csc_to_e1, K_csc_to_e1
):
    # Iterate CSC columns, merge M and K contributions
```

- [ ] **Step 3: Remove `E1_cached` (CSR) and `E1_csr_to_csc` pointer**

Only `E1_csc` is needed for `lu_real.factorize()`.

- [ ] **Step 4: Verify compilation and correctness**
Run: `pixi run mojo run -I src examples/single_price.mojo`

- [ ] **Step 5: Commit**
```bash
git add src/numerics/ode/radau.mojo generate_radau.py
git commit -m "perf: build/update E1 in CSC directly, eliminate CSR↔CSC conversion"
```

---

## Task 14: Performance Benchmark — Before/After

**Files:**
- Create: `examples/test_perf.mojo`

Create a dedicated benchmark that runs the pipeline with timing for each step, comparing old vs new code paths.

- [ ] **Step 1: Create benchmark file**

```mojo
def main() raises:
    # Run with old API (build_basis per call)
    # Run with new API (cached basis)
    # Print timing comparison per step
```

- [ ] **Step 2: Run benchmark**
Run: `pixi run mojo run -I src examples/test_perf.mojo`

- [ ] **Step 3: Record results** in the commit message

Expected improvements:
- Step 3 (assembly): 12.9s → ~5s (single basis build + single eval_tensor)
- Step 4 (initial condition): 3s → ~0.5s (reuse M + cached basis)
- Step 5 (ODE solve): 165s → ~80s (fewer LU refactorizations + no 2n complex LU)
- Step 6 (PDF): 1s → ~0.01s (cached two_basis)
- Total: 184s → ~90s

- [ ] **Step 4: Commit**
```bash
git add examples/test_perf.mojo
git commit -m "bench: add pipeline performance benchmark"
```

---

## Expected Outcome Summary

| Metric | Before | After | Improvement |
|---|---|---|---|
| `build_basis()` calls | 5+ | 1 | 5× fewer |
| `eval_tensor()` calls | 5+ | 1 | 5× fewer |
| `integ_weights()` calls | 3+ | 1 | 3× fewer |
| M assembly | 2× | 1× | 2× fewer |
| K assembly | 2× | 1× | 2× fewer |
| Complex LU (2n×2n) | ~25 factorizations | 0 | Eliminated |
| Real LU (n×n) | ~25 factorizations | ~15 (wider band) | 40% fewer |
| CSR→CSC conversion | ~50 (per LU refactor) | 0 (CSC-native) | Eliminated |
| `de_boor_cox` evals | 2× per (point, basis) | 1× | 2× fewer |
| `knots.copy()` | 10+ | 2 | 5× fewer |
| **Total wall time** | **~184s** | **~90s** | **~2× faster** |
