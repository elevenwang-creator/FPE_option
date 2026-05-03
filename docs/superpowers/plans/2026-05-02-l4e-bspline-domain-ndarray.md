# L4e: B-spline + Domain Pure NDArray Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all `List[Float64]` from the B-spline module and FPE domain/consumers chain, replacing with `NDArray[Float64]`.

**Architecture:** Bottom-up migration through 10 layers: NDArray helper → GaussLegendre → GenerateKnots → BSplineBasis → RecombinationBasis → TensorProductBasis → FPEDomain → consumers → delete _list helpers → update tests. Each layer compiles and passes tests before proceeding.

**Tech Stack:** Mojo v0.26.3 (Nightly), pixi for environment, NDArray[Float64] as sole container type.

**Spec:** `docs/superpowers/specs/2026-05-02-l4e-bspline-domain-ndarray-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/numerics/utils/constructors.mojo` | Modify | Add `from_list()`, delete `_list` helpers |
| `src/numerics/utils/copy.mojo` | Modify | Delete `_list` helpers |
| `src/numerics/utils/__init__.mojo` | Modify | Delete `_list` exports |
| `src/numerics/bspline/knots.mojo` | Rewrite | GaussLegendre + GenerateKnots → NDArray |
| `src/numerics/bspline/basis.mojo` | Rewrite | BSplineBasis → NDArray |
| `src/numerics/bspline/recombination.mojo` | Modify | eval_all/first_derivative_all → NDArray params |
| `src/numerics/bspline/tensor_product.mojo` | Modify | eval_tensor/partial_s/partial_v → NDArray params |
| `src/engines/fpe/domain.mojo` | Rewrite | FPEDomain fields + helpers → NDArray |
| `src/engines/fpe/galerkin.mojo` | Modify | Remove `_list_to_ndarr`, update NDArray access |
| `src/engines/fpe/pdf.mojo` | Modify | `len()` → `.len()` |
| `src/engines/fpe/initial_cond.mojo` | Modify | `len()` → `.len()` |
| `src/engines/calibrator/objective.mojo` | Modify | `len()` → `.len()` |
| `src/bindings/python_module.mojo` | Modify | NDArray access + `.to_list()` bridge |
| `src/server/pdf_cache.mojo` | Rewrite | PDFGrid → NDArray fields |
| `tests/test_bspline.mojo` | Modify | NDArray instead of List |
| `tests/test_zeros_3d.mojo` | Modify | Remove `_list` test cases |

---

### Task 1: Add `from_list()` to constructors.mojo

**Files:**
- Modify: `src/numerics/utils/constructors.mojo`
- Modify: `src/numerics/utils/__init__.mojo`

- [ ] **Step 1: Add `from_list` function to constructors.mojo**

Add at end of file (before the `_list` helpers section):

```mojo
def from_list(xs: List[Float64]) -> NDArray[Float64]:
    var n = len(xs)
    var arr = NDArray[Float64](n)
    for i in range(n):
        arr[i] = xs[i]
    return arr^
```

- [ ] **Step 2: Export `from_list` from `__init__.mojo`**

Add to the import line from `.constructors`:
```mojo
from .constructors import zeros, zeros_mat, zeros_3d, linspace, from_list
```

- [ ] **Step 3: Verify compilation**

Run: `pixi run mojo run -I src -e 'from numerics.utils import from_list; from numerics.utils.ndarray import NDArray; def main() raises: var x: List[Float64] = [1.0, 2.0, 3.0]; var a = from_list(x); print(a.len())'`
Expected: `3`

---

### Task 2: Migrate GaussLegendre to NDArray

**Files:**
- Rewrite: `src/numerics/bspline/knots.mojo` (GaussLegendre struct only)

- [ ] **Step 1: Rewrite GaussLegendre struct**

Replace `var nodes: List[Float64]` and `var weights: List[Float64]` with `NDArray[Float64]`.

For each `order` case, allocate NDArray and write by index. Example for order=2:

```mojo
if order == 2:
    var s = sqrt(1.0 / 3.0)
    self.nodes = NDArray[Float64](2)
    self.nodes[0] = -s
    self.nodes[1] = s
    self.weights = NDArray[Float64](2)
    self.weights[0] = 0.5
    self.weights[1] = 0.5
```

Same pattern for all 7 cases (order 2,3,4,5,6, else).

- [ ] **Step 2: Verify compilation**

Run: `pixi run mojo run -I src -e 'from numerics.bspline.knots import GaussLegendre; def main() raises: var gl = GaussLegendre(3); print(gl.nodes[0], gl.weights[0])'`
Expected: values printed (no crash)

---

### Task 3: Migrate GenerateKnots to NDArray

**Files:**
- Rewrite: `src/numerics/bspline/knots.mojo` (GenerateKnots struct)

This is the largest task. Each method changes from `List[Float64]` return to `NDArray[Float64]` return.

- [ ] **Step 1: Rewrite `normalize`**

```mojo
def normalize(self, x: NDArray[Float64]) -> NDArray[Float64]:
    var n = x.len()
    var min_val = x[0]
    var max_val = x[0]
    for i in range(n):
        if x[i] < min_val: min_val = x[i]
        if x[i] > max_val: max_val = x[i]
    var result = NDArray[Float64](n)
    for i in range(n):
        result[i] = (x[i] - min_val) / (max_val - min_val)
    return result^
```

- [ ] **Step 2: Delete local `linspace` method**

Remove the `linspace` method from GenerateKnots. Import the existing `linspace` from `numerics.utils` instead. Update all calls from `self.linspace(...)` to `linspace(...)`.

- [ ] **Step 3: Rewrite `chebyshev_knots`**

```mojo
def chebyshev_knots(self, n: Int, a: Float64, b: Float64) -> NDArray[Float64]:
    var total = n + 2
    var result = NDArray[Float64](total)
    result[0] = -1.0
    for i in range(n):
        var angle = (2.0 * Float64(i) + 1.0) * pi / (2.0 * Float64(n))
        result[i + 1] = cos(angle)
    result[total - 1] = 1.0
    for i in range(total):
        result[i] = (b - a) / 2.0 * result[i] + (a + b) / 2.0
    return result^
```

- [ ] **Step 4: Rewrite `func_parabolic`**

Returns `(Int, NDArray[Float64])`. Size of NDArray = `n_adj`.

**CRITICAL: Do NOT dedup x after sorting — duplicates are expected and must be preserved to match original behavior.** The original code uses `.append(factor)` which may create a duplicate with the last linspace value. The NDArray version must preserve this.

```mojo
def func_parabolic(self, n: Int, boundary: Tuple[Float64, Float64]) -> Tuple[Int, NDArray[Float64]]:
    # ... same algorithm as before ...
    # x: allocate NDArray[Float64](n_adj)
    # Copy linspace(0.0, upward, n_adj - 1) into x[0..n_adj-2]
    # x[n_adj-1] = factor
    # Bubble sort x in-place (NO dedup!)
    # y: allocate NDArray[Float64](n_adj), compute parabolic formula
    # Return (n_adj, y^)
```

Key: `lin_part = linspace(0.0, upward, n_adj - 1)` returns NDArray. Copy into x[0..n_adj-2], then x[n_adj-1] = factor. Sort x in-place with bubble sort. Compute y. Return.

- [ ] **Step 5: Rewrite `knots_concat`**

Two-pass approach:
1. First pass: count left_knots, median_knots, right_knots, build combined+sorted+rounded array, count unique
2. Second pass: fill NDArray with unique values

**CRITICAL: The original code has a rounding step before dedup** (knots.mojo:220-222):
```mojo
var rounded: List[Float64] = []
for i in range(len(points)):
    rounded.append(Float64(Int(points[i] * 1e8)) / 1e8)
```
This rounds values to 1e-8 precision, which affects dedup count. Must be preserved in NDArray version.

```mojo
def knots_concat(self, knots: NDArray[Float64], medim_knot: NDArray[Float64], left: Float64, right: Float64) -> NDArray[Float64]:
    # Count left (<= left), right (>= right)
    var n_left = 0
    for i in range(knots.len()):
        if abs(knots[i] - left) < 1e-10 or knots[i] < left: n_left += 1
    var n_right = 0
    for i in range(knots.len()):
        if abs(knots[i] - right) < 1e-10 or knots[i] > right: n_right += 1
    var n_medim = medim_knot.len()
    var total = n_left + n_medim + n_right
    # Build combined array
    var combined = NDArray[Float64](total)
    # ... fill left, medim, right ...
    # Bubble sort combined
    # Round: combined[i] = Float64(Int(combined[i] * 1e8)) / 1e8
    # Count unique (after rounding)
    # Fill result NDArray with unique values
    return result^
```

- [ ] **Step 6: Rewrite `generate_knots`**

```mojo
def generate_knots(self) -> NDArray[Float64]:
    # ... same logic but using NDArray throughout ...
    # internal_knots is NDArray from either linspace or knots_concat
    # IMPORTANT: self.normalize([boundary[0], boundary[1]]) must become:
    #   var boundary_arr = NDArray[Float64](2)
    #   boundary_arr[0] = boundary[0]; boundary_arr[1] = boundary[1]
    #   var boundary_normal = self.normalize(boundary_arr^)
    # IMPORTANT: sort(parabolic_knots) must become in-place bubble sort on NDArray
    #   (stdlib sort() only works on List, not NDArray)
    # Final: allocate NDArray of size 2*p + internal_knots.len()
    # Fill first p with boundary_normal[0], then internal_knots, then p with boundary_normal[1]
    return final_knots^
```

- [ ] **Step 7: Verify compilation**

Run: `pixi run mojo run -I src tests/test_bspline.mojo`
Note: test_bspline still uses List — this will fail. That's OK; verify the bspline/knots.mojo file compiles in isolation:

```bash
pixi run mojo run -I src -e 'from numerics.bspline.knots import GenerateKnots; def main() raises: var gen = GenerateKnots(n=8, degree=2, method="uniform"); var k = gen.generate_knots(); print(k.len())'
```
Expected: `8`

---

### Task 4: Migrate BSplineBasis to NDArray

**Files:**
- Rewrite: `src/numerics/bspline/basis.mojo`

- [ ] **Step 1: Change `knots` field and `__init__`**

```mojo
struct BSplineBasis[degree: Int](Copyable, Movable):
    var knots: NDArray[Float64]
    var num_basis: Int

    def __init__(out self, var knots: NDArray[Float64]):
        self.knots = knots^
        self.num_basis = self.knots.len() - Self.degree - 1
```

- [ ] **Step 2: Update `base_basis` and `de_boor_cox`**

No changes needed — they access `self.knots[i]` which works identically for NDArray.

- [ ] **Step 3: Update `eval_all` and `first_derivative_all` signatures**

```mojo
def eval_all(self, points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var n_pts = points.len()
    # ... same logic, access points[row] ...

def first_derivative_all(self, points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    # ... same logic ...
```

- [ ] **Step 4: Update `first_derivative_all` internal `BSplineBasis` construction**

`self.knots.copy()` returns `NDArray[Float64]` — compatible with new `__init__`.

- [ ] **Step 5: Verify compilation**

```bash
pixi run mojo run -I src -e 'from numerics.bspline.basis import BSplineBasis; from numerics.utils import linspace; def main() raises: var k = linspace(0.0, 1.0, 5); var b = BSplineBasis[1](k); print(b.num_basis)'
```
Expected: `3`

---

### Task 5: Migrate RecombinationBasis to NDArray

**Files:**
- Modify: `src/numerics/bspline/recombination.mojo`

- [ ] **Step 1: Change method signatures**

```mojo
def eval_all(self, points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var B = self.basis.eval_all(points)
    var R = self.recombination_matrix()
    return spgemm(B, R)

def first_derivative_all(self, points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var dB = self.basis.first_derivative_all(points)
    var R = self.recombination_matrix()
    return spgemm(dB, R)
```

- [ ] **Step 2: Verify compilation**

```bash
pixi run mojo run -I src -e 'from numerics.bspline.recombination import RecombinationBasis; from numerics.bspline.basis import BSplineBasis; from numerics.utils import linspace; def main() raises: var k = linspace(0.0, 1.0, 5); var b = BSplineBasis[1](k); var r = RecombinationBasis[1](b, left_cond="dirichlet", right_cond="dirichlet"); var pts = linspace(0.1, 0.9, 5); var mat = r.eval_all(pts); print(mat.nrows, mat.ncols)'
```

---

### Task 6: Migrate TensorProductBasis to NDArray

**Files:**
- Modify: `src/numerics/bspline/tensor_product.mojo`

- [ ] **Step 1: Change method signatures**

```mojo
def eval_tensor(self, s_points: NDArray[Float64], v_points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var Bs = self.basis_s.eval_all(s_points)
    var Bv = self.basis_v.eval_all(v_points)
    return kron(Bs, Bv)

def partial_s(self, s_points: NDArray[Float64], v_points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var dBs = self.basis_s.first_derivative_all(s_points)
    var Bv = self.basis_v.eval_all(v_points)
    return kron(dBs, Bv)

def partial_v(self, s_points: NDArray[Float64], v_points: NDArray[Float64]) -> CSRMatrix[DType.float64]:
    var Bs = self.basis_s.eval_all(s_points)
    var dBv = self.basis_v.first_derivative_all(v_points)
    return kron(Bs, dBv)
```

- [ ] **Step 2: Verify compilation**

```bash
pixi run mojo run -I src -e 'from numerics.bspline.tensor_product import TensorProductBasis; from numerics.bspline.recombination import RecombinationBasis; from numerics.bspline.basis import BSplineBasis; from numerics.utils import linspace; def main() raises: var k = linspace(0.0, 1.0, 5); var bs = RecombinationBasis[1](BSplineBasis[1](k.copy()), left_cond="neumann", right_cond="neumann"); var bv = RecombinationBasis[1](BSplineBasis[1](k.copy()), left_cond="neumann", right_cond="neumann"); var tp = TensorProductBasis[1,1](bs, bv); var pts = linspace(0.25, 0.75, 2); var B = tp.eval_tensor(pts, pts); print(B.nrows, B.ncols)'
```

---

### Task 7: Migrate FPEDomain to NDArray

**Files:**
- Rewrite: `src/engines/fpe/domain.mojo`

This is the second-largest task. All 6 fields change type, all helpers change return types.

- [ ] **Step 1: Rewrite `_sort_unique` for NDArray**

```mojo
def _sort_unique(x: NDArray[Float64]) -> NDArray[Float64]:
    var n = x.len()
    if n == 0:
        return NDArray[Float64](1)  # NDArray requires n > 0; shouldn't happen in practice
    var arr = x.copy()
    # Bubble sort
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                var tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
    # Count unique
    var unique_count = 1
    for i in range(1, n):
        if arr[i] > arr[i - 1] + 1e-9:
            unique_count += 1
    # Fill unique
    var out = NDArray[Float64](unique_count)
    out[0] = arr[0]
    var pos = 1
    for i in range(1, n):
        if arr[i] > out[pos - 1] + 1e-9:
            out[pos] = arr[i]
            pos += 1
    return out^
```

- [ ] **Step 2: Rewrite `_normalize` for NDArray**

```mojo
def _normalize(x: NDArray[Float64]) -> NDArray[Float64]:
    var gen = GenerateKnots(1, 1)
    return gen.normalize(x)
```

- [ ] **Step 3: Rewrite `_grid_create` for NDArray**

Uses `linspace` from `numerics.utils` instead of `linspace_list`. Strategy: build segments as NDArrays, compute total size, fill combined buffer, then `_sort_unique` to dedup.

```mojo
def _grid_create(
    mean: Float64, std_dev: Float64,
    bound: Tuple[Float64, Float64],
    num_insert: Int = 251,
    is_v: Bool = False,
) -> NDArray[Float64]:
    var lb = bound[0]; var ub = bound[1]
    var left_trail = num_insert * 20 // 100
    var num_interm = num_insert * 30 // 100
    var right_trail = num_insert * 50 // 100

    # Build each segment with linspace
    var seg1 = linspace(lb, mean - 5.0 * std_dev, left_trail)
    var seg2 = linspace(mean - 5.0 * std_dev, mean - std_dev / 5.0, num_interm // 3)
    var seg3 = linspace(mean - std_dev / 5.0, mean + std_dev / 5.0, num_interm // 3)
    var seg4 = linspace(mean + std_dev / 5.0, mean + 5.0 * std_dev, num_interm // 3)
    var seg5 = linspace(mean + 5.0 * std_dev, ub, right_trail)
    var seg6 = linspace(ub - 0.1, ub, num_insert * 10 // 100)

    # Compute total size conditionally (no placeholder NDArray needed)
    var total = seg1.len() + seg2.len() + seg3.len() + 1 + seg4.len() + seg5.len() + seg6.len()
    if is_v:
        total += num_insert * 20 // 100

    # Build combined array
    var combined = NDArray[Float64](total)
    var pos = 0
    if is_v:
        var zero_seg = linspace(0.0, 0.01, num_insert * 20 // 100)
        for i in range(zero_seg.len()): combined[pos] = zero_seg[i]; pos += 1
    for i in range(seg1.len()): combined[pos] = seg1[i]; pos += 1
    for i in range(seg2.len()): combined[pos] = seg2[i]; pos += 1
    for i in range(seg3.len()): combined[pos] = seg3[i]; pos += 1
    combined[pos] = mean; pos += 1
    for i in range(seg4.len()): combined[pos] = seg4[i]; pos += 1
    for i in range(seg5.len()): combined[pos] = seg5[i]; pos += 1
    for i in range(seg6.len()): combined[pos] = seg6[i]; pos += 1

    return _sort_unique(combined)^
```

- [ ] **Step 4: Rewrite `_compute_quad_points` for NDArray**

Two-pass: count valid intervals, then fill.

```mojo
def _compute_quad_points(grid: NDArray[Float64], num_gauss: Int) -> NDArray[Float64]:
    # Deduplicate grid first
    var unique_count = 1
    for i in range(1, grid.len()):
        if grid[i] > grid[i - 1] + 1e-9:
            unique_count += 1
    
    # Build unique array
    var unique = NDArray[Float64](unique_count)
    unique[0] = grid[0]
    var upos = 1
    for i in range(1, grid.len()):
        if grid[i] > unique[upos - 1] + 1e-9:
            unique[upos] = grid[i]
            upos += 1
    
    # Count valid intervals (where b > a)
    var n_intervals = unique_count - 1
    var valid = 0
    for i in range(n_intervals):
        if unique[i + 1] > unique[i]:
            valid += 1
    
    var gl = GaussLegendre(num_gauss)
    var total = valid * (1 + num_gauss)
    var points = NDArray[Float64](total)
    var pos = 0
    for i in range(n_intervals):
        var a = unique[i]
        var b = unique[i + 1]
        if b <= a: continue
        var half_span = 0.5 * (b - a)
        var mid = 0.5 * (a + b)
        points[pos] = a; pos += 1
        for j in range(gl.nodes.len()):
            points[pos] = half_span * gl.nodes[j] + mid; pos += 1
    
    # Trim if needed (shouldn't be, but safety)
    return points^
```

- [ ] **Step 5: Rewrite `_compute_quad_weights` for NDArray**

Same two-pass structure as quad_points.

- [ ] **Step 6: Update `FPEDomain` struct fields**

```mojo
struct FPEDomain[degree_s: Int = 3, degree_v: Int = 3](Copyable, Movable):
    var s_knots: NDArray[Float64]
    var v_knots: NDArray[Float64]
    var s_points: NDArray[Float64]
    var v_points: NDArray[Float64]
    var s_weights: NDArray[Float64]
    var v_weights: NDArray[Float64]
    var s_points_phys: NDArray[Float64]
    var v_points_phys: NDArray[Float64]
    var s_min: Float64
    var s_max: Float64
    var v_min: Float64
    var v_max: Float64
```

- [ ] **Step 7: Update `__init__`**

- `s_gen.generate_knots()` and `v_gen.generate_knots()` now return `NDArray[Float64]`
- `_grid_create` now returns `NDArray[Float64]`
- `_normalize` now takes/returns `NDArray[Float64]`
- `_compute_quad_points`/`_compute_quad_weights` now take/return `NDArray[Float64]`
- Jacobian weighting: `for i in range(self.s_weights.len()): self.s_weights[i] *= jacobian_s`
- `s_points_phys`/`v_points_phys`: allocate `NDArray[Float64](n)`, fill by index

- [ ] **Step 8: Update `integ_weights()` and `build_basis()`**

- `integ_weights()`: pass `self.s_weights`/`self.v_weights` directly to `DiagMatrix` (already NDArray)
- `build_basis()`: `self.s_knots.copy()`/`self.v_knots.copy()` now returns NDArray — compatible with new `BSplineBasis.__init__`

- [ ] **Step 9: Remove `_list_to_ndarr` bridge**

Delete the `_list_to_ndarr` function entirely.

- [ ] **Step 10: Update imports**

Replace `from numerics.utils import linspace_list` with `from numerics.utils import linspace, from_list`.

- [ ] **Step 11: Verify compilation**

```bash
pixi run mojo run -I src -e 'from engines.fpe.domain import FPEDomain; from engines.fpe.heston_params import HestonParams; def main() raises: var p = HestonParams(kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=1.0, S0=100.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0); var d = FPEDomain[3,3](p, n_s=8, n_v=8); print(d.s_points.len())'
```

---

### Task 8: Update consumers

**Files:**
- Modify: `src/engines/fpe/galerkin.mojo`
- Modify: `src/engines/fpe/pdf.mojo`
- Modify: `src/engines/fpe/initial_cond.mojo`
- Modify: `src/engines/calibrator/objective.mojo`
- Modify: `src/bindings/python_module.mojo`
- Modify: `src/server/pdf_cache.mojo`

- [ ] **Step 1: Update galerkin.mojo**

- Remove `_list_to_ndarr` function entirely
- Add `from numerics.utils.ndarray import NDArray` (needed for `s_sq` construction)
- `len(domain.s_points)` → `domain.s_points.len()`
- `len(domain.v_points)` → `domain.v_points.len()`
- `var s = domain.s_points_phys.copy()` — now returns NDArray (no change needed)
- `var v = domain.v_points_phys.copy()` — now returns NDArray (no change needed)
- `DiagMatrix(_list_to_ndarr(s.copy()))` → `DiagMatrix(s.copy())` (s already NDArray)
- `DiagMatrix(_list_to_ndarr(v.copy()))` → `DiagMatrix(v.copy())`
- `var s_sq: List[Float64] = []` → `var s_sq = NDArray[Float64](s.len()); for i in range(s.len()): s_sq[i] = s[i] * s[i]`
- `DiagMatrix(_list_to_ndarr(s_sq.copy()))` → `DiagMatrix(s_sq.copy())`

- [ ] **Step 2: Update pdf.mojo**

- `len(domain.s_points)` → `domain.s_points.len()`
- `len(domain.v_points)` → `domain.v_points.len()`
- `basis.eval_tensor(domain.s_points, domain.v_points)` — types now match (both NDArray), no change needed

- [ ] **Step 3: Update initial_cond.mojo**

- `len(domain.s_points_phys)` → `domain.s_points_phys.len()`
- `len(domain.v_points_phys)` → `domain.v_points_phys.len()`

- [ ] **Step 4: Update objective.mojo**

- `len(domain.s_points)` → `domain.s_points.len()`
- `len(domain.v_points)` → `domain.v_points.len()`

- [ ] **Step 5: Update python_module.mojo**

- `len(domain.s_points)` → `domain.s_points.len()`
- `len(domain.v_points)` → `domain.v_points.len()`
- `domain.s_points.copy()` returns NDArray — pass directly to PDFGrid (now NDArray fields, no `.to_list()` needed)
- `domain.v_points.copy()` returns NDArray — same, pass directly
- For `ds_weights`/`dv_weights` placeholder: allocate `NDArray[Float64](1)` as dummy (precompute_weights will overwrite)

- [ ] **Step 6: Update pdf_cache.mojo — PDFGrid fields**

Change:
```mojo
@fieldwise_init
struct PDFGrid(Copyable, Movable):
    var pdf: List[List[Float64]]
    var s_points: NDArray[Float64]
    var v_points: NDArray[Float64]
    var T: Float64
    var ds_weights: NDArray[Float64]
    var dv_weights: NDArray[Float64]
```

**Note:** `@fieldwise_init` requires all fields at construction. For `ds_weights`/`dv_weights`, callers must pass a placeholder `NDArray[Float64](1)` — `precompute_weights()` will overwrite with correctly-sized arrays.

Add import: `from numerics.utils import from_list`

- [ ] **Step 7: Update pdf_cache.mojo — `precompute_weights`**

```mojo
def precompute_weights(mut self):
    var n_s = self.s_points.len()
    self.ds_weights = NDArray[Float64](n_s)
    for i in range(n_s):
        if i == 0 or i == n_s - 1:
            self.ds_weights[i] = 1.0
        else:
            self.ds_weights[i] = (self.s_points[i + 1] - self.s_points[i - 1]) * 0.5
    var n_v = self.v_points.len()
    self.dv_weights = NDArray[Float64](n_v)
    for i in range(n_v):
        if i == 0 or i == n_v - 1:
            self.dv_weights[i] = 1.0
        else:
            self.dv_weights[i] = (self.v_points[i + 1] - self.v_points[i - 1]) * 0.5
```

- [ ] **Step 8: Update pdf_cache.mojo — `to_python_object`**

Use `.to_list()` for NDArray→Python conversion:
```mojo
var s_list = self.s_points.to_list()
for i in range(len(s_list)):
    _ = py_s.append(PythonObject(s_list[i]))
```

- [ ] **Step 9: Update pdf_cache.mojo — `from_python_object`**

Use `from_list()` for Python→NDArray conversion. Requires `from numerics.utils import from_list` (added in Step 6):
```mojo
var s_points: List[Float64] = []
# ... read from Python ...
var grid = PDFGrid(
    pdf=pdf^,
    s_points=from_list(s_points^),
    v_points=from_list(v_points^),
    T=Float64(py=py_obj["T"]),
    ds_weights=from_list(ds_weights^),
    dv_weights=from_list(dv_weights^),
)
```

- [ ] **Step 10: Verify compilation**

```bash
pixi run mojo run -I src -e 'from engines.fpe.galerkin import GalerkinAssembler; from engines.fpe.pdf import PDFComputer; from engines.fpe.initial_cond import InitialCondition; print("OK")'
```

---

### Task 9: Delete _list helpers

**Files:**
- Modify: `src/numerics/utils/constructors.mojo`
- Modify: `src/numerics/utils/copy.mojo`
- Modify: `src/numerics/utils/__init__.mojo`
- Modify: `src/numerics/__init__.mojo` (verify)
- Verify: `src/engines/nais/gpu_trainer.mojo` (already cleaned)

- [ ] **Step 1: Delete _list functions from constructors.mojo**

Remove: `zeros_list`, `zeros_mat_list`, `zeros_3d_list`, `linspace_list` (lines 35-74).

- [ ] **Step 2: Delete _list functions from copy.mojo**

Remove: `copy_vec_list`, `copy_mat_list`, `swap_rows_list` (lines 22-44).

Update docstring to remove "List-based versions available with _list suffix for legacy consumers."

- [ ] **Step 3: Clean `__init__.mojo` exports**

Remove all `_list` imports:
```mojo
from .constructors import zeros, zeros_mat, zeros_3d, linspace, from_list
from .copy import copy_vec, copy_mat, swap_rows
```

- [ ] **Step 4: Verify `numerics/__init__.mojo` is clean**

Already cleaned in prior work. Verify no `_list` exports remain.

- [ ] **Step 5: Verify no remaining `_list` imports across codebase**

```bash
cd /Users/knight/Agent/FPE_option && grep -r "linspace_list\|zeros_list\|zeros_mat_list\|copy_vec_list\|copy_mat_list" src/ --include="*.mojo"
```
Expected: no results

- [ ] **Step 6: Verify compilation**

```bash
pixi run mojo run -I src tests/test_ndarray.mojo
```

---

### Task 10: Update test files

**Files:**
- Modify: `tests/test_bspline.mojo`
- Modify: `tests/test_zeros_3d.mojo`

- [ ] **Step 1: Update test_bspline.mojo**

Replace all `List[Float64]` knot arrays with NDArray:
```mojo
from numerics.utils import linspace, from_list

# Before: var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
# After:
var knots = from_list(List[Float64]([0.0, 0.0, 0.5, 1.0, 1.0]))
```

Or use NDArray constructor directly:
```mojo
var knots = NDArray[Float64](5)
knots[0] = 0.0; knots[1] = 0.0; knots[2] = 0.5; knots[3] = 1.0; knots[4] = 1.0
```

Replace `sample_points: List[Float64] = [...]` with NDArray equivalents.

Replace `s_points: List[Float64] = [0.25, 0.75]` with `from_list(List[Float64]([0.25, 0.75]))` or NDArray constructor.

Note: `test_tensor_product_shapes` has pre-existing bug — leave it unchanged (still fails for the same reason).

- [ ] **Step 2: Update test_zeros_3d.mojo**

Remove the 3 `zeros_3d_list` test functions and the `zeros_3d_list` import. Keep `test_zeros_3d_ndarray_shape`.

```mojo
from numerics.utils import zeros_3d
from numerics.utils.ndarray import NDArray
from std.testing import assert_true, TestSuite


def test_zeros_3d_ndarray_shape() raises:
    var result = zeros_3d(2, 3, 4)
    var s = result.shape()
    assert_true(Int(s[0]) == 2, "dim0 should be 2")
    assert_true(Int(s[1]) == 3, "dim1 should be 3")
    assert_true(Int(s[2]) == 4, "dim2 should be 4")
    assert_true(result.len() == 24, "total elements should be 24")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 3: Run test_bspline.mojo**

```bash
pixi run mojo run -I src tests/test_bspline.mojo
```
Expected: 4/5 tests pass (tensor_product_shapes still fails — pre-existing bug)

- [ ] **Step 4: Run test_zeros_3d.mojo**

```bash
pixi run mojo run -I src tests/test_zeros_3d.mojo
```
Expected: 1 test pass

- [ ] **Step 5: Run full test suite**

```bash
for t in test_sparse test_sparse_coo_diag test_linalg test_ndarray test_zeros_3d test_ode test_nais_tracked_forward test_autograd_tape test_autograd_linear test_adam test_optim test_brownian_paths test_bspline; do
    result=$(pixi run mojo run -I src tests/${t}.mojo 2>&1 | grep -E "Summary" | head -1)
    echo "$t: $result"
done
```

Expected: all pass except test_bspline (1 pre-existing failure)
