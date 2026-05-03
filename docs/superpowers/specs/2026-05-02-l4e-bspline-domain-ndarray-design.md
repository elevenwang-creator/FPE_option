# L4e: B-spline + Domain Pure NDArray Migration

## Goal

Eliminate all `List[Float64]` from the B-spline module and FPE domain/consumers chain. Every field, parameter, and return type becomes `NDArray[Float64]`. No `List` anywhere in this dependency chain.

## Scope

Files to modify (13 source + 2 test):
- `src/numerics/bspline/knots.mojo` — GenerateKnots + GaussLegendre
- `src/numerics/bspline/basis.mojo` — BSplineBasis
- `src/numerics/bspline/recombination.mojo` — RecombinationBasis
- `src/numerics/bspline/tensor_product.mojo` — TensorProductBasis
- `src/numerics/bspline/__init__.mojo` — verify re-exports (no code change expected)
- `src/engines/fpe/domain.mojo` — FPEDomain + helpers
- `src/engines/fpe/galerkin.mojo` — GalerkinAssembler
- `src/engines/fpe/pdf.mojo` — PDFComputer
- `src/engines/fpe/initial_cond.mojo` — InitialCondition
- `src/engines/calibrator/objective.mojo` — uses `len(domain.s_points)`
- `src/bindings/python_module.mojo` — uses `domain.s_points.copy()` as List
- `src/server/pdf_cache.mojo` — PDFGrid uses `List[Float64]` for s_points, v_points
- `src/numerics/utils/constructors.mojo` — delete _list helpers
- `src/numerics/utils/copy.mojo` — delete _list helpers
- `src/numerics/utils/__init__.mojo` — delete _list exports
- `tests/test_bspline.mojo` — update to NDArray
- `tests/test_zeros_3d.mojo` — remove _list test cases

## Design

### Layer 1: NDArray helper

Add `from_list(xs: List[Float64]) -> NDArray[Float64]` as a free function in `numerics.utils.constructors`. Replaces the duplicate `_list_to_ndarr` in domain.mojo and galerkin.mojo. Used during migration for Python interop bridging.

### Layer 2: GaussLegendre

- `nodes: List[Float64]` → `nodes: NDArray[Float64]`
- `weights: List[Float64]` → `weights: NDArray[Float64]`
- Constructor: for each `order` case, pre-allocate NDArray of known size and write values by index
- Note: NDArray `[]` operator returns `Float64` directly — verified compatible with `gl.nodes[j]` access in domain.mojo

### Layer 3: GenerateKnots

All methods return `NDArray[Float64]`:

| Method | Size strategy |
|--------|--------------|
| `normalize(x: NDArray) -> NDArray` | `x.len()` |
| `linspace(start, stop, count) -> NDArray` | Use existing `linspace()` from constructors.mojo (delete local `linspace` method) |
| `func_parabolic(n, boundary) -> (Int, NDArray)` | Size = `n_adj` (n_adj-1 from linspace + 1 for factor append = n_adj) |
| `chebyshev_knots(n, a, b) -> NDArray` | `n + 2`. Collapse intermediates: allocate output, write [-1, ...nodes..., 1], transform in-place |
| `knots_concat(knots, median, left, right) -> NDArray` | Two-pass: count unique after sort+dedup, then fill NDArray |
| `generate_knots() -> NDArray` | Two-phase: (1) compute `internal_knots` NDArray, (2) allocate final of size `2*degree + internal_knots.len()`, fill |

### Layer 4: BSplineBasis

- `knots: List[Float64]` → `knots: NDArray[Float64]`
- `num_basis` computed from `knots.len() - degree - 1`
- `__init__(knots: NDArray[Float64])` replaces `__init__(knots: List[Float64])`
- `de_boor_cox(x, i)` — accesses `self.knots[idx]` (same API, NDArray supports `[]`)
- `eval_all(points: NDArray[Float64]) -> CSRMatrix` — iterate `points[row]`
- `first_derivative_all(points: NDArray[Float64]) -> CSRMatrix` — same
- `base_basis(x, i)` — accesses `self.knots[i]` (same API)
- Verification: `self.knots.copy()` at basis.mojo:116 returns `NDArray[Float64]` — compatible with new `BSplineBasis.__init__`

### Layer 5: RecombinationBasis

- `eval_all(points: NDArray[Float64]) -> CSRMatrix` — delegates to `self.basis.eval_all(points)`
- `first_derivative_all(points: NDArray[Float64]) -> CSRMatrix` — same

### Layer 6: TensorProductBasis

- `eval_tensor(s_points: NDArray, v_points: NDArray) -> CSRMatrix`
- `partial_s(s_points: NDArray, v_points: NDArray) -> CSRMatrix`
- `partial_v(s_points: NDArray, v_points: NDArray) -> CSRMatrix`

### Layer 7: FPEDomain

Fields change:
- `s_knots`, `v_knots`: `List[Float64]` → `NDArray[Float64]`
- `s_points`, `v_points`: `List[Float64]` → `NDArray[Float64]`
- `s_weights`, `v_weights`: `List[Float64]` → `NDArray[Float64]`
- `s_points_phys`, `v_points_phys`: `List[Float64]` → `NDArray[Float64]`

Helpers change:
- `_sort_unique(x: NDArray) -> NDArray` — two-pass: count unique, allocate exact-size NDArray, fill
- `_normalize_list(x: NDArray) -> NDArray` → rename to `_normalize(x: NDArray) -> NDArray`
- `_grid_create(...) -> NDArray` — uses `linspace` (NDArray) instead of `linspace_list`. Strategy: over-allocate (sum all segment sizes + 1 for mean), fill by writing into buffer with write pointer, then sort+dedup into exact-size NDArray via `_sort_unique`
- `_compute_quad_points(grid: NDArray, num_gauss) -> NDArray` — two-pass: (1) count valid intervals (where b > a), (2) allocate `valid_intervals * (1 + num_gauss)`, fill
- `_compute_quad_weights(grid: NDArray, num_gauss) -> NDArray` — same two-pass sizing as quad_points

`integ_weights()` updated: pass `self.s_weights`/`self.v_weights` directly to `DiagMatrix` (already NDArray).
`build_basis()` updated: `self.s_knots.copy()` / `self.v_knots.copy()` now returns `NDArray[Float64]` — compatible with new `BSplineBasis.__init__`.

Remove `_list_to_ndarr` bridge function entirely.

### Layer 8: Consumers

- `galerkin.mojo`: Remove `_list_to_ndarr`. Use `domain.s_points`/`domain.v_points` directly (already NDArray). `domain.s_points_phys`/`domain.v_points_phys` now NDArray — use `.len()` and `[]` index. `domain.s_weights`/`domain.v_weights` — already NDArray, pass directly to DiagMatrix.
- `pdf.mojo`: `len(domain.s_points)` → `domain.s_points.len()`.
- `initial_cond.mojo`: `len(domain.s_points_phys)` → `domain.s_points_phys.len()`. Index access `domain.s_points_phys[i]` unchanged.
- `objective.mojo`: `len(domain.s_points)` → `domain.s_points.len()`. `len(domain.v_points)` → `domain.v_points.len()`. Index access `domain.s_points[i]` unchanged.
- `python_module.mojo`: `len(domain.s_points)` → `domain.s_points.len()`. `domain.s_points.copy()` now returns `NDArray[Float64]` — use `.to_list()` bridge for `PDFGrid` interop.
- `pdf_cache.mojo`: `PDFGrid.s_points`/`v_points`/`ds_weights`/`dv_weights` → `NDArray[Float64]`. `precompute_weights()` uses NDArray index writes instead of `.append()`. `to_python_object()` uses `.to_list()` for Python serialization. `from_python_object()` uses `from_list()` for deserialization.

### Layer 9: Delete _list helpers

Remove from `src/numerics/utils/constructors.mojo`:
- `zeros_list`, `zeros_mat_list`, `zeros_3d_list`, `linspace_list`

Remove from `src/numerics/utils/copy.mojo`:
- `copy_vec_list`, `copy_mat_list`, `swap_rows_list`

Remove from `src/numerics/utils/__init__.mojo`:
- All `_list` imports and exports

Remove from `src/numerics/__init__.mojo` (already cleaned, verify).

Remove `linspace_list` import from `src/engines/nais/gpu_trainer.mojo` (already fixed, verify).

### Layer 10: Update tests

`test_bspline.mojo`:
- Build knot arrays using `NDArray` instead of `List` literals
- Pass `NDArray` points to `eval_tensor`, `eval_all`, etc.
- Note: test_tensor_product_shapes has pre-existing bug (ncols=1 vs expected=9) — do NOT fix, leave as-is

`test_zeros_3d.mojo`:
- Remove `zeros_3d_list` test cases (3 tests deleted)
- Keep NDArray test cases (1 test remains)

## Verification

After each layer, run:
```bash
pixi run mojo run -I src tests/test_bspline.mojo
pixi run mojo run -I src tests/test_zeros_3d.mojo
```

After Layer 7+, also run:
```bash
pixi run mojo run -I src tests/test_fpe_engine.mojo
```

After final layer, run full test suite to ensure no regressions.

## Risks

- `knots_concat` has complex dedup logic. Two-pass approach (count unique → fill NDArray) needs careful implementation.
- `func_parabolic` has sorting + conditional logic. Will use NDArray in-place sort (manual bubble sort like current code).
- `_grid_create` concatenates 6+ segments. Over-allocate then sort+dedup via `_sort_unique` — excess allocation is temporary, trimmed by dedup.
- `PDFGrid` migration affects Python interop — must verify `to_python_object()`/`from_python_object()` roundtrip.
- `GaussLegendre` constructor uses List literal syntax — each case needs manual NDArray allocation+write.

## Out of scope

- `GaussLegendre` node values are hard-coded per order — no algorithmic change, just storage type.
- `BSplineBasis.de_boor_cox` internal algorithm is unchanged — only `self.knots` access pattern differs.
- `test_tensor_product_shapes` pre-existing bug — not caused by this migration, not fixed here.
