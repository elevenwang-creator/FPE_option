# Move utils/linalg/sparse_lu into `numerics/utils/` directory

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `utils.mojo`, `linalg.mojo`, `linalg_gpu.mojo`, `sparse_lu.mojo` from `src/numerics/` flat files into a new `src/numerics/utils/` subdirectory package, with each file unchanged except for internal import path updates.

**Architecture:** Create `src/numerics/utils/` as a Mojo package directory. Move the 4 files in, update their internal cross-imports to use the new `numerics.utils.*` paths, add a `__init__.mojo` that re-exports everything for backward compatibility. Then update all 35+ consumer files that import from the old flat paths to use the new package paths. Finally, delete the old flat files and update `numerics/__init__.mojo`.

**Tech Stack:** Mojo v0.26.3+, pixi

---

## File Structure

### New directory: `src/numerics/utils/`
| File | Responsibility |
|------|---------------|
| `__init__.mojo` | Re-exports all public symbols from sub-modules |
| `fixed_size_vector.mojo` | `FixedSizeVector` struct (from current `utils.mojo` lines 20-267) |
| `helpers.mojo` | Free functions: `abs_f64`, `max_f64`, `min_f64`, `max_int`, `min_int`, `clamp_int`, `zeros`, `zeros_mat`, `zeros_3d`, `copy_vec`, `copy_mat`, `swap_rows`, `pow_pos`, `linspace` (from current `utils.mojo` lines 269-384) |
| `linalg.mojo` | `lu_solve`, `compute_jacobian`, `dense_matvec`, `sparse_matvec`, `copy_mat`, `copy_vec` (unchanged content, just moved) |
| `linalg_gpu.mojo` | GPU LU kernels (unchanged content, just moved) |
| `sparse_lu.mojo` | `SparseLU` struct (unchanged content, just moved) |

### Modified files (import path updates only):
- `src/numerics/__init__.mojo`
- `src/numerics/ode/radau.mojo`
- `src/numerics/optim/osqp.mojo`
- `src/numerics/optim/lm.mojo`
- `src/numerics/nn/adam.mojo`
- `src/numerics/nn/autograd.mojo`
- `src/numerics/nn/stable_linear.mojo`
- `src/sparse/csr.mojo`
- `src/server/interpolator.mojo`
- `src/engines/fpe/solver.mojo`
- `src/engines/fpe/gpu/executor.mojo`
- `src/engines/fpe/initial_cond.mojo` (if it imports from utils)
- `src/engines/calibrator/calibrator.mojo`
- `src/engines/calibrator/objective.mojo`
- `src/engines/nais/trainer.mojo`
- `src/engines/nais/variance.mojo`
- `src/engines/nais/nais_net.mojo`
- `src/engines/nais/volterra.mojo`
- `src/engines/nais/fbsde.mojo`
- `src/engines/nais/gpu_trainer.mojo`
- `src/engines/nais/inferencer.mojo`
- `benchmarks/bench_sparse_ops.mojo`
- `benchmarks/bench_radau.mojo`
- `examples/check_pou.mojo`
- `tests/test_sparse_lu.mojo`
- `tests/test_sparse_lu_pivot.mojo`
- `tests/test_sparse_lu_perf.mojo`
- `tests/test_linalg.mojo`
- `tests/test_radau.mojo`
- `tests/test_newton_debug.mojo`
- `tests/test_adam.mojo`
- `tests/test_gpu_vs_cpu.mojo`
- `tests/test_gpu_executor_dtype.mojo`
- `tests/test_zeros_3d.mojo`
- `tests/test_transpose.mojo`
- `tests/test_schur_verify.mojo`

### Deleted files:
- `src/numerics/utils.mojo` (replaced by `utils/` directory)

---

## Import Path Mapping

| Old Import | New Import |
|-----------|-----------|
| `from numerics.utils import X` | `from numerics.utils import X` (unchanged — re-exported via `__init__.mojo`) |
| `from numerics.linalg import X` | `from numerics.utils.linalg import X` |
| `from numerics.linalg_gpu import X` | `from numerics.utils.linalg_gpu import X` |
| `from numerics.sparse_lu import X` | `from numerics.utils.sparse_lu import X` |

Key insight: **`from numerics.utils import X` stays the same** because `utils/` is now a package and `__init__.mojo` re-exports everything. Only `linalg`, `linalg_gpu`, and `sparse_lu` paths change.

---

### Task 1: Split `utils.mojo` into `utils/fixed_size_vector.mojo` + `utils/helpers.mojo`

**Files:**
- Create: `src/numerics/utils/__init__.mojo`
- Create: `src/numerics/utils/fixed_size_vector.mojo`
- Create: `src/numerics/utils/helpers.mojo`
- Delete: `src/numerics/utils.mojo` (after move)

- [ ] **Step 1: Create `src/numerics/utils/` directory**
```bash
mkdir -p src/numerics/utils
```

- [ ] **Step 2: Write `src/numerics/utils/fixed_size_vector.mojo`**

Extract lines 20-267 from current `utils.mojo` (the `FixedSizeVector` struct) into this file. The struct needs the same imports it currently has:
```mojo
"""Cache-aligned fixed-size vector with SIMD operations."""

from std.memory import UnsafePointer, alloc, memcpy, memset_zero
from std.sys import simd_width_of

comptime CACHE_LINE_SIZE: Int = 64
comptime SIMD_WIDTH: Int = simd_width_of[DType.float64]()
comptime MAX_VECTOR_SIZE: Int = 1 << 30

@align(CACHE_LINE_SIZE)
struct FixedSizeVector(Copyable, Movable, Writable):
    # ... (full struct content from utils.mojo lines 22-267)
```

- [ ] **Step 3: Write `src/numerics/utils/helpers.mojo`**

Extract lines 269-384 from current `utils.mojo` (all free functions) into this file:
```mojo
"""Scalar utility functions and list constructors."""

from std.math import exp, log

@always_inline
def abs_f64(x: Float64) -> Float64:
    # ... (all free functions from utils.mojo lines 270-384)
```

- [ ] **Step 4: Write `src/numerics/utils/__init__.mojo`**

Re-export everything so `from numerics.utils import X` still works:
```mojo
from numerics.utils.fixed_size_vector import FixedSizeVector
from numerics.utils.helpers import (
    abs_f64, max_f64, min_f64, max_int, min_int, clamp_int,
    zeros, zeros_mat, zeros_3d, copy_vec, copy_mat, swap_rows,
    pow_pos, linspace,
)
```

- [ ] **Step 5: Verify `from numerics.utils import` still works**

Compile a consumer that uses `from numerics.utils import FixedSizeVector, abs_f64`:
```bash
pixi run mojo build -I src src/numerics/utils/__init__.mojo
```
Expected: compiles with no errors (only "no main" error for library).

- [ ] **Step 6: Delete old `src/numerics/utils.mojo`**
```bash
rm src/numerics/utils.mojo
```

- [ ] **Step 7: Run existing tests to verify backward compatibility**
```bash
pixi run mojo run -I src tests/test_sparse_lu.mojo
```
Expected: all tests pass.

- [ ] **Step 8: Commit**
```bash
git add src/numerics/utils/ src/numerics/utils.mojo
git commit -m "refactor: split numerics/utils.mojo into utils/ package directory"
```

---

### Task 2: Move `sparse_lu.mojo` into `utils/` directory

**Files:**
- Create: `src/numerics/utils/sparse_lu.mojo`
- Delete: `src/numerics/sparse_lu.mojo`
- Modify: all files that `from numerics.sparse_lu import X`

- [ ] **Step 1: Copy `src/numerics/sparse_lu.mojo` to `src/numerics/utils/sparse_lu.mojo`**

The file content stays the same, but its internal import must change:
- Old: `from numerics.utils import FixedSizeVector`
- New: `from numerics.utils.fixed_size_vector import FixedSizeVector`

- [ ] **Step 2: Update `src/numerics/utils/sparse_lu.mojo` internal import**

Change line 20 from:
```mojo
from numerics.utils import FixedSizeVector
```
to:
```mojo
from numerics.utils.fixed_size_vector import FixedSizeVector
```

- [ ] **Step 3: Add SparseLU to `src/numerics/utils/__init__.mojo`**

Add:
```mojo
from numerics.utils.sparse_lu import SparseLU
```

- [ ] **Step 4: Update `src/numerics/__init__.mojo`**

Change line 8 from:
```mojo
from numerics.linalg import lu_solve, dense_matvec, compute_jacobian, SparseLU
```
to:
```mojo
from numerics.utils.linalg import lu_solve, dense_matvec, compute_jacobian
from numerics.utils.sparse_lu import SparseLU
```
And remove line 7's `abs_f64, max_f64, min_f64, zeros, copy_vec, copy_mat` since they now come from `numerics.utils` package re-export.

- [ ] **Step 5: Update all consumers of `numerics.sparse_lu`**

Change `from numerics.sparse_lu import SparseLU` → `from numerics.utils.sparse_lu import SparseLU` in these files:
- `src/numerics/optim/osqp.mojo:26`
- `src/numerics/ode/radau.mojo:43`
- `src/numerics/linalg.mojo:9` (will be moved in Task 3, but update now)
- `tests/test_sparse_lu.mojo:3`
- `tests/test_sparse_lu_pivot.mojo:3`
- `tests/test_sparse_lu_perf.mojo:3`
- `tests/test_radau.mojo:8`
- `tests/test_newton_debug.mojo:5`

- [ ] **Step 6: Delete old `src/numerics/sparse_lu.mojo`**
```bash
rm src/numerics/sparse_lu.mojo
```

- [ ] **Step 7: Compile and test**
```bash
pixi run mojo build -I src src/numerics/optim/osqp.mojo
pixi run mojo run -I src tests/test_sparse_lu.mojo
pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo
```
Expected: all compile and pass.

- [ ] **Step 8: Commit**
```bash
git add -A src/numerics/ tests/
git commit -m "refactor: move sparse_lu.mojo into numerics/utils/ package"
```

---

### Task 3: Move `linalg.mojo` into `utils/` directory

**Files:**
- Create: `src/numerics/utils/linalg.mojo`
- Delete: `src/numerics/linalg.mojo`
- Modify: all files that `from numerics.linalg import X`

- [ ] **Step 1: Copy `src/numerics/linalg.mojo` to `src/numerics/utils/linalg.mojo`**

Update internal imports:
- Old: `from numerics.utils import abs_f64, zeros`
- New: `from numerics.utils.helpers import abs_f64, zeros`
- Old: `from numerics.sparse_lu import SparseLU`
- New: `from numerics.utils.sparse_lu import SparseLU`

- [ ] **Step 2: Add linalg symbols to `src/numerics/utils/__init__.mojo`**

Add:
```mojo
from numerics.utils.linalg import lu_solve, compute_jacobian, dense_matvec, sparse_matvec
```

- [ ] **Step 3: Update all consumers of `numerics.linalg`**

Change `from numerics.linalg import X` → `from numerics.utils.linalg import X` in:
- `src/numerics/optim/lm.mojo:1`
- `src/engines/fpe/solver.mojo:15`
- `tests/test_linalg.mojo:1`
- `src/numerics/__init__.mojo` (already updated in Task 2, verify)

- [ ] **Step 4: Delete old `src/numerics/linalg.mojo`**
```bash
rm src/numerics/linalg.mojo
```

- [ ] **Step 5: Compile and test**
```bash
pixi run mojo build -I src src/numerics/optim/lm.mojo
pixi run mojo run -I src tests/test_linalg.mojo
```
Expected: compile and pass.

- [ ] **Step 6: Commit**
```bash
git add -A src/numerics/ tests/
git commit -m "refactor: move linalg.mojo into numerics/utils/ package"
```

---

### Task 4: Move `linalg_gpu.mojo` into `utils/` directory

**Files:**
- Create: `src/numerics/utils/linalg_gpu.mojo`
- Delete: `src/numerics/linalg_gpu.mojo`
- Modify: all files that `from numerics.linalg_gpu import X`

- [ ] **Step 1: Copy `src/numerics/linalg_gpu.mojo` to `src/numerics/utils/linalg_gpu.mojo`**

Content unchanged — no internal imports from `numerics.*`.

- [ ] **Step 2: Add linalg_gpu symbols to `src/numerics/utils/__init__.mojo`**

Add:
```mojo
from numerics.utils.linalg_gpu import lu_decompose_gpu_kernel, lu_solve_gpu_kernel
```

- [ ] **Step 3: Update all consumers of `numerics.linalg_gpu`**

Change `from numerics.linalg_gpu import X` → `from numerics.utils.linalg_gpu import X` in:
- `src/engines/fpe/gpu/executor.mojo:33`

- [ ] **Step 4: Delete old `src/numerics/linalg_gpu.mojo`**
```bash
rm src/numerics/linalg_gpu.mojo
```

- [ ] **Step 5: Compile and test**
```bash
pixi run mojo build -I src src/engines/fpe/gpu/executor.mojo
```
Expected: compiles.

- [ ] **Step 6: Commit**
```bash
git add -A src/numerics/ src/engines/
git commit -m "refactor: move linalg_gpu.mojo into numerics/utils/ package"
```

---

### Task 5: Update `src/numerics/__init__.mojo` and final verification

**Files:**
- Modify: `src/numerics/__init__.mojo`

- [ ] **Step 1: Update `src/numerics/__init__.mojo`**

Replace with:
```mojo
# FPE Engine — numerics

from numerics.bspline import GenerateKnots
from numerics.bspline import BSplineBasis
from numerics.bspline import RecombinationBasis
from numerics.bspline import TensorProductBasis
from numerics.utils import (
    FixedSizeVector, abs_f64, max_f64, min_f64, zeros, copy_vec, copy_mat,
)
from numerics.utils.linalg import lu_solve, dense_matvec, compute_jacobian
from numerics.utils.sparse_lu import SparseLU
```

- [ ] **Step 2: Run full test suite**
```bash
pixi run mojo run -I src tests/test_sparse_lu.mojo
pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo
pixi run mojo run -I src tests/test_linalg.mojo
```
Expected: all pass.

- [ ] **Step 3: Compile all production modules**
```bash
pixi run mojo build -I src src/numerics/optim/osqp.mojo
pixi run mojo build -I src src/numerics/ode/radau.mojo
pixi run mojo build -I src src/engines/fpe/initial_cond.mojo
pixi run mojo build -I src src/engines/fpe/solver.mojo
pixi run mojo build -I src src/engines/fpe/gpu/executor.mojo
```
Expected: all compile (only "no main" errors).

- [ ] **Step 4: Commit**
```bash
git add src/numerics/__init__.mojo
git commit -m "refactor: update numerics/__init__.mojo for utils/ package structure"
```

---

### Task 6: Update remaining test/example/benchmark imports

**Files:**
- Modify: all test, example, and benchmark files

- [ ] **Step 1: Update test files**

Files that import `from numerics.utils import X` — **NO CHANGE needed** (re-exported via `__init__.mojo`).

Files that import `from numerics.sparse_lu import X` — already updated in Task 2.

Files that import `from numerics.linalg import X` — already updated in Task 3.

- [ ] **Step 2: Verify all tests compile and pass**

```bash
pixi run mojo run -I src tests/test_sparse_lu.mojo
pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo
pixi run mojo run -I src tests/test_linalg.mojo
pixi run mojo build -I src benchmarks/bench_sparse_ops.mojo
pixi run mojo build -I src benchmarks/bench_radau.mojo
pixi run mojo build -I src examples/check_pou.mojo
```
Expected: all compile and pass.

- [ ] **Step 3: Commit**
```bash
git add tests/ benchmarks/ examples/
git commit -m "refactor: update test/benchmark/example imports for utils/ package"
```
