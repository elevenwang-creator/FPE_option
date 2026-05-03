# NDArray Refactor Design — CPU Primary Storage + GPU View

## Summary

Replace the scattered `Vector`, `Buffer[T]`, `List[Float64]`, `List[Int]`, `List[List[Float64]]` data layer with a unified `NDArray[T]` type, using `IntTuple` for shape/strides metadata and `UnsafePointer` for contiguous data. Provide zero-copy stride-based views (transpose, slice, reshape) and one-line CPU→GPU transfer via `.to_device(ctx)`.

## Goals

1. **SIMD auto-vectorization** — All element-wise ops on NDArray use SIMD load/store (like current Vector, but generic over T and multi-dimensional)
2. **CPU-GPU zero-copy interop** — `ndarray.to_device(ctx)` returns a LayoutTensor view; eliminate manual `enqueue_create_buffer` + `enqueue_copy` boilerplate
3. **Layout optimization** — Contiguous row-major memory for 2D arrays (replace `List[List]`); stride-based views for transpose/slice with zero data copy

## Non-Goals

- Runtime-polymorphic dtypes (NDArray is parametric `NDArray[T]`, not dtype-erased)
- Automatic CPU↔GPU synchronization (explicit `.to_device()` / `.to_host()` calls)
- Replacing GPU kernel internals (kernels continue using `LayoutTensor` with comptime Layout)

---

## Architecture

### Core Type: `NDArray[T]`

```mojo
from layout import IntTuple
from std.memory import UnsafePointer, alloc, memcpy, memset_zero

@align(64)
struct NDArray[T: Copyable & Movable & ImplicitlyCopyable](Copyable, Movable, Writable):
    var _ptr: UnsafePointer[Self.T, MutExternalOrigin]
    var _shape: IntTuple    # e.g. IntTuple(3, 4) for 3x4 matrix
    var _strides: IntTuple  # in element-count units (not bytes)
    var _ndim: Int          # number of dimensions
    var _size: Int          # total element count
    var _owns: Bool         # RAII ownership flag
```

**Why `IntTuple` for shape/strides:**
- `RegisterPassable` + `ImplicitlyCopyable` — NDArray can be passed by value efficiently
- **Variadic constructor** — `IntTuple(nrows, ncols)`, `IntTuple(d0, d1, d2)` directly (unlike `IntArray` which only has `IntArray(size: Int)`)
- **Direct Layout interop** — `IntTuple` IS the underlying type for `Layout`; zero-conversion path to GPU
- `Equatable`, `Writable`, `Iterable`, `Sized` — richer trait set than `IntArray`
- Supports arbitrary dimensions (not limited to 4D) and hierarchical nesting
- Note: `IntTuple` is immutable after construction (no `__setitem__`), which is fine for shape/strides that are set once at construction

### Stride Convention

Strides are in **element-count units** (not byte offsets), matching NumPy:

```
# Row-major 3x4 Float64 matrix:
shape   = [3, 4]
strides = [4, 1]       # strides[0]=4 elements to next row, strides[1]=1 element to next col

# Column-major 3x4:
strides = [1, 3]

# Transpose view (zero-copy):
shape   = [4, 3]
strides = [1, 4]       # just swap the row-major strides
```

Address calculation: `ptr[i, j] = base_ptr + i * strides[0] + j * strides[1]`

### Memory Layout

```
NDArray[Float64] with shape [3, 4]:
+-------------------------------------------------------------+
| _ptr -> [a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 ]  |
|          ----- row 0 -----  ----- row 1 -----  ----- row 2 -----  |
+-------------------------------------------------------------+
_shape:   IntTuple [3, 4]    (~16B, register-passable)
_strides: IntTuple [4, 1]    (~16B, register-passable)
_ndim:    2
_size:    12
_owns:    True
```

---

## Replacement Map

| Current | Replaced By | Performance Impact |
|---------|-------------|-------------------|
| `Vector` (Float64, 1D, 289 lines) | `NDArray[Float64]` (1D) | Generic, multi-dim, same SIMD |
| `Buffer[T]` (74 lines) | `NDArray[T]` | Unified, adds SIMD ops |
| `List[Float64]` as array storage | `NDArray[Float64]` | SIMD + contiguous memory |
| `List[Int]` as index storage | `NDArray[Int]` | Contiguous, GPU-transferable |
| `List[List[Float64]]` as matrix | `NDArray[Float64]` (2D) | True row-major, cache-friendly |
| `List[List[List[Float64]]]` as 3D | `NDArray[Float64]` (3D) | Single allocation |
| `constructors.zeros_mat()` | `NDArray[Float64].zeros((n, m))` | One alloc, no nested Lists |
| `copy.copy_mat()` / `copy.copy_vec()` | `ndarray.copy()` | memcpy, one call |
| `copy.swap_rows()` | `ndarray.swap_rows(i, j)` | stride-aware, works on views |

### Files to DELETE after refactor

| File | Reason |
|------|--------|
| `utils/vector.mojo` | Replaced by `NDArray[Float64]` |
| `utils/buffer.mojo` | Replaced by `NDArray[T]` |
| `utils/constructors.mojo` | Replaced by `NDArray` factory methods |
| `utils/copy.mojo` | Replaced by `NDArray.copy()`, `NDArray.swap_rows()` |

### Files to MODIFY

| File | Change |
|------|--------|
| `utils/__init__.mojo` | Export `NDArray` instead of `Vector`/`Buffer`; add backward-compat aliases |
| `utils/scalar.mojo` | Keep as-is (pure scalar functions, no array dependency) |
| `utils/math_extra.mojo` | Keep as-is |
| `sparse/csr.mojo` | `data: List[Float64]` -> `NDArray[Float64]`, `indices: List[Int]` -> `NDArray[Int]`, `indptr: List[Int]` -> `NDArray[Int]` |
| `sparse/csc.mojo` | Same pattern as CSR |
| `sparse/coo.mojo` | `row/col/data` -> `NDArray`; keep dynamic `push` for construction; COO is `COOMatrix[dtype: DType]` so `data: NDArray[Scalar[dtype]]` |
| `sparse/diag.mojo` | `values: List[Float64]` -> `NDArray[Float64]` |
| `sparse/ops.mojo` | `alloc[Int]` workspace -> `NDArray[Int]`; `diag_scale` params -> `NDArray[Float64]` |
| `sparse/gpu_kernels.mojo` | No change (kernels already use LayoutTensor) |
| `numerics/sparse_lu.mojo` | 14x `Buffer[T]` -> `NDArray[T]`; `Vector` params -> `NDArray[Float64]` |
| `numerics/linalg.mojo` | `List[List[Float64]]` -> `NDArray[Float64]` 2D |
| `numerics/ode/radau.mojo` | ~20x `Vector` -> `NDArray[Float64]` |
| `numerics/optim/osqp.mojo` | 8x `Vector` -> `NDArray[Float64]` |

---

## File-by-File Implementation

### New File: `src/numerics/utils/ndarray.mojo`

The core NDArray implementation. Estimated ~500-600 lines.

#### Constructors

```mojo
def __init__(out self, shape: IntTuple, fill: Self.T = 0):
    # Compute size from shape, alloc, fill

def __init__(out self, *, copy: Self):
    # Deep copy: new allocation + memcpy

def __init__(out self, *, deinit take: Self):
    # Move: steal pointer, clear source

# 1D convenience (replaces Vector(n) / Buffer[T](n))
def __init__(out self, n: Int, fill: Self.T = 0):
    self = Self(IntTuple(n), fill)

# 2D convenience (replaces zeros_mat(nrows, ncols))
def __init__(out self, nrows: Int, ncols: Int, fill: Self.T = 0):
    self = Self(IntTuple(nrows, ncols), fill)

# From existing pointer (non-owning view)
def __init__(out self, ptr: UnsafePointer[Self.T, MutExternalOrigin], shape: IntTuple, strides: IntTuple):
    # View constructor -- _owns = False
```

#### Indexing

```mojo
def __getitem__(self, i: Int) -> Self.T:
    # 1D: ptr[i * strides[0]]

def __getitem__(self, i: Int, j: Int) -> Self.T:
    # 2D: ptr[i * strides[0] + j * strides[1]]

def __setitem__(mut self, i: Int, val: Self.T):
def __setitem__(mut self, i: Int, j: Int, val: Self.T):
```

#### SIMD Operations (1D, replaces Vector methods)

```mojo
def add_from(mut self, a: Self, b: Self):
    # self[i] = a[i] + b[i] with SIMD

def addassign(mut self, other: Self):
    # self[i] += other[i] with SIMD

def lin_comb_2(mut self, c1: Float64, v1: Self, c2: Float64, v2: Self):
    # self = c1*v1 + c2*v2 with SIMD

def lin_comb_3(mut self, c1: Float64, v1: Self, c2: Float64, v2: Self, c3: Float64, v3: Self):
    # self = c1*v1 + c2*v2 + c3*v3 with SIMD

def scale_assign(mut self, alpha: Float64):
    # self[i] *= alpha with SIMD

def scaled_norm_sq(self, scal: Self) -> Float64:
    # sum (self[i] / max(scal[i], 1e-300))^2 with SIMD

def sub_scaled(mut self, a: Self, alpha: Float64, b: Self):
    # self = a - alpha*b with SIMD

def update_scal(mut self, atol: Float64, rtol: Float64, y: Self):
    # self[i] = atol + rtol * |y[i]| with SIMD

def zero_out(mut self):
    # memset_zero

def copy_from_fixed(mut self, src: Self):
    # memcpy

def copy_from(mut self, src: List[Self.T]):
    # Element-wise copy from List (backward compat)
```

#### Parallel Operations

```mojo
def par_scale_assign(mut self, alpha: Float64):
def par_add(mut self, other: Self):
def par_map(mut self, f: fn(Self.T) -> Self.T):
def par_lin_comb(mut self, c1: Float64, v1: Self, c2: Float64, v2: Self):
```

#### View Operations (zero-copy)

```mojo
def T(self) -> Self:
    # Transpose view: construct new IntTuple with swapped shape/strides
    # For 2D: new_shape = IntTuple(shape[1], shape[0]), new_strides = IntTuple(strides[1], strides[0])
    # _owns = False, shares same _ptr

def reshape(mut self, new_shape: IntTuple) -> Self:
    # New view with different shape (requires contiguous data)

def row(self, i: Int) -> Self:
    # View of row i: shape=[ncols], strides=[strides[1]], offset=i*strides[0]

def col(self, j: Int) -> Self:
    # View of column j: shape=[nrows], strides=[strides[0]], offset=j*strides[1]

def slice(self, dim: Int, start: Int, end: Int) -> Self:
    # View of slice along dimension

def swap_rows(mut self, i: Int, j: Int):
    # For contiguous row-major: swap data between row i and row j
```

#### Utility Methods

```mojo
def len(self) -> Int:       return self._size
def ndim(self) -> Int:      return self._ndim
def shape(self) -> IntTuple: return self._shape
def strides(self) -> IntTuple: return self._strides
def ptr(self) -> UnsafePointer[Self.T, MutExternalOrigin]: return self._ptr
def copy(self) -> Self:
def to_list(self) -> List[Self.T]:
def write_to(self, mut writer: Some[Writer]):
```

#### GPU Interop

```mojo
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

struct NDArrayDeviceView[T: ...]:
    """Holds a DeviceBuffer + shape metadata. Created by NDArray.to_device()."""
    var buf: DeviceBuffer[Self.T]
    var shape: IntTuple

def to_device(ref self, ctx: DeviceContext) raises -> NDArrayDeviceView[Self.T]:
    """Copy data to GPU, return device view."""
    var dev_buf = ctx.enqueue_create_buffer[Self.T](self._size)
    ctx.synchronize()
    dev_buf.enqueue_copy_from(self._ptr)  # host->device via convenience method
    ctx.synchronize()
    return NDArrayDeviceView[Self.T](dev_buf, self._shape)

def from_device[T: ...](ctx: DeviceContext, view: NDArrayDeviceView[T]) -> NDArray[T]:
    """Copy data from GPU back to CPU NDArray."""
    var result = NDArray[T](view.shape)
    ctx.enqueue_copy(dst_buf=result._ptr, src_buf=view.buf)  # device->host
    ctx.synchronize()
    return result^
```

**Note on LayoutTensor construction from NDArrayDeviceView:**
LayoutTensor requires a `comptime Layout`, which cannot be created from runtime shape. The solution is the same pattern already used in the project -- `comptime if` dispatch in GPU call sites:

```mojo
# In the GPU dispatch code (e.g., executor.mojo):
comptime if has_apple_gpu_accelerator():
    comptime layout = Layout.row_major(METAL_MAX_N)
    var tensor = LayoutTensor[METAL_DTYPE, layout](dev_view.buf)
else:
    comptime layout = Layout.row_major(CUDA_MAX_N)
    var tensor = LayoutTensor[CUDA_DTYPE, layout](dev_view.buf)
```

This is unchanged from the current approach. NDArray's value is eliminating the manual buffer allocation + copy boilerplate, not replacing the comptime Layout dispatch.

---

## Phased Implementation Plan

### Phase 1: NDArray Core + utils/ Replacement (Priority: CRITICAL)

**Goal:** Create NDArray, replace Vector/Buffer, all tests pass.

| Step | File | Action | Estimated Lines |
|------|------|--------|----------------|
| 1.1 | `utils/ndarray.mojo` | Create NDArray[T] with constructors, indexing, basic ops | ~350 |
| 1.2 | `utils/ndarray.mojo` | Add SIMD operations (add_from, lin_comb_2/3, scale_assign, etc.) | ~200 |
| 1.3 | `utils/ndarray.mojo` | Add parallel operations (par_scale_assign, par_add, etc.) | ~80 |
| 1.4 | `utils/ndarray.mojo` | Add view operations (T, row, col, slice, swap_rows) | ~80 |
| 1.5 | `utils/ndarray.mojo` | Add GPU interop (to_device, from_device, NDArrayDeviceView) | ~60 |
| 1.6 | `utils/__init__.mojo` | Export NDArray; add `comptime Vector = NDArray[Float64]` alias; add `comptime Buffer[T] = NDArray[T]` alias | ~5 |
| 1.7 | `utils/constructors.mojo` | Rewrite to return NDArray: `zeros(n) -> NDArray[Float64]`, `zeros_mat(n,m) -> NDArray[Float64]`, `zeros_3d(d0,d1,d2) -> NDArray[Float64]`, `linspace -> NDArray[Float64]`. Uses `IntTuple(d0, d1, d2)` for shape construction | ~30 |
| 1.8 | `utils/copy.mojo` | Rewrite: `copy_vec` -> `ndarray.copy()`, `copy_mat` -> `ndarray.copy()`, `swap_rows` -> `ndarray.swap_rows()` | ~15 |
| 1.9 | Tests | Create `tests/test_ndarray.mojo` -- comprehensive NDArray unit tests | ~200 |
| 1.10 | Verify | Run all existing tests; Vector/Buffer aliases ensure backward compat | -- |

**Backward compatibility:** `Vector` becomes `comptime Vector = NDArray[Float64]`, `Buffer[T]` becomes `comptime Buffer[T] = NDArray[T]`. All existing `Vector(n)`, `Buffer[Int](n)`, `vector.add_from()` calls continue to work. After all consumers are migrated (Phase 3), these aliases can be deprecated.

**Estimated total:** ~770 lines new code, ~50 lines modified

### Phase 2: sparse/ Module Refactor (Priority: HIGH)

**Goal:** Replace all `List[Float64]`/`List[Int]` in sparse matrices with `NDArray`.

| Step | File | Action | Key Changes |
|------|------|--------|-------------|
| 2.1 | `sparse/csr.mojo` | Replace storage fields | `data: NDArray[Float64]`, `indices: NDArray[Int]`, `indptr: NDArray[Int]` |
| 2.2 | `sparse/csr.mojo` | Update constructors | Pre-allocate NDArray with known nnz; `__init__(nrows, ncols, nnz)` creates `NDArray[Float64](nnz)` |
| 2.3 | `sparse/csr.mojo` | Update spmv methods | Replace `self.data[p]` with `self._data_ptr[p]` (cached raw pointer for hot path); SIMD loads via `(self._data_ptr + p).load[width=W]()` |
| 2.4 | `sparse/csr.mojo` | Update `transpose()`, `to_dense()`, `from_dense()` | Use NDArray indexing; `from_dense` takes `NDArray[Float64]` 2D |
| 2.5 | `sparse/csc.mojo` | Same pattern as CSR | `data: NDArray[Float64]`, `indices: NDArray[Int]`, `colptr: NDArray[Int]` |
| 2.6 | `sparse/coo.mojo` | Hybrid approach | Construction uses `List` append (dynamic growth); `to_csr()`/`to_csc()` convert to NDArray at the end |
| 2.7 | `sparse/diag.mojo` | `values: NDArray[Float64]` | Direct replacement |
| 2.8 | `sparse/ops.mojo` | Replace `alloc[Int]` workspace with `NDArray[Int]` (RAII safe); `diag_scale` params -> `NDArray[Float64]` | Eliminates manual `.free()` calls |
| 2.9 | Tests | Update `test_sparse.mojo`, `test_sparse_coo_diag.mojo`, `test_sparse_lu.mojo`, `test_sparse_lu_perf.mojo` | -- |
| 2.10 | Verify | Run all sparse tests + dependent tests (radau, linalg, osqp) | -- |

**Critical optimization -- cached raw pointers in CSR:**
The spmv hot path accesses `data[p]`, `indices[p]`, `indptr[i]` millions of times. NDArray's `__getitem__` does `ptr + i * strides[0]` -- for 1D contiguous arrays strides[0]=1, so this optimizes to `ptr[i]`. But we should cache the raw pointer at the start of hot methods:

```mojo
def spmv_fixed(self, x: NDArray[Float64], mut y: NDArray[Float64]):
    var data_ptr = self.data.ptr()      # cache once
    var idx_ptr = self.indices.ptr()    # cache once
    var indptr_ptr = self.indptr.ptr()  # cache once
    var x_ptr = x.ptr()                 # cache once
    # ... hot loop uses raw pointers, same as current code
```

This ensures zero overhead vs current `List.__getitem__` or `Buffer.__getitem__`.

### Phase 3: numerics/ Module Refactor (Priority: HIGH)

**Goal:** Replace Vector/Buffer/List in numerics core modules.

| Step | File | Action |
|------|------|--------|
| 3.1 | `numerics/sparse_lu.mojo` | 14x `Buffer[T]` fields -> `NDArray[T]`; `Vector` params -> `NDArray[Float64]`; `solve()` returns `NDArray[Float64]` |
| 3.2 | `numerics/linalg.mojo` | `List[List[Float64]]` -> `NDArray[Float64]` 2D; `List[Float64]` -> `NDArray[Float64]` |
| 3.3 | `numerics/ode/radau.mojo` | ~20x `Vector` -> `NDArray[Float64]`; `_build_csr_to_csc_map` returns `NDArray[Int]` |
| 3.4 | `numerics/optim/osqp.mojo` | 8x `Vector` -> `NDArray[Float64]`; `List[Float64]` params -> `NDArray[Float64]` |
| 3.5 | `numerics/ode/rk45.mojo` | `List[Float64]` -> `NDArray[Float64]` for y0, t_eval, etc. |
| 3.6 | `numerics/nn/*.mojo` | `List[Float64]` -> `NDArray[Float64]` for weights, biases, gradients |
| 3.7 | `numerics/bspline/*.mojo` | `List[Float64]` -> `NDArray[Float64]` for knots, nodes, weights |
| 3.8 | Tests | Update all numerics tests |
| 3.9 | Verify | Full test suite pass |

### Phase 4: engines/ + server/ Refactor (Priority: MEDIUM, future)

| Step | File Group | Action |
|------|-----------|--------|
| 4.1 | `engines/fpe/*.mojo` | `List[List[Float64]]` -> `NDArray[Float64]` 2D for solver matrices |
| 4.2 | `engines/nais/*.mojo` | 12x `List[List[Float64]]` -> `NDArray[Float64]` for NN weights |
| 4.3 | `server/*.mojo` | `List[Float64]` -> `NDArray[Float64]` for pricing data |
| 4.4 | `bindings/*.mojo` | Update Python/C ABI to use NDArray |

---

## Performance Projections

### Quantitative Estimates

| Operation | Current Bottleneck | After NDArray | Expected Speedup |
|-----------|-------------------|---------------|-----------------|
| CSR SpMV | `List.__getitem__` per element (~20ns overhead) | Raw pointer access (~1ns) | **2-4x** |
| Dense matvec | `List[List]` row-indirection + no SIMD | Contiguous 2D + SIMD | **3-8x** |
| LU factorization | `Buffer.__getitem__` bounds check overhead | Cached raw pointer (same as current optimization in `factorize()`) | **1.0x** (already optimal) |
| LU solve | `Vector` -> `List` conversion copy | No conversion needed | **1.1-1.3x** |
| Radau step | 20x `Vector` allocation per step | Pre-allocated NDArray (same pattern) | **1.0x** (already optimal) |
| CPU->GPU transfer | 10+ lines manual alloc/copy per array | `.to_device(ctx)` 1 line | Code **-80%**, perf **1.0x** (same DMA) |
| Memory usage | `List` per-element overhead (~16B/element for ref-counted heap) | One allocation per array | **-30-50%** |

### Why SpMV Gets 2-4x

Current `CSRMatrix.spmv()` accesses `self.data[p]` where `data` is `List[Float64]`. Each `List.__getitem__` involves:
1. Bounds check (assert)
2. Pointer indirection to heap-allocated buffer
3. Another indirection through the List's internal dynamic array

With `NDArray[Float64]`, `self.data[p]` compiles to `self._ptr + p * self._strides[0]` -- for contiguous 1D (stride=1), this is `self._ptr[p]`, a single instruction. The hot path caches `self.data.ptr()` once, making it identical to raw `UnsafePointer` access.

For the SIMD inner loop in `spmv_fixed`, the current code already uses raw `UnsafePointer.load[width=W]()`. NDArray's `.ptr()` method returns the same pointer, so **SIMD performance is identical**. The speedup comes from eliminating the `List` -> `UnsafePointer` conversion step and the `List.__getitem__` overhead in non-SIMD fallback paths.

### Why Dense Matvec Gets 3-8x

Current `dense_matvec()` uses `A[i][j]` where `A: List[List[Float64]]`:
- Each `A[i]` is a separate heap allocation (pointer chase)
- Rows are not contiguous in memory (cache miss between rows)
- No SIMD

After NDArray 2D:
- `A[i, j]` -> `ptr + i * strides[0] + j * strides[1]` -- single computation, no pointer chase
- All rows contiguous in one allocation -- hardware prefetcher works across rows
- SIMD vectorization possible on inner loop

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **List.append() for dynamic construction** (COO, building CSR) | Phase 2.6: COO keeps `List` for construction; converts to NDArray at `to_csr()`/`to_csc()`. Add `NDArray.reserve(n)` + `NDArray.push(val)` for cases needing dynamic growth |
| **comptime Layout required for GPU** | NDArray doesn't try to create runtime Layout. GPU dispatch code uses existing `comptime if` pattern with `gpu_utils/dtype.mojo` constants |
| **Backward compatibility during migration** | Phase 1.6: `comptime Vector = NDArray[Float64]` alias. Existing code continues compiling. Remove aliases after Phase 3 |
| **Performance regression in hot paths** | Cached raw pointer pattern (Phase 2.3): `var p = ndarray.ptr()` at method start, identical to current `UnsafePointer` usage |
| **IntArray import from `layout` package** | IntTuple imported via `from layout import IntTuple`. Same package as `Layout`/`LayoutTensor`, already a project dependency. Note: IntArray is NOT used — it lacks variadic constructor and `__setitem__` is unnecessary for immutable shape/strides |
| **COO dynamic append** | Keep List for COO construction phase; convert to NDArray at finalization. Alternative: implement `NDArray.push()` with doubling buffer strategy |
| **COO genericity** | COOMatrix is parameterized as `COOMatrix[dtype: DType]` with `data: List[Scalar[dtype]]`. After migration: `data: NDArray[Scalar[dtype]]`. NDArray[T] is generic over T, so `NDArray[Scalar[float32]]` and `NDArray[Scalar[float64]]` both work naturally. CSR is Float64-only, no genericity needed |
| **enqueue_copy API** | Confirmed: `enqueue_copy` accepts `UnsafePointer` as source/destination (per Mojo docs). Also `DeviceBuffer.enqueue_copy_from(ptr)` convenience method available. Both approaches work for NDArray GPU interop |
| **Test regression** | Each Phase has a verify step. Run full test suite before proceeding to next Phase |

---

## File Structure After Refactor

```
src/numerics/utils/
  __init__.mojo          # Exports NDArray, backward-compat aliases
  ndarray.mojo           # NEW: Core NDArray[T] (~600 lines)
  scalar.mojo            # KEEP: abs_f64, max_f64, etc. (48 lines)
  math_extra.mojo        # KEEP: pow_pos (13 lines)
  vector.mojo            # DELETE after Phase 3 (replaced by NDArray)
  buffer.mojo            # DELETE after Phase 3 (replaced by NDArray)
  constructors.mojo      # REWRITE: wrappers around NDArray factories (~30 lines)
  copy.mojo              # REWRITE: wrappers around NDArray.copy() (~15 lines)

src/sparse/
  __init__.mojo          # Updated exports
  csr.mojo               # MODIFIED: NDArray storage
  csc.mojo               # MODIFIED: NDArray storage
  coo.mojo               # MODIFIED: List construction -> NDArray conversion
  diag.mojo              # MODIFIED: NDArray storage
  ops.mojo               # MODIFIED: NDArray workspace + params
  gpu_kernels.mojo       # UNCHANGED (already uses LayoutTensor)
```

---

## Testing Strategy

| Phase | Test File | What It Tests |
|-------|-----------|--------------|
| 1 | `tests/test_ndarray.mojo` (NEW) | NDArray constructors, 1D/2D/3D indexing, SIMD ops, views, copy, to_list, GPU transfer |
| 1 | Existing tests | Vector/Buffer backward compat via aliases |
| 2 | `tests/test_sparse.mojo` | CSR/CSC/COO/Diag with NDArray storage |
| 2 | `tests/test_sparse_coo_diag.mojo` | COO->CSR/CSC conversion |
| 2 | `tests/test_sparse_lu.mojo` | SparseLU with NDArray |
| 3 | `tests/test_radau_*.mojo` | Radau ODE with NDArray |
| 3 | `tests/test_linalg.mojo` | Dense/sparse linalg with NDArray |
| 3 | `tests/test_optim.mojo` | OSQP with NDArray |

---

## Implementation Order Summary

```
Phase 1: NDArray core + utils/ replacement
  |-- 1.1-1.5: Create ndarray.mojo
  |-- 1.6: Update __init__.mojo with aliases
  |-- 1.7-1.8: Rewrite constructors.mojo + copy.mojo
  +-- 1.9-1.10: Test + verify

Phase 2: sparse/ module refactor
  |-- 2.1-2.4: CSR matrix -> NDArray
  |-- 2.5: CSC matrix -> NDArray
  |-- 2.6: COO matrix (hybrid List->NDArray)
  |-- 2.7: DiagMatrix -> NDArray
  |-- 2.8: ops.mojo -> NDArray workspace
  +-- 2.9-2.10: Test + verify

Phase 3: numerics/ module refactor
  |-- 3.1: sparse_lu.mojo -> NDArray
  |-- 3.2: linalg.mojo -> NDArray
  |-- 3.3: radau.mojo -> NDArray
  |-- 3.4: osqp.mojo -> NDArray
  |-- 3.5-3.7: ode/nn/bspline -> NDArray
  +-- 3.8-3.9: Test + verify

Phase 4: engines/ + server/ (future)
+-- Gradual replacement of remaining List usages
```

---

## Verification Notes (2026-04-30)

### Issues Found & Resolved

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | `IntArray` lacks variadic constructor — `IntArray(3, 4)` doesn't compile | **Changed to `IntTuple`** — has `IntTuple(3, 4)`, `IntTuple(d0, d1, d2)`, plus direct Layout interop. Immutability is acceptable for set-once shape/strides |
| 2 | Design doc used `ctx.enqueue_copy(dev_buf, self._ptr)` with unclear API | **Confirmed valid** — Mojo docs state `enqueue_copy` accepts `UnsafePointer` as source. Also `DeviceBuffer.enqueue_copy_from(ptr)` convenience method available. Updated design to use `enqueue_copy_from` |
| 3 | COO is `COOMatrix[dtype: DType]` with `Scalar[dtype]` data — not addressed in original design | **Added to design** — `data: NDArray[Scalar[dtype]]` works naturally with NDArray[T] generic. Added to risk mitigation table |

### API Verification Summary

| API | Status | Notes |
|-----|--------|-------|
| `IntTuple` from `layout` | Confirmed | Variadic init, RegisterPassable, ImplicitlyCopyable, Equatable, Writable, Iterable, Sized |
| `IntArray` from `layout` | Not used | Only `IntArray(size: Int)`, no variadic. Mutable `__setitem__` unnecessary for shape/strides |
| `enqueue_create_buffer[dtype](count)` | Confirmed | Returns `DeviceBuffer[dtype]` |
| `enqueue_copy(dst_buf=, src_buf=)` | Confirmed | Accepts DeviceBuffer, HostBuffer, and UnsafePointer |
| `DeviceBuffer.enqueue_copy_from(ptr)` | Confirmed | Convenience method for host->device copy |
| `Layout.row_major(n)` / `Layout.row_major(n, m)` | Confirmed | Requires comptime n/m — unchanged from current approach |
| `IntTuple` -> `Layout` conversion | Direct | IntTuple IS the underlying storage for Layout; zero-conversion path |
