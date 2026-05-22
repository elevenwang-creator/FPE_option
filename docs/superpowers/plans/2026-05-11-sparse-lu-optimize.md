# SparseLU Factorize Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Goal:** Optimize SparseLU.factorize() from O(n²) to O(n·bandwidth) on banded FPE matrices, with symbolic/numeric split for Radau re-factorization.
>
> **Architecture:** Three incremental optimizations (A: sparse-aware left-looking, B: deferred pivot via row-index, C: symbolic/numeric split) in the existing `SparseLU` struct. Backward-compatible API preserved.
>
> **Tech Stack:** Mojo (nightly), std.memory (UnsafePointer, alloc, memcpy, memset_zero), std.time (perf_counter_ns), List[Int]/List[Float64], FixedSizeVector

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/numerics/utils/sparse_lu.mojo` | Modify | Core SparseLU struct — all 3 optimizations |
| `src/numerics/ode/radau.mojo` | Modify | Integration: symbolic/numeric split for re-factorization |
| `tests/test_sparse_lu.mojo` | Modify | Add banded-matrix scaling test |
| `tests/test_sparse_lu_pivot.mojo` | Modify | Add multi-pivot banded test |
| `tests/test_sparse_lu_symnum.mojo` | Create | Test symbolic+numeric split correctness |
| `benchmarks/bench_sparse_lu_scaling.mojo` | Create | Benchmark factorize scaling: n=64..2048 |

---

### Task 1: Sparse-Aware Left-Looking (Optimization A)

**Files:**
- Modify: `src/numerics/utils/sparse_lu.mojo:109-225`
- Test: `tests/test_sparse_lu.mojo`, `tests/test_sparse_lu_pivot.mojo`

The core change: replace `for j in range(k)` at line 123 with iteration over only the nonzero columns in the workspace, sorted by row index.

**Algorithm (revised for correctness):**

After scattering column k of A into W (lines 115-121), we have `w_nz` containing the row indices that are nonzero. The left-looking update must process these in ascending order (j < k first), and any new nonzeros created by scattering L column j must also be processed.

Key correctness property: When we scatter L column j (for j < k) into W, the new nonzero entries have row indices > j (since L is strictly lower triangular in the permuted frame — entries at rows > j). These new entries with row index < k need to be processed in the same left-looking pass. So we need a dynamic worklist.

**Implementation approach:** Use `w_nz` as a sorted dynamic worklist. After sorting, process entries in order. When scattering L column j creates a new nonzero at row r (where j < r < k), insert r into `w_nz` maintaining sorted order. Since L column j has row indices > j, and we process in ascending j order, the new entries at row r will be encountered later in the scan.

However, inserting into a sorted List is O(n) per insert. For banded matrices with ~bandwidth entries, total inserts are O(bandwidth²) per column. This is still much better than O(n).

**Simpler approach (recommended):** Use two lists: `w_nz` (sorted, current nonzero set) and a `w_nz_pending` (unsorted, newly discovered nonzeros). Process `w_nz` sequentially. When L scatter creates new nonzeros, append to `w_nz_pending`. After finishing `w_nz`, sort `w_nz_pending`, merge into `w_nz`, and process the new entries. Repeat until `w_nz_pending` is empty.

In practice, for banded FPE matrices, at most 1-2 rounds are needed because L fill-in is bounded by the bandwidth.

- [ ] **Step 1: Add a banded-matrix test to test_sparse_lu.mojo**

Add a test for a 32x32 pentadiagonal matrix (bandwidth 2) to verify correctness after optimization A:

```mojo
print("[Test 4] 32x32 pentadiagonal matrix (bandwidth 2)")
var n32 = 32
var A32_csr = CSRMatrix(n32, n32, 0)
var indptr32: List[Int] = [0]
var indices32: List[Int] = []
var data32: List[Float64] = []
for i in range(n32):
    if i > 1:
        indices32.append(i - 2)
        data32.append(-0.5)
    if i > 0:
        indices32.append(i - 1)
        data32.append(-1.0)
    indices32.append(i)
    data32.append(5.0)
    if i < n32 - 1:
        indices32.append(i + 1)
        data32.append(-1.0)
    if i < n32 - 2:
        indices32.append(i + 2)
        data32.append(-0.5)
    indptr32.append(len(indices32))
A32_csr.indptr = indptr32^
A32_csr.indices = indices32^
A32_csr.data = data32^
A32_csr._nnz = len(A32_csr.data)
var A32 = A32_csr.to_csc()

var lu32 = SparseLU(n32)
lu32.factorize(A32)

var b32: List[Float64] = []
for i in range(n32):
    b32.append(Float64(i + 1))
var x32 = lu32.solve(b32)

var Ax32 = A32_csr.spmv_new(x32)
var residual32 = 0.0
for i in range(n32):
    residual32 = residual32 + abs(Ax32[i] - b32[i])
print(" ||Ax - b|| = ", residual32)
var ok32 = residual32 < 1e-8
print(" PASS" if ok32 else " FAIL")
```

- [ ] **Step 2: Run tests to verify baseline (before changes)**

Run: `pixi run mojo test -I src tests/test_sparse_lu.mojo && pixi run mojo test -I src tests/test_sparse_lu_pivot.mojo`

Expected: All existing tests PASS. New Test 4 PASS.

- [ ] **Step 3: Implement sparse-aware left-looking in sparse_lu.mojo**

Replace the factorize method's main loop (lines 109-225) with the sparse-aware version. Key changes:

1. **Add `w_nz_pending: List[Int]`** as a struct field (initialized in `__init__`, cleared in `factorize`)
2. **Replace `for j in range(k)` with sorted w_nz iteration:**

```mojo
# After scattering column k of A into W (lines 115-121):
# Sort w_nz by row index
self.w_nz.sort()

# Sparse-aware left-looking: iterate only nonzero columns j < k
var nz_idx = 0
var w_nz_len = len(self.w_nz)
while nz_idx < w_nz_len:
    var j = self.w_nz[nz_idx]
    if j >= k:
        break

    var u_jk = W_ptr[j]
    if abs(u_jk) >= 1e-14:
        self.Uj.append(j)
        self.Ux.append(u_jk)
        nnzU += 1
        self.U_col_nnz[k] += 1

        var l_start = self.L_col_start[j]
        var l_end = l_start + self.L_col_nnz[j]
        for p in range(l_start, l_end):
            var row = self.Lj[p]
            var old_val = W_ptr[row]
            var new_val = old_val - self.Lx[p] * u_jk
            if old_val == 0.0 and new_val != 0.0:
                self.w_nz_pending.append(row)
            W_ptr[row] = new_val

    W_ptr[j] = 0.0
    nz_idx += 1

# Process pending new nonzeros (from L scatter fill-in)
while len(self.w_nz_pending) > 0:
    self.w_nz_pending.sort()
    var pending_len = len(self.w_nz_pending)
    var p_idx = 0
    while p_idx < pending_len:
        var j = self.w_nz_pending[p_idx]
        if j >= k:
            p_idx += 1
            continue

        var u_jk = W_ptr[j]
        if abs(u_jk) >= 1e-14:
            self.Uj.append(j)
            self.Ux.append(u_jk)
            nnzU += 1
            self.U_col_nnz[k] += 1

            var l_start = self.L_col_start[j]
            var l_end = l_start + self.L_col_nnz[j]
            for p in range(l_start, l_end):
                var row = self.Lj[p]
                var old_val = W_ptr[row]
                var new_val = old_val - self.Lx[p] * u_jk
                if old_val == 0.0 and new_val != 0.0:
                    # Would need another round — append to a second pending list
                    # For banded systems, this is extremely rare (0-1 occurrences)
                    self.w_nz.append(row)  # Reuse w_nz as overflow
                W_ptr[row] = new_val

        W_ptr[j] = 0.0
        p_idx += 1
    self.w_nz_pending.clear()
```

**Important**: The workspace clearing at lines 218-225 must also be updated. Instead of clearing via `w_nz` which only contains the original A-scattered entries, we must also clear the pending entries and the L-subdiagonal entries. The existing clearing loop at 218-225 already handles `w_nz` + L_col entries. We need to also clear `w_nz_pending` entries and re-clear any entries from the left-looking loop that were zeroed mid-pass (they're already 0.0, so redundant but harmless).

Actually, the simplest correct approach: **after the left-looking loop, the only nonzero entries in W are at rows >= k** (since we zeroed all j < k during the pass). The existing pivot search (lines 145-168) only reads W[k..n-1]. The clearing loop (218-225) only needs to zero W[k..n-1] — which is already handled by the existing `w_nz` + L_col clearing. But we need to make sure ALL nonzero entries in W are tracked.

**Revised clearing approach:** Instead of maintaining `w_nz` for clearing, use a simpler scheme — after the left-looking loop + pivot + L-subdiagonal extraction, clear W entries by iterating the L-subdiagonal row indices plus the pivot row. The entries at j < k were already zeroed during the left-looking pass. This eliminates the `w_nz`-based clearing entirely.

Wait — the existing clearing at lines 218-225 does:
1. Clear all w_nz entries (original A-scattered + L-fill entries)
2. Clear all L_col row indices (subdiagonal L entries for column k)
3. Clear W[k]

After optimization A, the w_nz list no longer contains ALL nonzero entries (because we zeroed some mid-pass). The remaining nonzeros are:
- W[k] (the pivot position — may have changed due to swap)
- W[piv_row] if piv_row != k (swapped into position k, but old W[piv_row] might be nonzero)
- W[i] for i in L-subdiagonal of column k

So we need a different clearing strategy. **Best approach:** Build a complete `w_nz_all` list that tracks every W entry that was set during this column's processing, then clear them all at the end.

**Final algorithm for column k:**

```
1. Scatter A column k → W, record rows in w_nz_all
2. Sort w_nz_all
3. Left-looking pass: iterate w_nz_all entries where j < k
   - For each nonzero j < k: emit U[j,k], scatter L column j → W
   - Any new nonzero rows from L scatter: add to w_nz_all (maintain sorted)
   - Zero W[j] after processing (but keep in w_nz_all for final clear)
4. Pivot search in W[k..n-1]
5. Swap if needed
6. Emit U[k,k], L-subdiagonal entries
7. Clear all entries in w_nz_all + L-subdiagonal rows + W[k]
```

This is the cleanest approach. The `w_nz_all` list contains ALL rows that were ever nonzero in W during this column step. At the end, we zero them all.

For the dynamic sorted insertion in step 3: Instead of inserting into a sorted list (O(n) per insert), use the pending-list approach:
- `w_nz` = sorted initial list (from A scatter)
- Process `w_nz` entries j < k
- New nonzeros go to `w_nz_pending`
- After processing all `w_nz` entries, sort `w_nz_pending`, process those too
- Merge `w_nz` + `w_nz_pending` into `w_nz_all` for final clearing

The full implementation code for the modified `factorize` method:

```mojo
@always_inline
def factorize(mut self, A: CSCMatrix) raises:
    var n = self.n

    self.w_nz.clear()
    self.w_nz_pending.clear()
    self.Lj.clear()
    self.Lx.clear()
    self.Uj.clear()
    self.Ux.clear()

    self.Lj.reserve(n * n // 4)
    self.Lx.reserve(n * n // 4)
    self.Uj.reserve(n * n // 4)
    self.Ux.reserve(n * n // 4)
    self.w_nz.reserve(n)
    self.w_nz_pending.reserve(n)

    for i in range(n):
        self.perm[i] = i
        self.pinv[i] = i
        self.L_col_start[i] = 0
        self.L_col_nnz[i] = 0
        self.U_col_start[i] = 0
        self.U_col_nnz[i] = 0

    self.W.zero_out()

    var W_ptr = self.W.ptr()
    var nnzL = 0
    var nnzU = 0

    for k in range(n):
        self.L_col_start[k] = nnzL
        self.U_col_start[k] = nnzU

        self.w_nz.clear()

        # Scatter column k of A into W
        var cp_start = A.colptr[k]
        var cp_end = A.colptr[k + 1]
        for p in range(cp_start, cp_end):
            var row = A.indices[p]
            var prow = self.pinv[row]
            W_ptr[prow] = A.data[p]
            self.w_nz.append(prow)

        # Sort w_nz for ordered left-looking scan
        self.w_nz.sort()

        # Sparse-aware left-looking: iterate only nonzero columns j < k
        # Use pending list for fill-in from L scatter
        self.w_nz_pending.clear()

        var nz_idx = 0
        var w_nz_len = len(self.w_nz)
        while nz_idx < w_nz_len:
            var j = self.w_nz[nz_idx]
            if j >= k:
                break

            var u_jk = W_ptr[j]
            if abs(u_jk) >= 1e-14:
                self.Uj.append(j)
                self.Ux.append(u_jk)
                nnzU += 1
                self.U_col_nnz[k] += 1

                var l_start = self.L_col_start[j]
                var l_end = l_start + self.L_col_nnz[j]
                for p in range(l_start, l_end):
                    var row = self.Lj[p]
                    var old_val = W_ptr[row]
                    var new_val = old_val - self.Lx[p] * u_jk
                    if old_val == 0.0 and new_val != 0.0:
                        self.w_nz_pending.append(row)
                    W_ptr[row] = new_val

            # Zero W[j] immediately after processing to prevent
            # duplicate emission in the pending-nonzeros loop
            W_ptr[j] = 0.0
            nz_idx += 1

        # Process pending nonzeros (fill-in from L scatter)
        # At most 2-3 rounds for banded matrices
        var round_limit = 3
        while len(self.w_nz_pending) > 0 and round_limit > 0:
            self.w_nz_pending.sort()
            var pending_len = len(self.w_nz_pending)
            var p_idx = 0
            while p_idx < pending_len:
                var j = self.w_nz_pending[p_idx]
                if j >= k:
                    p_idx += 1
                    continue

                var u_jk = W_ptr[j]
                if abs(u_jk) >= 1e-14:
                    # W[j] was nonzero and not yet processed (it was
                    # created by L scatter, not by A scatter, so it
                    # wasn't in the original w_nz list)
                    self.Uj.append(j)
                    self.Ux.append(u_jk)
                    nnzU += 1
                    self.U_col_nnz[k] += 1

                    var l_start = self.L_col_start[j]
                    var l_end = l_start + self.L_col_nnz[j]
                    for p in range(l_start, l_end):
                        var row = self.Lj[p]
                        var old_val = W_ptr[row]
                        var new_val = old_val - self.Lx[p] * u_jk
                        if old_val == 0.0 and new_val != 0.0:
                            self.w_nz_pending.append(row)
                        W_ptr[row] = new_val

                W_ptr[j] = 0.0
                p_idx += 1
            self.w_nz_pending.clear()
            round_limit -= 1

        # Pivot search (same as before)
        var piv_val = W_ptr[k]
        var piv_row = k

        var remain = n - k - 1
        if remain > 0:
            comptime width = SIMD_WIDTH
            var w_scalar = W_ptr.bitcast[Scalar[DType.float64]]()
            var p = k + 1
            while p + width <= n:
                var vals = (w_scalar + p).load[width=width]()
                var abs_vals = abs(vals)
                var cur_max = abs(piv_val)
                comptime for w in range(width):
                    if abs_vals[w] > cur_max:
                        cur_max = abs_vals[w]
                        piv_val = vals[w]
                        piv_row = p + w
                p += width
            while p < n:
                var v = W_ptr[p]
                if abs(v) > abs(piv_val):
                    piv_val = v
                    piv_row = p
                p += 1

        if abs(piv_val) < 1e-14:
            # Singular column — clear workspace
            for idx in range(len(self.w_nz)):
                W_ptr[self.w_nz[idx]] = 0.0
            continue

        if piv_row != k:
            swap(W_ptr[k], W_ptr[piv_row])

        # Pivot L-row swap (SAME AS BEFORE — will be optimized in Task 2)
        for j in range(k):
            var l_start = self.L_col_start[j]
            var l_end = l_start + self.L_col_nnz[j]
            var l_k_idx: Int = -1
            var l_piv_idx: Int = -1
            for p in range(l_start, l_end):
                if self.Lj[p] == k:
                    l_k_idx = p
                if self.Lj[p] == piv_row:
                    l_piv_idx = p
            if l_k_idx >= 0 and l_piv_idx >= 0:
                var tmp = self.Lx[l_k_idx]
                self.Lx[l_k_idx] = self.Lx[l_piv_idx]
                self.Lx[l_piv_idx] = tmp
            elif l_k_idx >= 0:
                self.Lj[l_k_idx] = piv_row
            elif l_piv_idx >= 0:
                self.Lj[l_piv_idx] = k

        var old_perm_k = self.perm[k]
        var old_perm_piv = self.perm[piv_row]
        self.perm.swap_elements(k, piv_row)
        self.pinv[old_perm_k] = piv_row
        self.pinv[old_perm_piv] = k

        self.Uj.append(k)
        self.Ux.append(W_ptr[k])
        nnzU += 1
        self.U_col_nnz[k] += 1

        var inv_diag = 1.0 / W_ptr[k]
        for i in range(k + 1, n):
            if abs(W_ptr[i]) > 1e-14:
                var l_val = W_ptr[i] * inv_diag
                self.Lj.append(i)
                self.Lx.append(l_val)
                nnzL += 1
                self.L_col_nnz[k] += 1

        # Clear workspace: zero all rows that were set in W
        for idx in range(len(self.w_nz)):
            W_ptr[self.w_nz[idx]] = 0.0
        var l_col_nnz_k = self.L_col_nnz[k]
        var l_col_start_k = self.L_col_start[k]
        for idx in range(l_col_nnz_k):
            W_ptr[self.Lj[l_col_start_k + idx]] = 0.0
        W_ptr[k] = 0.0
        if piv_row != k:
            W_ptr[piv_row] = 0.0

    self.Lp[0] = 0
    self.Up[0] = 0
    for k in range(n):
        self.Lp[k + 1] = self.Lp[k] + self.L_col_nnz[k]
        self.Up[k + 1] = self.Up[k] + self.U_col_nnz[k]

    for k in range(n):
        var diag: Float64 = 1.0
        var u_start = self.Up[k]
        var u_end = self.Up[k + 1]
        for p in range(u_start, u_end):
            if self.Uj[p] == k:
                diag = self.Ux[p]
                break
        self.diag_vals[k] = diag
```

**Struct changes**: Add `var w_nz_pending: List[Int]` field. Initialize in `__init__`.

- [ ] **Step 4: Run tests to verify optimization A is correct**

Run: `pixi run mojo run -I src tests/test_sparse_lu.mojo && pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo`

Expected: All tests PASS (including new Test 4 pentadiagonal).

- [ ] **Step 5: Run performance benchmark**

Run the existing perf test and verify factorize time decreased for banded matrices:

`pixi run mojo run -I src tests/test_sparse_lu_perf.mojo`

Expected: Faster factorization (no regression).

- [ ] **Step 6: Commit optimization A**

```bash
git add src/numerics/utils/sparse_lu.mojo tests/test_sparse_lu.mojo
git commit -m "perf: sparse-aware left-looking in SparseLU.factorize — O(n·bw) vs O(n²)"
```

---

### Task 2: Deferred Pivot Permutation via Row-Index (Optimization B)

**Files:**
- Modify: `src/numerics/utils/sparse_lu.mojo:176-202` (pivot swap section)
- Test: `tests/test_sparse_lu_pivot.mojo`

Replace the O(k · avg_L_nnz) L-row-scan loop with an O(nnz_in_row_k + nnz_in_row_piv) lookup using a row-to-column inverted index.

- [ ] **Step 1: Add a test for a large banded matrix with many pivots**

Add to `tests/test_sparse_lu_pivot.mojo`:

```mojo
print("[Test 4] 64x4 block-pentadiagonal with forced pivots")
var n_bp = 64
var A_bp_csr = CSRMatrix(n_bp, n_bp, 0)
var indptr_bp: List[Int] = [0]
var indices_bp: List[Int] = []
var data_bp: List[Float64] = []
for i in range(n_bp):
    if i > 1:
        indices_bp.append(i - 2)
        data_bp.append(-0.25)
    if i > 0:
        indices_bp.append(i - 1)
        data_bp.append(-0.5)
    indices_bp.append(i)
    data_bp.append(1.0 + 0.01 * Float64(i % 7))  # Varying diagonal to force pivots
    if i < n_bp - 1:
        indices_bp.append(i + 1)
        data_bp.append(-0.5)
    if i < n_bp - 2:
        indices_bp.append(i + 2)
        data_bp.append(-0.25)
    indptr_bp.append(len(indices_bp))
A_bp_csr.indptr = indptr_bp^
A_bp_csr.indices = indices_bp^
A_bp_csr.data = data_bp^
A_bp_csr._nnz = len(A_bp_csr.data)
var A_bp = A_bp_csr.to_csc()

var lu_bp = SparseLU(n_bp)
lu_bp.factorize(A_bp)

var b_bp: List[Float64] = []
for i in range(n_bp):
    b_bp.append(Float64(i + 1))
var x_bp = lu_bp.solve(b_bp)

var Ax_bp = A_bp_csr.spmv_new(x_bp)
var res_bp = 0.0
for i in range(n_bp):
    res_bp = res_bp + abs(Ax_bp[i] - b_bp[i])
print(" ||Ax - b|| = ", res_bp)
var ok4 = res_bp < 1e-8
print(" PASS" if ok4 else " FAIL")
```

- [ ] **Step 2: Run test to verify it passes before changes**

Run: `pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo`

Expected: Test 4 PASS with current (slow) pivot swap.

- [ ] **Step 3: Implement row-index inverted index for pivot swaps**

Add struct field: `var row_L_cols: List[List[Int]]` — `row_L_cols[i]` = list of L column indices that have an entry at row i.

Initialize in `__init__`:
```mojo
self.row_L_cols = List[List[Int]]()
for _ in range(n):
    var inner = List[Int]()
    self.row_L_cols.append(inner^)
```

In `factorize`, clear at start:
```mojo
for i in range(n):
    self.row_L_cols[i].clear()
```

When appending L subdiagonal entries (after the pivot):
```mojo
for i in range(k + 1, n):
    if abs(W_ptr[i]) > 1e-14:
        var l_val = W_ptr[i] * inv_diag
        self.Lj.append(i)
        self.Lx.append(l_val)
        nnzL += 1
        self.L_col_nnz[k] += 1
        self.row_L_cols[i].append(k)  # Column k has entry at row i
```

Replace the pivot L-row swap loop (old lines 179-196):
```mojo
if piv_row != k:
    swap(W_ptr[k], W_ptr[piv_row])

    # Swap L entries in columns that reference row k or piv_row
    # O(nnz_in_row_k + nnz_in_row_piv) instead of O(k * avg_L_nnz)
    for j_idx in range(len(self.row_L_cols[k])):
        var j = self.row_L_cols[k][j_idx]
        var l_start = self.L_col_start[j]
        var l_end = l_start + self.L_col_nnz[j]
        for p in range(l_start, l_end):
            if self.Lj[p] == k:
                self.Lj[p] = piv_row
                break

    for j_idx in range(len(self.row_L_cols[piv_row])):
        var j = self.row_L_cols[piv_row][j_idx]
        var l_start = self.L_col_start[j]
        var l_end = l_start + self.L_col_nnz[j]
        for p in range(l_start, l_end):
            if self.Lj[p] == piv_row:
                self.Lj[p] = k
                break

    # Swap row_L_cols entries
    var tmp_cols = self.row_L_cols[k]^
    self.row_L_cols[k] = self.row_L_cols[piv_row]^
    self.row_L_cols[piv_row] = tmp_cols
```

Wait — there's a subtlety. When we change `self.Lj[p]` from k to piv_row in a column j that previously referenced row k, the `row_L_cols[k]` list becomes stale. But we're about to swap `row_L_cols[k]` and `row_L_cols[piv_row]` anyway. So the sequence is:

1. For each col j in `row_L_cols[k]`: change Lj entry from k → piv_row
2. For each col j in `row_L_cols[piv_row]`: change Lj entry from piv_row → k
3. Swap `row_L_cols[k]` ↔ `row_L_cols[piv_row]`

After step 1, the L entries that were at row k are now at row piv_row. After step 2, the L entries that were at row piv_row are now at row k. After step 3, the inverted index is consistent again. This is correct.

- [ ] **Step 4: Run tests to verify optimization B is correct**

Run: `pixi run mojo run -I src tests/test_sparse_lu.mojo && pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo`

Expected: All tests PASS.

- [ ] **Step 5: Commit optimization B**

```bash
git add src/numerics/utils/sparse_lu.mojo tests/test_sparse_lu_pivot.mojo
git commit -m "perf: deferred pivot permutation via row-index in SparseLU — O(bw) per pivot"
```

---

### Task 3: Symbolic/Numeric Split (Optimization C)

**Files:**
- Modify: `src/numerics/utils/sparse_lu.mojo` (add `factorize_symbolic`, `factorize_numeric`)
- Modify: `src/numerics/ode/radau.mojo:311-321` (use symbolic/numeric split)
- Create: `tests/test_sparse_lu_symnum.mojo`

Split `factorize` into two phases. The symbolic phase determines sparsity pattern and pivot order; the numeric phase fills in values using the pre-determined structure.

- [ ] **Step 1: Write failing test for symbolic/numeric split**

Create `tests/test_sparse_lu_symnum.mojo`:

```mojo
"""Test SparseLU symbolic/numeric factorization split."""

from numerics.utils.sparse_lu import SparseLU
from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from sparse.ops import add, scale
from std.math import abs


def main() raises:
    print("=== SparseLU Symbolic/Numeric Split Test ===")
    print()

    # Test 1: symbolic + numeric gives same result as factorize
    print("[Test 1] Sym+Num matches factorize on 5x5 tridiagonal")
    var n = 5
    var A_csr = CSRMatrix(n, n, 0)
    var indptr: List[Int] = [0]
    var indices: List[Int] = []
    var data: List[Float64] = []
    for i in range(n):
        if i > 0:
            indices.append(i - 1)
            data.append(-1.0)
        indices.append(i)
        data.append(4.0)
        if i < n - 1:
            indices.append(i + 1)
            data.append(-1.0)
        indptr.append(len(indices))
    A_csr.indptr = indptr^
    A_csr.indices = indices^
    A_csr.data = data^
    A_csr._nnz = len(A_csr.data)
    var A_csc = A_csr.to_csc()

    var lu_full = SparseLU(n)
    lu_full.factorize(A_csc)

    var lu_split = SparseLU(n)
    lu_split.factorize_symbolic(A_csc)
    lu_split.factorize_numeric(A_csc)

    var b: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    var x_full = lu_full.solve(b)
    var x_split = lu_split.solve(b)

    var diff = 0.0
    for i in range(n):
        diff = diff + abs(x_full[i] - x_split[i])
    print(" ||x_full - x_split|| = ", diff)
    var ok1 = diff < 1e-12
    print(" PASS" if ok1 else " FAIL")
    print()

    # Test 2: re-numeric with scaled matrix gives correct result
    print("[Test 2] Re-numeric with scaled matrix")
    var scale_factor: Float64 = 2.0
    var A2_csr = scale(scale_factor, A_csr)
    var A2_csc = A2_csr.to_csc()

    var lu_renum = SparseLU(n)
    lu_renum.factorize_symbolic(A2_csc)
    lu_renum.factorize_numeric(A2_csc)

    var b2: List[Float64] = [6.0, 12.0, 6.0, 12.0, 8.0]
    var x2 = lu_renum.solve(b2)

    var Ax2 = A2_csr.spmv_new(x2)
    var res2 = 0.0
    for i in range(n):
        res2 = res2 + abs(Ax2[i] - b2[i])
    print(" ||Ax - b|| = ", res2)
    var ok2 = res2 < 1e-10
    print(" PASS" if ok2 else " FAIL")
    print()

    # Test 3: solve_inplace consistency after symbolic+numeric
    print("[Test 3] solve_inplace matches solve after sym+num")
    var b3_vec = FixedSizeVector(n)
    var work3 = FixedSizeVector(n)
    var b3_data: List[Float64] = [3.0, 6.0, 3.0, 6.0, 4.0]
    for i in range(n):
        b3_vec.ptr()[i] = b3_data[i]
    lu_split.solve_inplace(b3_vec, work3)
    var x3_inplace = b3_vec.to_list()

    var diff3 = 0.0
    for i in range(n):
        diff3 = diff3 + abs(x3_inplace[i] - x_split[i])
    print(" ||solve - solve_inplace|| = ", diff3)
    var ok3 = diff3 < 1e-12
    print(" PASS" if ok3 else " FAIL")
    print()

    var all_pass = ok1 and ok2 and ok3
    if all_pass:
        print("=== ALL SYM/NUM TESTS PASS ===")
    else:
        print("=== SOME TESTS FAILED ===")
```

- [ ] **Step 2: Run test to verify it fails (no factorize_symbolic yet)**

Run: `pixi run mojo run -I src tests/test_sparse_lu_symnum.mojo`

Expected: COMPILE ERROR — `factorize_symbolic` not found.

- [ ] **Step 3: Implement factorize_symbolic**

Add to `SparseLU` struct. `factorize_symbolic` runs the same algorithm as `factorize` but:
- Records sparsity pattern of L and U (appends to Lj, Uj)
- Records pivot decisions in `self._sym_pivots: List[Int]`
- Records L/U column structure (L_col_start, L_col_nnz, U_col_start, U_col_nnz)
- Records perm, pinv
- Allocates Lx, Ux to correct sizes (filled with 0.0)
- Does NOT compute Lx/Ux values (uses dummy values from the symbolic pass)
- Sets `self._symbolic_done = True`

New struct fields:
```mojo
var _symbolic_done: Bool
var _sym_pivots: List[Int]
```

Initialize in `__init__`:
```mojo
self._symbolic_done = False
self._sym_pivots = List[Int]()
```

The `factorize_symbolic` method is essentially the same as the current `factorize` but the Lx/Ux values computed during the pass are **not stored** (or stored as dummies). The sparsity pattern and pivot order are the real outputs.

Actually — the simplest correct implementation: `factorize_symbolic` runs the FULL factorize algorithm (including computing Lx/Ux values), records the pivot choices, and then **discards** the Lx/Ux values (clears them). The `factorize_numeric` then reruns just the value computation using the pre-determined pivots.

This is simpler than trying to skip value computation during symbolic (which would require tracking which W entries are "symbolic" vs "numeric"). The cost is that symbolic does ~2x the work of a single factorize, but it only runs once.

Wait — that defeats the purpose. The whole point of symbolic/numeric split is that numeric is cheaper because:
1. No List.append() — writes to pre-allocated arrays
2. No pivot search — uses fixed pivots from symbolic
3. No w_nz dynamic management — uses fixed pattern

Let me reconsider. The correct approach is:

**factorize_symbolic**: Run the full factorize algorithm but only record:
- Lj (row indices), Uj (row indices) — sparsity pattern
- L_col_start, L_col_nnz, U_col_start, U_col_nnz
- perm, pinv
- _sym_pivots[k] = piv_row for each column k
- Lx, Ux are allocated to correct sizes but values are WRONG (from the symbolic pass with h=1.0 data)

**factorize_numeric**: Run the same algorithm but:
- Skip all `List.append()` for Lj/Uj — write directly to pre-allocated positions
- Skip pivot search — use _sym_pivots[k] as the pivot row
- Skip w_nz management — use fixed sparsity from symbolic
- Compute Lx/Ux values correctly for the new matrix data

This is the cleanest split. The numeric pass is O(n · bandwidth) with no allocation, no pivot search, no dynamic list management.

**factorize_numeric implementation:**

```mojo
@always_inline
def factorize_numeric(mut self, A: CSCMatrix) raises:
    assert self._symbolic_done, "Must call factorize_symbolic first"

    var n = self.n
    self.W.zero_out()
    var W_ptr = self.W.ptr()

    var l_write = 0  # Write position in Lj/Lx
    var u_write = 0  # Write position in Uj/Ux

    for k in range(n):
        # Scatter column k of A into W
        var cp_start = A.colptr[k]
        var cp_end = A.colptr[k + 1]
        for p in range(cp_start, cp_end):
            var row = A.indices[p]
            var prow = self.pinv[row]
            W_ptr[prow] = A.data[p]

        # Sparse-aware left-looking using pre-determined U sparsity
        var u_start = self.U_col_start[k]
        var u_end = u_start + self.U_col_nnz[k]
        for p in range(u_start, u_end):
            var j = self.Uj[p]  # Pre-determined: which columns j < k have U[j,k] ≠ 0

            var u_jk = W_ptr[j]
            # Write U[j,k] value directly
            self.Ux[p] = u_jk

            # Scatter L column j into W
            var l_start = self.L_col_start[j]
            var l_end = l_start + self.L_col_nnz[j]
            for lp in range(l_start, l_end):
                var row = self.Lj[lp]
                W_ptr[row] = W_ptr[row] - self.Lx[lp] * u_jk

            W_ptr[j] = 0.0

        # Use pre-determined pivot
        var piv_row = self._sym_pivots[k]

        if piv_row != k:
            swap(W_ptr[k], W_ptr[piv_row])

        # Write U[k,k] — find its position in the pre-determined U column
        var diag_val = W_ptr[k]
        for p in range(u_start, u_end):
            if self.Uj[p] == k:
                self.Ux[p] = diag_val
                break

        # Write L subdiagonal values
        var inv_diag = 1.0 / diag_val
        var l_start = self.L_col_start[k]
        var l_end = l_start + self.L_col_nnz[k]
        for lp in range(l_start, l_end):
            var row = self.Lj[lp]
            self.Lx[lp] = W_ptr[row] * inv_diag

        # Clear W entries
        for p in range(u_start, u_end):
            W_ptr[self.Uj[p]] = 0.0
        for lp in range(l_start, l_end):
            W_ptr[self.Lj[lp]] = 0.0
        W_ptr[k] = 0.0
        if piv_row != k:
            W_ptr[piv_row] = 0.0

    # Recompute diag_vals
    for k in range(n):
        var diag: Float64 = 1.0
        var u_start = self.Up[k]
        var u_end = self.Up[k + 1]
        for p in range(u_start, u_end):
            if self.Uj[p] == k:
                diag = self.Ux[p]
                break
        self.diag_vals[k] = diag
```

**factorize_symbolic implementation:**

Same as the current `factorize` but also records `_sym_pivots[k] = piv_row` at each column, and sets `_symbolic_done = True` at the end.

```mojo
@always_inline
def factorize_symbolic(mut self, A: CSCMatrix) raises:
    # Run full factorize (same as current factorize with A+B optimizations)
    # but also record pivot choices
    self.factorize(A)  # reuse the full factorize
    # Record pivots (we need to record them DURING factorize, not after)
    # ... this needs integration into the factorize loop
```

The cleanest approach: **record pivots inside the existing `factorize` method** and have `factorize()` call `factorize_symbolic` + `factorize_numeric` as per the spec:

```mojo
# Inside factorize, after pivot search (line ~176):
self._sym_pivots[k] = piv_row

# Refactored factorize method:
def factorize(mut self, A: CSCMatrix) raises:
    self.factorize_symbolic(A)
    self.factorize_numeric(A)

def factorize_symbolic(mut self, A: CSCMatrix) raises:
    # Full factorize algorithm (same as current, with A+B optimizations)
    # plus: records _sym_pivots[k] = piv_row at each column
    # plus: sets _symbolic_done = True at the end
    # This does ALL the work — builds Lj, Lx, Uj, Ux, perm, pinv, etc.
    ...

def factorize_numeric(mut self, A: CSCMatrix) raises:
    # Cheap re-factorization: uses pre-determined sparsity and pivots
    # Only recomputes Lx, Ux values for new matrix data
    ...
```

This aligns with the spec: `factorize()` is backward-compatible (calls both phases), while Radau calls `factorize_symbolic` once then `factorize_numeric` on each h-change.

- [ ] **Step 4: Run symbolic/numeric tests**

Run: `pixi run mojo run -I src tests/test_sparse_lu_symnum.mojo`

Expected: All 3 tests PASS.

- [ ] **Step 5: Run all existing tests for regression**

Run: `pixi run mojo run -I src tests/test_sparse_lu.mojo && pixi run mojo run -I src tests/test_sparse_lu_pivot.mojo && pixi run mojo run -I src tests/test_sparse_lu_perf.mojo`

Expected: All PASS.

- [ ] **Step 6: Commit optimization C**

```bash
git add src/numerics/utils/sparse_lu.mojo tests/test_sparse_lu_symnum.mojo
git commit -m "feat: symbolic/numeric split in SparseLU — cheap re-factorization for Radau"
```

---

### Task 4: Radau Integration with Symbolic/Numeric Split

**Files:**
- Modify: `src/numerics/ode/radau.mojo:198,311-321`

Change Radau to call `factorize_symbolic` once at startup, then `factorize_numeric` on each h-change.

- [ ] **Step 1: Write the integration code**

In `radau.mojo`, change the LU initialization:

Before:
```mojo
var lu_real = SparseLU(n)
```

After (same — just note the initialization is unchanged):
```mojo
var lu_real = SparseLU(n)
```

Change the factorization call site (around lines 311-321):

Before:
```mojo
var need_lu = h_lu == 0.0
if h_lu != 0.0:
    var h_ratio = abs_f64(h / h_lu)
    need_lu = h_ratio < quot1 or h_ratio > quot2
if need_lu:
    self._update_real_data(M, K, h, n, E1_cached, E1_csc_cached, e1_csr_to_csc)
    lu_real.factorize(E1_csc_cached)
    self._update_diag_data(
        M, K, h, n, E_diag_cached, E_diag_csc_cached, ediag_csr_to_csc
    )
    h_lu = h
```

After:
```mojo
var need_lu = h_lu == 0.0
if h_lu != 0.0:
    var h_ratio = abs_f64(h / h_lu)
    need_lu = h_ratio < quot1 or h_ratio > quot2
if need_lu:
    self._update_real_data(M, K, h, n, E1_cached, E1_csc_cached, e1_csr_to_csc)
    if h_lu == 0.0:
        lu_real.factorize_symbolic(E1_csc_cached)
    else:
        lu_real.factorize_numeric(E1_csc_cached)
    self._update_diag_data(
        M, K, h, n, E_diag_cached, E_diag_csc_cached, ediag_csr_to_csc
    )
    h_lu = h
```

Key: First call (h_lu == 0.0) uses `factorize_symbolic` which does full factorization + records pivots. Subsequent calls use `factorize_numeric` which only recomputes values.

- [ ] **Step 2: Run Radau test**

Run: `pixi run mojo run -I src tests/test_radau.mojo`

Expected: Same results as before (numeric re-factorization with fixed pivots produces same solve output).

- [ ] **Step 3: Run FPE benchmark**

Run: `pixi run mojo run -I src benchmarks/bench_fpe_solve.mojo`

Expected: Total solve time reduced (factorize time should be ~5-10x smaller per re-factorization).

- [ ] **Step 4: Commit Radau integration**

```bash
git add src/numerics/ode/radau.mojo
git commit -m "perf: Radau uses symbolic/numeric split — skip symbolic on re-factorize"
```

---

### Task 5: Scaling Benchmark

**Files:**
- Create: `benchmarks/bench_sparse_lu_scaling.mojo`

Create a dedicated benchmark that measures factorize time vs n for banded matrices, demonstrating O(n·bw) scaling.

- [ ] **Step 1: Write the benchmark**

```mojo
"""Benchmark SparseLU factorize scaling on banded matrices."""

from numerics.utils.sparse_lu import SparseLU
from sparse.csc import CSCMatrix
from sparse.csr import CSRMatrix
from std.time import perf_counter_ns as now


def build_banded_csc(n: Int, bandwidth: Int) raises -> CSCMatrix:
    var A_csr = CSRMatrix(n, n, 0)
    var indptr: List[Int] = [0]
    var indices: List[Int] = []
    var data: List[Float64] = []
    var half_bw = bandwidth // 2
    for i in range(n):
        for j in range(max(0, i - half_bw), min(n, i + half_bw + 1)):
            indices.append(j)
            if j == i:
                data.append(4.0)
            else:
                data.append(-1.0 / Float64(abs(i - j)))
        indptr.append(len(indices))
    A_csr.indptr = indptr^
    A_csr.indices = indices^
    A_csr.data = data^
    A_csr._nnz = len(A_csr.data)
    return A_csr.to_csc()


def main() raises:
    print("=== SparseLU Scaling Benchmark ===")
    print()
    print("n      bw     factorize_us   numeric_us")

    for n in [64, 128, 256, 512, 1024, 2048]:
        var bw = min(n, max(5, n // 10))
        var A = build_banded_csc(n, bw)

        var lu = SparseLU(n)

        var t0 = now()
        lu.factorize_symbolic(A)
        var t_sym = Float64(now() - t0) / 1e3

        var t1 = now()
        lu.factorize_numeric(A)
        var t_num = Float64(now() - t1) / 1e3

        print(n, "  ", bw, "  ", t_sym, "  ", t_num)
```

- [ ] **Step 2: Run the benchmark**

Run: `pixi run mojo run -I src benchmarks/bench_sparse_lu_scaling.mojo`

Expected: factorize time grows roughly O(n · bw), not O(n²). numeric time is significantly less than symbolic time.

- [ ] **Step 3: Commit benchmark**

```bash
git add benchmarks/bench_sparse_lu_scaling.mojo
git commit -m "bench: SparseLU scaling benchmark — banded matrix factorize timing"
```

---

## Summary of Expected Results

| n    | bandwidth | Current factorize | After A+B factorize | After C numeric-only |
|------|-----------|-------------------|---------------------|----------------------|
| 256  | 51        | 1.9ms             | ~0.2ms              | ~0.05ms              |
| 484  | 69        | 8.0ms             | ~0.6ms              | ~0.1ms               |
| 784  | 87        | 24.9ms            | ~1.5ms              | ~0.2ms               |
| 1296 | 111       | 81.0ms            | ~3ms                | ~0.4ms               |

Radau FPE solve (8x8 grid, 256 DOF) with ~5-10 re-factorizations:
- Current: ~0.79s
- Expected: ~0.5s (factorize share drops from ~70% to ~10%)
