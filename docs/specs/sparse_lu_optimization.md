# SparseLU Factorize Optimization Spec

## Problem

`SparseLU.factorize()` is O(n²) on banded FPE matrices due to:
1. **P0**: `for j in range(k)` left-looking scan (line 123) iterates all prior columns, even though only `bandwidth` are nonzero
2. **P1**: Physical L-row swap on pivot (lines 179-196) scans all L columns for row indices k and piv_row
3. **P2**: No symbolic/numeric split — Radau re-factorizes from scratch when h changes, even though sparsity pattern is invariant

### Measured Bottleneck

| n_s | n   | bandwidth | factorize |
|-----|-----|-----------|-----------|
| 8   | 256 | 51        | 1.9ms     |
| 16  | 484 | 69        | 8.0ms     |
| 24  | 784 | 87        | 24.9ms    |
| 32  | 1296| 111       | 81.0ms    |

Scaling: ~O(n²). Target: O(n · bandwidth).

## Approach

All three optimizations (A, B, C) implemented incrementally in the existing `SparseLU` struct. Backward-compatible API preserved.

### A. Sparse-Aware Left-Looking (P0)

**Replace `for j in range(k)` with iteration over nonzero workspace columns.**

Current code (line 123):
```
for j in range(k):
    var u_jk = W_ptr[j]
    if abs(u_jk) < 1e-14:
        continue
    # ... process column j
```

This scans all k columns even though only ~bandwidth entries in W are nonzero after scattering column k of A and applying prior L columns.

**Fix**: After scattering column k (lines 115-121), sort `w_nz` by row index. Replace `for j in range(k)` with iteration over sorted `w_nz` where `row < k`:

```
w_nz.sort()  # sort by row index
var nz_idx = 0
while nz_idx < len(w_nz):
    var j = w_nz[nz_idx]
    if j >= k:
        break
    var u_jk = W_ptr[j]
    if abs(u_jk) >= 1e-14:
        # ... same body as before (emit U entry, scatter L column j)
    W_ptr[j] = 0.0  # clear immediately after processing
    nz_idx += 1
```

**Key subtlety**: When L column j is scattered into W, new nonzero rows may appear. These must be inserted into a "new_nz" list (not w_nz, to avoid iterator invalidation). After the w_nz loop, merge new_nz entries into w_nz for the workspace-clearing loop.

Actually, simpler approach: use two passes:
1. **Pass 1**: Iterate sorted w_nz entries with j < k. For each, emit U entry and scatter L column j into W. Track newly-created nonzeros in a separate `w_nz_new` list.
2. **Pass 2**: No second pass needed — the new nonzeros from L scatter are only for rows > k, which don't participate in the left-looking update for column k. They only contribute to future columns.

Wait — that's wrong. The standard left-looking algorithm needs the scatter from column j to be visible to subsequent columns j' > j within the same step k. But we're iterating j in order, so each scatter from column j updates W for the remaining columns. The key insight: **new nonzeros from scattering L column j have row indices > j (L is strictly lower triangular), so they don't affect the left-looking loop for column k** — they are rows that will become the subdiagonal part of column k of L.

So the approach is:
1. Sort `w_nz`
2. Iterate w_nz entries where j < k
3. For each, emit U[j,k] and scatter L column j into W (may create new w_nz entries for rows > k — but we don't need to process these in this loop)
4. Clear processed W entries immediately

**Expected speedup**: O(n · bandwidth) instead of O(n²). For n=1296, bandwidth=111: ~12x reduction in iterations.

### B. Deferred Pivot Permutation (P1)

**Eliminate the O(k · avg_L_nnz) L-row-scan loop on pivot.**

Current code (lines 179-196):
```
if piv_row != k:
    swap(W_ptr[k], W_ptr[piv_row])
    for j in range(k):  # O(k) scan of ALL L columns
        var l_start = self.L_col_start[j]
        var l_end = l_start + self.L_col_nnz[j]
        var l_k_idx: Int = -1
        var l_piv_idx: Int = -1
        for p in range(l_start, l_end):  # O(nnz in column j)
            if self.Lj[p] == k:
                l_k_idx = p
            if self.Lj[p] == piv_row:
                l_piv_idx = p
        # ... swap Lx or rename Lj entries
```

This is O(k · avg_L_nnz) per pivot — devastating for banded matrices where k grows to n.

**Fix**: Track the permutation as a composition of transpositions. Instead of physically swapping L entries, maintain an inverse permutation `pinv` that maps physical row index → logical row index. When we find a pivot at row `piv_row` instead of row `k`:

1. Swap `perm[k]` and `perm[piv_row]` (already done)
2. Swap `pinv[perm[k]]` and `pinv[perm[piv_row]]` (already done)
3. **Remove the entire `for j in range(k)` L-swap loop**

The L and U entries now store row indices in the **permuted** space. At solve time, the permutation is already applied via `perm[]` in the permute step. The key correctness argument:

- Before optimization: L stores logical row indices, and we physically swap L entries to reflect the pivot
- After optimization: L stores row indices in the permuted frame. When column j was factorized, rows k and piv_row hadn't been swapped yet (k > j at that time), so L column j's entries are correct as-is. The pivot swap only affects future columns (j > k), which haven't been factorized yet.

Wait — this needs more care. When piv_row > k, and we're processing column k, the pivot swap exchanges the roles of rows k and piv_row. For **already-factorized** columns j < k, the L entries in those columns may reference row k or row piv_row. If we don't swap them, then during solve, when we compute `y[i] -= L[i,j] * y[j]`, we'll apply the wrong L value to the wrong row.

The standard approach in sparse LU (e.g., CSparse's `lusolve`) is:

**Option B1: Don't swap L entries, but apply the permutation during solve.**

In the solve step, after permutation, the forward substitution operates on the permuted vector. The L matrix is stored with "local" row indices that are relative to the current permutation state at the time each column was factorized. This doesn't work directly because the permutation evolves during factorization.

**Option B2: Maintain a row permutation array and apply it to L row indices during solve.**

Store L row indices as-is (no swapping). Maintain `perm` array. During solve, when accessing `Lj[p]`, interpret it through the inverse permutation: the actual row is `pinv[Lj[p]]`.

Wait, let me think about this more carefully by looking at CSparse's approach...

In CSparse (`cs_lu`), the approach is:
1. During factorization, `pinv` is built as a row permutation
2. The input matrix A is permuted on-the-fly: `prow = pinv[row]` (already done in our code, line 119)
3. L row indices are stored in the permuted space
4. No physical L-row swap is needed because the permutation is already applied when scattering A
5. The pivot swap updates `perm`/`pinv` and swaps W entries, but **does NOT swap L entries in prior columns**

The reason this works: when we scatter column j of L into W at line 136-141, we use `row = self.Lj[p]` directly. If Lj[p] was set during the factorization of column j, and at that time the permutation was `perm_j`, then Lj[p] is in the frame of `perm_j`. But by the time we reach column k, the permutation has evolved to `perm_k`. So Lj[p] might refer to a different physical row than intended.

**The CSparse fix**: After the pivot swap, apply the inverse permutation to L row indices when they are used. Specifically, when scattering L column j into W:

```
# Instead of:
var row = self.Lj[p]
# Use:
var row = pinv[perm[Lj[p]]]  # or equivalently, some composition
```

Actually, in CSparse, the approach is simpler: **the row indices in L are stored in the permuted frame, and the permutation is final after factorization**. The key is that partial pivoting only swaps rows that haven't been factorized yet. So for column j (already factorized), the L entries in column j reference rows > j. If a later pivot at column k > j swaps row k with row piv_row > k, then:
- L column j might have an entry at row k or row piv_row
- After the swap, that entry is now at the "wrong" row

CSparse handles this by **not swapping at all during factorization** and instead applying the full permutation during solve. But this requires a different solve structure...

**Let me look at what SuperLU does**: SuperLU uses a "symbolic factorization" that determines the row permutation upfront (before numeric factorization), so there's no need for dynamic pivot swaps. But we need partial pivoting for numerical stability.

**Simplest correct approach (B1-revised)**: 

Instead of scanning ALL prior L columns on each pivot, maintain a **row-to-column map** that records which L columns contain entries at row k and row piv_row. This changes the O(k · avg_L_nnz) scan to O(nnz_in_row_k + nnz_in_row_piv).

But this requires row-wise access to L, which we don't currently have.

**Pragmatic approach (B-practical)**:

Given that A (sparse-aware left-looking) already reduces the `for j in range(k)` loop to O(bandwidth), the pivot L-swap loop `for j in range(k)` is the **same O(k) pattern** but with an inner loop over L column nnz. The total work is O(k · avg_L_nnz_per_column).

For banded FPE matrices, L has ~bandwidth entries per column, so this is O(k · bandwidth) which after A is comparable to the left-looking scan itself. So B is still important.

**Best approach for our codebase**: Build a `row_in_L` array — for each row i, store which L columns contain an entry at row i. This is an inverted index built incrementally as L entries are appended. On pivot, we only swap the L entries in columns that reference row k or row piv_row.

Implementation:
1. Add `var row_L_cols: List[List[Int]]` — row_L_cols[i] = list of L column indices that have an entry at row i
2. When appending `Lj.append(i)` and `Lx.append(l_val)`, also do `row_L_cols[i].append(k)` (column k has an entry at row i)
3. On pivot swap of rows k and piv_row:
   - For each col j in `row_L_cols[k]`: find entry at row k, change to piv_row
   - For each col j in `row_L_cols[piv_row]`: find entry at row piv_row, change to k
   - Swap `row_L_cols[k]` and `row_L_cols[piv_row]`

This is O(nnz_in_row_k + nnz_in_row_piv_row) per pivot — typically O(bandwidth) for banded matrices.

**Expected additional speedup**: 2-5x on top of A for banded systems.

### C. Symbolic/Numeric Split (P2)

**Split factorize into symbolic and numeric phases so Radau re-factorization skips symbolic work.**

The sparsity pattern of E1 = U1·M + h·K is invariant with h (only numerical values change). So:
- **Symbolic phase** (run once): Determine L/U column structure, total nnz, allocate Lj/Uj/Lx/Ux to exact sizes, compute pivot permutation
- **Numeric phase** (run per h-change): Fill Lx/Ux values into pre-allocated structure

**Implementation**:

Add two new methods:
```
def factorize_symbolic(mut self, A: CSCMatrix) raises:
    # Same as current factorize, but:
    # - Only tracks sparsity (which entries are nonzero, not their values)
    # - Records pivot decisions (which rows swap)
    # - Allocates Lj, Uj to exact sizes
    # - Allocates Lx, Ux to exact sizes (but leaves values as 0.0)
    # - Stores perm, pinv, L_col_start, L_col_nnz, etc.

def factorize_numeric(mut self, A: CSCMatrix) raises:
    # Precondition: factorize_symbolic was called previously
    # - Same algorithm as current factorize, but:
    # - No List.append() calls — writes directly to pre-allocated Lj/Lx/Uj/Ux
    # - No allocation (all sizes known from symbolic)
    # - Perm/pinv/L_col_start/L_col_nnz already set — reused as-is
```

The existing `factorize(A)` becomes:
```
def factorize(mut self, A: CSCMatrix) raises:
    self.factorize_symbolic(A)
    self.factorize_numeric(A)
```

**Radau integration**:
```
# At first h:
lu_real.factorize_symbolic(E1_csc_cached)  # once
lu_real.factorize_numeric(E1_csc_cached)   # first h

# On h change:
self._update_real_data(M, K, h, n, E1_cached, E1_csc_cached, e1_csr_to_csc)
lu_real.factorize_numeric(E1_csc_cached)   # skip symbolic!
```

**Key subtlety for P2**: The pivot choices in `factorize_numeric` must match those from `factorize_symbolic`. Partial pivoting selects the largest element, which depends on numerical values. If h changes, the pivot choices might differ.

**Resolution**: Use the **same permutation** determined by `factorize_symbolic`. This means `factorize_numeric` applies fixed pivots (from symbolic) rather than dynamic pivoting. This is safe because:
1. The sparsity pattern is the same
2. The pivot choices from symbolic used a representative h (h=1.0 or the first h)
3. For FPE matrices E1 = U1·M + h·K, the diagonal dominance pattern doesn't change drastically with h
4. If a pivot is numerically small (< 1e-14), we still detect it in numeric and skip (same as current singular-column handling)

Alternatively, if dynamic pivoting is essential for stability, the symbolic phase can use a **column ordering** (AMD or natural) instead of partial pivoting, and the numeric phase uses the fixed ordering. This is what SuperLU does (separate perm_c and perm_r).

For our use case (FPE with well-conditioned banded systems), fixed pivots from the symbolic phase are acceptable. If instability is ever observed, we can fall back to full `factorize()`.

**New struct fields for symbolic caching**:
```
var _symbolic_done: Bool        # True after factorize_symbolic
var _sym_nnzL: Int              # Total L nnz from symbolic
var _sym_nnzU: Int              # Total U nnz from symbolic
var _sym_pivots: List[Int]      # Pivot row for each column (from symbolic)
```

**Expected speedup for Radau**: Eliminates O(n · bandwidth) symbolic work per h-change. The numeric phase is also faster because it uses direct array writes instead of List.append() (no capacity checks, no memcpy on resize).

## Implementation Order

1. **A (sparse-aware left-looking)** — standalone, testable, biggest single win
2. **B (deferred permutation with row-index)** — builds on A, needs row_L_cols array
3. **C (symbolic/numeric split)** — builds on A+B, adds new methods + Radau integration

Each layer is independently testable — existing tests must pass after each.

## Test Plan

- `test_sparse_lu.mojo`: Existing tests pass unchanged (API backward-compatible)
- Add test: banded matrix factorize timing (n=256, 484, 784) — verify O(n·bw) scaling
- Add test: `factorize_symbolic` + `factorize_numeric` gives same L/U as `factorize`
- Add test: numeric re-factorization with different h gives correct results
- Radau FPE benchmark: compare before/after total solve time

## Expected Overall Speedup

| n    | bandwidth | Current | After A | After A+B | After A+B+C (re-factor) |
|------|-----------|---------|---------|-----------|--------------------------|
| 256  | 51        | 1.9ms   | ~0.4ms  | ~0.2ms    | ~0.05ms (numeric only)  |
| 484  | 69        | 8.0ms   | ~1.5ms  | ~0.6ms    | ~0.1ms                   |
| 784  | 87        | 24.9ms  | ~4ms    | ~1.5ms    | ~0.2ms                   |
| 1296 | 111       | 81.0ms  | ~10ms   | ~3ms      | ~0.4ms                   |

Estimates are approximate. A gives ~5x, A+B gives ~10-15x, A+B+C gives ~50-200x for re-factorization.
