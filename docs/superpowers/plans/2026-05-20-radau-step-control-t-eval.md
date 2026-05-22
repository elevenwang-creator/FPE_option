# Radau Step Controller Fix + t_eval Dense Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs in RadauSparseLinearSolver: (1) t_eval is ignored, (2) step controller diverges from Hairer's Fortran causing h to grow when it should shrink.

**Architecture:** Replace the single-n CONT buffer with a 4n collocation polynomial buffer. Add a `contr5` interpolation function for dense output. Fix step controller constants and logic to match Hairer's RADAU5 Fortran. Add t_eval-aware output recording.

**Tech Stack:** Mojo (nightly), pixi build system, existing test infrastructure in `tests/test_radau.mojo`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/numerics/ode/radau.mojo` | Modify | CONT buffer expansion, contr5(), step controller fixes, t_eval logic |
| `tests/test_radau.mojo` | Modify | Add t_eval test, step controller regression test |
| `src/engines/fpe/solver.mojo` | Modify | Adjust t_eval passing (already correct, verify) |
| `docs/superpowers/specs/2026-05-20-radau-step-control-t-eval-design.md` | No change | Design doc (already written) |

---

### Task 1: Expand CONT buffer from n to 4n

**Files:**
- Modify: `src/numerics/ode/radau.mojo:220` (CONT declaration)
- Modify: `src/numerics/ode/radau.mojo:314-316` (Z extrapolation using CONT)
- Modify: `src/numerics/ode/radau.mojo:458` (CONT computation)

The Fortran stores 4 blocks of n values in CONT: `CONT(1:n)` = y values, `CONT(n+1:2n)` = 1st coeff, `CONT(2n+1:3n)` = 2nd coeff, `CONT(3n+1:4n)` = 3rd coeff.

- [ ] **Step 1: Change CONT declaration from n to 4*n**

In `radau.mojo:220`, change:
```mojo
var CONT = FixedSizeVector(n)
```
to:
```mojo
var CONT = FixedSizeVector(4 * n)
```

- [ ] **Step 2: Update Z extrapolation to read from CONT blocks**

In `radau.mojo:314-316`, the Z extrapolation currently reads `CONT[k_idx]` directly.
It must read from the 4n layout: `CONT[i]` = y[i], `CONT[i+n]` = coeff1, `CONT[i+2n]` = coeff2, `CONT[i+3n]` = coeff3.

Change:
```mojo
Z1[k_idx] = c1r * CONT[k_idx]
Z2[k_idx] = c2r * CONT[k_idx]
Z3[k_idx] = c3r * CONT[k_idx]
```
to:
```mojo
var ak1 = CONT[k_idx + n]
var ak2 = CONT[k_idx + 2 * n]
var ak3 = CONT[k_idx + 3 * n]
Z1[k_idx] = c1r * (ak1 + (c1r - C2M1) * (ak2 + (c1r - C1M1) * ak3))
Z2[k_idx] = c2r * (ak1 + (c2r - C2M1) * (ak2 + (c2r - C1M1) * ak3))
Z3[k_idx] = c3r * (ak1 + (c3r - C2M1) * (ak2 + (c3r - C1M1) * ak3))
```

This matches the Fortran RADCOR (lines around the ELSE branch of STARTN check):
```fortran
C3Q=H/HOLD
C1Q=C1*C3Q
C2Q=C2*C3Q
DO I=1,N
  AK1=CONT(I+N)
  AK2=CONT(I+N2)
  AK3=CONT(I+N3)
  Z1I=C1Q*(AK1+(C1Q-C2M1)*(AK2+(C1Q-C1M1)*AK3))
  Z2I=C2Q*(AK1+(C2Q-C2M1)*(AK2+(C2Q-C1M1)*AK3))
  Z3I=C3Q*(AK1+(C3Q-C2M1)*(AK2+(C3Q-C1M1)*AK3))
```

- [ ] **Step 3: Update CONT computation after accepted step**

In `radau.mojo:458`, after the accepted step, compute the full 4n CONT buffer.
Replace:
```mojo
CONT.lin_comb_3(DD1, Z1, DD2, Z2, DD3, Z3)
```
with a loop that computes all 4 blocks (matching Fortran RADCOR accepted-step block):
```mojo
for k_idx in range(n):
    var z2i = Z2[k_idx]
    var z1i = Z1[k_idx]
    CONT[k_idx] = y[k_idx] + Z3[k_idx]
    CONT[k_idx + n] = (z2i - Z3[k_idx]) / C2M1
    var ak = (z1i - z2i) / C1MC2
    var acont3 = z1i / C1
    acont3 = (ak - acont3) / C2
    CONT[k_idx + 2 * n] = (ak - CONT[k_idx + n]) / C1M1
    CONT[k_idx + 3 * n] = CONT[k_idx + 2 * n] - acont3
```

Note: `CONT[i]` stores the new `y[i]` (= y_old[i] + Z3[i]). The Fortran does: `CONT(I) = Y(I)` after computing all coefficients (see RADCOR where it sets CONT(I)=Y(I) for SOLOUT). Since we update `y += Z3` at line 495, after that line `y[i]` equals the new value.

- [ ] **Step 4: Fix error estimate to use separate n-length CONT_ERR buffer**

The `M.spmv(CONT, M_CONT)` at line 460 and the error estimate at lines 461-463
require an n-length vector containing `DD1*Z1 + DD2*Z2 + DD3*Z3`. Since CONT
is now 4n, we need a dedicated n-length buffer for this.

Add a new variable (keep `CONT = FixedSizeVector(4 * n)` for polynomial, add):
```mojo
var CONT_ERR = FixedSizeVector(n)
```

Then after the CONT coefficient loop in Step 3, compute:
```mojo
CONT_ERR.lin_comb_3(DD1, Z1, DD2, Z2, DD3, Z3)
```

And change lines 460-463 from:
```mojo
M.spmv(CONT, M_CONT)
rhs_err.sub_scaled(M_CONT, h, w)
```
to:
```mojo
M.spmv(CONT_ERR, M_CONT)
rhs_err.sub_scaled(M_CONT, h, w)
```

The rest of the error estimate (lines 462-463) remains unchanged since it
uses `rhs_err` and `error_vec`, not CONT directly.

- [ ] **Step 5: Build and verify existing tests still pass**

Run: `pixi run mojo test -I src tests/test_radau.mojo`

Expected: All 6 existing tests should still pass (they don't use t_eval and the Z extrapolation change should produce identical results since the CONT coefficients encode the same information).

- [ ] **Step 6: Commit**

```bash
git add src/numerics/ode/radau.mojo
git commit -m "refactor: expand CONT buffer to 4n for collocation polynomial"
```

---

### Task 2: Add contr5 dense output interpolation function

**Files:**
- Modify: `src/numerics/ode/radau.mojo` (add `contr5` method)

- [ ] **Step 1: Write the contr5 function**

Add a standalone function (outside the struct) that matches Hairer's `CONTR5`:

```mojo
@always_inline
def contr5(i: Int, s: Float64, cont: FixedSizeVector, n: Int) -> Float64:
    return cont[i] + s * (cont[i + n] + (s - C2M1) * (cont[i + 2 * n] + (s - C1M1) * cont[i + 3 * n]))
```

This matches the Fortran:
```fortran
CONTR5=CONT(I)+S*(CONT(I+NN)+(S-C2M1)*(CONT(I+NN2)+(S-C1M1)*CONT(I+NN3)))
```

- [ ] **Step 2: Build and verify**

Run: `pixi run mojo build -I src examples/single_price.mojo`

Expected: 0 errors, 0 warnings.

- [ ] **Step 3: Commit**

```bash
git add src/numerics/ode/radau.mojo
git commit -m "feat: add contr5 dense output interpolation function"
```

---

### Task 3: Fix step controller constants and HOPT tracking

**Files:**
- Modify: `src/numerics/ode/radau.mojo:152-155` (controller constants)
- Modify: `src/numerics/ode/radau.mojo:281-295` (LU refactorization check)
- Modify: `src/numerics/ode/radau.mojo:486-520` (step size update logic)

- [ ] **Step 1: Fix quot1 and quot2 to match Fortran defaults**

In `radau.mojo:154-155`, change:
```mojo
var quot1: Float64 = 0.5
var quot2: Float64 = 2.5
```
to:
```mojo
var quot1: Float64 = 1.0
var quot2: Float64 = 1.2
```

These match the Fortran defaults `QUOT1=1.D0, QUOT2=1.2D0` (radau5.f:513,518).
The meaning: if `QUOT1 < HNEW/HOLD < QUOT2`, step size stays the same and LU is reused.
With `QUOT1=1.0, QUOT2=1.2`, only 0-20% step size growth skips LU refactorization.

- [ ] **Step 2: Add HOPT variable and tracking**

In the variable declarations section (around line 188-195), add:
```mojo
var h_opt: Float64 = h
```

After the accepted-step block (line 512, after `h_old = h`), before `h_new` computation at line 514, add `h_opt` tracking. The Fortran logic is:
```fortran
HNEW=POSNEG*MIN(ABS(HNEW),HMAXN)
HOPT=HNEW
HOPT=MIN(H,HNEW)
IF (REJECT) HNEW=POSNEG*MIN(ABS(HNEW),ABS(H))
REJECT=.FALSE.
```

Change lines 514-520 from:
```mojo
h_new = posneg * min_f64(abs_f64(h_new), abs_f64(t1 - t))
if self.max_step > 0.0:
    h_new = posneg * min_f64(abs_f64(h_new), self.max_step)
if reject:
    h_new = posneg * min_f64(abs_f64(h_new), abs_f64(h))
reject = False
h = h_new
```
to:
```mojo
h_new = posneg * min_f64(abs_f64(h_new), abs_f64(t1 - t))
if self.max_step > 0.0:
    h_new = posneg * min_f64(abs_f64(h_new), self.max_step)
h_opt = h_new
h_opt = min_f64(h, h_new)
if reject:
    h_new = posneg * min_f64(abs_f64(h_new), abs_f64(h))
reject = False
h = h_new
```

- [ ] **Step 3: Fix LU refactorization decision to include theta check**

In `radau.mojo:281-286`, change the need_lu check from:
```mojo
var need_lu: Bool
if h_lu == 0.0:
    need_lu = True
else:
    var h_ratio = abs_f64(h / h_lu)
    need_lu = h_ratio < quot1 or h_ratio > quot2
```
to:
```mojo
var need_lu: Bool
if h_lu == 0.0:
    need_lu = True
else:
    var h_ratio = abs_f64(h / h_lu)
    need_lu = h_ratio < quot1 or h_ratio > quot2
    if not need_lu and theta <= 0.001:
        need_lu = False
    elif not need_lu and theta > 0.001:
        need_lu = True
```

Wait, this needs more careful analysis. The Fortran has:
```fortran
IF (THETA.LE.THET.AND.QT.GE.QUOT1.AND.QT.LE.QUOT2) GOTO 30
H=HNEW
HHFAC=H
IF (THETA.LE.THET) GOTO 20
GOTO 10
```

Where `THET=0.001` (default), `QT=HNEW/HOLD`.
- If `theta <= thet` AND `quot1 <= qt <= quot2`: skip both Jacobian and LU (GOTO 30 = reuse everything)
- If `theta <= thet` but qt outside [quot1, quot2]: new LU needed, but reuse Jacobian (GOTO 20 = recompute LU only)
- If `theta > thet`: new Jacobian AND new LU needed (GOTO 10 = full recomputation)

In our linear system, there is no Jacobian (M and K are constant). So `theta` is irrelevant for Jacobian recomputation. But the step-size-dependent matrices E1 and E_diag DO depend on h, so the LU factorization needs updating when h changes significantly.

For our linear system case, the simplified logic should be:
```mojo
var need_lu: Bool
if h_lu == 0.0:
    need_lu = True
else:
    var h_ratio = abs_f64(h / h_lu)
    need_lu = h_ratio < quot1 or h_ratio > quot2
```

This is already correct! The theta check is for Jacobian recomputation (nonlinear systems), which we don't have. The `quot1/quot2` change from Task 3 Step 1 is the real fix — it tightens the dead band so LU is recomputed more often.

**Correction:** Do NOT add theta check to LU decision. The existing logic is correct for linear systems. The fix is just the `quot1/quot2` values.

- [ ] **Step 4: Fix rejection fallback to use h_new (not 0.5*h)**

In `radau.mojo:522-526`, the rejection block is:
```mojo
reject = True
if first:
    h = h * 0.1
else:
    h = h_new
```

The Fortran rejection block:
```fortran
REJECT=.TRUE.
LAST=.FALSE.
IF (FIRST) THEN
    H=H*0.1D0
    HHFAC=0.1D0
ELSE
    HHFAC=HNEW/H
    H=HNEW
END IF
```

Our code already matches the Fortran for the rejection path! The `h * 0.5` was in the Newton failure block (line 446), not the error-test rejection block.

Wait, re-reading: line 441-454 is the Newton convergence failure block:
```mojo
if newt_fail or not converged:
    reject = True
    if first:
        h = h * 0.1
    else:
        h = h * 0.5   # <-- THIS should be h_new for error rejection
```

But this is Newton failure, not error rejection. The Fortran handles Newton failure at label 78:
```fortran
78 H=H*0.5D0
   HHFAC=0.5D0
   REJECT=.TRUE.
   LAST=.FALSE.
   IF (CALJAC) GOTO 20
   GOTO 10
```

So `h = h * 0.5` is correct for Newton failure. The `h = h_new` path is for the error test rejection (err >= 1.0), which is at line 522-526.

**Correction:** The existing rejection code at lines 522-526 is correct. No change needed here.

- [ ] **Step 5: Build and verify tests pass**

Run: `pixi run mojo test -I src tests/test_radau.mojo`

Expected: Tests pass, but step counts may change (fewer steps due to tighter quot1/quot2, or more LU refactorizations).

- [ ] **Step 6: Commit**

```bash
git add src/numerics/ode/radau.mojo
git commit -m "fix: correct step controller quot1/quot2 defaults and add HOPT tracking"
```

---

### Task 4: Implement t_eval dense output in solve()

**Files:**
- Modify: `src/numerics/ode/radau.mojo:141-143` (output variables)
- Modify: `src/numerics/ode/radau.mojo:499-500` (output recording)
- Modify: `src/numerics/ode/radau.mojo:269-270` (step clamping for t_eval)
- Modify: `src/numerics/ode/radau.mojo:530-535` (return statement)

- [ ] **Step 1: Add t_eval tracking state**

After line 147 (where `t = t0`), add:
```mojo
var t_eval_idx: Int = 0
var t_eval_list: Optional[List[Float64]] = t_eval
```

Change the output initialization (lines 141-143). When `t_eval` is provided, we should NOT record every step — only record at t_eval points. When `t_eval` is None, record every step (backward compatible).

```mojo
var t_values: List[Float64] = [t0]
var y_values: List[List[Float64]] = []
```
remains the same (always record t0).

But we need a helper to record a solution at a given time. Add after the variable declarations:
```mojo
var _record_step: Bool = (t_eval_list is None)
```

- [ ] **Step 2: Clamp h to next t_eval point**

After the existing `t1` clamping at line 269-270, add t_eval clamping:
```mojo
if posneg * (t + 1.01 * h - t1) > 0.0:
    h = t1 - t

if t_eval_list is not None:
    var te = t_eval_list.value()[t_eval_idx]
    if posneg * (t + 1.01 * h - te) > 0.0 and posneg * (te - t) > 0.0:
        h = te - t
```

- [ ] **Step 3: Record output at t_eval points after accepted step**

Replace lines 499-500 (recording at every accepted step) with conditional logic:

```mojo
if t_eval_list is None:
    t_values.append(t)
    y_values.append(y.to_list())
else:
    while t_eval_idx < len(t_eval_list.value()):
        var te = t_eval_list.value()[t_eval_idx]
        if posneg * (te - t) > uround * max_f64(abs_f64(t), abs_f64(te)):
            break
        if abs_f64(te - t) <= uround * max_f64(abs_f64(t), abs_f64(te)):
            t_values.append(te)
            y_values.append(y.to_list())
        else:
            var s_val = (te - t_old) / h_old
            var y_interp: List[Float64] = []
            for k_idx in range(n):
                y_interp.append(contr5(k_idx, s_val, CONT, n))
            t_values.append(te)
            y_values.append(y_interp^)
        t_eval_idx += 1
```

Note: We need `t_old` and `h_old` for interpolation. `t_old` should be stored before the step updates `t`. Add before `t = t + h` (line 494):
```mojo
var t_old = t
```

- [ ] **Step 4: Build and verify**

Run: `pixi run mojo build -I src examples/single_price.mojo`

Expected: 0 errors, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add src/numerics/ode/radau.mojo
git commit -m "feat: implement t_eval dense output via contr5 interpolation"
```

---

### Task 5: Add t_eval test and step controller regression test

**Files:**
- Modify: `tests/test_radau.mojo` (add 2 new tests)

- [ ] **Step 1: Add t_eval interpolation test**

Add a new test (Test 7) that verifies t_eval works correctly:
```mojo
print()
print("[Test 7] t_eval interpolation: M=I, K=diag(1,2,3)")
print(" Requesting solution at t=[0.0, 0.25, 0.5, 0.75, 1.0]")
print()

var M7 = make_identity_csr(n)
var K7 = make_diag_csr(n, K1_diag)
var y0_7: List[Float64] = [1.0, 1.0, 1.0]

var sys7 = SimpleLinearSystem(M7^, K7^)
var solver7 = RadauSparseLinearSolver[SimpleLinearSystem](
    rtol=1e-3, atol=1e-6, max_step=0.0, first_step=0.1,
)
var t_eval_7: List[Float64] = [0.0, 0.25, 0.5, 0.75, 1.0]
var sol7 = solver7.solve(sys7^, (0.0, 1.0), y0_7, t_eval_7)

if not sol7.success:
    print(" FAILED: " + sol7.message)
else:
    print(" Output points: " + String(len(sol7.t)))
    var max_interp_err = 0.0
    for j in range(len(sol7.t)):
        var tj = sol7.t[j]
        var exact_j: List[Float64] = [exp(-1.0 * tj), exp(-2.0 * tj), exp(-3.0 * tj)]
        for i in range(n):
            var err = abs_f64(sol7.y[j][i] - exact_j[i])
            if err > max_interp_err:
                max_interp_err = err
    print(" Max interpolation error: " + String(max_interp_err))
    if len(sol7.t) == 5 and max_interp_err < 1e-3:
        print(" PASSED")
    else:
        print(" FAILED")
```

- [ ] **Step 2: Add step controller regression test**

Add a new test (Test 8) that verifies the step controller doesn't grow h excessively for stiff systems:
```mojo
print()
print("[Test 8] Step controller: stiff tridiagonal system")
print(" Verifies h doesn't grow beyond HMAX after quot1/quot2 fix")
print()

var n8 = 10
var M8 = make_identity_csr(n8)
var K8 = make_tridiag_csr(n8, -5.0, 10.0, -5.0)
var y0_8: List[Float64] = []
for i in range(n8):
    y0_8.append(1.0)

var sys8 = SimpleLinearSystem(M8^, K8^)
var solver8 = RadauSparseLinearSolver[SimpleLinearSystem](
    rtol=1e-6, atol=1e-8, max_step=0.0, first_step=0.001,
)
var sol8 = solver8.solve(sys8^, (0.0, 0.1), y0_8)

if not sol8.success:
    print(" FAILED: " + sol8.message)
else:
    print(" Steps: " + String(len(sol8.t)))
    var all_positive = True
    for i in range(n8):
        if sol8.y[len(sol8.y) - 1][i] < 0.0:
            all_positive = False
    var step_count_ok = len(sol8.t) > 1 and len(sol8.t) <= 200
    if all_positive and step_count_ok:
        print(" PASSED")
    else:
        print(" FAILED (steps=" + String(len(sol8.t)) + ")")
```

- [ ] **Step 3: Run all tests**

Run: `pixi run mojo test -I src tests/test_radau.mojo`

Expected: All 8 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_radau.mojo
git commit -m "test: add t_eval interpolation and step controller regression tests"
```

---

### Task 6: Verify end-to-end with FPE example

**Files:**
- No changes (run only)

- [ ] **Step 1: Build and run the single_price example**

Run: `pixi run mojo build -I src examples/single_price.mojo && pixi run mojo run -I src examples/single_price.mojo`

Expected: Successful run with reasonable output. Step counts may differ from before due to quot1/quot2 fix.

- [ ] **Step 2: Run full test suite**

Run: `pixi run mojo test -I src tests/test_radau.mojo`

Expected: All 8 tests pass.

- [ ] **Step 3: Verify no compiler warnings**

Run: `pixi run mojo build -I src examples/single_price.mojo 2>&1 | grep -i warning`

Expected: No output (0 warnings).
