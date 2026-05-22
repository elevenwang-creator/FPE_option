# Radau IIA Step Controller Fix + t_eval Dense Output

## Problem 1: t_eval has no effect

### Root Cause

`RadauSparseLinearSolver.solve()` accepts `t_eval: Optional[List[Float64]]`
but never reads it. Every accepted step is recorded into `t_values`/`y_values`
(radau.mojo:499-500), regardless of `t_eval`.

### Fix: Dense Output Interpolation (SciPy-style)

The solver steps freely with adaptive h. When the next `t_eval[i]` falls
within the current step `[t_old, t]`, we clamp h to land on `t_eval[i]`,
then record the solution. For multiple `t_eval` points within a single step,
use the CONT collocation polynomial for interpolation.

#### CONT Polynomial

Hairer's `CONTR5` function interpolates the solution within a completed step:

```
s = (x - x_sol) / h_sol
y_i(x) = CONT[i] + s * (CONT[i+n] + (s - C2M1) * (CONT[i+2n] + (s - C1M1) * CONT[i+3n]))
```

where the coefficients are computed from the Radau internal stage vectors:

```
CONT[i+n]   = (Z2[i] - Z3[i]) / C2M1
AK          = (Z1[i] - Z2[i]) / C1MC2
ACONT3      = Z1[i] / C1
ACONT3      = (AK - ACONT3) / C2
CONT[i+2n]  = (AK - CONT[i+n]) / C1M1
CONT[i+3n]  = CONT[i+2n] - ACONT3
```

Currently, only `CONT = DD1*Z1 + DD2*Z2 + DD3*Z3` is computed (line 458).
We need to store the full 4n collocation coefficients per accepted step.

#### Implementation

1. Replace the single `FixedSizeVector(n)` CONT with a
   `FixedSizeVector(4*n)` that stores all 4 coefficient blocks.
2. Add `contr5(i, s, cont, n)` function matching Hairer's `CONTR5`.
3. In the main loop, after each accepted step:
   - If `t_eval` is provided and `t_eval[next_idx]` falls in `(t_old, t]`,
     interpolate using `contr5` and append to output.
   - If `t_eval[next_idx] == t`, use the accepted step value directly.
4. Clamp h to the next `t_eval` boundary before stepping (similar to the
   existing `t1` clamping at line 269-270).

## Problem 2: Step size grows when grid gets finer

### Root Cause

After comparing our implementation with Hairer's Fortran RADAU5/RADCOR,
three bugs were found in the step controller:

#### Bug 1: Missing HOPT tracking

The Fortran stores `HOPT=HNEW` then `HOPT=MIN(H, HNEW)` before rejection
logic. Our code sets `h_new` but never computes `h_opt`. Without `h_opt`,
the step size recovery after rejection doesn't match the Fortran logic,
leading to incorrect step size selection.

Fortran (after accepted step):
```fortran
HNEW=POSNEG*MIN(ABS(HNEW),HMAXN)
HOPT=HNEW
HOPT=MIN(H,HNEW)
IF (REJECT) HNEW=POSNEG*MIN(ABS(HNEW),ABS(H))
REJECT=.FALSE.
```

Our code (line 514-520):
```mojo
h_new = posneg * min_f64(abs_f64(h_new), abs_f64(t1 - t))
if self.max_step > 0.0:
    h_new = posneg * min_f64(abs_f64(h_new), self.max_step)
if reject:
    h_new = posneg * min_f64(abs_f64(h_new), abs_f64(h))
reject = False
h = h_new
```

Missing: `h_opt = min_f64(h, h_new)` before rejection check.

#### Bug 2: Incorrect rejection fallback

After rejection, the Fortran does:
```fortran
IF (FIRST) THEN
    H=H*0.1D0
ELSE
    HHFAC=HNEW/H
    H=HNEW
END IF
```

Our code (line 522-528):
```mojo
if first:
    h = h * 0.1
else:
    h = h_new
```

This matches, but the `HHFAC` variable is missing. `HHFAC` is used in the
Fortran for index-2/3 DAE scaling (which we don't have), so this is not
a bug per se, but it's a divergence to track.

#### Bug 3: Overly wide QUOT1/QUOT2 dead band

The Fortran defaults are `QUOT1=1.0, QUOT2=1.2`, meaning the step size is
NOT changed if `1.0 < HNEW/HOLD < 1.2`. This prevents unnecessary LU
refactorizations.

Our code uses `quot1=0.5, quot2=2.5`, which is far too wide. When h changes
by up to 2.5x without triggering LU refactorization, the stale
factorization doesn't reflect the actual system dynamics. For stiff FPE
systems where the spectral radius grows with grid refinement, this causes
the error estimate to be too optimistic, allowing h to grow unchecked.

The fix: change `quot1=1.0, quot2=1.2` to match the Fortran defaults.

Additionally, the Fortran checks `IF (THETA.LE.THET.AND.QT.GE.QUOT1.AND.QT.LE.QUOT2) GOTO 30`
to skip Jacobian recomputation AND reuse the LU when h hasn't changed
significantly. Our code at line 281-295 only checks the h ratio for LU
refactorization, missing the `theta <= thet` condition for Jacobian reuse.

### Fix Summary

1. Add `h_opt` variable matching the Fortran's `HOPT`
2. Change `quot1` from `0.5` to `1.0`, `quot2` from `2.5` to `1.2`
3. Add `theta <= thet` check in the LU refactorization decision (optional,
   improves performance by avoiding unnecessary Jacobian recomputation)

## Files Changed

- `src/numerics/ode/radau.mojo`: Both fixes
- `src/numerics/ode/types.mojo`: No change needed (ODESolution already
  stores t and y lists)
- `src/engines/fpe/solver.mojo`: Adjust callers if t_eval semantics change

## Backward Compatibility

- When `t_eval=None`, the solver records every accepted step (unchanged behavior)
- When `t_eval` is provided, only the specified time points are returned
- Step controller changes produce different (more correct) h sequences;
  callers that set `max_step`/`first_step` manually may see different step counts
  but more accurate results
