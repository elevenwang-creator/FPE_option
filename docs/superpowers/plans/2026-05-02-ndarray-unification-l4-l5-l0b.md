# L4-L5-L0b NDArray Unification Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the NDArray unification by migrating FPE engine (L4), NN/NAIS/Calibrator (L5), and deleting `_list` exports (L0b).

**Architecture:** Bottom-up migration: first fix broken FPE files (objective.mojo, pdf.mojo), then merge StableLinear duplicate structs, then migrate calibrator to NDArray LM traits, then migrate NAIS engine (nais_net, volterra, trainer), finally delete `_list` exports from `__init__.mojo`.

**Tech Stack:** Mojo v0.26.3, pixi, NDArray[Float64]

---

## Dependency Graph

```
L4a (objective.mojo fix) ──→ L4b (pdf.mojo fix) ──→ L4c (test_fpe_engine.mojo fix) ──→ L4d (test_calibrator.mojo fix)
                                    │
L4e (domain.mojo NDArray) ──→ L4f (galerkin.mojo cleanup) ──→ L4g (DiagMatrix List bridge removal)
                                                              │
L5a (StableLinear merge) ──→ L5b (calibrator.mojo NDArray) ──→ L5c (nais_net.mojo NDArray)
                                                              │
                                              L5d (volterra.mojo NDArray) ──→ L5e (trainer.mojo NDArray)
                                                                                    │
                                                                              L0b (delete _list exports)
```

## Key Constraints

- NDArray[Float64] is not `ImplicitlyCopyable` — use `^` for move, `.copy()` for explicit copy
- `NDArray` requires `n > 0` in constructors — no zero-length arrays
- `NDArray.shape()` returns `IntTuple` — use `Int(arr.shape()[dim])`
- `len(ndarray)` does NOT work — use `Int(arr.shape()[0])` or `arr.len()`
- 2D NDArray access: `arr[row, col]`; row extraction: `arr.row(idx)` (returns 1D NDArray view)
- `CSRMatrix.spmv(x, mut y)` zero-inits y; `spmv_into` accumulates
- `scale` requires explicit dtype: `scale[DType.float64](alpha, M)`
- `diag_scale` requires explicit dtype: `diag_scale[DType.float64](M, Dinv, Dinv)`
- Mojo: `def` not `fn`, `var` not `let`, `comptime` not `alias`, `^` for move
- Run tests: `pixi run mojo run -I src tests/test_foo.mojo`

---

### Task 1: Fix objective.mojo — add map_s/v_to_physical + NDArray API

**Files:**
- Modify: `src/engines/fpe/domain.mojo` — add `map_s_to_physical()` and `map_v_to_physical()` methods
- Modify: `src/engines/calibrator/objective.mojo` — fix for NDArray solver output + new domain methods

**Context:** `objective.mojo` line 34 calls `domain.map_s_to_physical()` and line 46 calls `domain.map_v_to_physical()` — neither method exists on `FPEDomain`. Also, `solver.solve()` now returns `NDArray[Float64]` (2D, shape `(nt, n)`), but `objective.mojo` line 94 treats `q_path` as if `len()` works on it and accesses `q_path[len(q_path)-1]` as a flat vector.

**Step 1: Add physical mapping methods to FPEDomain**

In `src/engines/fpe/domain.mojo`, add two methods after `jacobian_factor()`:

```mojo
def map_s_to_physical(self, s_normalized: Float64) -> Float64:
    return self.s_min + s_normalized * (self.s_max - self.s_min)

def map_v_to_physical(self, v_normalized: Float64) -> Float64:
    return self.v_min + v_normalized * (self.v_max - self.v_min)
```

- [ ] **Step 2: Rewrite objective.mojo for NDArray API**

The key changes in `src/engines/calibrator/objective.mojo`:
1. `solver.solve()` returns `NDArray[Float64]` (2D shape `(nt, n)`) — use `sol.row(nt-1)` to get last time step as 1D NDArray
2. `PDFComputer.compute()` needs NDArray input — change `q_t` parameter to `NDArray[Float64]`
3. `q_path` is now `NDArray[Float64]` — use `Int(q_path.shape()[0])` instead of `len(q_path)`
4. Add `from numerics.utils.ndarray import NDArray` import
5. Fix `FPESolver` constructor: add `first_step=1e-4` parameter
6. Change `_integrate_call_price` to accept `NDArray[Float64]` for `pdf` instead of `List[List[Float64]]`

Rewrite `objective.mojo`:
```mojo
from numerics.utils import max_f64
from numerics.utils.ndarray import NDArray
from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.pdf import PDFComputer
from engines.fpe.solver import FPESolver


def _with_maturity(base: HestonParams, maturity: Float64) -> HestonParams:
    return HestonParams(
        kappa=base.kappa, theta=base.theta, sigma=base.sigma, rho=base.rho,
        r=base.r, T=maturity, S0=base.S0, V0=base.V0,
        S_min=base.S_min, S_max=base.S_max, V_min=base.V_min, V_max=base.V_max,
    )


def _integrate_call_price(
    domain: FPEDomain, pdf: NDArray[Float64], strike: Float64
) -> Float64:
    var price = 0.0
    var n_s = len(domain.s_points)
    var n_v = len(domain.v_points)

    for i in range(n_s):
        var s_phys = domain.map_s_to_physical(domain.s_points[i])
        var ds = 1.0
        if i > 0 and i < n_s - 1:
            var s_prev = domain.map_s_to_physical(domain.s_points[i - 1])
            var s_next = domain.map_s_to_physical(domain.s_points[i + 1])
            ds = 0.5 * (s_next - s_prev)

        var payoff = max_f64(s_phys - strike, 0.0)
        for j in range(n_v):
            var dv = 1.0
            if j > 0 and j < n_v - 1:
                var v_prev = domain.map_v_to_physical(domain.v_points[j - 1])
                var v_next = domain.map_v_to_physical(domain.v_points[j + 1])
                dv = 0.5 * (v_next - v_prev)
            price += payoff * pdf[i * n_v + j] * ds * dv

    return price


struct ObjectiveFunction[B: Int](Copyable, Movable):
    var market_prices: NDArray[Float64]
    var strikes: NDArray[Float64]
    var expiries: NDArray[Float64]

    def __init__(
        out self,
        var market_prices: NDArray[Float64],
        var strikes: NDArray[Float64],
        var expiries: NDArray[Float64],
    ):
        self.market_prices = market_prices^
        self.strikes = strikes^
        self.expiries = expiries^

    def __init__(out self, *, copy: Self):
        self.market_prices = copy.market_prices.copy()
        self.strikes = copy.strikes.copy()
        self.expiries = copy.expiries.copy()

    def compute(self, params: HestonParams) raises -> NDArray[Float64]:
        var n_opts = Int(self.market_prices.shape()[0])
        var residuals = NDArray[Float64](n_opts)
        var solver = FPESolver[Self.B](rtol=1e-5, atol=1e-7, max_step=0.02, first_step=1e-4)
        var pdf_comp = PDFComputer[Self.B]()

        for i in range(n_opts):
            var maturity = self.expiries[i]
            var local_params = _with_maturity(params, maturity)
            var domain = FPEDomain[3, 3](local_params, n_s=8, n_v=8)
            var sol = solver.solve(domain, local_params, [0.0, maturity])
            var nt = Int(sol.shape()[0])
            var q_terminal = sol.row(nt - 1).copy()
            var pdf = pdf_comp.compute(domain, q_terminal)
            var model_price = _integrate_call_price(domain, pdf, self.strikes[i])
            residuals[i] = model_price - self.market_prices[i]

        return residuals^
```

- [ ] **Step 3: Run test_calibrator.mojo to verify compilation**

Run: `pixi run mojo run -I src tests/test_calibrator.mojo 2>&1 | head -20`

Expected: Should compile (may still fail at runtime due to `PDFComputer.compute` not yet updated)

---

### Task 2: Fix pdf.mojo — convert to NDArray I/O

**Files:**
- Modify: `src/engines/fpe/pdf.mojo`

**Context:** `PDFComputer.compute()` currently takes `q_t: List[Float64]` and returns `List[List[Float64]]`. `objective.mojo` now calls it with `NDArray[Float64]` and expects `NDArray[Float64]` back. Also, `CSRMatrix.spmv` expects `NDArray[Float64]` input, not `List[Float64]`.

**Step 1: Rewrite pdf.mojo**

```mojo
from engines.fpe.domain import FPEDomain
from numerics.utils.ndarray import NDArray


struct PDFComputer[B: Int]:
    def __init__(out self):
        pass

    def compute(self, domain: FPEDomain, q_t: NDArray[Float64]) -> NDArray[Float64]:
        var basis = domain.build_basis()
        var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
        var n_s = len(domain.s_points)
        var n_v = len(domain.v_points)
        var n_grid = n_s * n_v
        var pdf_flat = NDArray[Float64](n_grid)
        Phi.spmv(q_t, pdf_flat)
        return pdf_flat^
```

Note: Returns a flat `NDArray[Float64]` of size `n_s * n_v`. The caller (`objective.mojo`) indexes as `pdf[i * n_v + j]`.

- [ ] **Step 2: Verify test_fpe_engine.mojo compiles after changes**

Run: `pixi run mojo run -I src tests/test_fpe_engine.mojo 2>&1 | head -20`

Expected: Still needs test file fix (Task 3), but source code should compile.

---

### Task 3: Fix test_fpe_engine.mojo

**Files:**
- Modify: `tests/test_fpe_engine.mojo`

**Context:** Test is broken because:
1. `len(q0)` doesn't work on `NDArray[Float64]` — need `Int(q0.shape()[0])` or `q0.len()`
2. `FPESolver` now requires `first_step` parameter
3. `solver.solve()` returns 2D `NDArray[Float64]` — `len(sol)` doesn't work, indexing as `sol[i]` doesn't work

**Step 1: Rewrite test_fpe_engine.mojo**

```mojo
from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from engines.fpe.solver import FPESolver
from numerics.utils.ndarray import NDArray
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def test_fpe_pipeline_small_grid() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.1,
        S0=60.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    assert_true(params.is_valid())

    var domain = FPEDomain[3, 3](params, n_s=8, n_v=8)
    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)

    assert_true(M.nrows == M.ncols)
    assert_true(M.nrows > 0)

    var q0 = InitialCondition[1]().compute(domain, params)
    var q0_sum = 0.0
    var n_q0 = Int(q0.shape()[0])
    for i in range(n_q0):
        q0_sum += q0[i]
    assert_true(_abs(q0_sum - 1.0) < 1e-2)

    var solver = FPESolver[1](rtol=1e-6, atol=1e-8, max_step=0.02, first_step=1e-4)
    var sol = solver.solve(domain, params, [0.0, 0.1])
    var nt = Int(sol.shape()[0])
    assert_true(nt >= 1)

    var last_row = sol.row(nt - 1)
    var n_last = Int(last_row.shape()[0])
    for i in range(n_last):
        assert_true(last_row[i] >= -1e-10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run the test**

Run: `pixi run mojo run -I src tests/test_fpe_engine.mojo 2>&1 | tail -10`

Expected: PASS (compiles + runs)

---

### Task 4: Fix test_calibrator.mojo for NDArray API

**Files:**
- Modify: `tests/test_calibrator.mojo`

**Context:** `ObjectiveFunction.__init__` now takes `NDArray[Float64]` args. `CalibratorResidual.__call__` and `CalibratorJacobian.__call__` still take `List[Float64]` because LM traits are NDArray-based. Need to update `CalibratorResidual`/`CalibratorJacobian` to match NDArray LM traits.

**IMPORTANT:** The LM optimizer traits (`ResidualCallable`, `JacobianCallable`) already use NDArray. But `calibrator.mojo` still uses `List[Float64]` for `CalibratorResidual.__call__` and `CalibratorJacobian.__call__`. These must be updated to NDArray to match the LM traits.

**Step 1: Rewrite calibrator.mojo for NDArray API**

```mojo
from engines.calibrator.objective import ObjectiveFunction
from engines.fpe.heston_params import HestonParams
from numerics.optim.lm import LevenbergMarquardt, ResidualCallable, JacobianCallable
from numerics.utils import abs_f64, max_f64, min_f64
from numerics.utils.ndarray import NDArray
from std.algorithm import parallelize


def _params_to_vec(p: HestonParams) -> NDArray[Float64]:
    var x = NDArray[Float64](5)
    x[0] = p.kappa
    x[1] = p.theta
    x[2] = p.sigma
    x[3] = p.rho
    x[4] = p.V0
    return x^


def _vec_to_params(x: NDArray[Float64], base: HestonParams) -> HestonParams:
    return HestonParams(
        kappa=max_f64(x[0], 1e-4),
        theta=max_f64(x[1], 1e-5),
        sigma=max_f64(x[2], 1e-4),
        rho=min_f64(0.999, max_f64(-0.999, x[3])),
        r=base.r,
        T=base.T,
        S0=base.S0,
        V0=max_f64(x[4], 1e-6),
        S_min=base.S_min,
        S_max=base.S_max,
        V_min=base.V_min,
        V_max=base.V_max,
    )


struct CalibratorResidual[B: Int](ResidualCallable):
    var obj: ObjectiveFunction[Self.B]
    var base: HestonParams

    def __init__(out self, obj: ObjectiveFunction[Self.B], base: HestonParams):
        self.obj = obj.copy()
        self.base = base.copy()

    def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]:
        return self.obj.compute(_vec_to_params(x, self.base))


struct CalibratorJacobian[B: Int](JacobianCallable):
    var obj: ObjectiveFunction[Self.B]
    var base: HestonParams

    def __init__(out self, obj: ObjectiveFunction[Self.B], base: HestonParams):
        self.obj = obj.copy()
        self.base = base.copy()

    def __call__(self, x: NDArray[Float64]) raises -> NDArray[Float64]:
        var r = self.obj.compute(_vec_to_params(x, self.base))
        var m = Int(r.shape()[0])
        var n = Int(x.shape()[0])
        var J = NDArray[Float64](m, n)
        for j in range(n):
            var eps = 1e-6 * (1.0 + abs_f64(x[j]))
            var xp = x.copy()
            var xm = x.copy()
            xp[j] = xp[j] + eps
            xm[j] = xm[j] - eps
            var rp = self.obj.compute(_vec_to_params(xp, self.base))
            var rm = self.obj.compute(_vec_to_params(xm, self.base))
            for i in range(m):
                J[i, j] = (rp[i] - rm[i]) / (2.0 * eps)
        return J^


@fieldwise_init
struct Calibrator[B: Int]:
    var max_iter: Int
    var tol: Float64

    def calibrate(
        self,
        var market_prices: NDArray[Float64],
        var strikes: NDArray[Float64],
        var expiries: NDArray[Float64],
        init_params: HestonParams,
    ) raises -> HestonParams:
        var lm = LevenbergMarquardt(
            max_iter=self.max_iter, tol=self.tol,
            lambda_init=1e-3, lambda_up=10.0, lambda_down=0.1,
        )
        var x = _params_to_vec(init_params)
        var residual = CalibratorResidual[Self.B](
            obj=ObjectiveFunction[Self.B](market_prices^, strikes^, expiries^),
            base=init_params,
        )
        var jacobian = CalibratorJacobian[Self.B](
            obj=ObjectiveFunction[Self.B](market_prices^, strikes^, expiries^),
            base=init_params,
        )
        var x_opt = lm.solve(residual, jacobian, x)
        return _vec_to_params(x_opt, init_params)

    def calibrate_batch(
        self,
        market_prices_list: List[NDArray[Float64]],
        strikes_list: List[NDArray[Float64]],
        expiries_list: List[NDArray[Float64]],
        init_params_list: List[HestonParams],
    ) raises -> List[HestonParams]:
        var batch_size = len(init_params_list)
        var results: List[HestonParams] = []
        for b in range(batch_size):
            var result = self.calibrate(
                market_prices_list[b]^, strikes_list[b]^,
                expiries_list[b]^, init_params_list[b],
            )
            results.append(result^)
        return results^
```

**Step 2: Update test_calibrator.mojo**

The test creates `ObjectiveFunction[1]` with `List[Float64]` args — must now use `NDArray[Float64]`.

Rewrite test:
```mojo
from engines.calibrator.calibrator import Calibrator
from engines.calibrator.objective import ObjectiveFunction
from engines.fpe.heston_params import HestonParams
from numerics.utils.ndarray import NDArray
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _rel_err(a: Float64, b: Float64) -> Float64:
    var denom = _abs(b)
    if denom < 1e-8:
        denom = 1.0
    return _abs(a - b) / denom


def _list_to_ndarr(xs: List[Float64]) -> NDArray[Float64]:
    var n = len(xs)
    var arr = NDArray[Float64](n)
    for i in range(n):
        arr[i] = xs[i]
    return arr^


def test_calibrator_converges_on_synthetic_market() raises:
    var true_params = HestonParams(
        kappa=1.1, theta=0.06, sigma=0.3, rho=-0.35, r=0.02, T=0.2,
        S0=100.0, V0=0.05, S_min=40.0, S_max=180.0, V_min=0.0, V_max=1.0,
    )
    var strikes = _list_to_ndarr([90.0, 100.0, 110.0])
    var expiries = _list_to_ndarr([0.1, 0.15, 0.2])
    var zero_market = _list_to_ndarr([0.0, 0.0, 0.0])

    var obj_zero = ObjectiveFunction[1](zero_market.copy(), strikes.copy(), expiries.copy())
    var market_prices = obj_zero.compute(true_params)

    var init_params = HestonParams(
        kappa=true_params.kappa * 1.01, theta=true_params.theta * 0.99,
        sigma=true_params.sigma * 1.01, rho=true_params.rho * 0.99,
        r=true_params.r, T=true_params.T, S0=true_params.S0,
        V0=true_params.V0 * 1.01, S_min=true_params.S_min,
        S_max=true_params.S_max, V_min=true_params.V_min, V_max=true_params.V_max,
    )

    var calibrator = Calibrator[1](max_iter=20, tol=1e-6)
    var fitted = calibrator.calibrate(market_prices, strikes, expiries, init_params)

    assert_true(_rel_err(fitted.kappa, true_params.kappa) < 0.10)
    assert_true(_rel_err(fitted.theta, true_params.theta) < 0.10)
    assert_true(_rel_err(fitted.sigma, true_params.sigma) < 0.10)
    assert_true(_rel_err(fitted.rho, true_params.rho) < 0.10)
    assert_true(_rel_err(fitted.V0, true_params.V0) < 0.10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 3: Run test**

Run: `pixi run mojo run -I src tests/test_calibrator.mojo 2>&1 | tail -10`

Expected: Compiles. May take a long time or fail at runtime (FPE solve + calibration is expensive).

---

### Task 5: Migrate domain.mojo and galerkin.mojo — remove _list_to_ndarr bridges

**Files:**
- Modify: `src/engines/fpe/domain.mojo` — convert `s_points`, `v_points`, `s_weights`, `v_weights`, `s_points_phys`, `v_points_phys` from `List[Float64]` to `NDArray[Float64]`
- Modify: `src/engines/fpe/galerkin.mojo` — remove `_list_to_ndarr` bridge, use NDArray directly

**Context:** `domain.mojo` stores grid data as `List[Float64]` but has to convert to `NDArray[Float64]` via `_list_to_ndarr` in `integ_weights()`. Similarly, `galerkin.mojo` uses `_list_to_ndarr` to convert `s`, `v`, `s_sq` before passing to `DiagMatrix`. If domain fields were NDArray, these bridges would be unnecessary.

**IMPORTANT:** The B-spline code (`knots.mojo`, `basis.mojo`, `recombination.mojo`, `tensor_product.mojo`) still uses `List[Float64]` for its I/O. These are deeply interconnected and use `List.append()` for dynamic construction. Migrating them is a separate, larger task. For now, we keep `s_knots`/`v_knots` as `List[Float64]` (needed by B-spline), but convert the computed fields to `NDArray[Float64]`.

**Step 1: Update FPEDomain fields to NDArray**

In `domain.mojo`, change:
- `s_points: List[Float64]` → `s_points: NDArray[Float64]`
- `v_points: List[Float64]` → `v_points: NDArray[Float64]`
- `s_weights: List[Float64]` → `s_weights: NDArray[Float64]`
- `v_weights: List[Float64]` → `v_weights: NDArray[Float64]`
- `s_points_phys: List[Float64]` → `s_points_phys: NDArray[Float64]`
- `v_points_phys: List[Float64]` → `v_points_phys: NDArray[Float64]`

Keep `s_knots`/`v_knots` as `List[Float64]`.

After computing `s_points` as `List[Float64]` from `_compute_quad_points`, convert:
```mojo
var s_points_list = _compute_quad_points(grid_s, num_gauss)
var s_weights_list = _compute_quad_weights(grid_s, num_gauss)
var v_points_list = _compute_quad_points(grid_v, num_gauss)
var v_weights_list = _compute_quad_weights(grid_v, num_gauss)

var jacobian_s = self.s_max - self.s_min
for i in range(len(s_weights_list)):
    s_weights_list[i] = s_weights_list[i] * jacobian_s

self.s_points = _list_to_ndarr(s_points_list)
self.s_weights = _list_to_ndarr(s_weights_list)
self.v_points = _list_to_ndarr(v_points_list)
self.v_weights = _list_to_ndarr(v_weights_list)

var n_s = Int(self.s_points.shape()[0])
var n_v = Int(self.v_points.shape()[0])
self.s_points_phys = NDArray[Float64](n_s)
for i in range(n_s):
    self.s_points_phys[i] = self.s_min + self.s_points[i] * (self.s_max - self.s_min)
self.v_points_phys = NDArray[Float64](n_v)
for i in range(n_v):
    self.v_points_phys[i] = self.v_min + self.v_points[i] * (self.v_max - self.v_min)
```

Remove `_list_to_ndarr` calls from `integ_weights()`:
```mojo
def integ_weights(self) -> CSRMatrix[DType.float64]:
    var sw_diag = DiagMatrix(self.s_weights.copy()).to_csr()
    var vw_diag = DiagMatrix(self.v_weights.copy()).to_csr()
    return kron(sw_diag, vw_diag)
```

Remove `_list_to_ndarr` function from domain.mojo (keep it if still used by `_grid_create` — check).

Actually: `_list_to_ndarr` is still needed internally to convert the `List[Float64]` outputs from `_compute_quad_points`/`_compute_quad_weights` into NDArray. Keep it as a private helper.

**Step 2: Update galerkin.mojo to remove _list_to_ndarr**

In `galerkin.mojo`, the `_list_to_ndarr` calls can be removed since `domain.s_points_phys` etc. are now `NDArray[Float64]`:

```mojo
var s = domain.s_points_phys.copy()
var v = domain.v_points_phys.copy()
var s_sq = NDArray[Float64](Int(s.shape()[0]))
for i in range(Int(s.shape()[0])):
    s_sq[i] = s[i] * s[i]

var s_diag = kron(DiagMatrix(s^).to_csr(), identity_csr(n_v))
var v_diag = kron(identity_csr(n_s), DiagMatrix(v^).to_csr())
var s_sq_diag = kron(DiagMatrix(s_sq^).to_csr(), identity_csr(n_v))
```

Remove the `_list_to_ndarr` function from galerkin.mojo.

**IMPORTANT:** `tensor_product.mojo` `eval_tensor()` and `partial_s/v()` still take `List[Float64]` args. We need to either:
a) Pass `domain.s_points.to_list()` / `domain.v_points.to_list()` at call sites (bridge)
b) Migrate tensor_product.mojo too (larger scope)

For now, use option (a) — call `.to_list()` at the 3 call sites in galerkin.mojo and initial_cond.mojo.

**Step 3: Verify compilation**

Run: `pixi run mojo run -I src tests/test_fpe_engine.mojo 2>&1 | tail -5`

Expected: Compiles + PASS

---

### Task 6: Remove DiagMatrix List[Float64] bridge constructor

**Files:**
- Modify: `src/sparse/diag.mojo` — remove `__init__(diag: List[Float64])`

**Context:** After Task 5, no code should be calling `DiagMatrix(List[Float64])`. All callers use `DiagMatrix(NDArray[Float64])` or `DiagMatrix(n: Int)`.

**Step 1: Search for remaining DiagMatrix(List) call sites**

Run: `rg "DiagMatrix\(" src/` to find all call sites.

**Step 2: Remove the List constructor from diag.mojo**

Remove lines 22-26:
```mojo
def __init__(out self, diag: List[Float64]):
    self.size = len(diag)
    self.values = NDArray[Float64](self.size)
    for i in range(self.size):
        self.values[i] = diag[i]
```

**Step 3: Verify all tests still pass**

Run: `pixi run mojo run -I src tests/test_sparse_coo_diag.mojo && pixi run mojo run -I src tests/test_sparse.mojo`

Expected: All pass

---

### Task 7: Merge StableLinear + StableLinearND into single NDArray-primary struct

**Files:**
- Modify: `src/numerics/nn/stable_linear.mojo` — delete `StableLinear` (List) and `make_stable_linear`, rename `StableLinearND` → `StableLinear`, `make_stable_linear_nd` → `make_stable_linear`
- Modify: `src/numerics/nn/__init__.mojo` — remove old exports
- Modify: `src/engines/nais/nais_net.mojo` — update to NDArray `StableLinear`
- Modify: `tests/test_nais_engine.mojo` — update to NDArray `StableLinear`

**Context:** `StableLinear` (List-based) is used by `nais_net.mojo`. `StableLinearND` (NDArray-based) is the NDArray duplicate. We need to merge them: make `StableLinear` NDArray-based and update `nais_net.mojo`.

**IMPORTANT:** This is the most complex task because `nais_net.mojo` heavily uses `List[Float64]` for weights, biases, and forward pass. Migrating nais_net.mojo to NDArray means rewriting most of it.

**Strategy:** First migrate `StableLinear` itself, then update `nais_net.mojo` to use the NDArray version. The NAIS net stores `List[List[Float64]]` for weights — these need to become `NDArray[Float64]` (2D).

**Step 1: Rewrite stable_linear.mojo — single NDArray-primary StableLinear**

Delete lines 1-135 (old List-based `StableLinear`, `_matmul_vec`, `make_stable_linear`).

Rename `StableLinearND` → `StableLinear`, `make_stable_linear_nd` → `make_stable_linear`.

Remove `zeros_list`/`zeros_mat_list` imports — they're no longer used.

**Step 2: Update nn/__init__.mojo**

```mojo
from numerics.nn.stable_linear import StableLinear, make_stable_linear
from numerics.nn.autograd import GradientTape
from numerics.nn.adam import Adam
```

**Step 3: Rewrite nais_net.mojo for NDArray**

This is the largest rewrite. Key changes:
- All `List[List[Float64]]` weights → `NDArray[Float64]` (2D)
- All `List[Float64]` biases → `NDArray[Float64]` (1D)
- `_make_weights` returns `NDArray[Float64]` (2D)
- `_linear(W, b, x)` takes NDArrays
- `_sin_vec(x)` takes NDArray
- `_add_vec(a, b)` takes NDArray
- `forward(t, x)` takes/returns NDArray
- `forward_tracked` can keep `List[Int]` for tape indices (tape internals stay List)

**Step 4: Update test_nais_engine.mojo**

Change `StableLinear` construction to use NDArray:
```mojo
var W = NDArray[Float64](3, 3)
W[0,0] = 0.1; W[0,1] = -0.2; W[0,2] = 0.3
...
var b = NDArray[Float64](3)
b[0] = 0.0; b[1] = 0.1; b[2] = -0.1
var layer = StableLinear(W=W^, b=b^, epsilon=0.01)
var y = layer.forward(x)
```

**Step 5: Run test**

Run: `pixi run mojo run -I src tests/test_nais_engine.mojo 2>&1 | tail -10`

Expected: PASS

---

### Task 8: Migrate volterra.mojo to NDArray

**Files:**
- Modify: `src/engines/nais/volterra.mojo`

**Context:** `volterra.mojo` uses `List[Float64]` and `zeros_list`/`zeros_3d_list`. Need to convert to NDArray. The 3D data (`List[List[List[Float64]]]`) needs to become `NDArray[Float64]` with 3D shape.

**IMPORTANT:** NDArray currently supports 1D and 2D. For 3D, we need to either:
a) Use a 1D NDArray with manual 3D indexing: `arr[i * d1 * d2 + j * d2 + k]`
b) Add 3D support to NDArray

For now, use option (a) — flat 1D NDArray with manual 3D indexing.

**Step 1: Rewrite volterra.mojo**

Replace `zeros_list`/`zeros_3d_list` with NDArray equivalents. Use `NDArray[Float64]` flat arrays with manual indexing for 3D data.

**Step 2: Update test_nais_engine.mojo volterra test**

- [ ] **Step 3: Run test**

Run: `pixi run mojo run -I src tests/test_nais_engine.mojo 2>&1 | tail -10`

---

### Task 9: Migrate trainer.mojo to NDArray

**Files:**
- Modify: `src/engines/nais/trainer.mojo`

**Context:** `trainer.mojo` uses `linspace_list` and `List[Float64]` heavily. After nais_net.mojo is NDArray, trainer.mojo must also be NDArray.

**Step 1: Rewrite trainer.mojo**

Replace `linspace_list` with `linspace` (if available) or build NDArray directly. Update `_flatten_net_params`/`_unflatten_net_params` to use NDArray.

**Step 2: Run test**

Run: `pixi run mojo run -I src tests/test_nais_engine.mojo 2>&1 | tail -10`

---

### Task 10: Delete _list exports from __init__.mojo (L0b)

**Files:**
- Modify: `src/numerics/utils/__init__.mojo` — remove `_list` function exports
- Modify: `src/numerics/__init__.mojo` — remove `_list` re-exports
- Modify: `src/numerics/utils/constructors.mojo` — delete `zeros_list`, `zeros_mat_list`, `zeros_3d_list`, `linspace_list`
- Modify: `src/numerics/utils/copy.mojo` — delete `copy_vec_list`, `copy_mat_list`, `swap_rows_list`

**Context:** After all consumers are migrated to NDArray, the `_list` helper functions are dead code.

**Step 1: Search for any remaining _list function usage**

Run: `rg "zeros_list|zeros_mat_list|zeros_3d_list|copy_vec_list|copy_mat_list|swap_rows_list|linspace_list" src/`

Expected: 0 matches (all migrated)

**Step 2: Delete _list functions from constructors.mojo and copy.mojo**

**Step 3: Remove _list exports from __init__.mojo files**

**Step 4: Run all tests**

Run: `pixi run mojo run -I src tests/test_sparse.mojo && pixi run mojo run -I src tests/test_linalg.mojo && pixi run mojo run -I src tests/test_ndarray.mojo && pixi run mojo run -I src tests/test_sparse_coo_diag.mojo && pixi run mojo run -I src tests/test_sparse_lu.mojo && pixi run mojo run -I src tests/test_optim.mojo && pixi run mojo run -I src tests/test_rk45.mojo && pixi run mojo run -I src tests/test_radau_optimized.mojo`

Expected: All pass

---

## Post-Completion Verification

After all tasks, run the full test suite:
```bash
for t in test_sparse test_linalg test_ndarray test_sparse_coo_diag test_sparse_lu test_optim test_rk45 test_radau_optimized test_fpe_engine test_nais_engine; do
    echo "=== $t ===" && pixi run mojo run -I src tests/${t}.mojo 2>&1 | tail -3
done
```

Also verify zero `List[Float64]` remains in `src/` (except B-spline internals and tape internals):
```bash
rg "List\[Float64\]" src/ --glob "*.mojo" | grep -v "knots.mojo" | grep -v "basis.mojo" | grep -v "recombination.mojo" | grep -v "tensor_product.mojo" | grep -v "autograd.mojo" | grep -v "coo.mojo" | grep -v "to_list"
```
