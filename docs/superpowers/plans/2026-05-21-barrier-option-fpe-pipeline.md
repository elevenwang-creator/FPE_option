# Barrier Option FPE Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragmented server layer with a direct FpeParams-to-price pipeline supporting 10 barrier/vanilla option types.

**Architecture:** `FpeParams` wraps `HestonParams` + grid/option params → `PricingEngine.price()` solves FPE, integrates `BarrierPayoff` over all strikes in a single pass, returns `List[PricingResult]`. No cache, no two-step flow. Python/C++ APIs accept flat params.

**Tech Stack:** Mojo (nightly), pixi, RadauIIA ODE solver, B-spline Galerkin FPE

**Spec:** `docs/superpowers/specs/2026-05-21-barrier-option-fpe-pipeline-design.md`

**Build:** `pixi run mojo build -I src <file>`
**Test:** `pixi run mojo run -I src <test_file>`
**Mojo skill:** Use `mojo-syntax` skill for all Mojo code

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/server/option_types.mojo` | Rewrite | `FpeParams` struct (replaces `OptionParams`, `RoughBergomiParams`, `NAISModel`) |
| `src/server/payoffs.mojo` | Rewrite | `Payoff` trait + `BarrierPayoff` struct (replaces 4 old structs) |
| `src/server/greeks.mojo` | Rewrite | `Greeks` struct using `BarrierPayoff`, multi-strike return |
| `src/server/pricing_engine.mojo` | Rewrite | `PDFGrid` (relocated from pdf_cache) + `PricingEngine` with direct FPE solve |
| `src/server/__init__.mojo` | Rewrite | Updated re-exports |
| `src/engines/fpe/domain.mojo` | Modify | Accept `s_left_cond`/`s_right_cond` params |
| `src/server/pricer.mojo` | Delete | Merged into pricing_engine.mojo |
| `src/server/pdf_cache.mojo` | Delete | No cache needed |
| `src/server/interpolator.mojo` | Keep | Retained, not used in v1 |
| `src/server/gpu_pricing_kernels.mojo` | Keep | GPU kernels unchanged for now |
| `src/bindings/python_module.mojo` | Rewrite | Single `py_price` function |
| `src/bindings/c_abi.mojo` | Rewrite | Single `fpe_price` function |
| `cpp/include/fpe_engine.h` | Rewrite | Updated C header |
| `python/examples/backtest.py` | Rewrite | New API example |
| `cpp/examples/live_trading.cpp` | Rewrite | New API example |
| `tests/test_fpe_params.mojo` | Create | FpeParams unit tests |
| `tests/test_barrier_payoff.mojo` | Create | BarrierPayoff unit tests |
| `tests/test_pricing_engine.mojo` | Create | End-to-end pricing tests |
| `tests/test_pricing_server.mojo` | Delete | Uses old Pricer/PricingRequest API |
| `tests/test_pdf_cache_serialization.mojo` | Delete | PDFCache removed |
| `tests/test_safe_serialization.mojo` | Delete | PDFCache removed |
| `tests/test_bindings.mojo` | Modify | Update to new API |

---

### Task 1: FpeParams struct

**Files:**
- Create: `src/server/option_types.mojo`
- Create: `tests/test_fpe_params.mojo`

- [ ] **Step 1: Write FpeParams tests**

```mojo
from server.option_types import FpeParams
from engines.fpe.heston_params import HestonParams
from std.testing import assert_true, assert_equal, TestSuite


def test_fpe_params_vanilla_valid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=0.0, option_type=8, strikes=[100.0])
    assert_true(p.is_valid())


def test_fpe_params_up_barrier_valid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=120.0, option_type=6, strikes=[100.0])
    assert_true(p.is_valid())


def test_fpe_params_down_barrier_valid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=80.0, option_type=0, strikes=[100.0])
    assert_true(p.is_valid())


def test_fpe_params_up_barrier_invalid_below_s0() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=90.0, option_type=6, strikes=[100.0])
    assert_true(not p.is_valid())


def test_fpe_params_vanilla_with_barrier_invalid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=120.0, option_type=8, strikes=[100.0])
    assert_true(not p.is_valid())


def test_revised_heston_up_barrier() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=120.0, option_type=6, strikes=[100.0])
    var revised = p.revised_heston()
    assert_equal(revised.S_max, 120.0)


def test_revised_heston_down_barrier() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=80.0, option_type=0, strikes=[100.0])
    var revised = p.revised_heston()
    assert_equal(revised.S_min, 80.0)


def test_revised_heston_vanilla_unchanged() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=0.0, option_type=8, strikes=[100.0])
    var revised = p.revised_heston()
    assert_equal(revised.S_min, 0.0)
    assert_equal(revised.S_max, 200.0)


def test_s_boundary_conditions_up() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=120.0, option_type=6, strikes=[100.0])
    assert_equal(p.s_left_cond(), "neumann")
    assert_equal(p.s_right_cond(), "dirichlet")


def test_s_boundary_conditions_down() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=80.0, option_type=0, strikes=[100.0])
    assert_equal(p.s_left_cond(), "dirichlet")
    assert_equal(p.s_right_cond(), "neumann")


def test_s_boundary_conditions_vanilla() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=0.0, option_type=8, strikes=[100.0])
    assert_equal(p.s_left_cond(), "dirichlet")
    assert_equal(p.s_right_cond(), "neumann")


def test_multi_strikes() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.5, S0=100.0, V0=0.1,
        S_min=0.0, S_max=200.0, V_min=0.0, V_max=1.0,
    )
    var p = FpeParams(heston=h, n_s=38, n_v=38, barrier=120.0, option_type=6, strikes=[95.0, 100.0, 105.0])
    assert_true(p.is_valid())
    assert_equal(len(p.strikes), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pixi run mojo run -I src tests/test_fpe_params.mojo 2>&1 | tail -5`
Expected: FAIL — `FpeParams` not found

- [ ] **Step 3: Write FpeParams implementation**

```mojo
from engines.fpe.heston_params import HestonParams


@fieldwise_init
struct FpeParams(Copyable, Movable, Writable):
    var heston: HestonParams
    var n_s: Int
    var n_v: Int
    var barrier: Float64
    var option_type: Int
    var strikes: List[Float64]

    def is_valid(self) -> Bool:
        if not self.heston.is_valid():
            return False
        if self.option_type < 0 or self.option_type > 9:
            return False
        if len(self.strikes) == 0:
            return False
        for i in range(len(self.strikes)):
            if self.strikes[i] <= 0.0:
                return False
        if self.option_type <= 3:
            if self.barrier <= 0.0 or self.barrier >= self.heston.S0:
                return False
        elif self.option_type <= 7:
            if self.barrier <= 0.0 or self.barrier <= self.heston.S0:
                return False
        else:
            if self.barrier != 0.0:
                return False
        return True

    def revised_heston(self) -> HestonParams:
        var h = self.heston
        if self.option_type <= 3:
            return HestonParams(
                kappa=h.kappa, theta=h.theta, sigma=h.sigma, rho=h.rho,
                r=h.r, T=h.T, S0=h.S0, V0=h.V0,
                S_min=self.barrier, S_max=h.S_max, V_min=h.V_min, V_max=h.V_max,
            )
        elif self.option_type <= 7:
            return HestonParams(
                kappa=h.kappa, theta=h.theta, sigma=h.sigma, rho=h.rho,
                r=h.r, T=h.T, S0=h.S0, V0=h.V0,
                S_min=h.S_min, S_max=self.barrier, V_min=h.V_min, V_max=h.V_max,
            )
        else:
            return h

    def s_left_cond(self) -> String:
        if self.option_type <= 3:
            return "dirichlet"
        elif self.option_type <= 7:
            return "neumann"
        else:
            return "dirichlet"

    def s_right_cond(self) -> String:
        if self.option_type <= 3:
            return "neumann"
        elif self.option_type <= 7:
            return "dirichlet"
        else:
            return "neumann"
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pixi run mojo run -I src tests/test_fpe_params.mojo`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**
```bash
git add src/server/option_types.mojo tests/test_fpe_params.mojo
git commit -m "feat: add FpeParams struct replacing OptionParams"
```

---

### Task 2: BarrierPayoff struct

**Files:**
- Rewrite: `src/server/payoffs.mojo`
- Create: `tests/test_barrier_payoff.mojo`

- [ ] **Step 1: Write BarrierPayoff tests**

```mojo
from server.payoffs import BarrierPayoff
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def assert_close(a: Float64, b: Float64, tol: Float64 = 1e-12) raises:
    assert_true(_abs(a - b) <= tol, "Expected " + String(b) + " got " + String(a))


def test_down_and_in_call() raises:
    var p = BarrierPayoff(option_type=0, strikes=[100.0], barrier=80.0)
    var vals = p.evaluate(70.0)
    assert_close(vals[0], 0.0)
    vals = p.evaluate(90.0)
    assert_close(vals[0], 0.0)
    vals = p.evaluate(60.0)
    assert_close(vals[0], 0.0)


def test_down_and_out_call() raises:
    var p = BarrierPayoff(option_type=2, strikes=[100.0], barrier=80.0)
    var vals = p.evaluate(120.0)
    assert_close(vals[0], 20.0)
    vals = p.evaluate(70.0)
    assert_close(vals[0], 0.0)


def test_up_and_out_call() raises:
    var p = BarrierPayoff(option_type=6, strikes=[100.0], barrier=120.0)
    var vals = p.evaluate(110.0)
    assert_close(vals[0], 10.0)
    vals = p.evaluate(130.0)
    assert_close(vals[0], 0.0)


def test_up_and_in_put() raises:
    var p = BarrierPayoff(option_type=5, strikes=[100.0], barrier=120.0)
    var vals = p.evaluate(130.0)
    assert_close(vals[0], 0.0)
    vals = p.evaluate(90.0)
    assert_close(vals[0], 0.0)


def test_european_call() raises:
    var p = BarrierPayoff(option_type=8, strikes=[100.0], barrier=0.0)
    var vals = p.evaluate(120.0)
    assert_close(vals[0], 20.0)
    vals = p.evaluate(80.0)
    assert_close(vals[0], 0.0)


def test_european_put() raises:
    var p = BarrierPayoff(option_type=9, strikes=[100.0], barrier=0.0)
    var vals = p.evaluate(80.0)
    assert_close(vals[0], 20.0)
    vals = p.evaluate(120.0)
    assert_close(vals[0], 0.0)


def test_multi_strike() raises:
    var p = BarrierPayoff(option_type=8, strikes=[95.0, 100.0, 105.0], barrier=0.0)
    var vals = p.evaluate(110.0)
    assert_close(vals[0], 15.0)
    assert_close(vals[1], 10.0)
    assert_close(vals[2], 5.0)


def test_multi_strike_barrier() raises:
    var p = BarrierPayoff(option_type=6, strikes=[95.0, 100.0, 105.0], barrier=120.0)
    var vals = p.evaluate(110.0)
    assert_close(vals[0], 15.0)
    assert_close(vals[1], 10.0)
    assert_close(vals[2], 5.0)
    vals = p.evaluate(130.0)
    assert_close(vals[0], 0.0)
    assert_close(vals[1], 0.0)
    assert_close(vals[2], 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pixi run mojo run -I src tests/test_barrier_payoff.mojo 2>&1 | tail -5`
Expected: FAIL — `BarrierPayoff` not found or wrong API

- [ ] **Step 3: Write BarrierPayoff implementation**

```mojo
trait Payoff:
    def __init__(out self):
        ...

    def evaluate(self, S: Float64) -> List[Float64]:
        ...

    def name(self) -> StaticString:
        ...


struct BarrierPayoff(Payoff):
    var option_type: Int
    var strikes: List[Float64]
    var barrier: Float64

    def __init__(out self, option_type: Int, strikes: List[Float64], barrier: Float64):
        self.option_type = option_type
        self.strikes = strikes^
        self.barrier = barrier

    def evaluate(self, S: Float64) -> List[Float64]:
        var is_call = (self.option_type % 2 == 0)
        var active = self._is_active(S)
        var result: List[Float64] = []
        for i in range(len(self.strikes)):
            if not active:
                result.append(0.0)
            elif is_call:
                var val = S - self.strikes[i]
                if val > 0.0:
                    result.append(val)
                else:
                    result.append(0.0)
            else:
                var val = self.strikes[i] - S
                if val > 0.0:
                    result.append(val)
                else:
                    result.append(0.0)
        return result^

    def _is_active(self, S: Float64) -> Bool:
        if self.option_type == 0 or self.option_type == 1:
            return S <= self.barrier
        elif self.option_type == 2 or self.option_type == 3:
            return S > self.barrier
        elif self.option_type == 4 or self.option_type == 5:
            return S >= self.barrier
        elif self.option_type == 6 or self.option_type == 7:
            return S < self.barrier
        else:
            return True

    def name(self) -> StaticString:
        return "BarrierPayoff"
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pixi run mojo run -I src tests/test_barrier_payoff.mojo`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**
```bash
git add src/server/payoffs.mojo tests/test_barrier_payoff.mojo
git commit -m "feat: add BarrierPayoff replacing 4 separate payoff structs"
```

---

### Task 3: FPEDomain boundary condition params

**Files:**
- Modify: `src/engines/fpe/domain.mojo:190-289` (FPEDomain struct + build_basis)
- Modify: `src/engines/fpe/domain.mojo:168-188` (FPECachedBasis — no change needed)

- [ ] **Step 1: Add `s_left_cond` and `s_right_cond` fields and params to FPEDomain**

In `FPEDomain.__init__`, add two `String` params with defaults matching current behavior:
- Add `s_left_cond: String = "dirichlet"` and `s_right_cond: String = "neumann"` params
- Add `var s_left_cond: String` and `var s_right_cond: String` fields
- In `build_basis()`, pass them to `RecombinationBasis[Self.degree_s]` instead of hardcoded values

- [ ] **Step 2: Build to verify no breakage**
Run: `pixi run mojo build -I src examples/single_price.mojo`
Expected: 0 errors

- [ ] **Step 3: Run existing FPE engine test**
Run: `pixi run mojo run -I src tests/test_fpe_engine.mojo`
Expected: PASS (default params preserve existing behavior)

- [ ] **Step 4: Commit**
```bash
git add src/engines/fpe/domain.mojo
git commit -m "feat: add configurable boundary conditions to FPEDomain"
```

---

### Task 4: PricingEngine with direct FPE solve

**Files:**
- Rewrite: `src/server/pricing_engine.mojo`
- Create: `tests/test_pricing_engine.mojo`

This task also deletes `src/server/pricer.mojo` and `src/server/pdf_cache.mojo`, and relocates `PDFGrid` into `pricing_engine.mojo`.

- [ ] **Step 1: Write pricing engine test**

```mojo
from server.option_types import FpeParams
from server.pricing_engine import PricingEngine, PricingResult
from engines.fpe.heston_params import HestonParams
from std.testing import assert_true, TestSuite


def _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def test_european_call_small_grid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.1, S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var params = FpeParams(heston=h, n_s=8, n_v=8, barrier=0.0, option_type=8, strikes=[60.0])
    var engine = PricingEngine(rtol=1e-4, atol=1e-6)
    var results = engine.price(params)
    assert_true(len(results) == 1)
    assert_true(results[0].success)
    assert_true(results[0].price >= 0.0)


def test_up_and_out_call_small_grid() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.1, S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var params = FpeParams(heston=h, n_s=8, n_v=8, barrier=80.0, option_type=6, strikes=[60.0])
    var engine = PricingEngine(rtol=1e-4, atol=1e-6)
    var results = engine.price(params)
    assert_true(len(results) == 1)
    assert_true(results[0].success)
    assert_true(results[0].price >= 0.0)


def test_multi_strike_european() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.1, S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var params = FpeParams(heston=h, n_s=8, n_v=8, barrier=0.0, option_type=8, strikes=[55.0, 60.0, 65.0])
    var engine = PricingEngine(rtol=1e-4, atol=1e-6)
    var results = engine.price(params)
    assert_true(len(results) == 3)
    for i in range(3):
        assert_true(results[i].success)
        assert_true(results[i].price >= 0.0)


def test_invalid_params() raises:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.1, S0=60.0, V0=0.1,
        S_min=0.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )
    var params = FpeParams(heston=h, n_s=8, n_v=8, barrier=90.0, option_type=6, strikes=[60.0])
    var engine = PricingEngine(rtol=1e-4, atol=1e-6)
    var results = engine.price(params)
    assert_true(len(results) == 1)
    assert_true(not results[0].success)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pixi run mojo run -I src tests/test_pricing_engine.mojo 2>&1 | tail -5`
Expected: FAIL — `PricingEngine` API mismatch

- [ ] **Step 3: Write PricingEngine implementation**

The new `pricing_engine.mojo` contains:
1. `PDFGrid` struct (relocated from `pdf_cache.mojo`, simplified — no Python serialization, no `precompute_weights` as separate step)
2. `PricingResult` struct (kept from `option_types.mojo`)
3. `PricingEngine` struct with `price(fpe_params: FpeParams) -> List[PricingResult]`

Key logic in `PricingEngine.price()`:
1. Validate `FpeParams` → return failed result if invalid
2. `revised = fpe_params.revised_heston()`
3. `domain = FPEDomain[3,3](revised, n_s, n_v, s_left_cond=fpe_params.s_left_cond(), s_right_cond=fpe_params.s_right_cond())`
4. `solver = FPESolver[1](rtol=self.rtol, atol=self.atol, max_step=revised.T/5.0)`
5. `sol = solver.solve(domain, revised)` — solve FPE
6. Build `PDFGrid` from last time step of `sol`
7. Compute trapezoidal weights
8. Create `BarrierPayoff(fpe_params.option_type, fpe_params.strikes, fpe_params.barrier)`
9. Integrate: for each (i,j) grid point, accumulate `payoff(S) * pdf[i][j] * ds_weights[i] * dv_weights[j]` into price accumulator per strike
10. Compute Greeks via finite differences (reuse grid, perturb S/V)
11. Return `List[PricingResult]`

- [ ] **Step 4: Run test to verify it passes**
Run: `pixi run mojo run -I src tests/test_pricing_engine.mojo`
Expected: All 4 tests PASS

- [ ] **Step 5: Delete old files and update imports**
```bash
rm src/server/pricer.mojo
rm src/server/pdf_cache.mojo
```
Update `src/server/__init__.mojo` to remove old re-exports and add new ones.

- [ ] **Step 6: Build to verify no dangling imports**
Run: `pixi run mojo build -I src examples/single_price.mojo`
Expected: 0 errors (some warnings OK)

- [ ] **Step 7: Run all surviving tests**
Run: `pixi run mojo run -I src tests/test_fpe_params.mojo && pixi run mojo run -I src tests/test_barrier_payoff.mojo && pixi run mojo run -I src tests/test_pricing_engine.mojo`
Expected: All PASS

- [ ] **Step 8: Commit**
```bash
git add -A src/server/ tests/test_pricing_engine.mojo
git commit -m "feat: PricingEngine with direct FPE solve, delete pricer.mojo and pdf_cache.mojo"
```

---

### Task 5: Greeks update

**Files:**
- Rewrite: `src/server/greeks.mojo`
- Modify: `tests/test_pricing_engine.mojo` (add Greeks assertions)

- [ ] **Step 1: Rewrite Greeks to use BarrierPayoff**

Key changes:
- `_price_at(grid, payoff: BarrierPayoff) -> List[Float64]` — integrates all strikes at once
- `compute_delta(grid, interp, S, V, payoff: BarrierPayoff) -> List[Float64]` — finite difference on all strikes
- `compute_gamma`, `compute_vega`, `compute_theta` — same pattern
- Remove `K` and `barrier` from all method signatures

- [ ] **Step 2: Update pricing_engine.mojo to use new Greeks API**
- Replace `compute_delta/gamma/vega` calls with new signatures
- Pass `BarrierPayoff` instead of separate K/barrier

- [ ] **Step 3: Add Greeks assertions to pricing engine test**
- Check `results[i].delta` and `results[i].gamma` are finite numbers
- No exact value check (depends on solver convergence)

- [ ] **Step 4: Run tests**
Run: `pixi run mojo run -I src tests/test_pricing_engine.mojo && pixi run mojo run -I src tests/test_fpe_engine.mojo`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/server/greeks.mojo src/server/pricing_engine.mojo tests/test_pricing_engine.mojo
git commit -m "refactor: Greeks use BarrierPayoff, multi-strike return values"
```

---

### Task 6: Python binding

**Files:**
- Rewrite: `src/bindings/python_module.mojo`
- Rewrite: `python/examples/backtest.py`

- [ ] **Step 1: Write new python_module.mojo**

Single `py_price(params_obj: PythonObject) -> PythonObject`:
- Extract all fields from flat dict
- Map `option_type` string to Int using lookup dict
- Accept `K` as scalar or list → convert to `List[Float64]`
- Construct `FpeParams`, call `PricingEngine.price()`
- Return dict with `prices`, `deltas`, `gammas`, `vegas`, `success` lists

- [ ] **Step 2: Write new backtest.py**

```python
import fpe_engine

result = fpe_engine.price({
    "kappa": 1.2, "theta": 0.05, "sigma": 0.35, "rho": -0.4,
    "r": 0.05, "T": 0.5, "S0": 100.0, "V0": 0.1,
    "K": [95.0, 100.0, 105.0],
    "barrier": 120.0, "option_type": "up_and_out_call",
    "n_s": 8, "n_v": 8, "rtol": 1e-4, "atol": 1e-6,
})
for i, K in enumerate([95.0, 100.0, 105.0]):
    print(f"K={K}: price={result['prices'][i]:.4f}, delta={result['deltas'][i]:.4f}")
```

- [ ] **Step 3: Build Python module**
Run: `pixi run mojo build -I src src/bindings/python_module.mojo`
Expected: 0 errors

- [ ] **Step 4: Commit**
```bash
git add src/bindings/python_module.mojo python/examples/backtest.py
git commit -m "feat: new Python binding with single price() API"
```

---

### Task 7: C++ binding

**Files:**
- Rewrite: `src/bindings/c_abi.mojo`
- Rewrite: `cpp/include/fpe_engine.h`
- Rewrite: `cpp/examples/live_trading.cpp`

- [ ] **Step 1: Write new c_abi.mojo**

Single `fpe_price()` export:
- Takes Heston params + `K: UnsafePointer[Float64]`, `n_strikes: Int32` + `option_type: Int32` + output arrays
- Constructs `FpeParams`, calls `PricingEngine.price()`
- Writes results to output arrays

- [ ] **Step 2: Write new fpe_engine.h**

```c
int32_t fpe_price(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    const double* K, int32_t n_strikes,
    double barrier, int32_t option_type,
    int32_t n_s, int32_t n_v,
    double rtol, double atol,
    double* out_prices, double* out_deltas,
    double* out_gammas, double* out_vegas
);
```

- [ ] **Step 3: Write new live_trading.cpp**

```c
#include "../include/fpe_engine.h"
#include <stdio.h>

int main() {
    double strikes[] = {95.0, 100.0, 105.0};
    double prices[3], deltas[3], gammas[3], vegas[3];
    int32_t result = fpe_price(
        1.2, 0.05, 0.35, -0.4, 0.05, 0.5, 100.0, 0.1,
        strikes, 3, 120.0, 6,
        8, 8, 1e-4, 1e-6,
        prices, deltas, gammas, vegas
    );
    if (result == 0) {
        for (int i = 0; i < 3; i++) {
            printf("K=%.0f: price=%.6f, delta=%.6f\n", strikes[i], prices[i], deltas[i]);
        }
    } else {
        fprintf(stderr, "Pricing failed\n");
    }
    return 0;
}
```

- [ ] **Step 4: Build C ABI module**
Run: `pixi run mojo build -I src src/bindings/c_abi.mojo`
Expected: 0 errors

- [ ] **Step 5: Commit**
```bash
git add src/bindings/c_abi.mojo cpp/include/fpe_engine.h cpp/examples/live_trading.cpp
git commit -m "feat: new C ABI with single fpe_price() function"
```

---

### Task 8: Update __init__.mojo and cleanup old tests

**Files:**
- Rewrite: `src/server/__init__.mojo`
- Delete: `tests/test_pricing_server.mojo`
- Delete: `tests/test_pdf_cache_serialization.mojo`
- Delete: `tests/test_safe_serialization.mojo`
- Modify: `tests/test_bindings.mojo` (update imports if needed)

- [ ] **Step 1: Update server/__init__.mojo**

```mojo
from server.option_types import FpeParams
from server.payoffs import Payoff, BarrierPayoff
from server.greeks import Greeks
from server.pricing_engine import PricingEngine, PricingResult, PDFGrid
from server.interpolator import Interpolator
```

- [ ] **Step 2: Delete obsolete test files**
```bash
rm tests/test_pricing_server.mojo
rm tests/test_pdf_cache_serialization.mojo
rm tests/test_safe_serialization.mojo
```

- [ ] **Step 3: Update test_bindings.mojo if it imports old types**
Check and fix any references to `PricingRequest`, `Pricer`, `PDFCache`.

- [ ] **Step 4: Full build verification**
Run: `pixi run mojo build -I src examples/single_price.mojo && pixi run mojo build -I src examples/profile_galerkin.mojo`
Expected: 0 errors

- [ ] **Step 5: Full test run**
Run: `pixi run mojo run -I src tests/test_fpe_params.mojo && pixi run mojo run -I src tests/test_barrier_payoff.mojo && pixi run mojo run -I src tests/test_pricing_engine.mojo && pixi run mojo run -I src tests/test_radau.mojo && pixi run mojo run -I src tests/test_fpe_engine.mojo`
Expected: All PASS

- [ ] **Step 6: Commit**
```bash
git add -A src/server/ tests/
git commit -m "chore: update re-exports, delete obsolete tests"
```

---

### Task 9: GPU pricing kernel update (optional, low priority)

**Files:**
- Modify: `src/server/gpu_pricing_kernels.mojo`

The GPU kernel currently hard-codes EuropeanCall + barrier knock-out. It can be updated to support `option_type` dispatch in a future iteration. For now, leave it as-is since the CPU pipeline is the priority. Add a TODO comment noting the kernel needs updating to match `BarrierPayoff` logic.

- [ ] **Step 1: Add TODO comment to gpu_pricing_kernels.mojo**
- [ ] **Step 2: Commit**
```bash
git add src/server/gpu_pricing_kernels.mojo
git commit -m "chore: add TODO for GPU kernel option_type dispatch"
```

---

## Dependency Graph

```
Task 1 (FpeParams) ──────────┐
                              ├──► Task 4 (PricingEngine) ──► Task 6 (Python) ──► Task 8 (cleanup)
Task 2 (BarrierPayoff) ──────┤                         ──► Task 7 (C++)     ──► Task 8
                              │
Task 3 (FPEDomain BC) ────────┘──► Task 4
                              │
Task 5 (Greeks) ──────────────┘──► Task 4 (co-dependent, can be done together)
```

Tasks 1, 2, 3 are independent and can be parallelized. Task 4 depends on all three. Tasks 6, 7 depend on Task 4. Task 8 depends on Tasks 6, 7. Task 9 is optional.
