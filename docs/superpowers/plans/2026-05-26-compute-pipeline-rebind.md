# ComputePipeline 统一计算管线 + Python 绑定重构

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 `ComputePipeline` 统一管理 Mojo 侧 FPE 计算管线，废弃 `_fpe_state.mojo` session 模式，Python 端用 `ConvertibleToPython` 简化数据传递，C++ 通过 C ABI 直接访问管线。

**Architecture:** 三层分离：(1) `server/compute_pipeline.mojo` 纯 Mojo 核心结构体，无 Python 依赖；(2) `bindings/_convert.mojo` + `_py_pipeline.mojo` Python 绑定层，用 `ConvertibleToPython` trait 实现自动 `PythonObject` 转换；(3) `bindings/c_abi.mojo` 保持 C++ 访问，改用 `ComputePipeline` 指针。Python 端废弃 FPEPricer session 管理，改为无状态 `Compute` 上下文对象。

**Tech Stack:** Mojo nightly, `std.python.conversions.ConvertibleToPython`, `std.collections.OwnedKwargsDict`, `std.python.bindings.PythonModuleBuilder/PythonTypeBuilder`

---

## File Structure Map

```
创建:
  src/server/compute_pipeline.mojo      — 核心结构体，无 Python
  src/bindings/_params.mojo             — 统一 kwargs → FpeParams 解析
  src/bindings/_convert.mojo            — ConvertibleToPython 结果类型
  src/bindings/_py_pipeline.mojo        — PyComputePipeline 包装 (PythonTypeBuilder)

修改:
  src/bindings/_fpe_native.mojo         — 重写，废弃 session API
  src/bindings/c_abi.mojo               — 改用 ComputePipeline
  python/fpe_engine/pricer.py           — 简化为 Compute 上下文
  python/fpe_engine/__init__.py         — 更新导出
  tests/test_pricer_stepwise.py         — 更新 Python 测试

删除:
  src/bindings/_fpe_state.mojo          — 整个文件废弃

新建 Mojo 测试:
  tests/test_compute_pipeline.mojo      — 纯 Mojo 管线测试
  tests/test_params.mojo                — 统一参数解析测试
  tests/test_convert.mojo               — ConvertibleToPython 类型测试
```

---

### Task 1: Create `server/compute_pipeline.mojo` — 纯 Mojo 核心结构体

**Files:**
- Create: `src/server/compute_pipeline.mojo`

**Context:** 当前 `Pricer.price()`（`pricer.mojo:76-116`）每次调用重建 `FPEDomain`、`FPESolver`、做完整求解。`FPEState`（`_fpe_state.mojo:6-44`）在绑定层缓存了部分中间结果，但没有正式的统一接口。`ComputePipeline` 替代两者，内置惰性缓存。

该结构体在 `server/` 层，**不能 import `std.python`**，只能使用原生 Mojo 类型。

- [ ] **Step 1: Write failing Mojo test**

File: `tests/test_compute_pipeline.mojo`

```mojo
from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams
from server.compute_pipeline import ComputePipeline
from std.testing import TestSuite, assert_true, assert_false, assert_equal

def _make_fp() -> FpeParams:
    var h = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=0.1,
        S0=60.0, V0=0.1, S_min=0.0, S_max=180.0, V_min=0.0, V_max=1.0,
    )
    return FpeParams(heston=h^, n_s=16, n_v=16, barrier=0.0, option_type=8, strikes=[60.0])

def test_knots() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var (s, v) = pipe.knots()
    assert_true(len(s) > 0)
    assert_true(len(v) > 0)

def test_grid_points() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var (s, v, sw, vw) = pipe.grid_points()
    assert_true(len(s) > 0)
    assert_true(len(v) > 0)
    assert_equal(len(s), len(sw))
    assert_equal(len(v), len(vw))

def test_basis_1d() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var (Bs, dBs, Bv, dBv) = pipe.basis_1d()
    assert_true(Bs.nrows > 0)

def test_initial_condition() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var q0 = pipe.initial_condition()
    assert_true(len(q0) > 0)
    # Verify caching: second call returns same result
    var q0b = pipe.initial_condition()
    assert_equal(len(q0b), len(q0))

def test_solve() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var sol = pipe.solve()
    assert_true(len(sol) > 0)
    assert_true(len(sol[0]) > 0)

def test_pdf() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var pdf = pipe.pdf()
    assert_true(len(pdf) > 0)

def test_price_at() raises:
    var fp = _make_fp()
    var pipe = ComputePipeline(fp^)
    var prices = pipe.price_at([60.0])
    assert_equal(len(prices), 1)
    assert_true(prices[0] >= 0.0)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run mojo test -I src tests/test_compute_pipeline.mojo`
Expected: compilation error — `ComputePipeline` not found

- [ ] **Step 3: Implement `ComputePipeline`**

File: `src/server/compute_pipeline.mojo`

```mojo
from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain, FPECachedBasis
from engines.fpe.galerkin import mass_from_cached, stiffness_from_cached
from engines.fpe.initial_cond import initial_condition_from_cached
from engines.fpe.pdf import pdf_from_cached
from engines.fpe.solver import FPESolver
from server.option_types import FpeParams, PricingResult
from server.payoffs import BarrierPayoff
from server.pricer import PDFGrid, _price_at
from server.simd_utils import vec_central_diff, vec_second_diff, vec_scale
from sparse.csr import CSRMatrix
from sparse.kron import kron
from std.math import exp

struct ComputePipeline(Movable):
    var fp: FpeParams
    var heston: HestonParams
    var domain: FPEDomain[3, 3]
    var cached: FPECachedBasis[3, 3]
    var M: CSRMatrix
    var K_mat: CSRMatrix
    var q0: Optional[List[Float64]]
    var solution: Optional[List[List[Float64]]]
    var pdf_grid: Optional[List[List[Float64]]]

    def __init__(out self, var fp: FpeParams, num_insert: Int = 251):
        if not fp.is_valid():
            raise Error("invalid FPE parameters")
        self.fp = fp^
        self.heston = self.fp.revised_heston()
        self.domain = FPEDomain[3, 3](
            self.heston,
            n_s=self.fp.n_s, n_v=self.fp.n_v,
            num_insert=num_insert,
            s_left_cond=self.fp.s_left_cond(),
            s_right_cond=self.fp.s_right_cond(),
        )
        self.cached = self.domain.cached_basis()
        self.M = mass_from_cached(self.cached)
        self.K_mat = stiffness_from_cached(self.cached, self.heston)
        self.q0 = None
        self.solution = None
        self.pdf_grid = None

    fn knots(self) -> Tuple[List[Float64], List[Float64]]:
        return Tuple(self.domain.s_knots.copy(), self.domain.v_knots.copy())

    fn grid_points(self) -> Tuple[List[Float64], List[Float64], List[Float64], List[Float64]]:
        return Tuple(
            self.cached.s_points_phys.copy(),
            self.cached.v_points_phys.copy(),
            self.cached.s_weights.copy(),
            self.cached.v_weights.copy(),
        )

    fn basis_1d(self) -> Tuple[CSRMatrix, CSRMatrix, CSRMatrix, CSRMatrix]:
        return Tuple(
            self.cached.Bs.copy(), self.cached.dBs.copy(),
            self.cached.Bv.copy(), self.cached.dBv.copy(),
        )

    fn basis_2d(self) -> CSRMatrix:
        return kron(self.cached.Bs, self.cached.Bv)

    fn initial_condition(self) -> List[Float64]:
        if self.q0 == None:
            var q0 = initial_condition_from_cached(self.cached, self.heston, self.M.copy())
            self.q0 = q0^
        return self.q0.value().copy()

    fn _ensure_solve(self):
        if self.solution == None:
            if self.q0 == None:
                var q0 = initial_condition_from_cached(self.cached, self.heston, self.M.copy())
                self.q0 = q0^
            var solver = FPESolver[1](
                rtol=1e-4, atol=1e-6,
                max_step=self.heston.T / 5.0,
                first_step=1e-6,
            )
            var sol = solver.solve(self.domain, self.heston)
            self.solution = sol^

    fn solve(self) -> List[List[Float64]]:
        self._ensure_solve()
        var sol_val = self.solution.value()
        var result: List[List[Float64]] = List[List[Float64]](cap=len(sol_val))
        for t in range(len(sol_val)):
            var t_copy: List[Float64] = List[Float64](cap=len(sol_val[t]))
            for i in range(len(sol_val[t])):
                t_copy.append(sol_val[t][i])
            result.append(t_copy^)
        return result^

    fn pdf(self) -> List[List[Float64]]:
        self._ensure_solve()
        if self.pdf_grid == None:
            var sol_val = self.solution.value()
            var q_T = sol_val[len(sol_val) - 1].copy()
            var pdf = pdf_from_cached(self.cached, q_T)
            self.pdf_grid = pdf^
        var pdf_val = self.pdf_grid.value()
        var result: List[List[Float64]] = List[List[Float64]](cap=len(pdf_val))
        for i in range(len(pdf_val)):
            var row_copy: List[Float64] = List[Float64](cap=len(pdf_val[i]))
            for j in range(len(pdf_val[i])):
                row_copy.append(pdf_val[i][j])
            result.append(row_copy^)
        return result^

    fn price_at(self, strikes: List[Float64]) -> List[Float64]:
        if len(strikes) == 0:
            return List[Float64]()
        self._ensure_solve()
        if self.pdf_grid == None:
            var sol_val = self.solution.value()
            var q_T = sol_val[len(sol_val) - 1].copy()
            var pdf = pdf_from_cached(self.cached, q_T)
            self.pdf_grid = pdf^
        var pdf_val = self.pdf_grid.value().copy()
        var payoff = BarrierPayoff(
            option_type=self.fp.option_type,
            strikes=strikes.copy(),
            barrier=self.fp.barrier,
        )
        var grid = PDFGrid(
            pdf=pdf_val^,
            s_points=self.cached.s_points_phys.copy(),
            v_points=self.cached.v_points_phys.copy(),
            T=self.heston.T,
            ds_weights=self.cached.s_weights.copy(),
            dv_weights=self.cached.v_weights.copy(),
        )
        var prices = _price_at(grid, payoff)
        var discount = exp(-self.heston.r * self.heston.T)
        return vec_scale(prices, discount)

    fn greeks(self, strikes: List[Float64],
              rel_s: Float64 = 0.01, rel_v: Float64 = 0.1) -> Tuple[List[Float64], List[Float64], List[Float64]]:
        if len(strikes) == 0:
            return Tuple(List[Float64](), List[Float64](), List[Float64]())
        var h_s = self.heston.S0 * rel_s
        var h_v = self.heston.V0 * rel_v
        if h_s < 1e-8: h_s = 1e-8
        if h_v < 1e-12: h_v = 1e-12

        var p_base = self.price_at(strikes)

        fn _bumped_pipe(fp: FpeParams, dS: Float64, dV: Float64,
                        num_insert: Int) -> ComputePipeline:
            var h = fp.heston.copy()
            h.S0 = h.S0 + dS
            h.V0 = h.V0 + dV
            var new_fp = FpeParams(
                heston=h^, n_s=fp.n_s, n_v=fp.n_v,
                barrier=fp.barrier, option_type=fp.option_type,
                strikes=fp.strikes.copy(),
            )
            return ComputePipeline(new_fp^, num_insert=num_insert)

        var up_s_pipe = _bumped_pipe(self.fp, h_s, 0.0, 251)
        var dn_s_pipe = _bumped_pipe(self.fp, -h_s, 0.0, 251)
        var up_v_pipe = _bumped_pipe(self.fp, 0.0, h_v, 251)
        var dn_v_pipe = _bumped_pipe(self.fp, 0.0, -h_v, 251)

        var p_up_s = up_s_pipe.price_at(strikes)
        var p_dn_s = dn_s_pipe.price_at(strikes)
        var p_up_v = up_v_pipe.price_at(strikes)
        var p_dn_v = dn_v_pipe.price_at(strikes)

        var deltas = vec_central_diff(p_up_s^, p_dn_s^, h_s)
        var gammas = vec_second_diff(p_up_s^, p_base^, p_dn_s^, h_s)
        var vegas = vec_central_diff(p_up_v^, p_dn_v^, h_v)
        return Tuple(deltas^, gammas^, vegas^)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run mojo test -I src tests/test_compute_pipeline.mojo`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/server/compute_pipeline.mojo tests/test_compute_pipeline.mojo
git commit -m "feat: add ComputePipeline struct for unified FPE computation"
```

---

### Task 2: Create `bindings/_params.mojo` — 统一参数解析

**Files:**
- Create: `src/bindings/_params.mojo`

**Context:** 当前 `py_create_session`、`py_price`、`py_greeks` 三个函数各有一份 50+ 行的重复 `Float64(py=obj.get(...))` 参数解析代码。统一为 `fpe_params_from_kwargs()`。

- [ ] **Step 1: Write failing test**

File: `tests/test_params.mojo`

```mojo
from bindings._params import fpe_params_from_kwargs, option_type_from_py
from std.python import Python, PythonObject
from std.testing import TestSuite, assert_true, assert_equal

def test_option_type_from_string() raises:
    assert_equal(option_type_from_py(PythonObject("european_call")), 8)
    assert_equal(option_type_from_py(PythonObject("european_put")), 9)

def test_option_type_from_int() raises:
    assert_equal(option_type_from_py(PythonObject(0)), 0)
    assert_equal(option_type_from_py(PythonObject(9)), 9)

def test_fpe_params_from_kwargs() raises:
    var builtins = Python.import_module("builtins")
    var kwargs = Python.dict(
        kappa=PythonObject(1.2), theta=PythonObject(0.05),
        sigma=PythonObject(0.35), rho=PythonObject(-0.4),
        r=PythonObject(0.05), T=PythonObject(0.5),
        S0=PythonObject(60.0), V0=PythonObject(0.1),
        n_s=PythonObject(16), n_v=PythonObject(16),
        K=PythonObject(60.0),
        option_type=PythonObject("european_call"),
    )
    var fp = fpe_params_from_kwargs(kwargs)
    assert_true(fp.is_valid())
    assert_equal(fp.heston.kappa, 1.2)
    assert_equal(fp.heston.S0, 60.0)
    assert_equal(fp.option_type, 8)

def test_fpe_params_from_kwargs_list_K() raises:
    var builtins = Python.import_module("builtins")
    var k_list = Python.list()
    _ = k_list.append(PythonObject(50.0))
    _ = k_list.append(PythonObject(60.0))
    var kwargs = Python.dict(
        S0=PythonObject(60.0), V0=PythonObject(0.1),
        n_s=PythonObject(16), n_v=PythonObject(16),
        K=k_list, option_type=PythonObject("european_call"),
    )
    var fp = fpe_params_from_kwargs(kwargs)
    assert_equal(len(fp.strikes), 2)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run mojo test -I src tests/test_params.mojo`
Expected: compilation error

- [ ] **Step 3: Implement `fpe_params_from_kwargs`**

File: `src/bindings/_params.mojo`

```mojo
from std.collections import OwnedKwargsDict
from std.python import Python, PythonObject
from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams

comptime MAX_STRIKES: Int = 1024

def _option_type_from_string(type_str: String) raises -> Int:
    if type_str == "down_and_in_call":      return 0
    elif type_str == "down_and_in_put":     return 1
    elif type_str == "down_and_out_call":   return 2
    elif type_str == "down_and_out_put":    return 3
    elif type_str == "up_and_in_call":      return 4
    elif type_str == "up_and_in_put":       return 5
    elif type_str == "up_and_out_call":     return 6
    elif type_str == "up_and_out_put":      return 7
    elif type_str == "european_call":       return 8
    elif type_str == "european_put":        return 9
    else:
        raise Error("unknown option_type: " + type_str)

def option_type_from_py(ot_obj: PythonObject) raises -> Int:
    var builtins = Python.import_module("builtins")
    if builtins.isinstance(ot_obj, builtins.str):
        return _option_type_from_string(String(py=ot_obj))
    elif builtins.isinstance(ot_obj, builtins.int):
        var val = Int(py=ot_obj)
        if val < 0 or val > 9:
            raise Error("option_type must be 0-9, got " + String(val))
        return val
    else:
        raise Error("option_type must be str or int")

fn _get_float(kwargs: PythonObject, key: String, default: Float64) raises -> Float64:
    if kwargs.__contains__(key):
        return Float64(py=kwargs[key])
    return default

fn _get_int(kwargs: PythonObject, key: String, default: Int) raises -> Int:
    if kwargs.__contains__(key):
        return Int(py=kwargs[key])
    return default

def fpe_params_from_kwargs(kwargs: PythonObject) raises -> FpeParams:
    var kappa = _get_float(kwargs, "kappa", 1.2)
    var theta = _get_float(kwargs, "theta", 0.05)
    var sigma = _get_float(kwargs, "sigma", 0.35)
    var rho = _get_float(kwargs, "rho", -0.4)
    var r_rate = _get_float(kwargs, "r", 0.05)
    var T = _get_float(kwargs, "T", 0.5)
    var S0 = _get_float(kwargs, "S0", 100.0)
    var V0 = _get_float(kwargs, "V0", 0.1)
    var n_s = _get_int(kwargs, "n_s", 38)
    var n_v = _get_int(kwargs, "n_v", 38)
    var barrier = _get_float(kwargs, "barrier", 0.0)

    if n_s < 4 or n_s > 256:
        raise Error("n_s must be in [4, 256], got " + String(n_s))
    if n_v < 4 or n_v > 256:
        raise Error("n_v must be in [4, 256], got " + String(n_v))

    var option_type_int = option_type_from_py(kwargs["option_type"])

    var builtins = Python.import_module("builtins")
    var K_obj = kwargs["K"] if kwargs.__contains__("K") else PythonObject(100.0)
    var strikes: List[Float64] = []
    if builtins.isinstance(K_obj, builtins.list):
        var k_len = Int(py=builtins.len(K_obj))
        if k_len > MAX_STRIKES:
            raise Error("K list too large, max " + String(MAX_STRIKES))
        for i in range(k_len):
            strikes.append(Float64(py=K_obj[i]))
    elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
        strikes.append(Float64(py=K_obj))
    else:
        raise Error("K must be float, int, or list of floats")

    if len(strikes) == 0:
        raise Error("K must not be empty")

    var heston = HestonParams(
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        r=r_rate, T=T, S0=S0, V0=V0,
        S_min=0.0, S_max=S0 * 3.0, V_min=0.0, V_max=1.0,
    )
    if not heston.is_valid():
        raise Error("invalid Heston parameters")

    var fp = FpeParams(
        heston=heston^, n_s=n_s, n_v=n_v, barrier=barrier,
        option_type=option_type_int, strikes=strikes^,
    )
    if not fp.is_valid():
        raise Error("invalid FPE parameters (check barrier/option_type combo)")
    return fp^
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run mojo test -I src tests/test_params.mojo`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/bindings/_params.mojo tests/test_params.mojo
git commit -m "feat: add unified fpe_params_from_kwargs for Python kwargs parsing"
```

---

### Task 3: Create `bindings/_convert.mojo` — `ConvertibleToPython` 结果类型

**Files:**
- Create: `src/bindings/_convert.mojo`

**Context:** 参考 Mojo nightly `ConvertibleToPython` trait 文档（`https://mojolang.org/nightly/docs/std/python/conversions/ConvertibleToPython/`），定义可以在 `bindings/` 层自动转为 `PythonObject` 的结果包装类型。这些类型只存在于绑定层，`server/` 不受污染。

Nightly 新增：所有 `ConvertibleToPython` 类型隐式转换为 `PythonObject`，所以 `return PyKnotsResult(...)` 在返回类型为 `PythonObject` 的函数中自动工作。

- [ ] **Step 1: Write failing test**

File: `tests/test_convert.mojo`

```mojo
from bindings._convert import PyKnotsResult, PyGridPointsResult, PyCsrMatrix, PyListResult, PyGrid2DResult, PyPriceResult
from std.python import Python, PythonObject
from std.testing import TestSuite, assert_true, assert_equal

def test_knots_to_python() raises:
    var s_list = List[Float64](1.0, 2.0, 3.0)
    var v_list = List[Float64](4.0, 5.0)
    var r = PyKnotsResult(s=s_list^, v=v_list^)
    var py = PythonObject(r)
    var builtins = Python.import_module("builtins")
    assert_equal(Int(py=builtins.len(py["s"])), 3)
    assert_equal(Int(py=builtins.len(py["v"])), 2)

def test_grid_points_to_python() raises:
    var r = PyGridPointsResult(
        s=List[Float64](1.0), v=List[Float64](2.0),
        s_weights=List[Float64](0.5), v_weights=List[Float64](0.5),
    )
    var py = PythonObject(r)

def test_csr_to_python() raises:
    var r = PyCsrMatrix(
        data=List[Float64](1.0, 2.0), indices=List[Int32](0, 1),
        indptr=List[Int32](0, 1, 2), nrows=2, ncols=2,
    )
    var py = PythonObject(r)
    assert_equal(Int(py=py["nrows"]), 2)

def test_list_result() raises:
    var r = PyListResult(data=List[Float64](1.0, 2.0, 3.0))
    var py = PythonObject(r)
    var builtins = Python.import_module("builtins")
    assert_equal(Int(py=builtins.len(py)), 3)

def test_grid_2d_result() raises:
    var inner = List[Float64](1.0, 2.0)
    var outer = List[List[Float64]]()
    outer.append(inner^)
    var r = PyGrid2DResult(data=outer^)
    var py = PythonObject(r)

def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
```

- [ ] **Step 2: Run to verify fails**

Run: `pixi run mojo test -I src tests/test_convert.mojo`
Expected: compilation error

- [ ] **Step 3: Implement convert types**

File: `src/bindings/_convert.mojo`

```mojo
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleToPython

struct PyCsrMatrix(ConvertibleToPython, Movable, Copyable):
    var data: List[Float64]
    var indices: List[Int32]
    var indptr: List[Int32]
    var nrows: Int
    var ncols: Int

    def to_python_object(var self) raises -> PythonObject:
        var py_data = Python.list()
        var py_indices = Python.list()
        var py_indptr = Python.list()
        for i in range(len(self.data)):
            _ = py_data.append(PythonObject(self.data[i]))
        for i in range(len(self.indices)):
            _ = py_indices.append(PythonObject(self.indices[i]))
        for i in range(len(self.indptr)):
            _ = py_indptr.append(PythonObject(self.indptr[i]))
        return Python.dict(
            data=py_data, indices=py_indices, indptr=py_indptr,
            nrows=PythonObject(self.nrows), ncols=PythonObject(self.ncols),
        )

struct PyKnotsResult(ConvertibleToPython, Movable, Copyable):
    var s: List[Float64]
    var v: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_s = Python.list()
        var py_v = Python.list()
        for i in range(len(self.s)):
            _ = py_s.append(PythonObject(self.s[i]))
        for i in range(len(self.v)):
            _ = py_v.append(PythonObject(self.v[i]))
        return Python.dict(s=py_s, v=py_v)

struct PyGridPointsResult(ConvertibleToPython, Movable, Copyable):
    var s: List[Float64]
    var v: List[Float64]
    var s_weights: List[Float64]
    var v_weights: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_s = Python.list()
        var py_v = Python.list()
        var py_sw = Python.list()
        var py_vw = Python.list()
        for i in range(len(self.s)):
            _ = py_s.append(PythonObject(self.s[i]))
        for i in range(len(self.v)):
            _ = py_v.append(PythonObject(self.v[i]))
        for i in range(len(self.s_weights)):
            _ = py_sw.append(PythonObject(self.s_weights[i]))
        for i in range(len(self.v_weights)):
            _ = py_vw.append(PythonObject(self.v_weights[i]))
        return Python.dict(s=py_s, v=py_v, s_weights=py_sw, v_weights=py_vw)

struct PyBasis1DResult(ConvertibleToPython, Movable, Copyable):
    var Bs: PyCsrMatrix
    var dBs: PyCsrMatrix
    var Bv: PyCsrMatrix
    var dBv: PyCsrMatrix

    def to_python_object(var self) raises -> PythonObject:
        return Python.dict(
            Bs=PythonObject(self.Bs), dBs=PythonObject(self.dBs),
            Bv=PythonObject(self.Bv), dBv=PythonObject(self.dBv),
        )

struct PyListResult(ConvertibleToPython, Movable, Copyable):
    var data: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_data = Python.list()
        for i in range(len(self.data)):
            _ = py_data.append(PythonObject(self.data[i]))
        return py_data

struct PyGrid2DResult(ConvertibleToPython, Movable, Copyable):
    var data: List[List[Float64]]

    def to_python_object(var self) raises -> PythonObject:
        var py_outer = Python.list()
        for row in range(len(self.data)):
            var py_row = Python.list()
            for col in range(len(self.data[row])):
                _ = py_row.append(PythonObject(self.data[row][col]))
            _ = py_outer.append(py_row)
        return py_outer

struct PyPriceResult(ConvertibleToPython, Movable, Copyable):
    var prices: List[Float64]
    var deltas: List[Float64]
    var gammas: List[Float64]
    var vegas: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_prices = Python.list()
        var py_deltas = Python.list()
        var py_gammas = Python.list()
        var py_vegas = Python.list()
        for i in range(len(self.prices)):
            _ = py_prices.append(PythonObject(self.prices[i]))
            _ = py_deltas.append(PythonObject(self.deltas[i]))
            _ = py_gammas.append(PythonObject(self.gammas[i]))
            _ = py_vegas.append(PythonObject(self.vegas[i]))
        return Python.dict(
            prices=py_prices, deltas=py_deltas,
            gammas=py_gammas, vegas=py_vegas,
        )

struct PyGreeksResult(ConvertibleToPython, Movable, Copyable):
    var delta: List[Float64]
    var gamma: List[Float64]
    var vega: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_delta = Python.list()
        var py_gamma = Python.list()
        var py_vega = Python.list()
        for i in range(len(self.delta)):
            _ = py_delta.append(PythonObject(self.delta[i]))
        for i in range(len(self.gamma)):
            _ = py_gamma.append(PythonObject(self.gamma[i]))
        for i in range(len(self.vega)):
            _ = py_vega.append(PythonObject(self.vega[i]))
        return Python.dict(delta=py_delta, gamma=py_gamma, vega=py_vega)
```

- [ ] **Step 4: Run to verify passes**

Run: `pixi run mojo test -I src tests/test_convert.mojo`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/bindings/_convert.mojo tests/test_convert.mojo
git commit -m "feat: add ConvertibleToPython result types for Python binding"
```

---

### Task 4: Create `bindings/_py_pipeline.mojo` — PyComputePipeline 包装

**Files:**
- Create: `src/bindings/_py_pipeline.mojo`

**Context:** 将 `ComputePipeline` 通过 `PythonTypeBuilder` 注册为 Python 类型。由于 `ComputePipeline` 在 `server/` 中无 Python 依赖，`PyComputePipeline` 是包装结构体，内部持有 `ComputePipeline`。每个 `py_self` 方法使用 `downcast_value_ptr[Self]()` 获取内部指针，调用 `ComputePipeline` 方法后通过 `ConvertibleToPython` 类型自动转换。

- [ ] **Step 1: Implement `PyComputePipeline`**

File: `src/bindings/_py_pipeline.mojo`

```mojo
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, PythonTypeBuilder
from server.compute_pipeline import ComputePipeline
from server.option_types import FpeParams
from bindings._params import fpe_params_from_kwargs
from bindings._convert import (
    PyKnotsResult, PyGridPointsResult, PyBasis1DResult, PyCsrMatrix,
    PyListResult, PyGrid2DResult, PyPriceResult, PyGreeksResult,
)
from sparse.csr import CSRMatrix
from sparse.kron import kron

@fieldwise_init
struct PyComputePipeline(Writable, Movable):
    var inner: ComputePipeline

    @staticmethod
    def py_init(out self, args: PythonObject, kwargs: PythonObject) raises:
        var fp = fpe_params_from_kwargs(kwargs)
        self = PyComputePipeline(inner=ComputePipeline(fp^))

    @staticmethod
    def knots(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var (s, v) = self_ptr[].inner.knots()
        return PythonObject(PyKnotsResult(s=s^, v=v^))

    @staticmethod
    def grid_points(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var (s, v, sw, vw) = self_ptr[].inner.grid_points()
        return PythonObject(PyGridPointsResult(s=s^, v=v^, s_weights=sw^, v_weights=vw^))

    @staticmethod
    def basis_1d(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var (Bs, dBs, Bv, dBv) = self_ptr[].inner.basis_1d()
        return PythonObject(PyBasis1DResult(
            Bs=_csr_to_py(Bs), dBs=_csr_to_py(dBs),
            Bv=_csr_to_py(Bv), dBv=_csr_to_py(dBv),
        ))

    @staticmethod
    def basis_2d(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var B_2d = self_ptr[].inner.basis_2d()
        return PythonObject(_csr_to_py(B_2d^))

    @staticmethod
    def initial_condition(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var q0 = self_ptr[].inner.initial_condition()
        return PythonObject(PyListResult(data=q0^))

    @staticmethod
    def solve(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var sol = self_ptr[].inner.solve()
        return PythonObject(PyGrid2DResult(data=sol^))

    @staticmethod
    def pdf(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var pdf = self_ptr[].inner.pdf()
        return PythonObject(PyGrid2DResult(data=pdf^))

    @staticmethod
    def payoff_price(py_self: PythonObject, K_obj: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var builtins = Python.import_module("builtins")
        var strikes: List[Float64] = []
        if builtins.isinstance(K_obj, builtins.list):
            var k_len = Int(py=builtins.len(K_obj))
            for i in range(k_len):
                strikes.append(Float64(py=K_obj[i]))
        elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
            strikes.append(Float64(py=K_obj))
        else:
            raise Error("K must be float, int, or list of floats")
        var prices = self_ptr[].inner.price_at(strikes)
        return PythonObject(PyListResult(data=prices^))

    @staticmethod
    def greeks(py_self: PythonObject, K_obj: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var builtins = Python.import_module("builtins")
        var strikes: List[Float64] = []
        if builtins.isinstance(K_obj, builtins.list):
            var k_len = Int(py=builtins.len(K_obj))
            for i in range(k_len):
                strikes.append(Float64(py=K_obj[i]))
        elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
            strikes.append(Float64(py=K_obj))
        else:
            raise Error("K must be float, int, or list of floats")
        var (deltas, gammas, vegas) = self_ptr[].inner.greeks(strikes)
        return PythonObject(PyGreeksResult(delta=deltas^, gamma=gammas^, vega=vegas^))

    def write_to(self, mut writer: Some[Writer]):
        t"PyComputePipeline(...)".write_to(writer)


fn _csr_to_py(var mat: CSRMatrix) -> PyCsrMatrix:
    var data: List[Float64] = List[Float64](cap=len(mat.data))
    var indices: List[Int32] = List[Int32](cap=len(mat.indices))
    var indptr: List[Int32] = List[Int32](cap=len(mat.indptr))
    for i in range(len(mat.data)):
        data.append(mat.data[i])
    for i in range(len(mat.indices)):
        indices.append(Int32(mat.indices[i]))
    for i in range(len(mat.indptr)):
        indptr.append(Int32(mat.indptr[i]))
    return PyCsrMatrix(data=data^, indices=indices^, indptr=indptr^, nrows=mat.nrows, ncols=mat.ncols)
```

- [ ] **Step 2: Run Mojo compile check**

Run: `pixi run mojo build -I src src/bindings/_py_pipeline.mojo`
Expected: compiles without error

- [ ] **Step 3: Commit**

```bash
git add src/bindings/_py_pipeline.mojo
git commit -m "feat: add PyComputePipeline wrapper with PythonTypeBuilder binding"
```

---

### Task 5: Rewrite `bindings/_fpe_native.mojo` — 废弃 session API

**Files:**
- Modify: `src/bindings/_fpe_native.mojo` (full rewrite)
- Delete: `src/bindings/_fpe_state.mojo`

**Context:** 删除所有 session 相关函数（`py_init`、`py_destroy`、`py_create_session`、`py_close`），替换为：(1) `PyComputePipeline` Python 类型注册；(2) `py_price` 一步到位含 Greeks 的函数。

- [ ] **Step 1: Rewrite `_fpe_native.mojo`**

File: `src/bindings/_fpe_native.mojo`

```mojo
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from collections import OwnedKwargsDict

from server.option_types import FpeParams
from server.pricing_engine import PricingEngine
from server.compute_pipeline import ComputePipeline
from bindings._params import fpe_params_from_kwargs
from bindings._convert import PyPriceResult, PyGrid2DResult
from bindings._py_pipeline import PyComputePipeline


def py_price(kwargs: OwnedKwargsDict[PythonObject]) raises -> PythonObject:
    var fp = fpe_params_from_kwargs(kwargs)
    var engine = PricingEngine()
    var results = engine.price(fp)

    var prices: List[Float64] = List[Float64](cap=len(results))
    var deltas: List[Float64] = List[Float64](cap=len(results))
    var gammas: List[Float64] = List[Float64](cap=len(results))
    var vegas: List[Float64] = List[Float64](cap=len(results))

    for i in range(len(results)):
        prices.append(results[i].price)
        deltas.append(results[i].delta)
        gammas.append(results[i].gamma)
        vegas.append(results[i].vega)

    return PythonObject(PyPriceResult(
        prices=prices^, deltas=deltas^,
        gammas=gammas^, vegas=vegas^,
    ))


@export
def PyInit__fpe_native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_fpe_native")
        module.def_function[py_price]("price")
        _ = module.add_type[PyComputePipeline]("Compute")\
            .def_py_init[PyComputePipeline.py_init]()\
            .def_method[PyComputePipeline.knots]("knots")\
            .def_method[PyComputePipeline.grid_points]("grid_points")\
            .def_method[PyComputePipeline.basis_1d]("basis_1d")\
            .def_method[PyComputePipeline.basis_2d]("basis_2d")\
            .def_method[PyComputePipeline.initial_condition]("initial_condition")\
            .def_method[PyComputePipeline.solve]("solve")\
            .def_method[PyComputePipeline.pdf]("pdf")\
            .def_method[PyComputePipeline.payoff_price]("payoff_price")\
            .def_method[PyComputePipeline.greeks]("greeks")
        return module.finalize()
    except e:
        print("Failed to init _fpe_native module: ", e)
        return PythonObject(None)
```

**注意**：`OwnedKwargsDict` 在 `def_function` 中接收 kwargs。根据 nightly 文档，`py_price(kwargs: OwnedKwargsDict[PythonObject])` 会自动接收 Python 端 `_native_price({"kappa": 1.2, ...})` 的字典参数作为关键字参数。

但更准确地说，Python 端调用 `_native_price(kappa=1.2, S0=60.0, ...)` 时，Mojo 侧收到 `kwargs` 字典。Python 端传 dict 时，需要解包：`_native_price(**params_dict)`。

- [ ] **Step 2: Delete `_fpe_state.mojo`**

手动删除文件：`src/bindings/_fpe_state.mojo`

- [ ] **Step 3: Run build to verify**

Run: `pixi run mojo build -I src src/bindings/_fpe_native.mojo`
Expected: compiles without error

- [ ] **Step 4: Commit**

```bash
git rm src/bindings/_fpe_state.mojo
git add src/bindings/_fpe_native.mojo
git commit -m "refactor: rewrite _fpe_native.mojo, remove session API, add PyComputePipeline type"
```

---

### Task 6: Update `bindings/c_abi.mojo` — C++ 接口改用 `ComputePipeline`

**Files:**
- Modify: `src/bindings/c_abi.mojo`

**Context:** 当前 C ABI 函数 `fpe_price` 接受 18 个平铺参数且内部重复构造 `HestonParams` + `FpeParams`。添加基于 `ComputePipeline` 指针的 C API。

- [ ] **Step 1: Add pipeline C API exports**

在 `src/bindings/c_abi.mojo` 末尾追加以下函数：

```mojo
from server.compute_pipeline import ComputePipeline
from engines.fpe.heston_params import HestonParams
from server.option_types import FpeParams

@export("fpe_pipeline_create", ABI="C")
def fpe_pipeline_create(
    kappa: Float64, theta: Float64, sigma: Float64, rho: Float64,
    r: Float64, T: Float64, S0: Float64, V0: Float64,
    n_s: Int32, n_v: Int32, barrier: Float64, option_type: Int32,
    num_insert: Int32,
) raises -> UnsafePointer[ComputePipeline, MutExternalOrigin]:
    var heston = HestonParams(
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        r=r, T=T, S0=S0, V0=V0,
        S_min=0.0, S_max=S0 * 3.0, V_min=0.0, V_max=1.0,
    )
    if not heston.is_valid():
        return UnsafePointer[ComputePipeline, MutExternalOrigin](unsafe_from_address=0)
    var strikes = List[Float64](100.0)
    var fp = FpeParams(
        heston=heston^, n_s=Int(n_s), n_v=Int(n_v),
        barrier=barrier, option_type=Int(option_type),
        strikes=strikes^,
    )
    var layout = Layout[ComputePipeline](count=1)
    var ptr = alloc[ComputePipeline](layout)
    ptr.init_pointee_move(ComputePipeline(fp^, num_insert=Int(num_insert)))
    return ptr

@export("fpe_pipeline_destroy", ABI="C")
def fpe_pipeline_destroy(ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]) raises:
    ptr.destroy_pointee()
    free(ptr, Layout[ComputePipeline](count=1))

@export("fpe_pipeline_price", ABI="C")
def fpe_pipeline_price(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin],
    K_ptr: UnsafePointer[Float64, MutExternalOrigin],
    n_strikes: Int32,
    out_ptr: UnsafePointer[Float64, MutExternalOrigin],
) raises -> Int32:
    var strikes: List[Float64] = List[Float64](cap=Int(n_strikes))
    for i in range(Int(n_strikes)):
        strikes.append(K_ptr[i])
    var prices = ptr[].price_at(strikes)
    var count = min(len(prices), Int(n_strikes))
    for i in range(count):
        out_ptr[i] = prices[i]
    return Int32(count)
```

- [ ] **Step 2: Build to verify**

Run: `pixi run mojo build -I src src/bindings/c_abi.mojo`
Expected: compiles without error

- [ ] **Step 3: Commit**

```bash
git add src/bindings/c_abi.mojo
git commit -m "feat: add C ABI pipeline create/destroy/price functions using ComputePipeline"
```

---

### Task 7: Update Python side — 简化 `FPEPricer` 为 `Compute` 上下文

**Files:**
- Modify: `python/fpe_engine/pricer.py`
- Modify: `python/fpe_engine/__init__.py`
- Modify: `tests/test_pricer_stepwise.py`

**Context:** Python 端废弃 `FPEPricer`（300 行 session 管理 + 手工缓存），替换为 `Compute` — 薄封装类，底层使用 Mojo `PyComputePipeline`。

- [ ] **Step 1: Rewrite `pricer.py`**

File: `python/fpe_engine/pricer.py`

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Self

import numpy as np
from scipy import sparse

from ._fpe_native import Compute as _NativeCompute


@dataclass
class PriceResult:
    prices: np.ndarray
    deltas: np.ndarray
    gammas: np.ndarray
    vegas: np.ndarray


@dataclass
class KnotsResult:
    s: np.ndarray
    v: np.ndarray


@dataclass
class GridPointsResult:
    s: np.ndarray
    v: np.ndarray
    s_weights: np.ndarray
    v_weights: np.ndarray


@dataclass
class Basis1DResult:
    Bs: sparse.csr_matrix
    dBs: sparse.csr_matrix
    Bv: sparse.csr_matrix
    dBv: sparse.csr_matrix


@dataclass
class GreeksResult:
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray


def _csr_from_mojo(d: dict) -> sparse.csr_matrix:
    data = np.array(d["data"], dtype=np.float64)
    indices = np.array(d["indices"], dtype=np.int32)
    indptr = np.array(d["indptr"], dtype=np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=(d["nrows"], d["ncols"]))


_OPTION_TYPES = {
    "down_and_in_call": 0, "down_and_in_put": 1,
    "down_and_out_call": 2, "down_and_out_put": 3,
    "up_and_in_call": 4, "up_and_in_put": 5,
    "up_and_out_call": 6, "up_and_out_put": 7,
    "european_call": 8, "european_put": 9,
}


@dataclass
class FpeParams:
    kappa: float = 1.2
    theta: float = 0.05
    sigma: float = 0.35
    rho: float = -0.4
    r: float = 0.05
    T: float = 0.5
    S0: float = 100.0
    V0: float = 0.1
    n_s: int = 38
    n_v: int = 38
    option_type: str | int = "european_call"
    K: float | list | None = None
    barrier: float = 0.0

    def __post_init__(self):
        if isinstance(self.option_type, str):
            if self.option_type not in _OPTION_TYPES:
                raise ValueError(f"unknown option_type '{self.option_type}'")
        else:
            if not (0 <= self.option_type <= 9):
                raise ValueError(f"option_type must be 0-9, got {self.option_type}")
        if not (4 <= self.n_s <= 256):
            raise ValueError(f"n_s must be in [4, 256], got {self.n_s}")
        if not (4 <= self.n_v <= 256):
            raise ValueError(f"n_v must be in [4, 256], got {self.n_v}")


def _normalize_K(K: float | int | list | np.ndarray) -> list[float]:
    if isinstance(K, np.ndarray):
        return K.tolist()
    if isinstance(K, (int, float)):
        return [float(K)]
    return [float(k) for k in K]


class Compute:
    """FPE computation context.

    Thin Python wrapper around Mojo ``PyComputePipeline``.
    Each property access triggers a Mojo call with Python-side caching.
    """

    def __init__(self, **kwargs):
        self._pipe = _NativeCompute(**kwargs)
        self._knots_cache: KnotsResult | None = None
        self._grid_points_cache: GridPointsResult | None = None
        self._basis_1d_cache: Basis1DResult | None = None
        self._basis_2d_cache: sparse.csr_matrix | None = None
        self._ic_cache: np.ndarray | None = None
        self._solve_cache: list[np.ndarray] | None = None
        self._pdf_cache: np.ndarray | None = None

    @property
    def knots(self) -> KnotsResult:
        if self._knots_cache is None:
            raw = self._pipe.knots()
            self._knots_cache = KnotsResult(
                s=np.array(raw["s"], dtype=np.float64),
                v=np.array(raw["v"], dtype=np.float64),
            )
        return self._knots_cache

    @property
    def grid_points(self) -> GridPointsResult:
        if self._grid_points_cache is None:
            raw = self._pipe.grid_points()
            self._grid_points_cache = GridPointsResult(
                s=np.array(raw["s"], dtype=np.float64),
                v=np.array(raw["v"], dtype=np.float64),
                s_weights=np.array(raw["s_weights"], dtype=np.float64),
                v_weights=np.array(raw["v_weights"], dtype=np.float64),
            )
        return self._grid_points_cache

    @property
    def basis_1d(self) -> Basis1DResult:
        if self._basis_1d_cache is None:
            raw = self._pipe.basis_1d()
            self._basis_1d_cache = Basis1DResult(
                Bs=_csr_from_mojo(raw["Bs"]),
                dBs=_csr_from_mojo(raw["dBs"]),
                Bv=_csr_from_mojo(raw["Bv"]),
                dBv=_csr_from_mojo(raw["dBv"]),
            )
        return self._basis_1d_cache

    @property
    def basis_2d(self) -> sparse.csr_matrix:
        if self._basis_2d_cache is None:
            raw = self._pipe.basis_2d()
            self._basis_2d_cache = _csr_from_mojo(raw)
        return self._basis_2d_cache

    @property
    def initial_condition(self) -> np.ndarray:
        if self._ic_cache is None:
            self._ic_cache = np.array(self._pipe.initial_condition(), dtype=np.float64)
        return self._ic_cache

    @property
    def solve(self) -> list[np.ndarray]:
        if self._solve_cache is None:
            raw = self._pipe.solve()
            self._solve_cache = [np.array(t, dtype=np.float64) for t in raw]
        return self._solve_cache

    @property
    def pdf(self) -> np.ndarray:
        if self._pdf_cache is None:
            raw = self._pipe.pdf()
            self._pdf_cache = np.array(raw, dtype=np.float64)
        return self._pdf_cache

    def payoff_price(self, K: float | int | list | np.ndarray) -> np.ndarray:
        K_list = _normalize_K(K)
        raw = self._pipe.payoff_price(K_list)
        return np.array(raw, dtype=np.float64)

    def greeks(self, K: float | int | list | np.ndarray) -> GreeksResult:
        K_list = _normalize_K(K)
        raw = self._pipe.greeks(K_list)
        return GreeksResult(
            delta=np.array(raw["delta"], dtype=np.float64),
            gamma=np.array(raw["gamma"], dtype=np.float64),
            vega=np.array(raw["vega"], dtype=np.float64),
        )
```

- [ ] **Step 2: Update `__init__.py`**

File: `python/fpe_engine/__init__.py`

```python
"""FPE Option Pricing Engine — Python binding.

Usage:
    import fpe_engine as fpe

    # One-shot (with Greeks)
    result = fpe.price(S0=60.0, K=[100.0], ...)
    # result.prices, result.deltas, result.gammas, result.vegas

    # Stepwise access to intermediates
    pipe = fpe.Compute(S0=60.0, V0=0.1, T=0.6, r=0.1)
    ks = pipe.knots
    gp = pipe.grid_points
    pdf = pipe.pdf
    prices = pipe.payoff_price(100.0)
    g = pipe.greeks([80.0, 100.0, 120.0])
"""

import logging

import numpy as np

_logger = logging.getLogger("fpe_engine")

try:
    from ._fpe_native import price as _native_price
    from ._fpe_native import Compute as NativeCompute
    _NATIVE_AVAILABLE = True
except ImportError as e:
    _logger.warning("Mojo FPE engine not available: %s", e)
    _NATIVE_AVAILABLE = False
except Exception as e:
    _logger.error("Unexpected error loading Mojo FPE engine: %s", e)
    _NATIVE_AVAILABLE = False

if _NATIVE_AVAILABLE:
    from .pricer import (
        Compute, FpeParams,
        PriceResult, KnotsResult, GridPointsResult,
        Basis1DResult, GreeksResult,
    )


def is_available() -> bool:
    return _NATIVE_AVAILABLE


_OPTION_TYPES = {
    "down_and_in_call": 0, "down_and_in_put": 1,
    "down_and_out_call": 2, "down_and_out_put": 3,
    "up_and_in_call": 4, "up_and_in_put": 5,
    "up_and_out_call": 6, "up_and_out_put": 7,
    "european_call": 8, "european_put": 9,
}


def price(
    kappa: float = 1.2, theta: float = 0.05, sigma: float = 0.35,
    rho: float = -0.4, r: float = 0.1, T: float = 0.6,
    S0: float = 60.0, V0: float = 0.1,
    K: list[float] | float = 100.0, barrier: float = 0.0,
    option_type: str | int = "european_call",
    n_s: int = 38, n_v: int = 38,
    rtol: float = 1e-4, atol: float = 1e-6,
) -> PriceResult:
    if not _NATIVE_AVAILABLE:
        raise RuntimeError("Mojo FPE engine not available")

    if isinstance(K, (int, float)):
        K = [float(K)]
    else:
        K = [float(k) for k in K]

    if isinstance(option_type, str):
        if option_type not in _OPTION_TYPES:
            raise ValueError(f"unknown option_type '{option_type}'")
        option_type_int = _OPTION_TYPES[option_type]
    else:
        option_type_int = int(option_type)

    kwargs = {
        "kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
        "r": r, "T": T, "S0": S0, "V0": V0,
        "K": K, "barrier": barrier, "option_type": option_type_int,
        "n_s": n_s, "n_v": n_v, "rtol": rtol, "atol": atol,
    }
    result = _native_price(kwargs)
    return PriceResult(
        prices=np.array(result["prices"], dtype=np.float64),
        deltas=np.array(result["deltas"], dtype=np.float64),
        gammas=np.array(result["gammas"], dtype=np.float64),
        vegas=np.array(result["vegas"], dtype=np.float64),
    )
```

- [ ] **Step 3: Update Python tests**

File: `tests/test_pricer_stepwise.py`

```python
"""Integration tests for Compute (stepwise API) and price (one-shot API)."""

import numpy as np
import pytest
from scipy import sparse

from fpe_engine import Compute, price, KnotsResult, GridPointsResult, Basis1DResult, GreeksResult, PriceResult


@pytest.fixture
def ctx():
    return Compute(S0=60.0, V0=0.1, T=0.6, r=0.1, n_s=8, n_v=8)


class TestInvalidParams:
    def test_invalid_n_s(self):
        with pytest.raises(Exception):
            Compute(n_s=1)

    def test_invalid_option_type(self):
        with pytest.raises(Exception):
            Compute(option_type="invalid")


class TestKnots:
    def test_returns_knots_result(self, ctx):
        k = ctx.knots
        assert isinstance(k, KnotsResult)
        assert isinstance(k.s, np.ndarray)
        assert isinstance(k.v, np.ndarray)
        assert k.s.dtype == np.float64
        assert len(k.s) > 0
        assert len(k.v) > 0

    def test_cached(self, ctx):
        k1 = ctx.knots
        k2 = ctx.knots
        assert k1 is k2


class TestGridPoints:
    def test_returns_grid_points_result(self, ctx):
        gp = ctx.grid_points
        assert isinstance(gp, GridPointsResult)
        assert isinstance(gp.s, np.ndarray)
        assert gp.s.dtype == np.float64
        assert len(gp.s) > 0

    def test_s_weights(self, ctx):
        gp = ctx.grid_points
        assert isinstance(gp.s_weights, np.ndarray)
        assert gp.s_weights.dtype == np.float64
        assert len(gp.s_weights) == len(gp.s)
        assert np.all(gp.s_weights >= 0.0)

    def test_cached(self, ctx):
        gp1 = ctx.grid_points
        gp2 = ctx.grid_points
        assert gp1 is gp2


class TestBasis1D:
    def test_returns_basis_1d_result(self, ctx):
        b = ctx.basis_1d
        assert isinstance(b, Basis1DResult)
        assert isinstance(b.Bs, sparse.csr_matrix)
        assert b.Bs.shape[0] > 0

    def test_cached(self, ctx):
        b1 = ctx.basis_1d
        b2 = ctx.basis_1d
        assert b1 is b2


class TestBasis2D:
    def test_returns_csr_matrix(self, ctx):
        b = ctx.basis_2d
        assert isinstance(b, sparse.csr_matrix)
        assert b.shape[0] > 0


class TestInitialCondition:
    def test_returns_ndarray(self, ctx):
        q0 = ctx.initial_condition
        assert isinstance(q0, np.ndarray)
        assert q0.dtype == np.float64


class TestSolve:
    def test_returns_list_of_ndarray(self, ctx):
        sol = ctx.solve
        assert isinstance(sol, list)
        assert len(sol) > 0
        for t in sol:
            assert isinstance(t, np.ndarray)
            assert t.dtype == np.float64


class TestPDF:
    def test_returns_2d_array(self, ctx):
        pdf = ctx.pdf
        assert isinstance(pdf, np.ndarray)
        assert pdf.ndim == 2
        assert pdf.dtype == np.float64


class TestPayoffPrice:
    def test_single_strike(self, ctx):
        p = ctx.payoff_price([100.0])
        assert isinstance(p, np.ndarray)
        assert len(p) == 1

    def test_multiple_strikes(self, ctx):
        p = ctx.payoff_price([80.0, 100.0, 120.0])
        assert isinstance(p, np.ndarray)
        assert len(p) == 3

    def test_k_variation(self, ctx):
        p1 = ctx.payoff_price([80.0])
        p2 = ctx.payoff_price([120.0])
        assert p1[0] != p2[0]


class TestPriceOneShot:
    def test_returns_price_result(self):
        pr = price(S0=60.0, K=100.0, T=0.6, n_s=8, n_v=8)
        assert isinstance(pr, PriceResult)
        assert isinstance(pr.prices, np.ndarray)
        assert isinstance(pr.deltas, np.ndarray)

    def test_multiple_strikes(self):
        pr = price(S0=60.0, K=[80.0, 100.0, 120.0], T=0.6, n_s=8, n_v=8)
        assert len(pr.prices) == 3


class TestGreeks:
    def test_single_strike(self, ctx):
        g = ctx.greeks([100.0])
        assert isinstance(g, GreeksResult)
        assert isinstance(g.delta, np.ndarray)
        assert g.delta.shape == (1,)
        assert g.gamma.shape == (1,)
        assert g.vega.shape == (1,)

    def test_multiple_strikes(self, ctx):
        g = ctx.greeks([65.0, 70.0, 75.0, 80.0])
        assert g.delta.shape == (4,)
        assert g.gamma.shape == (4,)
        assert g.vega.shape == (4,)

    def test_delta_negative(self, ctx):
        g = ctx.greeks([65.0, 100.0])
        assert np.all(g.delta < 0)  # put delta normally negative
```

- [ ] **Step 4: Build Mojo module for Python**

Run: `pixi run mojo build -I src src/bindings/_fpe_native.mojo --emit shared-lib -o python/fpe_engine/_fpe_native.so`
Expected: builds `.so` file

- [ ] **Step 5: Run Python tests**

Run: `pixi run python -m pytest tests/test_pricer_stepwise.py -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add python/fpe_engine/pricer.py python/fpe_engine/__init__.py tests/test_pricer_stepwise.py
git rm python/fpe_engine/pricer.py.old 2>/dev/null || true
git commit -m "refactor: simplify Python API, remove FPEPricer session, add Compute context"
```

---

### Task 8: Update existing Mojo tests — 修复 import 引用

**Files:**
- Modify: `tests/test_bindings.mojo`

**Context:** 确认现有测试不依赖 `_fpe_state`。

- [ ] **Step 1: Verify test_bindings.mojo**

查看 `tests/test_bindings.mojo` 确认没有 `import bindings._fpe_state` 引用。已有代码只使用 `server.option_types` 和 `server.pricing_engine`，不需要修改。

- [ ] **Step 2: Run all Mojo tests**

Run:
```bash
pixi run mojo test -I src tests/test_bindings.mojo
pixi run mojo test -I src tests/test_pricing_engine.mojo
pixi run mojo test -I src tests/test_fpe_params.mojo
pixi run mojo test -I src tests/test_compute_pipeline.mojo
pixi run mojo test -I src tests/test_params.mojo
pixi run mojo test -I src tests/test_convert.mojo
```

Expected: all pass

- [ ] **Step 3: Commit**

```bash
git commit -m "test: verify existing tests still pass after ComputePipeline refactor"
```

---

## Execution Order / Dependency Graph

```
Task 1 (server/compute_pipeline.mojo) ──────────────────┐
                                                         ├──▶ Task 4 (py_pipeline) ──▶ Task 5 (_fpe_native) ──▶ Task 7 (Python)
Task 2 (bindings/_params.mojo) ──────────────────────────┘                            │
Task 3 (bindings/_convert.mojo) ─────────────────────────┘                            ├──▶ Task 6 (c_abi.mojo)
                                                                                      └──▶ Task 8 (test fixes)
```

Task 1,2,3 可以并行执行。Task 4,6 依赖它们完成。Task 5 依赖 Task 4。Task 7 依赖 Task 5 编译出 `.so`。Task 8 最后收尾。

---

## Rollback Strategy

每个 Task 完成并提交后，通过 `git revert <hash>` 可精确回滚单个步骤。

如果 Task 5（删除 `_fpe_state.mojo`）导致问题：
```bash
git revert <task5-commit-hash>
# 恢复 _fpe_state.mojo 和旧 _fpe_native.mojo
```

如果 Python 端测试失败：
```bash
cd python && pixi run mojo build -I ../src ../src/bindings/_fpe_native.mojo --emit shared-lib -o fpe_engine/_fpe_native.so
```
确保重建 `.so` 文件。
