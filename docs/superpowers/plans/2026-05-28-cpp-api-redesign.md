# C++ API Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the session-managed C ABI with a heap-struct C ABI + C++ RAII wrapper, making C++ end-user code as simple as Python's `Compute` class.

**Architecture:** New `c_abi.mojo` exports flat C functions that return heap-allocated result structs (`{double* data, int32_t len}`). A C++ header-only `FpeCompute` class wraps these with RAII + lazy caching, mirroring Python's `Compute` API. `ComputePipeline` struct is unchanged.

**Tech Stack:** Mojo (C ABI @export), C17/C++17, cmake, pixi

---

## File Structure

| File | Responsibility |
|---|---|
| `src/bindings/c_abi.mojo` | **Rewrite** — new heap-struct C ABI functions |
| `cpp/include/fpe_engine.h` | **Rewrite** — C result structs + function declarations |
| `cpp/include/fpe_compute.hpp` | **New** — C++ RAII wrapper class `FpeCompute` |
| `cpp/examples/demo.cpp` | **New** — simplified C++ example |
| `cpp/examples/pipeline_demo.cpp` | **Delete** |
| `cpp/examples/live_trading.cpp` | **Delete** |
| `cpp/examples/CMakeLists.txt` | **Update** — build `demo` only |
| `scripts/build_cpp.sh` | **Update** — echo new binary name + copy new header |

**Unchanged:** `compute_pipeline.mojo`, `_fpe_native.mojo`, `_py_pipeline.mojo`, `pricer.py`, Python tests

---

### Task 1: Rewrite C Header — `cpp/include/fpe_engine.h`

**Files:**
- Modify: `cpp/include/fpe_engine.h` (full rewrite)

- [ ] **Step 1: Write the new C header**

Replace the entire file with:

```c
#ifndef FPE_ENGINE_H
#define FPE_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpeCompute FpeCompute;

struct FpeVecResult {
    double* data;
    int32_t len;
};

struct FpeVec2Result {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
};

struct FpeGridPtsResult {
    double* s_data;
    int32_t s_len;
    double* v_data;
    int32_t v_len;
    double* sw_data;
    double* vw_data;
};

struct FpeMatResult {
    double* data;
    int32_t n_rows;
    int32_t n_cols;
};

struct FpeGreeksResult {
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

struct FpeOneshotResult {
    double* price;
    double* delta;
    double* gamma;
    double* vega;
    int32_t len;
};

extern FpeCompute* fpe_compute_create(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    int32_t n_s, int32_t n_v,
    double barrier, int32_t option_type,
    int32_t num_insert
);

extern void fpe_compute_destroy(FpeCompute* ptr);

extern struct FpeVec2Result fpe_compute_knots(FpeCompute* ptr);

extern struct FpeGridPtsResult fpe_compute_grid_points(FpeCompute* ptr);

extern struct FpeVecResult fpe_compute_initial_condition(FpeCompute* ptr);

extern struct FpeMatResult fpe_compute_solve(FpeCompute* ptr);

extern struct FpeMatResult fpe_compute_pdf(FpeCompute* ptr);

extern struct FpeVecResult fpe_compute_price(
    FpeCompute* ptr,
    const double* K, int32_t n_K
);

extern struct FpeGreeksResult fpe_compute_greeks(
    FpeCompute* ptr,
    const double* K, int32_t n_K,
    double rel_s, double rel_v
);

extern struct FpeOneshotResult fpe_price_oneshot(
    double kappa, double theta, double sigma, double rho,
    double r, double T, double S0, double V0,
    const double* K, int32_t n_K,
    double barrier, int32_t option_type,
    int32_t n_s, int32_t n_v,
    int32_t num_insert
);

extern void fpe_compute_free_vec(struct FpeVecResult* r);
extern void fpe_compute_free_vec2(struct FpeVec2Result* r);
extern void fpe_compute_free_grid_pts(struct FpeGridPtsResult* r);
extern void fpe_compute_free_mat(struct FpeMatResult* r);
extern void fpe_compute_free_greeks(struct FpeGreeksResult* r);
extern void fpe_compute_free_oneshot(struct FpeOneshotResult* r);

#ifdef __cplusplus
}
#endif

#endif
```

- [ ] **Step 2: Commit**

```bash
git add cpp/include/fpe_engine.h
git commit -m "refactor: rewrite C header with heap-struct result types"
```

---

### Task 2: Rewrite C ABI — `src/bindings/c_abi.mojo`

**Files:**
- Modify: `src/bindings/c_abi.mojo` (full rewrite)

- [ ] **Step 1: Write the new C ABI Mojo file**

Replace the entire file with:

```mojo
from server.option_types import FpeParams, PricingResult
from server.pricing_engine import PricingEngine
from server.compute_pipeline import ComputePipeline
from engines.fpe.heston_params import HestonParams
from std.memory import alloc, free, Layout


@fieldwise_init
struct FpeVecResult(TrivialRegisterPassable):
    var data: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


@fieldwise_init
struct FpeVec2Result(TrivialRegisterPassable):
    var s_data: UnsafePointer[Float64, MutExternalOrigin]
    var s_len: Int32
    var v_data: UnsafePointer[Float64, MutExternalOrigin]
    var v_len: Int32


@fieldwise_init
struct FpeGridPtsResult(TrivialRegisterPassable):
    var s_data: UnsafePointer[Float64, MutExternalOrigin]
    var s_len: Int32
    var v_data: UnsafePointer[Float64, MutExternalOrigin]
    var v_len: Int32
    var sw_data: UnsafePointer[Float64, MutExternalOrigin]
    var vw_data: UnsafePointer[Float64, MutExternalOrigin]


@fieldwise_init
struct FpeMatResult(TrivialRegisterPassable):
    var data: UnsafePointer[Float64, MutExternalOrigin]
    var n_rows: Int32
    var n_cols: Int32


@fieldwise_init
struct FpeGreeksResult(TrivialRegisterPassable):
    var delta: UnsafePointer[Float64, MutExternalOrigin]
    var gamma: UnsafePointer[Float64, MutExternalOrigin]
    var vega: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


@fieldwise_init
struct FpeOneshotResult(TrivialRegisterPassable):
    var price: UnsafePointer[Float64, MutExternalOrigin]
    var delta: UnsafePointer[Float64, MutExternalOrigin]
    var gamma: UnsafePointer[Float64, MutExternalOrigin]
    var vega: UnsafePointer[Float64, MutExternalOrigin]
    var len: Int32


def _list_to_heap(var src: List[Float64]) -> UnsafePointer[Float64, MutExternalOrigin]:
    var n = len(src)
    if n == 0:
        return UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=0)
    var ptr = alloc[Float64](n)
    for i in range(n):
        ptr[i] = src[i]
    return ptr


def _mat_to_heap(data: List[List[Float64]]) -> Tuple[UnsafePointer[Float64, MutExternalOrigin], Int, Int]:
    var n_rows = len(data)
    if n_rows == 0:
        return (
            UnsafePointer[Float64, MutExternalOrigin](unsafe_from_address=0),
            0,
            0,
        )
    var n_cols = len(data[0])
    var total = n_rows * n_cols
    var ptr = alloc[Float64](total)
    for i in range(n_rows):
        var row = data[i].copy()
        var base = i * n_cols
        for j in range(min(len(row), n_cols)):
            ptr[base + j] = row[j]
    return (ptr, n_rows, n_cols)


def _free_heap(ptr: UnsafePointer[Float64, MutExternalOrigin], n: Int):
    if ptr.address != 0:
        free(ptr, Layout[Float64](count=n))


@export("fpe_compute_create", ABI="C")
def fpe_compute_create(
    kappa: Float64, theta: Float64, sigma: Float64, rho: Float64,
    r: Float64, T: Float64, S0: Float64, V0: Float64,
    n_s: Int32, n_v: Int32,
    barrier: Float64, option_type: Int32,
    num_insert: Int32,
) -> UnsafePointer[ComputePipeline, MutExternalOrigin]:
    var heston = HestonParams(
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        r=r, T=T, S0=S0, V0=V0,
        S_min=0.0, S_max=S0 * 3.0, V_min=0.0, V_max=1.0,
    )
    if not heston.is_valid():
        return UnsafePointer[ComputePipeline, MutExternalOrigin](
            unsafe_from_address=0
        )
    var strikes = [100.0]
    var fp = FpeParams(
        heston=heston^, n_s=Int(n_s), n_v=Int(n_v),
        barrier=barrier, option_type=Int(option_type),
        strikes=strikes^,
    )
    var ptr = alloc[ComputePipeline](1)
    try:
        ptr.init_pointee_move(ComputePipeline(fp^, num_insert=Int(num_insert)))
        return ptr
    except:
        free(ptr, Layout[ComputePipeline](count=1))
        return UnsafePointer[ComputePipeline, MutExternalOrigin](
            unsafe_from_address=0
        )


@export("fpe_compute_destroy", ABI="C")
def fpe_compute_destroy(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
):
    if ptr.address != 0:
        ptr.destroy_pointee()
        free(ptr, Layout[ComputePipeline](count=1))


@export("fpe_compute_knots", ABI="C")
def fpe_compute_knots(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) -> FpeVec2Result:
    var tup = ptr[].knots()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var s_len = len(s)
    var v_len = len(v)
    return FpeVec2Result(
        s_data=_list_to_heap(s^), s_len=Int32(s_len),
        v_data=_list_to_heap(v^), v_len=Int32(v_len),
    )


@export("fpe_compute_grid_points", ABI="C")
def fpe_compute_grid_points(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) -> FpeGridPtsResult:
    var tup = ptr[].grid_points()
    var s = tup[0].copy()
    var v = tup[1].copy()
    var sw = tup[2].copy()
    var vw = tup[3].copy()
    return FpeGridPtsResult(
        s_data=_list_to_heap(s^), s_len=Int32(len(s)),
        v_data=_list_to_heap(v^), v_len=Int32(len(v)),
        sw_data=_list_to_heap(sw^), vw_data=_list_to_heap(vw^),
    )


@export("fpe_compute_initial_condition", ABI="C")
def fpe_compute_initial_condition(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeVecResult:
    var q0 = ptr[].initial_condition()
    var n = len(q0)
    return FpeVecResult(data=_list_to_heap(q0^), len=Int32(n))


@export("fpe_compute_solve", ABI="C")
def fpe_compute_solve(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeMatResult:
    var sol = ptr[].solve()
    var heap_tup = _mat_to_heap(sol)
    return FpeMatResult(
        data=heap_tup[0], n_rows=Int32(heap_tup[1]), n_cols=Int32(heap_tup[2]),
    )


@export("fpe_compute_pdf", ABI="C")
def fpe_compute_pdf(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin]
) raises -> FpeMatResult:
    var pdf_grid = ptr[].pdf()
    var heap_tup = _mat_to_heap(pdf_grid)
    return FpeMatResult(
        data=heap_tup[0], n_rows=Int32(heap_tup[1]), n_cols=Int32(heap_tup[2]),
    )


@export("fpe_compute_price", ABI="C")
def fpe_compute_price(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin],
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
) raises -> FpeVecResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var prices = ptr[].price_at(strikes)
    var n = len(prices)
    return FpeVecResult(data=_list_to_heap(prices^), len=Int32(n))


@export("fpe_compute_greeks", ABI="C")
def fpe_compute_greeks(
    ptr: UnsafePointer[ComputePipeline, MutExternalOrigin],
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    rel_s: Float64,
    rel_v: Float64,
) raises -> FpeGreeksResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var g_tup = ptr[].greeks(strikes, rel_s=rel_s, rel_v=rel_v)
    var deltas = g_tup[0].copy()
    var gammas = g_tup[1].copy()
    var vegas = g_tup[2].copy()
    return FpeGreeksResult(
        delta=_list_to_heap(deltas^),
        gamma=_list_to_heap(gammas^),
        vega=_list_to_heap(vegas^),
        len=Int32(len(deltas)),
    )


@export("fpe_price_oneshot", ABI="C")
def fpe_price_oneshot(
    kappa: Float64, theta: Float64, sigma: Float64, rho: Float64,
    r: Float64, T: Float64, S0: Float64, V0: Float64,
    K_ptr: UnsafePointer[Float64, MutAnyOrigin],
    n_K: Int32,
    barrier: Float64, option_type: Int32,
    n_s: Int32, n_v: Int32,
    num_insert: Int32,
) raises -> FpeOneshotResult:
    var strikes: List[Float64] = List[Float64](capacity=Int(n_K))
    for i in range(Int(n_K)):
        strikes.append(K_ptr[i])
    var heston = HestonParams(
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        r=r, T=T, S0=S0, V0=V0,
        S_min=0.0, S_max=S0 * 3.0, V_min=0.0, V_max=1.0,
    )
    var fp = FpeParams(
        heston=heston^, n_s=Int(n_s), n_v=Int(n_v),
        barrier=barrier, option_type=Int(option_type),
        strikes=strikes^,
    )
    var engine = PricingEngine(num_insert=Int(num_insert))
    var results = engine.price(fp)
    var count = len(results)
    var p_ptr = alloc[Float64](count)
    var d_ptr = alloc[Float64](count)
    var g_ptr = alloc[Float64](count)
    var v_ptr = alloc[Float64](count)
    for i in range(count):
        p_ptr[i] = results[i].price
        d_ptr[i] = results[i].delta
        g_ptr[i] = results[i].gamma
        v_ptr[i] = results[i].vega
    return FpeOneshotResult(
        price=p_ptr, delta=d_ptr, gamma=g_ptr, vega=v_ptr,
        len=Int32(count),
    )


@export("fpe_compute_free_vec", ABI="C")
def fpe_compute_free_vec(r: FpeVecResult):
    _free_heap(r.data, Int(r.len))


@export("fpe_compute_free_vec2", ABI="C")
def fpe_compute_free_vec2(r: FpeVec2Result):
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))


@export("fpe_compute_free_grid_pts", ABI="C")
def fpe_compute_free_grid_pts(r: FpeGridPtsResult):
    _free_heap(r.s_data, Int(r.s_len))
    _free_heap(r.v_data, Int(r.v_len))


@export("fpe_compute_free_mat", ABI="C")
def fpe_compute_free_mat(r: FpeMatResult):
    if r.data.address != 0:
        var total = Int(r.n_rows) * Int(r.n_cols)
        _free_heap(r.data, total)


@export("fpe_compute_free_greeks", ABI="C")
def fpe_compute_free_greeks(r: FpeGreeksResult):
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))


@export("fpe_compute_free_oneshot", ABI="C")
def fpe_compute_free_oneshot(r: FpeOneshotResult):
    _free_heap(r.price, Int(r.len))
    _free_heap(r.delta, Int(r.len))
    _free_heap(r.gamma, Int(r.len))
    _free_heap(r.vega, Int(r.len))
```

- [ ] **Step 2: Build the shared library to verify it compiles**

Run: `pixi run mojo build -I src --emit shared-lib -o /tmp/libfpe_engine_test.dylib src/bindings/c_abi.mojo`

Expected: successful compilation, no errors

- [ ] **Step 3: Commit**

```bash
git add src/bindings/c_abi.mojo
git commit -m "refactor: rewrite C ABI with heap-struct result types"
```

---

### Task 3: Create C++ RAII Wrapper — `cpp/include/fpe_compute.hpp`

**Files:**
- Create: `cpp/include/fpe_compute.hpp`

- [ ] **Step 1: Write the header-only C++ wrapper**

Create `cpp/include/fpe_compute.hpp`:

```cpp
#ifndef FPE_COMPUTE_HPP
#define FPE_COMPUTE_HPP

#include "fpe_engine.h"
#include <vector>
#include <optional>
#include <cstddef>
#include <cstdio>

namespace fpe {

struct KnotsResult {
    std::vector<double> s;
    std::vector<double> v;
};

struct GridPointsResult {
    std::vector<double> s;
    std::vector<double> v;
    std::vector<double> s_weights;
    std::vector<double> v_weights;
};

struct GreeksResult {
    std::vector<double> delta;
    std::vector<double> gamma;
    std::vector<double> vega;
};

struct OneshotResult {
    std::vector<double> price;
    std::vector<double> delta;
    std::vector<double> gamma;
    std::vector<double> vega;
};

class FpeCompute {
    FpeCompute* ptr_;
    mutable std::optional<KnotsResult> knots_;
    mutable std::optional<GridPointsResult> grid_points_;
    mutable std::optional<std::vector<double>> initial_condition_;
    mutable std::optional<std::vector<std::vector<double>>> solve_;
    mutable std::optional<std::vector<std::vector<double>>> pdf_;

    static std::vector<double> vec_from_c(FpeVecResult&& r) {
        std::vector<double> v(r.data, r.data + r.len);
        fpe_compute_free_vec(&r);
        return v;
    }

    static KnotsResult knots_from_c(FpeVec2Result&& r) {
        KnotsResult k;
        k.s.assign(r.s_data, r.s_data + r.s_len);
        k.v.assign(r.v_data, r.v_data + r.v_len);
        fpe_compute_free_vec2(&r);
        return k;
    }

    static GridPointsResult grid_from_c(FpeGridPtsResult&& r) {
        GridPointsResult g;
        g.s.assign(r.s_data, r.s_data + r.s_len);
        g.v.assign(r.v_data, r.v_data + r.v_len);
        g.s_weights.assign(r.sw_data, r.sw_data + r.s_len);
        g.v_weights.assign(r.vw_data, r.vw_data + r.v_len);
        fpe_compute_free_grid_pts(&r);
        return g;
    }

    static std::vector<std::vector<double>> mat_from_c(FpeMatResult&& r) {
        std::vector<std::vector<double>> m;
        if (r.n_rows == 0 || r.n_cols == 0 || r.data == nullptr) {
            fpe_compute_free_mat(&r);
            return m;
        }
        m.resize(r.n_rows);
        for (int32_t i = 0; i < r.n_rows; i++) {
            m[i].assign(r.data + i * r.n_cols, r.data + (i + 1) * r.n_cols);
        }
        fpe_compute_free_mat(&r);
        return m;
    }

    static GreeksResult greeks_from_c(FpeGreeksResult&& r) {
        GreeksResult g;
        g.delta.assign(r.delta, r.delta + r.len);
        g.gamma.assign(r.gamma, r.gamma + r.len);
        g.vega.assign(r.vega, r.vega + r.len);
        fpe_compute_free_greeks(&r);
        return g;
    }

public:
    FpeCompute(
        double kappa, double theta, double sigma, double rho,
        double r, double T, double S0, double V0,
        int32_t n_s, int32_t n_v,
        double barrier, int32_t option_type,
        int32_t num_insert = 50
    ) : ptr_(nullptr) {
        ptr_ = fpe_compute_create(
            kappa, theta, sigma, rho,
            r, T, S0, V0,
            n_s, n_v, barrier, option_type, num_insert
        );
        if (!ptr_) {
            fprintf(stderr, "FpeCompute: failed to create pipeline\n");
        }
    }

    ~FpeCompute() {
        if (ptr_) {
            fpe_compute_destroy(ptr_);
            ptr_ = nullptr;
        }
    }

    FpeCompute(const FpeCompute&) = delete;
    FpeCompute& operator=(const FpeCompute&) = delete;
    FpeCompute(FpeCompute&& o) noexcept : ptr_(o.ptr_) {
        o.ptr_ = nullptr;
        knots_ = std::move(o.knots_);
        grid_points_ = std::move(o.grid_points_);
        initial_condition_ = std::move(o.initial_condition_);
        solve_ = std::move(o.solve_);
        pdf_ = std::move(o.pdf_);
    }
    FpeCompute& operator=(FpeCompute&& o) noexcept {
        if (this != &o) {
            if (ptr_) fpe_compute_destroy(ptr_);
            ptr_ = o.ptr_;
            o.ptr_ = nullptr;
            knots_ = std::move(o.knots_);
            grid_points_ = std::move(o.grid_points_);
            initial_condition_ = std::move(o.initial_condition_);
            solve_ = std::move(o.solve_);
            pdf_ = std::move(o.pdf_);
        }
        return *this;
    }

    bool valid() const { return ptr_ != nullptr; }

    KnotsResult knots() const {
        if (!knots_) {
            knots_ = knots_from_c(fpe_compute_knots(ptr_));
        }
        return knots_.value();
    }

    GridPointsResult grid_points() const {
        if (!grid_points_) {
            grid_points_ = grid_from_c(fpe_compute_grid_points(ptr_));
        }
        return grid_points_.value();
    }

    std::vector<double> initial_condition() {
        if (!initial_condition_) {
            initial_condition_ = vec_from_c(fpe_compute_initial_condition(ptr_));
        }
        return initial_condition_.value();
    }

    std::vector<std::vector<double>> solve() {
        if (!solve_) {
            solve_ = mat_from_c(fpe_compute_solve(ptr_));
        }
        return solve_.value();
    }

    std::vector<std::vector<double>> pdf() {
        if (!pdf_) {
            pdf_ = mat_from_c(fpe_compute_pdf(ptr_));
        }
        return pdf_.value();
    }

    std::vector<double> price(const std::vector<double>& K) {
        return vec_from_c(fpe_compute_price(ptr_, K.data(), int32_t(K.size())));
    }

    GreeksResult greeks(const std::vector<double>& K, double rel_s = 0.01, double rel_v = 0.1) {
        return greeks_from_c(fpe_compute_greeks(ptr_, K.data(), int32_t(K.size()), rel_s, rel_v));
    }

    static OneshotResult price_oneshot(
        double kappa, double theta, double sigma, double rho,
        double r, double T, double S0, double V0,
        const std::vector<double>& K,
        double barrier, int32_t option_type,
        int32_t n_s, int32_t n_v,
        int32_t num_insert = 50
    ) {
        auto raw = fpe_price_oneshot(
            kappa, theta, sigma, rho,
            r, T, S0, V0,
            K.data(), int32_t(K.size()),
            barrier, option_type, n_s, n_v, num_insert
        );
        OneshotResult o;
        o.price.assign(raw.price, raw.price + raw.len);
        o.delta.assign(raw.delta, raw.delta + raw.len);
        o.gamma.assign(raw.gamma, raw.gamma + raw.len);
        o.vega.assign(raw.vega, raw.vega + raw.len);
        fpe_compute_free_oneshot(&raw);
        return o;
    }
};

}

#endif
```

- [ ] **Step 2: Commit**

```bash
git add cpp/include/fpe_compute.hpp
git commit -m "feat: add C++ RAII wrapper FpeCompute with lazy caching"
```

---

### Task 4: Create C++ Demo — `cpp/examples/demo.cpp`

**Files:**
- Create: `cpp/examples/demo.cpp`

- [ ] **Step 1: Write the simplified demo**

Create `cpp/examples/demo.cpp`:

```cpp
#include <cstdio>
#include <vector>
#include "fpe_compute.hpp"

int main() {
    printf("=== FPE Compute Pipeline - C++ API ===\n\n");

    std::vector<double> K = {65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0};

    printf("[1] Create pipeline - European Call\n");
    fpe::FpeCompute eu(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        16, 16, 0.0, 8
    );
    if (!eu.valid()) { printf("ERROR: failed to create European pipeline\n"); return 1; }

    printf("[2] Create pipeline - Down-and-Out Barrier Call\n");
    fpe::FpeCompute bar(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        16, 16, 50.0, 2
    );
    if (!bar.valid()) { printf("ERROR: failed to create barrier pipeline\n"); return 1; }

    printf("\n[3] Knots\n");
    auto k_eu = eu.knots();
    printf("[European] s_knots(%zu):", k_eu.s.size());
    for (size_t i = 0; i < k_eu.s.size() && i < 6; i++) printf(" %.4f", k_eu.s[i]);
    if (k_eu.s.size() > 6) printf(" ...");
    printf("\n");
    auto k_bar = bar.knots();
    printf("[Barrier]  s_knots(%zu):", k_bar.s.size());
    for (size_t i = 0; i < k_bar.s.size() && i < 6; i++) printf(" %.4f", k_bar.s[i]);
    if (k_bar.s.size() > 6) printf(" ...");
    printf("\n");

    printf("\n[4] Grid Points\n");
    auto gp = eu.grid_points();
    printf("[European] %zu s-pts, %zu v-pts\n", gp.s.size(), gp.v.size());

    printf("\n[5] Initial Condition\n");
    auto q0 = eu.initial_condition();
    printf("[European] q0 length=%zu, q0[0]=%.6f, q0[%zu]=%.6f\n",
           q0.size(), q0[0], q0.size() - 1, q0[q0.size() - 1]);

    printf("\n[6] Solve\n");
    auto sol = eu.solve();
    printf("[European] solution: %zu time steps, %zu DOF per step\n",
           sol.size(), sol.empty() ? 0 : sol[0].size());

    printf("\n[7] PDF\n");
    auto pdf = eu.pdf();
    printf("[European] PDF: %zu rows x %zu cols\n",
           pdf.size(), pdf.empty() ? 0 : pdf[0].size());

    printf("\n[8] Pricing\n");
    auto prices_eu = eu.price(K);
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_eu[i]);
    auto prices_bar = bar.price(K);
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f\n", K[i], prices_bar[i]);

    printf("\n[9] Greeks\n");
    auto g_eu = eu.greeks(K);
    printf("[European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], g_eu.delta[i], g_eu.gamma[i], g_eu.vega[i]);
    auto g_bar = bar.greeks(K);
    printf("[Down-and-Out Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], g_bar.delta[i], g_bar.gamma[i], g_bar.vega[i]);

    printf("\n[10] One-shot pricing (PricingEngine)\n");
    auto os = fpe::FpeCompute::price_oneshot(
        1.2, 0.05, 0.35, -0.4,
        0.1, 0.6, 60.0, 0.1,
        K, 0.0, 8, 16, 16
    );
    printf("[One-shot European Call]:\n");
    for (size_t i = 0; i < K.size(); i++)
        printf("  K=%.1f: price=%.6f delta=%.6f gamma=%.6f vega=%.6f\n",
               K[i], os.price[i], os.delta[i], os.gamma[i], os.vega[i]);

    printf("\nDone.\n");
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add cpp/examples/demo.cpp
git commit -m "feat: add simplified C++ demo using FpeCompute wrapper"
```

---

### Task 5: Delete old C++ examples and update CMakeLists.txt

**Files:**
- Delete: `cpp/examples/pipeline_demo.cpp`
- Delete: `cpp/examples/live_trading.cpp`
- Modify: `cpp/examples/CMakeLists.txt`

- [ ] **Step 1: Delete old examples**

```bash
rm cpp/examples/pipeline_demo.cpp cpp/examples/live_trading.cpp
```

- [ ] **Step 2: Update CMakeLists.txt**

Replace entire `cpp/examples/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(fpe_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT FPE_PREFIX)
  message(FATAL_ERROR
    "FPE_PREFIX not set. Pass -DFPE_PREFIX=<pixi env prefix>\n"
    " e.g.: cmake -DFPE_PREFIX=$(pixi env prefix) .."
  )
endif()

set(FPE_LIB_DIR "${FPE_PREFIX}/lib")
set(FPE_INC_DIR "${FPE_PREFIX}/include")

add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${FPE_INC_DIR})
target_link_directories(demo PRIVATE ${FPE_LIB_DIR})
target_link_libraries(demo PRIVATE fpe_engine)
set_target_properties(demo PROPERTIES
  BUILD_RPATH "${FPE_LIB_DIR}"
  INSTALL_RPATH "${FPE_LIB_DIR}"
)
```

- [ ] **Step 3: Commit**

```bash
git add -A cpp/examples/
git commit -m "refactor: replace pipeline_demo + live_trading with demo, update cmake"
```

---

### Task 6: Update `scripts/build_cpp.sh`

**Files:**
- Modify: `scripts/build_cpp.sh`

- [ ] **Step 1: Update the build script**

In `scripts/build_cpp.sh`, change step 2 to also copy the new header, and update the echo at the bottom:

Replace the echo lines at the bottom (lines 50-52):

Old:
```
echo " C++: ${BUILD_DIR}/pipeline_demo"
```

New:
```
echo " C++: ${BUILD_DIR}/demo"
```

Also add a copy for the new header after the existing `cp` in step 2 (after line 30):

Old:
```bash
cp cpp/include/fpe_engine.h "${PREFIX}/include/"
```

New:
```bash
cp cpp/include/fpe_engine.h "${PREFIX}/include/"
cp cpp/include/fpe_compute.hpp "${PREFIX}/include/"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_cpp.sh
git commit -m "chore: update build_cpp.sh for new demo binary + fpe_compute.hpp header"
```

---

### Task 7: Full build + smoke test

**Files:** None (verification only)

- [ ] **Step 1: Clean old build artifacts**

```bash
rm -rf cpp/examples/build
```

- [ ] **Step 2: Run the full build script**

```bash
bash scripts/build_cpp.sh
```

Expected: all 5 steps complete successfully, `demo` binary built in `cpp/examples/build/`

- [ ] **Step 3: Run the demo**

```bash
cpp/examples/build/demo
```

Expected: output showing all 10 sections (create, knots, grid_points, initial_condition, solve, pdf, pricing, greeks, oneshot) with no errors. Timeout: 300s max.

- [ ] **Step 4: Run Python tests to verify no regression**

```bash
pixi run mojo test -I src tests/
```

Expected: all 21 tests pass

- [ ] **Step 5: Commit final state if any fixes were needed**

```bash
git add -A
git commit -m "fix: adjustments from build+test verification"
```

---

## Self-Review

**Spec coverage check:**
- C ABI heap-struct result types → Task 1 (C header) + Task 2 (Mojo) ✓
- `fpe_compute_create/destroy` → Task 2 ✓
- `fpe_compute_knots/grid_points/initial_condition/solve/pdf/price/greeks` → Task 2 ✓
- `fpe_price_oneshot` returns price+greeks → Task 2 ✓
- Free functions for all result types → Task 2 ✓
- C++ RAII wrapper `FpeCompute` → Task 3 ✓
- C++ demo with pipeline + oneshot → Task 4 ✓
- Delete old examples → Task 5 ✓
- Build system updates → Task 5 + Task 6 ✓
- `ComputePipeline` unchanged → no task touches it ✓

**Placeholder scan:** No TBD/TODO/fill-in-later found. All code blocks are complete.

**Type consistency:**
- `FpeVecResult` fields: `data` + `len` — consistent across C header, Mojo struct, C++ wrapper ✓
- `FpeVec2Result` fields: `s_data`, `s_len`, `v_data`, `v_len` — consistent ✓
- `FpeGridPtsResult` fields: `s_data`, `s_len`, `v_data`, `v_len`, `sw_data`, `vw_data` — consistent ✓
- `FpeMatResult` fields: `data`, `n_rows`, `n_cols` — consistent ✓
- `FpeGreeksResult` fields: `delta`, `gamma`, `vega`, `len` — consistent ✓
- `FpeOneshotResult` fields: `price`, `delta`, `gamma`, `vega`, `len` — consistent ✓
- Free function names match result struct names ✓
- C++ wrapper method names match Python `Compute` class ✓
