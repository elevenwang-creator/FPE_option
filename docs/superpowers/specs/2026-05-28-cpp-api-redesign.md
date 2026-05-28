# C++ API Redesign: Remove c_abi.mojo Session Management, Add Heap-Struct C ABI + C++ RAII Wrapper

## Problem

The current C ABI (`src/bindings/c_abi.mojo`) uses a session-managed pattern with opaque `FpePipeline*` pointers, caller-allocated buffers with capacity tracking, and packed int encoding (`s_len * 10000 + v_len`) to return multiple sizes. This makes C++ end-user code complex — manual buffer allocation, capacity management, and pointer lifecycle.

Goal: Make C++ end-user API as simple as Python's `Compute` class.

## Design Decisions

1. **Remove** the current `c_abi.mojo` session-management pattern entirely
2. **Keep** `ComputePipeline` struct unchanged in `compute_pipeline.mojo`
3. **New C ABI**: flat functions returning heap-allocated result structs (`{double* data, int32_t len}`)
4. **C++ header-only wrapper**: `FpeCompute` class with RAII + lazy caching, mirroring Python `Compute`
5. **One-shot API**: `fpe_price_oneshot` using `PricingEngine`, returns price+greeks
6. **Export `solve()`**: Full time-stepping solution available to C++ users
7. **Single demo**: Replace `pipeline_demo.cpp` + `live_trading.cpp` with one `demo.cpp`

## C ABI — Result Structs

All result structs are `TrivialRegisterPassable` so they cross the C boundary safely. Each owns heap-allocated data that must be freed via the corresponding `fpe_compute_free_*` function.

```c
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
```

## C ABI — Exported Functions

### Lifecycle

| Function | Signature | Notes |
|---|---|---|
| `fpe_compute_create` | `(kappa,theta,sigma,rho, r,T,S0,V0, n_s,n_v, barrier,option_type, num_insert) -> FpeCompute*` | Heap-allocates `ComputePipeline` |
| `fpe_compute_destroy` | `(FpeCompute*) -> void` | Frees `ComputePipeline` |

### Pipeline Data Access

| Function | Returns | Notes |
|---|---|---|
| `fpe_compute_knots` | `FpeVec2Result` | s_knots + v_knots |
| `fpe_compute_grid_points` | `FpeGridPtsResult` | s_pts, v_pts, s_weights, v_weights |
| `fpe_compute_initial_condition` | `FpeVecResult` | q0 vector (triggers IC computation) |
| `fpe_compute_solve` | `FpeMatResult` | Full time-stepping solution (triggers solver) |
| `fpe_compute_pdf` | `FpeMatResult` | Row-major flat array + dims (triggers solve+pdf) |
| `fpe_compute_price` | `(FpeCompute*, K_ptr, n_K) -> FpeVecResult` | Prices for given strikes |
| `fpe_compute_greeks` | `(FpeCompute*, K_ptr, n_K, rel_s, rel_v) -> FpeGreeksResult` | delta/gamma/vega arrays |

### One-Shot (PricingEngine-based)

| Function | Returns | Notes |
|---|---|---|
| `fpe_price_oneshot` | `(kappa,...,K,n_K,barrier,option_type,n_s,n_v,num_insert) -> FpeOneshotResult` | Uses `PricingEngine`, returns price+delta+gamma+vega per strike |

### Free Functions

| Function | Frees |
|---|---|
| `fpe_compute_free_vec` | `FpeVecResult.data` |
| `fpe_compute_free_vec2` | `FpeVec2Result.s_data` + `.v_data` |
| `fpe_compute_free_grid_pts` | `FpeGridPtsResult` all 4 arrays |
| `fpe_compute_free_mat` | `FpeMatResult.data` |
| `fpe_compute_free_greeks` | `FpeGreeksResult` 3 arrays |
| `fpe_compute_free_oneshot` | `FpeOneshotResult` 4 arrays |

## C Header — `cpp/include/fpe_engine.h`

Rewrite to declare all result structs + function prototypes. Remove old `FpePriceResult`, `fpe_price`, `fpe_pipeline_*`. `extern "C"` block for C++ compatibility.

## C++ Header-Only Wrapper — `cpp/include/fpe_compute.hpp`

```cpp
struct KnotsResult { std::vector<double> s, v; };
struct GridPointsResult { std::vector<double> s, v, s_weights, v_weights; };
struct GreeksResult { std::vector<double> delta, gamma, vega; };
struct OneshotResult { std::vector<double> price, delta, gamma, vega; };

class FpeCompute {
    FpeCompute_* ptr_;
    // lazy caches
    mutable std::optional<KnotsResult> knots_;
    mutable std::optional<GridPointsResult> grid_points_;
    mutable std::optional<std::vector<double>> initial_condition_;
    mutable std::optional<std::vector<std::vector<double>>> solve_;
    mutable std::optional<std::vector<std::vector<double>>> pdf_;

public:
    FpeCompute(double kappa, ..., int num_insert = 50);
    ~FpeCompute();

    KnotsResult knots() const;                    // cached
    GridPointsResult grid_points() const;         // cached
    std::vector<double> initial_condition();      // cached, triggers IC
    std::vector<std::vector<double>> solve();     // cached, triggers solver
    std::vector<std::vector<double>> pdf();       // cached, triggers solve+pdf

    std::vector<double> price(const std::vector<double>& K);
    GreeksResult greeks(const std::vector<double>& K, double rel_s=0.01, double rel_v=0.1);

    static OneshotResult price_oneshot(double kappa, ..., const std::vector<double>& K, ...);
};
```

Each method: calls C function → copies data into C++ owned memory → calls `fpe_compute_free_*` → caches result.
RAII destructor calls `fpe_compute_destroy`.

## C++ Example — `cpp/examples/demo.cpp`

Single example demonstrating both APIs:

1. Pipeline usage: `FpeCompute c(...)` → `c.knots()`, `c.grid_points()`, `c.initial_condition()`, `c.solve()`, `c.pdf()`, `c.price(K)`, `c.greeks(K)`
2. One-shot usage: `FpeCompute::price_oneshot(...)`

Removes: `pipeline_demo.cpp`, `live_trading.cpp`.

## Build System

- `scripts/build_cpp.sh`: unchanged structure (build shared lib → copy headers → cmake)
- `cpp/examples/CMakeLists.txt`: builds `demo` from `demo.cpp`
- `pixi.toml`: no changes needed

## Files Changed

| File | Action |
|---|---|
| `src/bindings/c_abi.mojo` | Rewrite — new heap-struct C ABI |
| `cpp/include/fpe_engine.h` | Rewrite — new structs + function decls |
| `cpp/include/fpe_compute.hpp` | New — C++ RAII wrapper |
| `cpp/examples/demo.cpp` | New — simplified example |
| `cpp/examples/pipeline_demo.cpp` | Delete |
| `cpp/examples/live_trading.cpp` | Delete |
| `cpp/examples/CMakeLists.txt` | Update — build `demo` only |
| `scripts/build_cpp.sh` | Minor update — echo new binary name |

## No Changes

- `src/server/compute_pipeline.mojo` — untouched
- `src/bindings/_fpe_native.mojo` — Python bindings untouched
- `src/bindings/_py_pipeline.mojo` — Python bindings untouched
- `python/fpe_engine/pricer.py` — Python API untouched
- `tests/test_pricer_stepwise.py` — Python tests untouched
