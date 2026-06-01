# FPE Option Pricing Engine — Sisyphus Work Plan

> Converted from IMPLEMENTATION_PLAN.md v3
> Language: Mojo v0.26.2+ with MAX AI Kernels
> Design: Unified parametric `FPESolver[B]` — write once, CPU/GPU via comptime

---

## TODOs

### Phase 1: Project Setup + Sparse Math Library (Weeks 1–3)

- [x] **P1-T1**: Initialize pixi project — create `pixi.toml`, `mojoproject.toml`, verify `mojo build` succeeds
- [x] **P1-T2**: Create full directory structure under `src/` (sparse, numerics, engines, server, bindings)
- [x] **P1-T3**: MAX Kernels smoke test — write `tests/test_max_integration.mojo` verifying `matmul`, `gemv`, `rfft` imports work
- [x] **P1-T4**: `src/sparse/csr.mojo` — `CSRMatrix[dtype]` struct: data/indices/indptr, `spmv()`, `to_gpu()`
- [x] **P1-T5**: `src/sparse/coo.mojo` — `COOMatrix[dtype]` struct + `to_csr()` (sort + compress)
- [x] **P1-T6**: `src/sparse/diag.mojo` — `DiagMatrix[dtype]` struct + diagonal-vector multiply
- [x] **P1-T7**: `src/sparse/ops.mojo` — `kron(A,B)`, `spgemm(A,B)`, `spmm(A,D)` operations
- [x] **P1-T8**: `src/sparse/gpu_kernels.mojo` — `spmv_kernel` (one thread/row) + `batch_spmv_kernel` (grid_dim=(nrows,B))
- [x] **P1-T9**: `tests/test_sparse.mojo` — unit tests for all sparse ops vs scipy.sparse reference `.npz`

### Phase 2: B-Spline + ODE + Optimizer (Weeks 4–7)

- [x] **P2-T1**: `tests/reference/generate_reference.py` — generate all `.npz` reference files from Python FPE_Solver
- [x] **P2-T2**: `src/numerics/bspline/knots.mojo` — `GenerateKnots` (uniform + non-uniform, SIMD linspace)
- [x] **P2-T3**: `src/numerics/bspline/basis.mojo` — `BSplineBasis[degree]` with `comptime for` De Boor-Cox + SIMD batch eval
- [x] **P2-T4**: `src/numerics/bspline/recombination.mojo` — `RecombinationBasis` (Dirichlet/Neumann BCs)
- [x] **P2-T5**: `src/numerics/bspline/tensor_product.mojo` — `TensorProductBasis` via `kron()`
- [x] **P2-T6**: `tests/test_bspline.mojo` — validate all basis outputs vs Python reference to 1e-10
- [x] **P2-T7**: `src/numerics/ode/types.mojo` — `ODESystem` trait, `ODESolution` struct
- [x] **P2-T8**: `src/numerics/ode/rk45.mojo` — Dormand-Prince RK45 with `comptime` Butcher tableaux
- [x] **P2-T9**: `src/numerics/ode/radau.mojo` — RadauIIA stiff solver (LU solve per step)
- [x] **P2-T10**: `tests/test_ode.mojo` — validate ODE solvers vs scipy.integrate reference
- [x] **P2-T11**: `src/numerics/optim/osqp.mojo` — ADMM-based QP solver for non-negative initial condition
- [x] **P2-T12**: `src/numerics/optim/lm.mojo` — Levenberg-Marquardt using MAX `qr_factorization`
- [x] **P2-T13**: `tests/test_optim.mojo` — validate OSQP vs CVXPY reference, LM convergence test

### Phase 3: FPE Engine — Unified Parametric Solver (Weeks 8–12)

- [x] **P3-T1**: `src/engines/fpe/heston_params.mojo` — `HestonParams` + `HestonParamsBatch[B]` + Feller condition check
- [x] **P3-T2**: `src/engines/fpe/domain.mojo` — `FPEDomain`: grid generation, basis construction (shared across batch)
- [x] **P3-T3**: `src/engines/fpe/galerkin.mojo` — `GalerkinAssembler[B]`: batch-aware mass `M` + stiffness `K` assembly
- [x] **P3-T4**: `src/engines/fpe/initial_cond.mojo` — `InitialCondition[B]`: bivariate Gaussian delta approx + OSQP `q₀≥0`
- [x] **P3-T5**: `src/engines/fpe/solver.mojo` — **`FPESolver[B]`**: unified `comptime if batch_size==1` CPU / `has_accelerator()` GPU dispatch
- [x] **P3-T6**: `src/engines/fpe/pdf.mojo` — `PDFComputer[B]`: `pdf = Φ @ q(t)` — batch-aware reshape to `(B, n_s, n_v, n_t)`
- [x] **P3-T7**: `tests/test_fpe_engine.mojo` — full pipeline test: `FPESolver[1]` output matches Python FPE_Solver to 1e-8
- [x] **P3-T8**: `src/server/pdf_cache.mojo` — `PDFCache`: `Dict[UInt64, PDFGrid]`, `store()`, `get()`, disk serialization
- [x] **P3-T9**: `src/server/interpolator.mojo` — bicubic interpolation on S×V grid, SIMD-vectorized via `vectorize`
- [x] **P3-T10**: `src/server/payoffs.mojo` — `Payoff` trait + `BarrierUpAndOut`, `BarrierDownAndIn`, `EuropeanCall`, `EuropeanPut`
- [x] **P3-T11**: `src/server/greeks.mojo` — `Greeks[B]`: finite-difference Greeks, batch-aware
- [x] **P3-T12**: `src/server/pricer.mojo` — **`Pricer[B]`**: unified `comptime` CPU SIMD (B=1) / GPU one-thread-per-option (B>1)
- [x] **P3-T13**: `src/server/pricing_engine.mojo` — `PricingEngine` with unified `price[B]()` + `calibrate[B]()` entry points
- [x] **P3-T14**: `tests/test_pricing_server.mojo` — `price[1]` sub-ms benchmark + `price[1000]` GPU batch test
- [x] **P3-T15**: `src/engines/calibrator/objective.mojo` + `calibrator.mojo` — `Calibrator[B]` using `FPESolver[B]` + LM
- [x] **P3-T16**: `tests/test_calibrator.mojo` — calibrate to synthetic market data, verify convergence

### Phase 4: NAIS Engine (Weeks 13–17)

- [x] **P4-T1**: `src/numerics/nn/stable_linear.mojo` — `StableLinear` weight constraint (R^TR norm) using MAX `matmul`
- [x] **P4-T2**: `src/numerics/nn/autograd.mojo` — reverse-mode autodiff tape for FBSDE gradient computation
- [x] **P4-T3**: `src/numerics/nn/adam.mojo` — Adam optimizer using MAX `gemv` for param updates
- [x] **P4-T4**: `tests/test_nn.mojo` — forward/backward pass matches TF NaisNet reference
- [x] **P4-T5**: `src/engines/nais/nais_net.mojo` — `NaisNet`: Linear + StableLinear + skip connections + sin activation
- [x] **P4-T6**: `src/engines/nais/volterra.mojo` — `VolterraProcess[B]`: hybrid scheme FFT convolution using MAX `rfft`/`irfft`
- [x] **P4-T7**: `src/engines/nais/variance.mojo` — `VarianceProcess[B]`: rough Bergomi `ε(t)·exp(η·X̃ - 0.5η²t^{2H})`
- [x] **P4-T8**: `src/engines/nais/fbsde.mojo` — `FBSDELoss[B]`: forward-backward SDE loss with Z, Z̃ computation
- [x] **P4-T9**: `src/engines/nais/trainer.mojo` — `Trainer[B]`: GPU training loop, Adam + gradient tape
- [x] **P4-T10**: `src/engines/nais/inferencer.mojo` — `Inferencer[B]`: GPU inference `(t,S,V)→(price,φ,Du)`
- [x] **P4-T11**: `src/server/vol_surface.mojo` — `VolSurfaceGenerator`: sweep K×T grid → implied vol surface
- [x] **P4-T12**: `tests/test_nais_engine.mojo` — forward pass matches TF, loss converges, Greeks match Python

### Phase 5: Bindings + Production (Weeks 18–22)

- [x] **P5-T1**: `src/bindings/python_module.mojo` — `PyInit_fpe_engine` exposing `price[B]`, `calibrate[B]`, NAIS functions
- [x] **P5-T2**: `src/bindings/c_abi.mojo` — `@export` C functions: `fpe_init`, `fpe_price_single`, `fpe_price_batch`, `fpe_calibrate`
- [x] **P5-T3**: `cpp/include/fpe_engine.h` — C header for live trading engine integration
- [x] **P5-T4**: `cpp/examples/live_trading.cpp` — C++ example: init, price single, price batch
- [x] **P5-T5**: `python/fpe_engine/__init__.py` + `python/examples/backtest.py` — Python wrapper + backtest example
- [x] **P5-T6**: `benchmarks/` — full benchmark suite: `price[1]` <1ms, `price[1000]` <10ms, calibrate <25s
- [x] **P5-T7**: Integration tests — end-to-end: Python → Mojo → price barrier option, match notebook reference
- [x] **P5-T8**: Documentation + packaging — README, API docs, pixi publish config

---

## Final Verification Wave

- [x] **F1**: Performance gate — `price[1]` p99 <1ms, `price[1000]` <10ms, `calibrate[64]` <25s
- [x] **F2**: Correctness gate — all outputs match Python reference to specified tolerances
- [x] **F3**: Binding gate — Python and C++ examples run end-to-end without errors
- [x] **F4**: Mojo philosophy gate — no duplicate CPU/GPU code paths; all compute via `FPESolver[B]`/`Pricer[B]`
