# FPE Option Pricing Engine

> High-performance exotic option pricing engine using Fokker-Planck PDE + B-spline Galerkin discretization, built in Mojo with MAX AI Kernels.

[![CI](https://github.com/elevenwang-creator/FPE_option/actions/workflows/ci.yml/badge.svg)](https://github.com/elevenwang-creator/FPE_option/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Mojo](https://img.shields.io/badge/Mojo-%E2%89%A51.0.0b2-orange)
[![Platform](https://img.shields.io/badge/Platform-macOS%20ARM64%20%7C%20Linux%20x86--64-green)]()

---

## Project Overview

A production-grade option pricing engine (~10,713 lines of Mojo, 85 source files) that ports three Python reference implementations into high-performance Mojo:

| Component | Python Reference | Mojo Source | Mojo Lines | Purpose |
|-----------|-----------------|-------------|------------|---------|
| **FPE Solver** | `docs/python_reference/FPE_Solver_Final_Version.py` | `src/engines/fpe/` | ~1,607 | Heston pricing via B-spline Galerkin + RadauIIA |
| **NAIS-Net** | `docs/python_reference/NAIS_rBM.py` | `src/engines/nais/` | ~1,371 | Neural FBSDE solver for rough Bergomi |
| **Pricing Demo** | `docs/python_reference/Barrier_Call_Option_Pricing.ipynb` | `src/server/` | ~766 | Sub-ms single + batch pricing frontend |
| **Sparse Math** | — | `src/sparse/` | ~1,449 | Custom CSR/CSC with SIMD SpMV, Kronecker, SpGEMM |
| **Domain Numerics** | — | `src/numerics/` | ~3,953 | B-splines, ODE, optimization, NN runtime, linear algebra |

### Key Features

- **B-spline Galerkin FPE solver**: Tensor-product cubic B-splines on (S,V) domain
- **RadauIIA implicit ODE integrator**: Order-5 stiff solver for semi-discrete FPE
- **Dual bindings**: Python (research/backtest) + C/C++ (live trading)
- **NAIS-Net**: Neural FBSDE solver for rough Bergomi model (upcoming)
- **GPU acceleration**: Metal/CUDA/HIP via `_gpu.mojo` variants + standalone GPU executor
- **Full Mojo-native**: Zero Python/scipy in production path; SIMD + comptime dispatch

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              FPE OPTION PRICING ENGINE                                 │
│                                    ~10,713 lines of Mojo                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 5: BINDINGS (6 files, 939 lines)   Python Extension · C ABI                    │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 4: PRICING SERVER (8 files, 766 lines)  ComputePipeline · Pricer · Greeks ·     │
│                                                  Payoffs · Interpolator                 │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 3: ENGINES (29 files, 3,431 lines)                                             │
│            ├─ engines/fpe/        B-spline FPE: Galerkin + RadauIIA solver              │
│            ├─ engines/fpe/gpu/    GPU chain executor                                     │
│            ├─ engines/nais/       NAIS-Net neural FBSDE                                 │
│            └─ engines/calibrator/ Heston LM calibration                                │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 2: DOMAIN NUMERICS (25 files, 3,953 lines)                                      │
│            B-Spline basis · RadauIIA/RK45 ODE · OSQP/LM optim · Autograd · Linalg      │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 1: SPARSE MATH (13 files, 1,449 lines)                                          │
│            CSRMatrix · CSCMatrix · DiagMatrix · kron · spgemm · spmm · scratch          │
│  ────────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 0: MAX AI KERNELS + STDLIB    matmul · gemv · rfft/irfft · Philox PRNG          │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
FPE_option/
├── pixi.toml                    # Mojo >=1.0.0b2, MAX >=26.3
├── recipe.yaml                  # Conda build recipe
├── pyproject.toml               # Python packaging metadata
│
├── src/                         # 85 .mojo files, ~10,713 lines
│   │
│   ├── bindings/                # Layer 5 (6 files, 939 lines)
│   │   ├── __init__.mojo
│   │   ├── _convert.mojo        # Mojo-Python type conversion
│   │   ├── _fpe_native.mojo     # Mojo Python extension (PyInit_fpe_engine)
│   │   ├── _params.mojo         # Parameter struct marshaling
│   │   ├── _py_pipeline.mojo    # Compute class exposed to Python
│   │   └── c_abi.mojo          # C/C++ FFI exports (fpe_init, fpe_price, etc.)
│   │
│   ├── server/                  # Layer 4 (8 files, 766 lines)
│   │   ├── __init__.mojo        # Exports: FpeParams, PricingEngine, Pricer, etc.
│   │   ├── compute_pipeline.mojo # Stepwise pipeline (knots → grid → solve → pdf → price)
│   │   ├── greeks.mojo          # Greeks (Δ/Γ/ν) via finite differences
│   │   ├── interpolator.mojo    # Bicubic/bilinear interpolation
│   │   ├── option_types.mojo    # FpeParams, PricingResult structs
│   │   ├── payoffs.mojo         # BarrierPayoff trait with integrate()
│   │   ├── pricer.mojo          # Pricer + PDFGrid + _price_at quadrature
│   │   └── pricing_engine.mojo  # Top-level orchestrator
│   │
│   ├── engines/                 # Layer 3 (29 files, 3,431 lines)
│   │   │
│   │   ├── fpe/                 # FPE engine (11 files, 1,607 lines)
│   │   │   ├── __init__.mojo    # Exports: HestonParams, FPEDomain, FPESolver, etc.
│   │   │   ├── solver.mojo      # FPESolver — RadauIIA time integration
│   │   │   ├── galerkin.mojo    # Mass/stiffness matrix assembly (CPU)
│   │   │   ├── galerkin_gpu.mojo# GPU Galerkin assembly
│   │   │   ├── domain.mojo      # FPEDomain: knots, grid, basis, quadrature weights
│   │   │   ├── domain_gpu.mojo  # GPU domain construction
│   │   │   ├── heston_params.mojo # HestonParams struct + Feller validation
│   │   │   ├── initial_cond.mojo  # Initial condition (CPU)
│   │   │   ├── initial_cond_gpu.mojo # GPU initial condition
│   │   │   ├── pdf.mojo         # PDF reconstruction from coefficients (CPU)
│   │   │   └── pdf_gpu.mojo     # GPU PDF reconstruction
│   │   │
│   │   ├── fpe/gpu/             # GPU executor (2 files, 469 lines)
│   │   │   ├── __init__.mojo
│   │   │   └── executor.mojo    # GPUFullChainExecutor — full GPU batch chain
│   │   │
│   │   ├── nais/                # NAIS-Net neural FBSDE (11 files, 1,371 lines)
│   │   │   ├── __init__.mojo
│   │   │   ├── nais_net.mojo    # 6-layer residual network
│   │   │   ├── volterra.mojo    # Fractional BM via FFT
│   │   │   ├── variance.mojo    # Rough Bergomi variance process
│   │   │   ├── fbsde.mojo       # Forward-backward SDE loss
│   │   │   ├── trainer.mojo     # CPU training loop
│   │   │   ├── inferencer.mojo  # Online inference + implied vol surface
│   │   │   ├── utils.mojo       # NAIS shared utilities
│   │   │   ├── gpu_trainer.mojo # GPU training executor
│   │   │   ├── gpu_forward_kernels.mojo # GPU forward pass
│   │   │   └── gpu_train_kernels.mojo   # GPU training kernels
│   │   │
│   │   └── calibrator/          # Heston calibration (4 files, 452 lines)
│   │       ├── __init__.mojo
│   │       ├── calibrator.mojo  # Levenberg-Marquardt optimizer
│   │       ├── objective.mojo   # Objective function (CPU)
│   │       └── objective_gpu.mojo # GPU objective function
│   │
│   ├── numerics/                # Layer 2 (25 files, 3,953 lines)
│   │   │
│   │   ├── bspline/             # B-spline module (6 files, 728 lines)
│   │   │   ├── __init__.mojo
│   │   │   ├── basis.mojo       # De Boor-Cox + SIMD evaluation
│   │   │   ├── knots.mojo       # Uniform/Chebyshev knot generation
│   │   │   ├── knots_gpu.mojo   # GPU knot generation
│   │   │   ├── recombination.mojo # Boundary condition enforcement
│   │   │   └── tensor_product.mojo # 2D tensor product basis
│   │   │
│   │   ├── ode/                 # ODE integrators (4 files, 1,623 lines)
│   │   │   ├── __init__.mojo
│   │   │   ├── radau.mojo       # RadauIIA (order 5) + BackwardEuler
│   │   │   ├── radau_gpu.mojo   # GPU RadauIIA
│   │   │   └── types.mojo       # ODESystem trait
│   │   │
│   │   ├── optim/               # Optimization (3 files, 298 lines)
│   │   │   ├── __init__.mojo
│   │   │   ├── osqp.mojo        # ADMM QP solver
│   │   │   └── lm.mojo          # Levenberg-Marquardt
│   │   │
│   │   ├── nn/                  # Neural network runtime (4 files, 316 lines)
│   │   │   ├── __init__.mojo
│   │   │   ├── stable_linear.mojo # Spectral-norm constrained layer
│   │   │   ├── autograd.mojo    # Reverse-mode autodiff tape
│   │   │   └── adam.mojo        # Adam optimizer
│   │   │
│   │   └── utils/               # Utilities (7 files, 979 lines)
│   │       ├── __init__.mojo
│   │       ├── helpers.mojo     # linspace, zeros, copy, swap, clamp
│   │       ├── simd_utils.mojo  # SIMD load/store/convert helpers
│   │       ├── fixed_size_vector.mojo # Fixed-size vector arithmetic
│   │       ├── linalg.mojo      # Dense LU solve with partial pivoting
│   │       ├── linalg_gpu.mojo  # GPU linear algebra kernels
│   │       └── sparse_lu.mojo   # Sparse LU decomposition
│   │
│   ├── sparse/                  # Layer 1 (13 files, 1,449 lines)
│   │   ├── __init__.mojo
│   │   ├── csr.mojo            # CSRMatrix with SIMD SpMV
│   │   ├── csc.mojo            # CSCMatrix
│   │   ├── diag.mojo           # Diagonal matrix
│   │   ├── add.mojo            # Sparse matrix add
│   │   ├── scale.mojo          # Sparse matrix scaling
│   │   ├── diag_scale.mojo     # Diagonal scaling
│   │   ├── diag_mul.mojo       # Diagonal multiplication
│   │   ├── kron.mojo           # Kronecker product
│   │   ├── kron_spmv.mojo      # Kronecker-powered SpMV
│   │   ├── spgemm.mojo         # Sparse × sparse → sparse
│   │   ├── scratch.mojo        # Scratch allocation workspace
│   │   └── gpu_kernels.mojo    # GPU SpMV kernels
│   │
│   └── gpu_utils/              # GPU utilities (3 files, 116 lines)
│       ├── __init__.mojo
│       ├── detect.mojo          # Multi-backend GPU detection
│       └── dtype.mojo           # METAL/CUDA/HIP type constants
│
├── tests/                       # 57 test files (56 Mojo + 1 Python)
├── benchmarks/                  # Performance benchmarks
├── python/fpe_engine/           # Python package (conda)
│   ├── __init__.py              # fpe.price() + fpe.Compute() API
│   ├── pricer.py                # Python-side wrappers
│   ├── _version.py              # Version from git tag
│   └── _fpe_native.so           # Compiled Mojo extension
├── python/examples/             # Jupyter notebooks + scripts
├── cpp/                         # C++ header + demo
│   ├── include/fpe_engine.h     # C ABI header
│   ├── include/fpe_compute.hpp  # C++ convenience wrapper
│   └── examples/demo.cpp        # Live trading demo
└── docs/                        # Documentation
```

---

## Complete File Inventory

### Layer 5: Bindings (6 files, 939 lines)

| File | Lines | Description |
|------|-------|-------------|
| `bindings/__init__.mojo` | 1 | Package marker |
| `bindings/_convert.mojo` | 143 | Mojo ↔ Python type conversion (lists, dicts, errors) |
| `bindings/_fpe_native.mojo` | 64 | Python extension entry: `PyInit_fpe_engine` |
| `bindings/_params.mojo` | 119 | Parameter marshaling between Python dict and FpeParams |
| `bindings/_py_pipeline.mojo` | 165 | `Compute` class exposed to Python: knots, grid_points, pdf, payoff_price, greeks |
| `bindings/c_abi.mojo` | 438 | C/C++ FFI: `fpe_init`, `fpe_price`, `fpe_free`, version query; `abi("C")` exports |

### Layer 4: Pricing Server (8 files, 766 lines)

| File | Lines | Description |
|------|-------|-------------|
| `server/__init__.mojo` | 5 | Exports: FpeParams, PricingResult, PricingEngine, Pricer, PDFGrid, Greeks |
| `server/compute_pipeline.mojo` | 171 | **Stepwise pipeline**: `ComputePipeline` with `knots()`, `grid_points()`, `basis_1d()`, `basis_2d()`, `initial_condition()`, `solve()`, `pdf()`, `price_at()`, `greeks()` |
| `server/greeks.mojo` | 53 | Greeks (Δ/Γ/ν) via central finite differences |
| `server/interpolator.mojo` | 198 | Bicubic (Catmull-Rom) and bilinear interpolation on 2D grid |
| `server/option_types.mojo` | 79 | `FpeParams` (Heston + grid + barrier + strikes) and `PricingResult` structs |
| `server/payoffs.mojo` | 53 | `BarrierPayoff` trait: integrate() for 10 option types |
| `server/pricer.mojo` | 155 | `Pricer` with full pipeline (domain → Galerkin → solve → pdf → price), `PDFGrid` |
| `server/pricing_engine.mojo` | 52 | `PricingEngine` orchestrator: pricer dispatch + Greeks |

### Layer 3: Engines (29 files, 3,431 lines)

#### FPE Engine (11 files, 1,607 lines)

| File | Lines | Description |
|------|-------|-------------|
| `engines/fpe/__init__.mojo` | 10 | Exports: HestonParams, FPEDomain, FPECachedBasis, FPESolver, PDFComputer |
| `engines/fpe/solver.mojo` | 105 | **FPESolver**: RadauIIA time integration with CSR SpMV |
| `engines/fpe/galerkin.mojo` | 125 | **Galerkin assembly**: mass matrix M and stiffness matrix K (CPU) |
| `engines/fpe/galerkin_gpu.mojo` | 63 | GPU Galerkin: dot-product-based matrix assembly |
| `engines/fpe/domain.mojo` | 294 | **FPEDomain**: knot generation, grid points, basis construction, quadrature weights |
| `engines/fpe/domain_gpu.mojo` | 81 | GPU domain: knot generation, boundary conditions |
| `engines/fpe/heston_params.mojo` | 81 | Heston model parameters struct with Feller condition check |
| `engines/fpe/initial_cond.mojo` | 130 | **Initial condition**: bivariate Gaussian via OSQP NNLS solve (CPU) |
| `engines/fpe/initial_cond_gpu.mojo` | 179 | GPU initial condition: OSQP on GPU |
| `engines/fpe/pdf.mojo` | 47 | PDF reconstruction: `pdf = Φ @ q(t)` (CPU) |
| `engines/fpe/pdf_gpu.mojo` | 81 | GPU PDF reconstruction |

#### FPE GPU Excecutor (2 files, 469 lines)

| File | Lines | Description |
|------|-------|-------------|
| `engines/fpe/gpu/__init__.mojo` | 17 | Package marker |
| `engines/fpe/gpu/executor.mojo` | 452 | **GPUFullChainExecutor**: full GPU batch chain (knots → grid → basis → matrix → solve → integrate) |

#### NAIS Engine (11 files, 1,371 lines)

| File | Lines | Description |
|------|-------|-------------|
| `engines/nais/__init__.mojo` | 10 | Exports: NaisNet, VolterraProcess, VarianceProcess, FBSDELoss, Trainer, Inferencer |
| `engines/nais/nais_net.mojo` | 416 | **NAIS-Net**: 6-layer residual network with stable linear layers |
| `engines/nais/volterra.mojo` | 131 | Fractional Brownian motion via Volterra + FFT convolution |
| `engines/nais/variance.mojo` | 39 | Rough Bergomi variance: `ε(t)·exp(η·X̃ - 0.5η²t^{2H})` |
| `engines/nais/fbsde.mojo` | 172 | Forward-backward SDE loss with tracked autodiff |
| `engines/nais/trainer.mojo` | 252 | CPU training loop with finite-difference gradients |
| `engines/nais/inferencer.mojo` | 107 | Online inference: `(t,S,V) → (price, delta, implied vol)` |
| `engines/nais/utils.mojo` | 72 | NAIS shared utilities (weights, schedule) |
| `engines/nais/gpu_trainer.mojo` | 60 | GPU training executor |
| `engines/nais/gpu_forward_kernels.mojo` | 130 | GPU batch forward pass |
| `engines/nais/gpu_train_kernels.mojo` | 32 | GPU training kernels (FBSDE loss, gradients) |

#### Calibrator (4 files, 452 lines)

| File | Lines | Description |
|------|-------|-------------|
| `engines/calibrator/__init__.mojo` | 4 | Exports: Calibrator |
| `engines/calibrator/calibrator.mojo` | 136 | Heston LM calibration: `calibrate()` with Jacobian |
| `engines/calibrator/objective.mojo` | 102 | Objective function: sum of squared pricing errors |
| `engines/calibrator/objective_gpu.mojo` | 185 | GPU objective + LM step kernels |

### Layer 2: Domain Numerics (25 files, 3,953 lines)

#### B-Spline Module (6 files, 728 lines)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/bspline/__init__.mojo` | 7 | Exports: GenerateKnots, BSplineBasis, RecombinationBasis, TensorProductBasis |
| `numerics/bspline/basis.mojo` | 213 | **B-spline evaluation**: De Boor-Cox with SIMD, eval_all collocation |
| `numerics/bspline/knots.mojo` | 282 | Knot vector: uniform, Chebyshev, from_data; `s_min`/`s_max` clamping |
| `numerics/bspline/knots_gpu.mojo` | 126 | GPU knot generation |
| `numerics/bspline/recombination.mojo` | 66 | Boundary condition recombination matrix |
| `numerics/bspline/tensor_product.mojo` | 24 | 2D tensor product basis for (S,V) grid |

#### ODE Solvers (4 files, 1,623 lines)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/ode/__init__.mojo` | 4 | Exports: ODESystem, RadauIIA, BackwardEuler |
| `numerics/ode/radau.mojo` | 1,151 | **RadauIIA**: 3-stage implicit RK (order 5), Newton iteration, Rosenbrock/W function, adaptive step control; BackwardEuler fallback |
| `numerics/ode/radau_gpu.mojo` | 452 | GPU RadauIIA solver |
| `numerics/ode/types.mojo` | 12 | ODESystem trait: `rhs(t, y, dydt)` + `dim()` |

#### Optimizer (3 files, 298 lines)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/optim/__init__.mojo` | 4 | Exports: OSQP, LevenbergMarquardt |
| `numerics/optim/lm.mojo` | 121 | **Levenberg-Marquardt** nonlinear least squares |
| `numerics/optim/osqp.mojo` | 173 | **OSQP**: ADMM-based QP solver for non-negative least squares |

#### Neural Network Runtime (4 files, 316 lines)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/nn/__init__.mojo` | 5 | Exports: StableLinear, GradientTape, Adam |
| `numerics/nn/autograd.mojo` | 145 | **Reverse-mode autodiff** tape: record_value/add/mul/sin/linear, backward |
| `numerics/nn/stable_linear.mojo` | 102 | Spectral-norm constrained linear layer `W = I - RᵀR` |
| `numerics/nn/adam.mojo` | 57 | Adam adaptive learning rate optimizer |

#### Utilities (7 files, 979 lines)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/utils/__init__.mojo` | 13 | Exports: linspace, zeros, copy, swap, clamp, simd helpers, linalg, sparse_lu |
| `numerics/utils/helpers.mojo` | 90 | `linspace`, `zeros`, `copy_vec`/`copy_mat`, `swap_rows`, `abs`/`max`/`min`/`clamp` |
| `numerics/utils/simd_utils.mojo` | 107 | SIMD load/store/convert/accumulate helpers |
| `numerics/utils/fixed_size_vector.mojo` | 265 | Fixed-size vector: element-wise ops, reductions, SIMD dot product |
| `numerics/utils/linalg.mojo` | 58 | Dense LU solve with partial pivoting |
| `numerics/utils/linalg_gpu.mojo` | 123 | GPU linear algebra: LU, solve, matvec |
| `numerics/utils/sparse_lu.mojo` | 323 | **Sparse LU decomposition** with partial pivoting + Schur complement |

### Layer 1: Sparse Math (13 files, 1,449 lines)

| File | Lines | Description |
|------|-------|-------------|
| `sparse/__init__.mojo` | 13 | Exports: CSRMatrix, CSCMatrix, DiagMatrix, all ops |
| `sparse/csr.mojo` | 438 | **CSRMatrix**: SIMD SpMV, zero-allocation `spmv_into`, transpose, assemble |
| `sparse/csc.mojo` | 20 | **CSCMatrix**: column-compressed storage |
| `sparse/diag.mojo` | 60 | Diagonal matrix with SIMD multiplication and inversion |
| `sparse/add.mojo` | 156 | Sparse matrix addition: `C = A + B` with merge-sort |
| `sparse/scale.mojo` | 72 | Sparse matrix scaling: `C = αA` |
| `sparse/diag_scale.mojo` | 75 | Diagonal scaling: `diag @ A` and `A @ diag` |
| `sparse/diag_mul.mojo` | 87 | Diagonal×diagonal and diagonal×CSR multiplication |
| `sparse/kron.mojo` | 113 | **Kronecker product**: `kron(A, B)` for tensor-product assembly |
| `sparse/kron_spmv.mojo` | 133 | **Kronecker SpMV**: `(A⊗B) @ x` without materializing the product |
| `sparse/spgemm.mojo` | 157 | **Sparse GEMM**: `C = A @ B` with symbolic+numeric phases |
| `sparse/scratch.mojo` | 64 | Scratch allocation workspace for SpMV accumulator |
| `sparse/gpu_kernels.mojo` | 65 | GPU SpMV kernels (single + batch) |

### GPU Utilities (3 files, 116 lines)

| File | Lines | Description |
|------|-------|-------------|
| `gpu_utils/__init__.mojo` | 3 | Package marker |
| `gpu_utils/detect.mojo` | 64 | **Multi-backend GPU detection**: Metal/CUDA/HIP/generic |
| `gpu_utils/dtype.mojo` | 49 | Cross-platform dtype/layout constants |

---

## Core Components Detail

### 1. Sparse Mathematics (`src/sparse/`, 13 files, 1,449 lines)

Custom sparse matrix operations optimized for FPE Galerkin assembly.

```
┌──────────────────────────────────────────────────────────────────┐
│  sparse/                                                         │
│  ├── csr.mojo (438) ───────────► CSRMatrix                       │
│  │   ├── spmv()               SIMD vectorized row dot products   │
│  │   ├── spmv_into()          Zero-allocation for ODE inner loops│
│  │   ├── transpose()          O(nnz) without dense round-trip    │
│  │   └── assemble()           Build CSR from triplets             │
│  │                                                               │
│  ├── csc.mojo (20) ────────────► CSCMatrix                       │
│  │   └── Column-compressed format for column access              │
│  │                                                               │
│  ├── diag.mojo (60) ───────────► DiagMatrix                      │
│  │   └── matvec()             SIMD diagonal scaling              │
│  │                                                               │
│  ├── kron.mojo (113) ───────────► kron(A, B)                     │
│  │   └── O(nnz_A × nnz_B) for tensor-product assembly            │
│  │                                                               │
│  ├── kron_spmv.mojo (133) ──────► (A⊗B) @ x                     │
│  │   └── SpMV without materializing Kronecker product            │
│  │                                                               │
│  ├── spgemm.mojo (157) ─────────► C = A @ B                      │
│  │   ├── symbolic_phase()     Predict nnz pattern                │
│  │   └── numeric_phase()      Compute values                     │
│  │                                                               │
│  ├── add.mojo (156) ────────────► C = A + B                      │
│  ├── scale.mojo (72) ───────────► C = αA                         │
│  ├── diag_scale.mojo (75) ──────► diag @ A or A @ diag           │
│  ├── diag_mul.mojo (87) ────────► Diag@Diag, Diag@CSR            │
│  ├── scratch.mojo (64) ─────────► Scratch workspace              │
│  └── gpu_kernels.mojo (65) ─────► GPU SpMV                       │
│      ├── spmv_kernel()         One thread per row                │
│      └── batch_spmv_kernel()   grid=(nrows, B)                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Optimization**: `spmv_into` eliminates List allocation overhead in ODE inner loops where `rhs()` is called hundreds of times. `kron_spmv` avoids materializing the full Kronecker product matrix.

### 2. B-Spline Module (`src/numerics/bspline/`, 6 files, 728 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  numerics/bspline/                                               │
│  ├── basis.mojo (213)                                            │
│  │   ├── BSplineBasis[degree]  De Boor-Cox with SIMD            │
│  │   ├── de_boor_cox()         Recursive evaluation              │
│  │   ├── _de_boor_cox_simd()   SIMD batch evaluation             │
│  │   ├── evaluate_batch_simd() Multiple points simultaneously    │
│  │   ├── first_derivative_all() Basis function derivatives        │
│  │   └── eval_all()            Sparse collocation matrix          │
│  │                                                               │
│  ├── knots.mojo (282) ───────────► GenerateKnots                  │
│  │   ├── uniform()              Clamped uniform knots             │
│  │   ├── chebyshev()            Chebyshev nodes                   │
│  │   ├── from_data()            Quantile knots                    │
│  │   └── clamp_s_min/s_max()   Domain boundary enforcement        │
│  │                                                               │
│  ├── knots_gpu.mojo (126) ───────► GPU knot generation            │
│  ├── recombination.mojo (66) ────► RecombinationBasis             │
│  │   └── Boundary conditions via recombination matrix             │
│  └── tensor_product.mojo (24) ───► TensorProductBasis             │
│       └── 2D basis: kron(Bs, Bv)                                 │
└──────────────────────────────────────────────────────────────────┘
```

**Key Optimization**: `evaluate_batch_simd` processes multiple evaluation points simultaneously using SIMD vectors.

### 3. ODE Integrators (`src/numerics/ode/`, 4 files, 1,623 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  numerics/ode/                                                    │
│  ├── radau.mojo (1,151)                                          │
│  │   ├── RadauIIA[System]      3-stage implicit RK, order 5      │
│  │   │   ├── Comptime Butcher tableau constants                   │
│  │   │   ├── Newton iteration with analytical Jacobian            │
│  │   │   ├── Rosenbrock/W function for step rejection             │
│  │   │   └── Adaptive step size control (PI controller)          │
│  │   │                                                           │
│  │   └── BackwardEuler[System]  Simple stiff fallback             │
│  │       └── Fixed-point iteration                               │
│  │                                                               │
│  ├── radau_gpu.mojo (452) ────────► GPU RadauIIA                  │
│  │   └── GPU-aware Newton + SpMV                                 │
│  │                                                               │
│  └── types.mojo (12) ─────────────► ODESystem trait               │
│       └── rhs(t, y, dydt) + dim() interface                     │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features**: Comptime Butcher tableau constants eliminate runtime table lookups. RadauIIA handles stiff FPE semi-discrete systems. GPU variant supports batch ODE solves.

### 4. FPE Engine (`src/engines/fpe/`, 11+2 files, ~2,076 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  engines/fpe/                                                    │
│  ├── solver.mojo (105)                                           │
│  │   ├── FPESolver              RadauIIA + CSR SpMV             │
│  │   ├── FPESparseSystem        ODESystem wrapping CSR SpMV     │
│  │   └── solve()                Time integration from 0 to T    │
│  │                                                               │
│  ├── domain.mojo (294) ───────────► FPEDomain[deg_s, deg_v]      │
│  │   ├── build_basis()          Tensor product B-spline basis   │
│  │   ├── quadrature_weights()   Gauss-Legendre weights           │
│  │   └── cached_basis()         Pre-compute basis for reuse      │
│  │                                                               │
│  ├── galerkin.mojo (125) ────────► Galerkin assembly             │
│  │   ├── mass_from_cached()     M = ΦᵀWΦ                        │
│  │   └── stiffness_from_cached() K = drift + diffusion terms    │
│  │                                                               │
│  ├── initial_cond.mojo (130) ────► Initial condition             │
│  │   └── initial_condition_from_cached() Bivariate Gaussian      │
│  │                                                               │
│  ├── pdf.mojo (47) ───────────────► PDFComputer                  │
│  │   └── pdf_from_cached()      pdf = Φ @ q(t)                   │
│  │                                                               │
│  ├── heston_params.mojo (81) ────► HestonParams                  │
│  │   └── κ, θ, σ, ρ, r, T, S₀, V₀ + Feller validation          │
│  │                                                               │
│  ├── *_gpu.mojo (4 files, 404 lines)  GPU counterparts           │
│  │   ├── domain_gpu.mojo        GPU basis/grid                   │
│  │   ├── galerkin_gpu.mojo      GPU matrix assembly              │
│  │   ├── initial_cond_gpu.mojo  GPU initial condition            │
│  │   └── pdf_gpu.mojo           GPU PDF reconstruction           │
│  │                                                               │
│  └── gpu/executor.mojo (452)     GPUFullChainExecutor            │
│      └── Full GPU batch pipeline: knots → grid → basis →        │
│          matrix → solve → integrate                              │
└──────────────────────────────────────────────────────────────────┘
```

### 5. NAIS Engine (`src/engines/nais/`, 11 files, 1,371 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  engines/nais/                                                   │
│  ├── nais_net.mojo (416)                                         │
│  │   ├── NaisNet            6-layer residual architecture        │
│  │   │   ├── Layer 1: Linear → sin                               │
│  │   │   ├── Layers 2-4: [StableLinear + skip + sin] × 3         │
│  │   │   ├── Layer 5: u = W₅h + b₅                              │
│  │   │   └── Layer 6: φ = W₆h + b₆                              │
│  │   │                                                           │
│  │   ├── forward(t, x)           Plain forward pass              │
│  │   └── forward_tracked(...)    With autodiff tape              │
│  │                                                               │
│  ├── volterra.mojo (131)                                         │
│  │   ├── VolterraProcess[B]  Fractional Brownian motion          │
│  │   │   ├── generate()      O(N²) direct convolution           │
│  │   │   └── generate_fft()  O(N log N) FFT convolution        │
│  │   └── Uses MAX kernels: rfft, irfft                          │
│  │                                                               │
│  ├── variance.mojo (39) ────────► VarianceProcess[B]             │
│  │   └── Rough Bergomi: ε(t)·exp(η·X̃ - 0.5η²t^{2H})            │
│  │                                                               │
│  ├── fbsde.mojo (172) ──────────► FBSDELoss[B]                  │
│  │   └── Forward-backward SDE loss computation                   │
│  │                                                               │
│  ├── trainer.mojo (252) ─────────► Trainer[B]                    │
│  │   └── CPU training loop with finite-difference grads          │
│  │                                                               │
│  ├── inferencer.mojo (107) ──────► Inferencer[B]                 │
│  │   ├── (t,S,V) → (price, delta, implied vol)                  │
│  │   ├── _implied_vol_newton()   Newton's method for IV          │
│  │   └── vol_surface()           Implied vol surface generation  │
│  │                                                               │
│  ├── utils.mojo (72) ────────────► Shared utilities               │
│  └── gpu_*.mojo ────────────────► GPU training kernels            │
│      ├── gpu_trainer.mojo (60)            GPU training executor   │
│      ├── gpu_forward_kernels.mojo (130)   GPU batch forward      │
│      └── gpu_train_kernels.mojo (32)      GPU FBSDE loss/grad    │
└──────────────────────────────────────────────────────────────────┘
```

### 6. Neural Network Runtime (`src/numerics/nn/`, 4 files, 316 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  numerics/nn/                                                    │
│  ├── autograd.mojo (145)                                         │
│  │   ├── Tape            Reverse-mode autodiff tape              │
│  │   │   ├── record_value()  Input value                        │
│  │   │   ├── record_add()    Addition                           │
│  │   │   ├── record_mul()    Multiplication                     │
│  │   │   ├── record_sin()    Sine                               │
│  │   │   ├── record_linear()  W@x + b                          │
│  │   │   └── backward()      Gradient backpropagation            │
│  │   │                                                           │
│  │   └── GradientTape    Finite-difference alternative           │
│  │                                                               │
│  ├── stable_linear.mojo (102) ► StableLinear                     │
│  │   └── W = I - RᵀR (spectral norm constraint)                │
│  │                                                               │
│  └── adam.mojo (57) ────────────► Adam optimizer                │
│       └── m, v moments + bias correction                        │
└──────────────────────────────────────────────────────────────────┘
```

### 7. GPU Utilities (`src/gpu_utils/`, 3 files, 116 lines)

```
┌──────────────────────────────────────────────────────────────────┐
│  gpu_utils/                                                      │
│  ├── detect.mojo (64)                                            │
│  │   ├── detect_gpu_backend()    Returns: metal/cuda/hip/cpu    │
│  │   ├── is_gpu_available()      Boolean check                  │
│  │   └── get_device_api_name()   API string for DeviceContext   │
│  │                                                               │
│  └── dtype.mojo (49) ────────────► METAL_DTYPE, CUDA_DTYPE      │
│       └── Layout constants per backend                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ComputePipeline (stepwise API)                    │
│                                                                      │
│  [1] knots() ───► (s_knots, v_knots)           B-spline knot vectors │
│  [2] grid_points() ──► (s_pts, v_pts, w_s, w_v)  Collocation points  │
│  [3] basis_1d() ──► (Bs, dBs, Bv, dBv)       1D basis matrices      │
│  [4] basis_2d() ──► kron(Bs, Bv)              2D tensor basis       │
│  [5] initial_condition() ──► q0               Galerkin coefficients │
│  [6] solve() ──► sol[t_n]                     RadauIIA time stepping │
│  [7] pdf() ──► PDF(S_i, V_j)                  Terminal distribution  │
│  [8] price_at(K) ──► prices                   Payoff integration     │
│  [9] greeks(K) ──► (Δ, Γ, ν)                 Finite difference      │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Pipeline

1. **FPE domain** (`FPEDomain[3,3]`): Tensor-product cubic B-splines on (S,V)
2. **Galerkin assembly**: Mass matrix `M` + stiffness matrix `K` (sparse CSR)
3. **Initial condition**: Bivariate Gaussian projected via `Mq₀ = Φᵀf`
4. **Time integration**: RadauIIA solving `M·dq/dt = K·q` from 0→T
5. **Payoff integration**: `price = e^{-rT} ∫∫ payoff(S)·PDF(S,V) dS dV`

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Mojo >=1.0.0b2 | Zero-cost abstractions, SIMD, GPU |
| **GPU Compute** | MAX AI Kernels >=26.3 | matmul, gemv, rfft, irfft |
| **GPU Targets** | Metal, CUDA, HIP | Apple Silicon, NVIDIA, AMD |
| **Package Manager** | pixi | Reproducible environments |
| **Python (dev)** | matplotlib, scipy, numpy, pytest | Testing + analysis |
| **Benchmarks** | Mojo std.benchmark | Performance profiling |

### Mojo Features Used

| Feature | Usage | Files |
|---------|-------|-------|
| `comptime` | Compile-time dispatch (B-spline degree, GPU/CPU) | solver, detect, basis |
| `SIMD` | Vectorized B-spline, sparse, linear algebra | basis, csr, radau, helpers |
| `traits` | ODESystem, Payoff, Copyable/Movable interfaces | ode/types, payoffs |
| `struct` | Zero-overhead value types | All files |
| `has_accelerator()` | Compile-time GPU detection | detect.mojo |
| `has_apple_gpu_accelerator()` | Metal-specific path | detect.mojo, dtype.mojo |
| `has_nvidia_gpu_accelerator()` | CUDA-specific path | detect.mojo |
| `has_amd_gpu_accelerator()` | HIP-specific path | detect.mojo |
| `vectorize`, `parallelize` | SIMD and multi-core | basis, csr, helpers, pricer, galerkin |
| `@fieldwise_init` | Bulk initialization | ~40 structs across codebase |
| `@always_inline` | Zero call overhead | interpolator |
| `abi("C")` | C-compatible FFI exports | c_abi.mojo (16 functions) |

---

## Python API

```python
import fpe_engine as fpe

# One-shot pricing with Greeks
result = fpe.price(
    S0=60.0, V0=0.1, T=0.6, r=0.1,
    kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
    K=[65, 70, 75, 80, 85, 90, 95, 100],
    barrier=50.0,
    option_type="down_and_out_call",
    n_s=38, n_v=38,
)
print(result.prices, result.deltas, result.gammas, result.vegas)

# Stepwise access to pipeline intermediates
pipe = fpe.Compute(
    S0=60.0, V0=0.1, T=0.6, r=0.1,
)
ks = pipe.knots           # B-spline knot vectors
gp = pipe.grid_points     # Collocation points + quadrature weights
pdf = pipe.pdf            # Terminal PDF
prices = pipe.payoff_price(100.0)  # Barrier + vanilla payoff
greeks = pipe.greeks([80.0, 100.0, 120.0])
```

### Option Types

| ID | Name | Barrier |
|----|------|---------|
| 0 | Down-and-In Call | S_min = barrier |
| 1 | Down-and-In Put | S_min = barrier |
| 2 | Down-and-Out Call | S_min = barrier |
| 3 | Down-and-Out Put | S_min = barrier |
| 4 | Up-and-In Call | S_max = barrier |
| 5 | Up-and-In Put | S_max = barrier |
| 6 | Up-and-Out Call | S_max = barrier |
| 7 | Up-and-Out Put | S_max = barrier |
| 8 | European Call | none |
| 9 | European Put | none |

---

## Performance Benchmarks

| Metric | Native Mojo | Python Binding | C++ Demo | vs Python Ref |
|--------|-------------|----------------|----------|---------------|
| 8-strike Heston pricing | 3.44 s | 4.22 s | 4.18 s | 10.2× faster |
| In-out parity (DIC+DOC=Van) | 0.0 error | — | — | Exact |
| DOC vs Python ref error | <0.3% | — | — | — |

*Benchmark params: `num_insert=251`, `s_max=150`, `n_s=38`, `n_v=38`*

---

## Dependencies

```toml
# pixi.toml — production
[dependencies]
max = ">=26.3"
mojo = ">=1.0.0b2.dev2026050805,<2"
numpy = ">=2.4.3,<3"
scipy = ">=1.17.1,<2"
matplotlib = ">=3.10.8,<4"
seaborn = ">=0.13.2,<0.14"
polars = ">=1.41.0,<2"
pytest = ">=9.0.3,<10"
ipykernel = ">=7.1.0,<8"
cvxpy = ">=1.8.2,<2"
plotly = ">=6.6.0,<7"
nbformat = ">=5.10.4,<6"
nbconvert = ">=7.17.1,<8"
```

---

## Code Statistics

| Category | Files | Lines |
|----------|-------|-------|
| Sparse Math | 13 | 1,449 |
| Numerics | 25 | 3,953 |
| Engines | 29 | 3,431 |
| Server | 8 | 766 |
| Bindings | 6 | 939 |
| GPU Utils | 3 | 116 |
| **Total (src/)** | **85** | **10,713** |
| **Tests** | **57** | — |

---

## Getting Started

### Prerequisites

- macOS ARM64 (Apple Silicon) or Linux x86-64
- pixi package manager
- Mojo >=1.0.0b2
- MAX SDK >=26.3

### Installation

```bash
git clone https://github.com/elevenwang-creator/FPE_option.git
cd FPE_option
pixi install
```

### Running Tests

```bash
pixi run test        # Python integration tests
pixi run test-mojo   # Mojo unit tests
```

### Usage Example (Mojo)

```python
from engines.fpe.heston_params import HestonParams
from server.pricing_engine import PricingEngine
from server.option_types import FpeParams

var heston = HestonParams(
    kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
    r=0.1, T=0.6, S0=60.0, V0=0.1,
)
var fp = FpeParams(
    heston=heston, n_s=38, n_v=38,
    barrier=50.0, option_type=2, strikes=List[Float64](100.0),
)
var engine = PricingEngine()
var result = engine.price(fp)
```

### Usage Example (Python)

```python
import fpe_engine as fpe

result = fpe.price(
    S0=60.0, V0=0.1, T=0.6, r=0.1,
    K=[65, 70, 75, 80, 85, 90, 95, 100],
    barrier=50.0, option_type="down_and_out_call",
)
print(result.prices)
```

---

## Upcoming Features

- **NAIS-Net v2**: GPU-accelerated neural FBSDE solver for rough Bergomi
- **GPU full-chain pricing**: Metal/CUDA batch pricing via `GPUFullChainExecutor`
- **Heston calibration**: LM optimization on GPU

---

## Key Design Patterns

### 1. Cached Basis Reuse

```python
var domain = FPEDomain[3, 3](heston, n_s, n_v, num_insert)
var cached = domain.cached_basis()  # Pre-compute once

var M = mass_from_cached(cached)    # Reuse for assembly
var K = stiffness_from_cached(cached, heston)
var q0 = initial_condition_from_cached(cached, heston, M)
```

The `FPECachedBasis` holds pre-computed collocation matrices, quadrature weights, and grid points — reused across all pipeline stages.

### 2. Comptime-Dispatched GPU Kernels

```python
# GPU kernels live alongside CPU code in _gpu.mojo variants
from engines.fpe.galerkin import mass_from_cached       # CPU
from engines.fpe.galerkin_gpu import mass_gpu_kernel    # GPU
```

GPU code is organized as `_gpu.mojo` files at the same level, not in separate subdirectories. The `GPUFullChainExecutor` in `engines/fpe/gpu/` orchestrates the full chain.

### 3. SIMD-Vectorized Quadrature

```python
# Hoist payoff out of V-loop
for i in range(n_s):
    var payoff_val = payoff.value(S)
    if payoff_val == 0.0:
        continue
    var pdf_row = SIMD_LOAD[simd_width](grid.pdf, i * n_v)
    var dv = SIMD_LOAD[simd_width](grid.dv_weights, 0)
    v_sum += (pdf_row * dv).reduce_add()
```

---

## License

MIT License — See [LICENSE](../LICENSE) for details.

---

*Generated with DeepWiki-style documentation*
