# FPE Option Pricing Engine

> High-performance exotic option pricing engine using Fokker-Planck PDE + NAIS-Net neural solver, built entirely in Mojo with MAX AI Kernels.

[![Mojo](https://img.shields.io/badge/Mojo-v0.26.3-red)](https://docs.modular.com/mojo/)
[![MAX SDK](https://img.shields.io/badge/MAX-26.3.0-blue)](https://docs.modular.com/max/)
[![License](https://img.shields.io/badge/License-Proprietary-blue)](./LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20ARM64%20%7C%20Linux%20x86--64-green)]()

---

## Project Overview

This is a production-grade option pricing engine (~7,088 lines of Mojo) that reconstructs three Python codebases into high-performance Mojo:

| Component | Source | Lines | Purpose |
|-----------|--------|-------|---------|
| **FPE Solver** | `FPE_Solver_Final_Version.py` | ~1,153 | Heston model pricing via B-spline Galerkin |
| **NAIS-Net** | `NAIS_rBM.py` | ~448 | Neural FBSDE solver for rough Bergomi |
| **Pricing Server** | `BarrierOptionPricing.ipynb` | — | Sub-ms single + GPU batch pricing |

### Key Features

- **Unified Compute Model**: Write once, deploy to CPU (batch=1) or GPU (batch=N)
- **Three Runtime Modes**: CPU single pricing (<1ms), GPU batch pricing, GPU batch calibration
- **GPU Portability**: NVIDIA (CUDA), AMD (HIP), Apple Silicon (Metal) via Mojo's GPU abstraction
- **Full Mojo-Native**: No Python/scipy/TensorFlow dependencies in production path
- **Dual Bindings**: Python (research/backtest) + C++ (live trading)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FPE OPTION PRICING ENGINE                                │
│                                    ~7,088 lines of Mojo                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 5: BINDINGS                    Python Extension · C ABI                       │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 4: PRICING SERVER              PricingEngine · Pricer · PDFCache · Payoffs   │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 3: ENGINES                     FPE Engine · NAIS Engine · Calibrator         │
│            ├─ FPE/gpu/                GPU kernels for Heston batch pricing            │
│            └─ nais/gpu_*              GPU kernels for NAIS training/inference         │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 2: DOMAIN NUMERICS             B-Spline · ODE · Optimizer · NN Runtime       │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 1: SPARSE MATH (Custom)         CSRMatrix · COOMatrix · DiagMatrix · ops      │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│  LAYER 0: MAX AI KERNELS + STDLIB     matmul · gemv · rfft/irfft · Philox PRNG     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
FPE_option/
├── pixi.toml                    # Mojo v0.26.3, MAX SDK 26.3.0
├── mojoproject.toml
│
├── src/                         # ~62 .mojo files, ~7,088 lines
│   │
│   ├── bindings/                # Layer 5: Language bindings (3 files)
│   │   ├── __init__.mojo
│   │   ├── c_abi.mojo          # C/C++ FFI exports
│   │   └── python_module.mojo  # PyInit_fpe_engine
│   │
│   ├── server/                  # Layer 4: Pricing server (9 files, ~1,160 lines)
│   │   ├── __init__.mojo
│   │   ├── pricing_engine.mojo  # Top-level orchestrator (42 lines)
│   │   ├── pricer.mojo          # Unified pricer CPU/GPU/parallel (447 lines)
│   │   ├── pdf_cache.mojo       # PDF grid cache with disk I/O (161 lines)
│   │   ├── interpolator.mojo   # Bicubic/bilinear interpolation (187 lines)
│   │   ├── greeks.mojo          # Greeks (Δ/Γ/ν/θ) finite diff (135 lines)
│   │   ├── payoffs.mojo         # Barrier/European payoff traits (83 lines)
│   │   ├── vol_surface.mojo     # Implied vol surface (20 lines)
│   │   └── gpu_pricing_kernels.mojo # GPU batch pricing kernel (77 lines)
│   │
│   ├── engines/                 # Layer 3: Core engines (18 files, ~2,400 lines)
│   │   │
│   │   ├── fpe/                # Fokker-Planck Equation engine
│   │   │   ├── __init__.mojo
│   │   │   ├── solver.mojo     # Unified solver CPU/GPU dispatch (236 lines)
│   │   │   ├── galerkin.mojo   # Mass/stiffness matrix assembly (158 lines)
│   │   │   ├── domain.mojo     # Knots, grid, basis construction
│   │   │   ├── heston_params.mojo # Heston parameters struct
│   │   │   ├── initial_cond.mojo  # Initial condition (bivariate Gaussian)
│   │   │   └── pdf.mojo        # PDF reconstruction from coefficients
│   │   │
│   │   ├── fpe/gpu/            # GPU FPE kernels (6 files)
│   │   │   ├── __init__.mojo
│   │   │   ├── executor.mojo   # Full GPU chain orchestrator (172 lines)
│   │   │   ├── domain.mojo     # GPU knots/grid/basis/boundary
│   │   │   ├── matrix.mojo     # GPU sparse matrix/delta/initial
│   │   │   ├── solver.mojo     # GPU LU/RADAU5 kernels
│   │   │   ├── integration.mojo # GPU integration kernel
│   │   │   └── calibration.mojo # GPU loss/LM optimization
│   │   │
│   │   ├── nais/               # NAIS-Net (Neural FBSDE)
│   │   │   ├── __init__.mojo
│   │   │   ├── nais_net.mojo   # 6-layer residual network (387 lines)
│   │   │   ├── volterra.mojo   # Fractional BM via FFT (139 lines)
│   │   │   ├── variance.mojo   # Rough Bergomi variance process
│   │   │   ├── fbsde.mojo      # Forward-backward SDE loss
│   │   │   ├── trainer.mojo    # CPU training loop
│   │   │   ├── inferencer.mojo # Online inference
│   │   │   ├── gpu_trainer.mojo # GPU training executor
│   │   │   ├── gpu_forward_kernels.mojo # GPU forward pass
│   │   │   └── gpu_train_kernels.mojo  # GPU training kernels
│   │   │
│   │   └── calibrator/         # Heston calibration
│   │       ├── __init__.mojo
│   │       ├── calibrator.mojo # Levenberg-Marquardt optimizer
│   │       └── objective.mojo  # Calibration objective function
│   │
│   ├── numerics/                # Layer 2: Domain numerics (18 files, ~1,700 lines)
│   │   │
│   │   ├── __init__.mojo
│   │   ├── utils.mojo          # linspace, zeros, copy utilities (136 lines)
│   │   ├── linalg.mojo         # LU solve with partial pivoting
│   │   │
│   │   ├── bspline/            # B-spline module
│   │   │   ├── __init__.mojo
│   │   │   ├── basis.mojo      # De Boor-Cox + SIMD evaluation (188 lines)
│   │   │   ├── knots.mojo     # Knot vector generation
│   │   │   ├── recombination.mojo # Boundary condition enforcement
│   │   │   └── tensor_product.mojo # 2D tensor product basis
│   │   │
│   │   ├── ode/                # ODE solvers
│   │   │   ├── __init__.mojo
│   │   │   ├── radau.mojo      # RadauIIA implicit solver (377 lines)
│   │   │   ├── rk45.mojo      # Runge-Kutta 45 explicit
│   │   │   └── types.mojo      # ODESystem trait definitions
│   │   │
│   │   ├── optim/              # Optimization
│   │   │   ├── __init__.mojo
│   │   │   ├── osqp.mojo      # ADMM QP solver (81 lines)
│   │   │   └── lm.mojo        # Levenberg-Marquardt (100 lines)
│   │   │
│   │   └── nn/                 # Neural network components
│   │       ├── __init__.mojo
│   │       ├── stable_linear.mojo # Spectral-norm constrained layer
│   │       ├── autograd.mojo   # Reverse-mode autodiff tape (149 lines)
│   │       └── adam.mojo       # Adam optimizer
│   │
│   ├── sparse/                  # Layer 1: Custom sparse math (6 files, ~608 lines)
│   │   ├── __init__.mojo
│   │   ├── csr.mojo           # CSRMatrix with SIMD spmv (166 lines)
│   │   ├── coo.mojo           # COOMatrix with merge-sort to CSR (137 lines)
│   │   ├── diag.mojo          # Diagonal matrix (63 lines)
│   │   ├── ops.mojo           # kron, spgemm, spmm, add, scale (171 lines)
│   │   └── gpu_kernels.mojo    # GPU SpMV kernels (66 lines)
│   │
│   └── gpu_utils/              # GPU utilities (3 files)
│       ├── __init__.mojo
│       ├── detect.mojo         # Multi-backend GPU detection (59 lines)
│       ├── dtype.mojo          # METAL_*/CUDA_* type constants
│       └── host_utils.mojo     # DeviceContext creation helpers
│
├── tests/
├── benchmarks/
├── examples/
├── python/                      # Python package
└── cpp/                         # C++ examples
```

---

## Complete File Inventory

### Layer 5: Bindings (3 files)

| File | Lines | Description |
|------|-------|-------------|
| `bindings/__init__.mojo` | — | Package marker |
| `bindings/c_abi.mojo` | — | C/C++ FFI exports: `fpe_init`, `fpe_price_single`, `fpe_price_batch`, `fpe_calibrate` |
| `bindings/python_module.mojo` | — | Python extension: `PyInit_fpe_engine` with `price_single`, `price_batch`, `solve_fpe` |

### Layer 4: Pricing Server (9 files, ~1,160 lines)

| File | Lines | Description |
|------|-------|-------------|
| `server/__init__.mojo` | 8 | Exports: PDFCache, Interpolator, Payoffs, Greeks, Pricer, PricingEngine |
| `server/pricing_engine.mojo` | 42 | **Top-level orchestrator**: PDF cache lookup → pricer dispatch |
| `server/pricer.mojo` | 447 | **Unified pricer** with CPU/GPU/parallel dispatch, pre-computed quadrature weights |
| `server/pdf_cache.mojo` | 161 | PDF grid cache with JSON serialization/deserialization |
| `server/interpolator.mojo` | 187 | Bicubic (Catmull-Rom) and bilinear interpolation on 2D PDF grid |
| `server/greeks.mojo` | 135 | Greeks (Δ/Γ/ν/θ) via central finite differences on PDF grid |
| `server/payoffs.mojo` | 83 | Payoff traits: `BarrierUpAndOut`, `BarrierDownAndIn`, `EuropeanCall`, `EuropeanPut` |
| `server/vol_surface.mojo` | 20 | Implied volatility surface generation from NAIS-Net inference |
| `server/gpu_pricing_kernels.mojo` | 77 | GPU kernel for batch payoff integration (block-per-option) |

### Layer 3: Engines (18 files, ~2,400 lines)

#### FPE Engine (12 files)

| File | Lines | Description |
|------|-------|-------------|
| `engines/fpe/__init__.mojo` | — | Exports: FPEDomain, GalerkinAssembler, InitialCondition, FPESolver, PDFComputer, HestonParams |
| `engines/fpe/solver.mojo` | 236 | **Unified FPE solver** with comptime CPU/GPU dispatch (RadauIIA + sparse SpMV) |
| `engines/fpe/galerkin.mojo` | 158 | **Galerkin assembly**: mass matrix M and stiffness matrix K for Heston FPE |
| `engines/fpe/domain.mojo` | — | FPE computational domain: knot generation, grid points, basis construction |
| `engines/fpe/heston_params.mojo` | — | Heston model parameters struct with validation and Feller condition check |
| `engines/fpe/initial_cond.mojo` | — | Initial condition: bivariate Gaussian with OSQP NNLS solve |
| `engines/fpe/pdf.mojo` | — | PDF reconstruction from Galerkin coefficients |

#### FPE GPU Kernels (6 files)

| File | Lines | Description |
|------|-------|-------------|
| `engines/fpe/gpu/__init__.mojo` | — | Package marker |
| `engines/fpe/gpu/executor.mojo` | 172 | **Full GPU chain executor** orchestrating batch pricing on GPU |
| `engines/fpe/gpu/domain.mojo` | — | GPU kernels: knot generation, grid construction, basis functions, boundary conditions |
| `engines/fpe/gpu/matrix.mojo` | — | GPU kernels: sparse matrix assembly, delta function, initial condition |
| `engines/fpe/gpu/solver.mojo` | — | GPU kernels: LU decomposition, RADAU5 ODE solver |
| `engines/fpe/gpu/integration.mojo` | — | GPU integration kernel for option pricing |
| `engines/fpe/gpu/calibration.mojo` | — | GPU kernels: loss computation, Levenberg-Marquardt optimization |

#### NAIS Engine (8 files)

| File | Lines | Description |
|------|-------|-------------|
| `engines/nais/__init__.mojo` | — | Exports: NaisNet, VolterraProcess, VarianceProcess, FBSDELoss, Trainer, Inferencer |
| `engines/nais/nais_net.mojo` | 387 | **NAIS-Net architecture**: 6-layer residual network with stable linear layers |
| `engines/nais/volterra.mojo` | 139 | Fractional Brownian motion via Volterra representation (direct + FFT convolution) |
| `engines/nais/variance.mojo` | — | Rough Bergomi variance process: `ε(t)·exp(η·X̃ - 0.5η²t^{2H})` |
| `engines/nais/fbsde.mojo` | — | Forward-backward SDE loss with tracked autodiff |
| `engines/nais/trainer.mojo` | — | CPU training loop with finite-difference gradients |
| `engines/nais/inferencer.mojo` | — | Online inference: `(t,S,V) → (price, delta, implied vol)` |
| `engines/nais/gpu_trainer.mojo` | — | GPU training executor for NAIS-Net |
| `engines/nais/gpu_forward_kernels.mojo` | — | GPU batch forward pass kernel for NAIS-Net |
| `engines/nais/gpu_train_kernels.mojo` | — | GPU training kernels: FBSDE loss, gradient descent |

#### Calibrator (3 files)

| File | Lines | Description |
|------|-------|-------------|
| `engines/calibrator/__init__.mojo` | — | Exports: Calibrator, ObjectiveFunction |
| `engines/calibrator/calibrator.mojo` | — | Heston parameter calibration using Levenberg-Marquardt |
| `engines/calibrator/objective.mojo` | — | Calibration objective: sum of squared pricing errors vs market prices |

### Layer 2: Domain Numerics (18 files, ~1,700 lines)

#### B-Spline Module (5 files)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/bspline/__init__.mojo` | — | Exports: GenerateKnots, BSplineBasis, RecombinationBasis, TensorProductBasis |
| `numerics/bspline/basis.mojo` | 188 | **B-spline evaluation** via De Boor-Cox algorithm with SIMD vectorization |
| `numerics/bspline/knots.mojo` | — | Knot vector generation: uniform, Chebyshev, non-uniform |
| `numerics/bspline/recombination.mojo` | — | Boundary condition enforcement via recombination matrix |
| `numerics/bspline/tensor_product.mojo` | — | 2D tensor product basis for (S,V) grid with Kronecker products |

#### ODE Solvers (4 files)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/ode/__init__.mojo` | — | Exports: ODESystem, ODESolution, RungeKutta45, BackwardEuler, RadauIIA |
| `numerics/ode/radau.mojo` | 377 | **RadauIIA** implicit RK solver (order 5) for stiff FPE; BackwardEuler fallback |
| `numerics/ode/rk45.mojo` | — | Explicit Runge-Kutta 45 solver with adaptive step size |
| `numerics/ode/types.mojo` | — | ODESystem trait and ODESolution struct definitions |

#### Optimizer (3 files)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/optim/__init__.mojo` | 4 | Exports: ProjectedGradient, OSQP, LevenbergMarquardt |
| `numerics/optim/lm.mojo` | 100 | **Levenberg-Marquardt** nonlinear least squares solver |
| `numerics/optim/osqp.mojo` | 81 | **OSQP solver** (ADMM-based QP) for non-negative least squares |

#### Neural Network Runtime (4 files)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/nn/__init__.mojo` | — | Exports: StableLinear, GradientTape, Adam |
| `numerics/nn/autograd.mojo` | 149 | **Reverse-mode autodiff tape**: record_value/add/mul/sin/linear, backward |
| `numerics/nn/stable_linear.mojo` | — | Spectral-norm constrained linear layer for NAIS-Net stability |
| `numerics/nn/adam.mojo` | — | Adam adaptive learning rate optimizer |

#### Utilities (2 files)

| File | Lines | Description |
|------|-------|-------------|
| `numerics/__init__.mojo` | — | Exports: BSplineBasis, RecombinationBasis, TensorProductBasis, linalg, utils |
| `numerics/utils.mojo` | 136 | Shared utilities: linspace, zeros, copy_vec/mat, swap_rows, abs/max/min/clamp |
| `numerics/linalg.mojo` | — | LU solve with partial pivoting, dense matvec |

### Layer 1: Sparse Math (6 files, ~608 lines)

| File | Lines | Description |
|------|-------|-------------|
| `sparse/__init__.mojo` | 5 | Exports: COOMatrix, CSRMatrix, DiagMatrix, sparse operations |
| `sparse/csr.mojo` | 166 | **CSRMatrix** with SIMD SpMV, zero-allocation `spmv_into`, transpose |
| `sparse/coo.mojo` | 137 | **COOMatrix** with merge-sort conversion to CSR |
| `sparse/diag.mojo` | 63 | Diagonal matrix with SIMD multiplication and inversion |
| `sparse/ops.mojo` | 171 | Sparse operations: **add**, **scale**, **transpose**, **spgemm**, **spmm**, **kron** |
| `sparse/gpu_kernels.mojo` | 66 | GPU SpMV kernels (single and batch) |

### GPU Utilities (4 files)

| File | Lines | Description |
|------|-------|-------------|
| `gpu_utils/__init__.mojo` | — | Exports: GPU detection, context creation, dtype utilities |
| `gpu_utils/detect.mojo` | 59 | **Multi-backend GPU detection**: Metal/CUDA/HIP/generic/CPU |
| `gpu_utils/dtype.mojo` | — | Cross-platform dtype/layout constants (METAL_*, CUDA_*, HIP_*) |
| `gpu_utils/host_utils.mojo` | — | Automatic DeviceContext creation based on detected backend |

---

## Core Components Detail

### 1. Sparse Mathematics (`src/sparse/`, 608 lines)

Custom sparse matrix operations optimized for FPE assembly.

```
┌─────────────────────────────────────────────────────────────┐
│  sparse/                                                    │
│  ├── csr.mojo (166) ─────► CSRMatrix                        │
│  │   ├── spmv()        SIMD vectorized row dot products    │
│  │   ├── spmv_into()   Zero-allocation for ODE inner loops  │
│  │   └── transpose()   O(nnz) without dense round-trip     │
│  │                                                          │
│  ├── coo.mojo (137) ─────► COOMatrix                       │
│  │   ├── append()      Efficient triplet accumulation      │
│  │   └── to_csr()      Merge-sort → CSR conversion         │
│  │                                                          │
│  ├── diag.mojo (63) ─────► DiagonalMatrix                   │
│  │   └── matvec()      SIMD diagonal scaling               │
│  │                                                          │
│  ├── ops.mojo (171) ─────► kron, spgemm, spmm, add, scale  │
│  │   ├── kron()        Kronecker: O(nnz_A × nnz_B)        │
│  │   ├── spgemm()      Sparse × sparse → sparse            │
│  │   ├── spmm()        Sparse × dense → dense               │
│  │   ├── add()         O(nnz) merge-sort add               │
│  │   └── scale()       O(nnz) direct scaling               │
│  │                                                          │
│  └── gpu_kernels.mojo (66) ─► GPU SpMV                      │
│      ├── spmv_kernel()    One thread per row               │
│      └── batch_spmv_kernel() grid=(nrows, B)               │
└─────────────────────────────────────────────────────────────┘
```

**Key Optimization**: `spmv_into` eliminates List allocation overhead in ODE inner loops where `rhs()` is called hundreds of times.

### 2. B-Spline Module (`src/numerics/bspline/`, 188+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  numerics/bspline/                                          │
│  ├── basis.mojo (188)                                       │
│  │   ├── BSplineBasis[degree]  De Boor-Cox with SIMD       │
│  │   ├── de_boor_cox()        Recursive evaluation         │
│  │   ├── _de_boor_cox_simd()  SIMD batch evaluation        │
│  │   ├── evaluate_batch_simd() Multiple points simultaneously│
│  │   ├── first_derivative_all() Basis function derivatives  │
│  │   └── eval_all()           Sparse collocation matrix      │
│  │                                                          │
│  ├── knots.mojo ─────────► GenerateKnots                     │
│  │   ├── uniform()           Clamped uniform knots          │
│  │   ├── chebyshev()         Chebyshev nodes                │
│  │   └── from_data()         Quantile-based knots           │
│  │                                                          │
│  ├── recombination.mojo ► RecombinationBasis               │
│  │   └── Boundary conditions via recombination matrix        │
│  │                                                          │
│  └── tensor_product.mojo ► TensorProductBasis              │
│      └── 2D basis for (S,V) grid using Kronecker products  │
└─────────────────────────────────────────────────────────────┘
```

**Key Optimization**: `evaluate_batch_simd` processes multiple evaluation points simultaneously using SIMD vectors.

### 3. ODE Integrators (`src/numerics/ode/`, 377+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  numerics/ode/                                               │
│  ├── radau.mojo (377)                                       │
│  │   ├── RadauIIA[System]  3-stage implicit RK, order 5    │
│  │   │   ├── Comptime Butcher tableau constants             │
│  │   │   ├── Newton iteration for implicit stages           │
│  │   │   └── Adaptive step size control                    │
│  │   │                                                    │
│  │   └── BackwardEuler[System]  Simple stiff fallback       │
│  │       └── Richardson extrapolation                      │
│  │                                                          │
│  ├── rk45.mojo ─────────► RungeKutta45                      │
│  │   └── Dormand-Prince embedded RK method                  │
│  │                                                          │
│  └── types.mojo ────────► ODESystem trait                   │
│      └── rhs(t, y, dydt) + dim() interface                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Feature**: `comptime` Butcher tableau for RadauIIA eliminates runtime table lookups.

### 4. FPE Engine (`src/engines/fpe/`, ~400+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  engines/fpe/                                               │
│  ├── solver.mojo (236)                                      │
│  │   ├── FPESolver[B]  Unified solver: B=1→CPU, B>1→GPU   │
│  │   │   ├── _integrate_cpu_sparse()  RadauIIA + sparse   │
│  │   │   ├── _solve_gpu_batch()       GPU parallel path   │
│  │   │   └── _solve_cpu_parallel()    CPU multi-core       │
│  │   │                                                    │
│  │   ├── FPESparseSystem  ODESystem using CSR spmv        │
│  │   └── FPEDenseSystem   Fallback dense system            │
│  │                                                          │
│  ├── galerkin.mojo (158)                                    │
│  │   ├── GalerkinAssembler[B]                              │
│  │   ├── mass_matrix()   M = ΦᵀWΦ (sparse)                │
│  │   └── stiffness_matrix() K = drift + diffusion terms   │
│  │                                                          │
│  ├── domain.mojo ────────► FPEDomain                        │
│  │   ├── build_basis()    Tensor product B-spline basis   │
│  │   └── quadrature_weights() Gauss-Legendre weights      │
│  │                                                          │
│  ├── heston_params.mojo ► HestonParams                     │
│  │   └── κ, θ, σ, ρ, r, T, S₀, V₀ + Feller validation    │
│  │                                                          │
│  ├── initial_cond.mojo ► InitialCondition[B]               │
│  │   └── Bivariate Gaussian + OSQP NNLS                   │
│  │                                                          │
│  └── pdf.mojo ─────────► PDFComputer[B]                    │
│      └── pdf = Φ @ q(t) reconstruction                     │
│                                                             │
│  └── gpu/                                                   │
│      ├── executor.mojo (172)                                │
│      │   └── GPUFullChainExecutor[B]                        │
│      │       ├── execute_batch_pricing()  Full GPU chain   │
│      │       └── execute_calibration_logic()  GPU LM      │
│      ├── domain.mojo     GPU knots/grid/basis/boundary      │
│      ├── matrix.mojo     GPU sparse/delta/initial           │
│      ├── solver.mojo    GPU LU/RADAU5                       │
│      ├── integration.mojo GPU payoff integration            │
│      └── calibration.mojo GPU loss/LM optimization          │
└─────────────────────────────────────────────────────────────┘
```

### 5. NAIS Engine (`src/engines/nais/`, ~600+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  engines/nais/                                              │
│  ├── nais_net.mojo (387)                                    │
│  │   ├── NaisNet    6-layer residual architecture          │
│  │   │   ├── Layer 1: Linear → sin                          │
│  │   │   ├── Layers 2-4: [StableLinear + skip + sin] × 3   │
│  │   │   ├── Layer 5: u = W₅h + b₅                         │
│  │   │   └── Layer 6: φ = W₆h + b₆                         │
│  │   │                                                    │
│  │   ├── forward(t, x)           Plain forward pass        │
│  │   └── forward_tracked(...)    With autodiff tape        │
│  │                                                          │
│  ├── volterra.mojo (139)                                    │
│  │   ├── VolterraProcess[B]  Fractional Brownian motion     │
│  │   │   ├── generate()      O(N²) direct convolution      │
│  │   │   └── generate_fft()  O(N log N) FFT convolution   │
│  │   └── Uses MAX kernels: rfft, irfft                     │
│  │                                                          │
│  ├── variance.mojo ────► VarianceProcess[B]                │
│  │   └── ε(t)·exp(η·X̃ - 0.5η²t^{2H})                     │
│  │                                                          │
│  ├── fbsde.mojo ───────► FBSDELoss[B]                      │
│  │   └── Forward-backward SDE loss computation              │
│  │                                                          │
│  ├── trainer.mojo ─────► Trainer[B]                         │
│  │   └── CPU training loop with finite-difference grads    │
│  │                                                          │
│  ├── inferencer.mojo ──► Inferencer[B]                      │
│  │   └── (t,S,V) → (price, delta, implied vol)             │
│  │                                                          │
│  └── gpu_*.mojo ───────► GPU training kernels               │
│      ├── gpu_trainer.mojo       GPU training executor       │
│      ├── gpu_forward_kernels.mojo  GPU batch forward        │
│      └── gpu_train_kernels.mojo   GPU FBSDE loss/gradients  │
└─────────────────────────────────────────────────────────────┘
```

### 6. Neural Network Runtime (`src/numerics/nn/`, ~200+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  numerics/nn/                                               │
│  ├── autograd.mojo (149)                                    │
│  │   ├── Tape            Reverse-mode autodiff tape        │
│  │   │   ├── record_value()  Input value                   │
│  │   │   ├── record_add()    Addition                      │
│  │   │   ├── record_mul()    Multiplication                │
│  │   │   ├── record_sin()    Sine                          │
│  │   │   ├── record_linear()  W@x + b                     │
│  │   │   └── backward()      Gradient backpropagation       │
│  │   │                                                    │
│  │   └── GradientTape    Finite-difference alternative     │
│  │                                                          │
│  ├── stable_linear.mojo ► StableLinear                      │
│  │   └── W = I - RᵀR (spectral norm constraint)           │
│  │                                                          │
│  └── adam.mojo ─────────► Adam optimizer                   │
│      └── m, v moments + bias correction                   │
└─────────────────────────────────────────────────────────────┘
```

### 7. GPU Utilities (`src/gpu_utils/`, ~100+ lines)

```
┌─────────────────────────────────────────────────────────────┐
│  gpu_utils/                                                 │
│  ├── detect.mojo (59)                                       │
│  │   ├── detect_gpu_backend()  Returns: metal/cuda/hip/cpu │
│  │   ├── is_gpu_available()    Boolean check               │
│  │   └── get_device_api_name()  API string for DeviceContext│
│  │                                                          │
│  ├── dtype.mojo ───────► METAL_DTYPE, CUDA_DTYPE, etc.     │
│  │                      Layout constants per backend        │
│  │                                                          │
│  └── host_utils.mojo ► create_device_context()             │
│      └── Automatic backend selection                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Three Runtime Modes

### Mode 1: CPU Single Pricing (<1ms target)

```
Input: (S, K, T, barrier_type, barrier_level, param_hash)
   │
   ▼
[1] PDFCache lookup (Dict[param_hash → PDFGrid])    O(1), <1μs
   │
   ▼
[2] Bicubic interpolation on S×V grid               SIMD, <50μs
   │
   ▼
[3] Pre-computed trapezoidal quadrature            Reused per option
   │
   ▼
[4] Payoff integration: ∫∫ payoff(S)·PDF dS dV    <100μs
   │   • Payoff hoisted out of V-loop (was O(n_s×n_v), now O(n_s))
   │   • SIMD inner loop over variance dimension
   │
   ▼
[5] Greeks: Δ, Γ, ν, θ via central finite diff    <200μs
   │
   ▼
Output: (price, Δ, Γ, ν, Θ)                       Total: <400μs
```

### Mode 2: GPU Batch Pricing

```
Input: N pricing requests + shared HestonParams
   │
   ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPUFullChainExecutor[B] — Full GPU chain, no CPU fallback       │
│                                                                  │
│  [1] generate_knots_gpu_kernel    GPU knot generation            │
│  [2] grid_gpu_kernel              GPU grid construction          │
│  [3] basis_gpu_kernel             GPU B-spline basis             │
│  [4] boundary_gpu_kernel          GPU boundary conditions        │
│  [5] spmatrix_gpu_kernel          GPU sparse matrix assembly      │
│  [6] delta_gpu_kernel             GPU delta function             │
│  [7] initial_gpu_kernel           GPU initial condition           │
│  [8] lu_gpu_kernel                GPU LU decomposition           │
│  [9] radau5_gpu_kernel            GPU RADAU5 ODE solver          │
│  [10] integrate_gpu_kernel         GPU payoff integration         │
└──────────────────────────────────────────────────────────────────┘
   │
   ▼
Output: B × (price, Δ, Γ, ν, Θ)
```

### Mode 3: GPU Batch Calibration

```
Input: Market prices + strikes + expiries + init params (B batches)
   │
   ▼
[1] GPU batch pricing (per LM iteration)
   │
   ▼
[2] loss_gpu_kernel        Compute Σ(model - market)²
   │
   ▼
[3] lm_optimization_gpu_kernel  Levenberg-Marquardt step
   │
   ▼
Output: B × calibrated (κ, θ, σ, ρ, V₀)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Mojo v0.26.3 | Zero-cost abstractions, SIMD, GPU |
| **GPU Compute** | MAX AI Kernels 26.3.0 | matmul, gemv, rfft, irfft |
| **GPU Targets** | Metal, CUDA, HIP | Apple Silicon, NVIDIA, AMD |
| **Package Manager** | pixi | Reproducible environments |
| **Testing** | Mojo std.testing | Unit tests |
| **Benchmarks** | Mojo std.benchmark | Performance profiling |

### Mojo Features Used

| Feature | Usage | Files |
|---------|-------|-------|
| `comptime` | Compile-time batch dimension, GPU/CPU dispatch | solver, pricer, detect |
| `SIMD` | Vectorized B-spline, sparse operations | basis, csr, radau |
| `traits` | ODESystem, Payoff interfaces | ode/types, payoffs |
| `struct` | Zero-overhead value types | All files |
| `has_accelerator()` | Compile-time GPU detection | solver, pricer, detect |
| `has_apple_gpu_accelerator()` | Metal-specific path | pricer, detect |
| `has_nvidia_gpu_accelerator()` | CUDA-specific path | detect |
| `has_amd_gpu_accelerator()` | HIP-specific path | detect |
| `vectorize`, `parallelize` | SIMD and multi-core | basis, pricer, galerkin |
| `@fieldwise_init` | Bulk initialization | Most structs |
| `@always_inline` | Zero call overhead | pricer |

---

## Performance Targets

| Metric | Target | vs Python |
|--------|--------|-----------|
| Single pricing (cached PDF) | <400μs | 100× faster |
| Batch pricing (1000 options) | <10ms | 500× faster |
| FPE solve (single) | <1s | 60× faster |
| GPU calibration (64 param sets) | <25s | 10× faster |

---

## Dependencies

```toml
# pixi.toml
[dependencies]
max = "==26.3.0.dev2026040405"
mojo = "==0.26.3.0.dev2026040405"
matplotlib = ">=3.10.8,<4"
scipy = ">=1.17.1,<2"
numpy = ">=2.4.3,<3"
```

---

## Code Statistics

| Category | Files | Lines |
|----------|-------|-------|
| Sparse Math | 6 | 608 |
| Numerics | 18 | ~1,700 |
| Engines | 18 | ~2,400 |
| Server | 9 | ~1,160 |
| Bindings | 3 | — |
| GPU Utils | 4 | ~100 |
| **Total** | **58** | **~7,088** |

---

## Getting Started

### Prerequisites

- macOS ARM64 (Apple Silicon) or Linux x86-64
- pixi package manager
- Mojo v0.26.3+
- MAX SDK 26.3.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/FPE_option.git
cd FPE_option

# Install dependencies
pixi install

# Build the project
pixi run build
```

### Running Tests

```bash
# Run all tests
pixi run test

# Run GPU benchmarks
pixi run gpu-bench
```

### Usage Example

```mojo
from engines.fpe.solver import FPESolver
from engines.fpe.heston_params import HestonParams
from server.pricing_engine import PricingEngine

# Create Heston parameters
var params = HestonParams(
    kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
    r=0.05, T=1.0, S0=100.0, V0=0.04
)

# Solve FPE (CPU single)
var solver = FPESolver[1]()
var pdf = solver.solve(domain, params, t_eval)

# Price barrier option
var engine = PricingEngine()
var result = engine.price[1](requests)
```

---

## Key Design Patterns

### 1. Unified Parametric Compute

```mojo
struct FPESolver[B: Int]:
    # B=1: CPU single-stream with RadauIIA + sparse spmv
    # B>1 + GPU: GPU parallel (one thread-block per batch)
    # B>1 + CPU: CPU parallel via parallelize[]
    
    comptime if B == 1:
        return self._integrate_cpu_sparse(M, K, q0, t_eval)
    else:
        comptime if has_accelerator():
            return self._solve_gpu_batch(M, K, q0, t_eval)
        else:
            return self._solve_cpu_parallel(M, K, q0, t_eval)
```

### 2. Multi-Backend GPU Support

```mojo
from gpu_utils.detect import detect_gpu_backend

comptime if has_apple_gpu_accelerator():
    # Metal path
    var tensor = LayoutTensor[METAL_DTYPE, METAL_VEC_LAYOUT](buffer)
else:
    # CUDA/HIP path
    var tensor = LayoutTensor[CUDA_DTYPE, CUDA_VEC_LAYOUT](buffer)
```

### 3. SIMD-Vectorized Inner Loops

```mojo
# Pre-compute quadrature weights once
var ds_weights = self._compute_trap_weights(grid.s_points)

# Hoist payoff out of V-loop
for i in range(n_s):
    var payoff_val = self._payoff_value(req, S)
    if payoff_val == 0.0:
        continue
    
    # SIMD inner loop
    var j = 0
    while j + simd_width <= n_v:
        var pdf_vals = SIMD[DType.float64, simd_width]()
        var dv_vals = SIMD[DType.float64, simd_width]()
        for k in range(simd_width):
            pdf_vals[k] = grid.pdf[i][j + k]
            dv_vals[k] = dv_weights[j + k]
        v_sum += (pdf_vals * dv_vals).reduce_add()
        j += simd_width
```

---

## License

Proprietary — All rights reserved.

---

*Generated with DeepWiki-style documentation*
