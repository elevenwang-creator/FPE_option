# FPE Option Pricing Engine — Module Call Graph & Data Pipeline

> Detailed dependency relationships and data flow between all modules with direct links to source code.

---

## Table of Contents

1. [Top-Level Data Flow](#1-top-level-data-flow)
2. [Layer 0 → Layer 1: MAX Kernels](#2-layer-0--layer-1-max-kernels)
3. [Layer 1 → Layer 2: Sparse Math → Numerics](#3-layer-1--layer-2-sparse-math--numerics)
4. [Layer 2 → Layer 3: Numerics → Engines](#4-layer-2--layer-3-numerics--engines)
5. [Layer 3 → Layer 4: Engines → Server](#5-layer-3--layer-4-engines--server)
6. [Layer 4 → Layer 5: Server → Bindings](#6-layer-4--layer-5-server--bindings)
7. [Complete Dependency Tree](#7-complete-dependency-tree)
8. [Entry Points & Usage](#8-entry-points--usage)

---

## 1. Top-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              TOP-LEVEL DATA FLOW                                                │
│                                                                                                                  │
│   User Input                                                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  HestonParams | PricingRequest | MarketData | NaisNet Weights                                              │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                    │                                                             │
│                                                    ▼                                                             │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Layer 4: Pricing Server  ───────────────────────────────────────────────────────────────────────  │   │
│   │  ┌────────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌────────────────┐                   │   │
│   │  │ PricingEngine │───►│ PDFCache        │───►│ Pricer[B]     │───►│ Payoffs       │                   │   │
│   │  │ (src/server/  │    │ (src/server/    │    │ (src/server/  │    │ (src/server/  │                   │   │
│   │  │  pricing_     │    │  pdf_cache.     │    │  pricer.      │    │  payoffs.     │                   │   │
│   │  │  engine.mojo) │    │  mojo)          │    │  mojo)        │    │  mojo)        │                   │   │
│   │  └────────────────┘    └─────────────────┘    └────────────────┘    └────────────────┘                   │   │
│   │         │                   │                    │                    │                                     │   │
│   │         │                   │                    │                    ▼                                     │   │
│   │         │                   │                    │            ┌────────────────┐                           │   │
│   │         │                   │                    │            │ Interpolator   │                           │   │
│   │         │                   │                    │            │ (src/server/   │                           │   │
│   │         │                   │                    │            │  interpolator.  │                           │   │
│   │         │                   │                    │            │  mojo)         │                           │   │
│   │         │                   │                    │            └────────────────┘                           │   │
│   │         │                   │                    │                    │                                     │   │
│   │         │                   │                    ▼                    ▼                                     │   │
│   │         │                   │            ┌────────────────┐    ┌────────────────┐                           │   │
│   │         │                   │            │ Greeks[B]      │    │ PricingResult │                           │   │
│   │         │                   │            │ (src/server/   │    │ price,Δ,Γ,ν,θ │                           │   │
│   │         │                   │            │  greeks.mojo)  │    └────────────────┘                           │   │
│   │         │                   │            └────────────────┘                                                │   │
│   │         └───────────────────┴────────────────────────────────────────────────────────────────────────────┘   │
│   │                            │                                                                                 │
│   │                            ▼                                                                                 │
│   │   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│   │   │  Layer 3: Engines  ──────────────────────────────────────────────────────────────────────────────   │ │
│   │   │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐              │ │
│   │   │  │ FPESolver[B]  │    │ NaisNet        │    │ Calibrator[B] │    │ Inferencer    │              │ │
│   │   │  │ (src/engines/  │    │ (src/engines/  │    │ (src/engines/ │    │ (src/engines/ │              │ │
│   │   │  │  fpe/solver.   │    │  nais/nais_    │    │  calibrator/  │    │  nais/        │              │ │
│   │   │  │  mojo)         │    │  net.mojo)     │    │  calibrator.  │    │  inferencer.  │              │ │
│   │   │  └───────┬────────┘    │  │             │    │  mojo)        │    │  mojo)        │              │ │
│   │   │          │             │  │             │    └───────┬────────┘    └───────┬────────┘              │ │
│   │   │          │             │  ▼             │            │                       │                       │ │
│   │   │          │             │ ┌────────────────┐       │                       │                       │ │
│   │   │          │             │ │ VolterraProc   │       │                       │                       │ │
│   │   │          │             │ │ VarianceProc   │       │                       │                       │ │
│   │   │          │             │ │ FBSDELoss      │       │                       │                       │ │
│   │   │          │             │ └────────────────┘       │                       │                       │ │
│   │   │          │             │             │           │                       │                       │ │
│   │   │          │             │             ▼           │                       │                       │ │
│   │   │          │             │ ┌────────────────────┐  │                       │                       │ │
│   │   │          │             │ │ Trainer[B]          │  │                       │                       │ │
│   │   │          │             │ │ gpu_trainer.mojo   │  │                       │                       │ │
│   │   │          │             │ └────────────────────┘  │                       │                       │ │
│   │   └──────────┼─────────────┼────────────────────────┼───────────────────────┼───────────────────────┘ │
│   │              │             │                        │                       │                             │
│   │              ▼             ▼                        ▼                       ▼                             │
│   │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │   │  Layer 2: Domain Numerics  ────────────────────────────────────────────────────────────────────── │   │
│   │   │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐           │   │
│   │   │  │ BSplineBasis  │    │ RadauIIA       │    │ OSQP           │    │ StableLinear   │           │   │
│   │   │  │ (src/numerics/│    │ (src/numerics/ │    │ (src/numerics/│    │ (src/numerics/│           │   │
│   │   │  │  bspline/     │    │  ode/radau.    │    │  optim/osqp.   │    │  nn/           │           │   │
│   │   │  │  basis.mojo)  │    │  mojo)         │    │  mojo)        │    │  stable_       │           │   │
│   │   │  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘    │  linear.mojo)  │           │   │
│   │   │          │                     │                     │            └───────┬────────┘           │   │
│   │   │          │                     │                     │                      │                    │   │
│   │   │          │                     ▼                     ▼                      ▼                    │   │
│   │   │          │             ┌────────────────┐    ┌────────────────┐    ┌────────────────┐          │   │
│   │   │          │             │ ODESystem     │    │ Levenberg-     │    │ Tape           │          │   │
│   │   │          │             │ (trait)       │    │ Marquardt      │    │ (autograd)     │          │   │
│   │   │          │             │ src/numerics/  │    │ (src/numerics/│    │ (src/numerics/│          │   │
│   │   │          │             │  ode/types.   │    │  optim/lm.    │    │  nn/           │          │   │
│   │   │          │             │  mojo)        │    │  mojo)        │    │  autograd.    │          │   │
│   │   │          │             └────────────────┘    └────────────────┘    │  mojo)        │          │   │
│   │   │          │                                                       └────────────────┘          │   │
│   │   └──────────┼───────────────────────────────────────────────────────────────────────────────────────┘   │
│   │              │                                                                                             │
│   │              ▼                                                                                             │
│   │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │   │  Layer 1: Sparse Math  ───────────────────────────────────────────────────────────────────────── │   │
│   │   │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐                                 │   │
│   │   │  │ CSRMatrix      │    │ COOMatrix      │    │ Operations     │                                 │   │
│   │   │  │ (src/sparse/   │    │ (src/sparse/   │    │ (src/sparse/  │                                 │   │
│   │   │  │  csr.mojo)     │    │  coo.mojo)     │    │  ops.mojo)    │                                 │   │
│   │   │  └────────────────┘    └────────────────┘    │  kron, spgemm, │                                 │   │
│   │   │                                           │  add, scale   │                                 │   │
│   │   │                                           └────────────────┘                                 │   │
│   │   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│   │              │                                                                                             │
│   │              ▼                                                                                             │
│   │   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │   │  Layer 0: MAX AI Kernels + Mojo Stdlib  ──────────────────────────────────────────────────────── │   │
│   │   │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐          │   │
│   │   │  │ kernels.linalg │    │ kernels.nn     │    │ std.algorithm │    │ std.gpu        │          │   │
│   │   │  │ .matmul       │    │ .rfft/irfft   │    │ .vectorize    │    │ .host          │          │   │
│   │   │  └────────────────┘    └────────────────┘    │ .parallelize  │    └────────────────┘          │   │
│   │   │                                              └────────────────┘                                 │   │
│   │   └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│   │                                                                                                              │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                                  │
│   Output                                                                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  PricingResult { price, delta, gamma, vega, theta } | PDFGrid | CalibratedParams | ImpliedVolSurface     │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 0 → Layer 1: MAX Kernels

### MAX AI Kernels Usage Map

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MAX AI KERNELS (Layer 0)                                        │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ kernels.linalg                                                                   │   │
│  │ ├── matmul ──────────────────────────────────────────────────────────────────┐ │   │
│  │ │   └── Used by:                                                              │ │   │
│  │ │       ├── src/engines/nais/nais_net.mojo      (NaisNet forward)          │ │   │
│  │ │       ├── src/numerics/nn/stable_linear.mojo (StableLinear Wx+b)         │ │   │
│  │ │       └── src/engines/nais/gpu_forward_kernels.mojo (GPU batch)          │ │   │
│  │ └──────────────────────────────────────────────────────────────────────────┘ │   │
│  │ ├── gemv ──────────────────────────────────────────────────────────────────┐ │   │
│  │ │   └── Used by:                                                              │ │   │
│  │ │       └── src/numerics/nn/stable_linear.mojo                              │ │   │
│  │ └──────────────────────────────────────────────────────────────────────────┘ │   │
│  │ └── qr_factorization ─────────────────────────────────────────────────────┐ │   │
│  │     └── Used by:                                                              │ │   │
│  │         └── src/numerics/optim/lm.mojo           (Levenberg-Marquardt)   │ │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ kernels.nn                                                                       │   │
│  │ ├── rfft ──────────────────────────────────────────────────────────────────┐ │   │
│  │ │   └── Used by:                                                              │ │   │
│  │ │       └── src/engines/nais/volterra.mojo     (FFT convolution)           │ │   │
│  │ └──────────────────────────────────────────────────────────────────────────┘ │   │
│  │ ├── irfft ─────────────────────────────────────────────────────────────────┐ │   │
│  │ │   └── Used by:                                                              │ │   │
│  │ │       └── src/engines/nais/volterra.mojo     (FFT convolution)           │ │   │
│  │ └──────────────────────────────────────────────────────────────────────────┘ │   │
│  │ └── activations ───────────────────────────────────────────────────────────┐ │   │
│  │     └── Used by:                                                              │ │   │
│  │         └── src/engines/nais/nais_net.mojo    (sin activations)          │ │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: Sparse Math                                       │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ src/sparse/csr.mojo (166 lines)                                               │   │
│  │ ├── struct CSRMatrix[dtype: DType]                                             │   │
│  │ │   ├── spmv(x) ──────────────── O(nnz) sparse matrix-vector multiply       │   │
│  │ │   ├── spmv_into(x, y) ──────── Zero-allocation in-place (ODE inner loop)  │   │
│  │ │   └── transpose() ───────────── O(nnz) without dense round-trip          │   │
│  │ └── from_dense(dense) ────────── Dense → sparse conversion                  │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                  │
│                                    ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ src/sparse/coo.mojo (137 lines)                                               │   │
│  │ ├── struct COOMatrix[dtype: DType]                                           │   │
│  │ │   ├── append(row, col, val) ──── Triplet accumulation                     │   │
│  │ │   └── to_csr() ───────────────── Merge-sort → CSR conversion              │   │
│  │ └── from_dense(dense)                                                        │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                  │
│                                    ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ src/sparse/ops.mojo (171 lines)                                               │   │
│  │ ├── kron(A, B) ────────────────── Kronecker: O(nnz_A × nnz_B)                │   │
│  │ ├── spgemm(A, B) ─────────────── Sparse × sparse → sparse                    │   │
│  │ ├── spmm(A, D) ────────────────── Sparse × dense → dense                      │   │
│  │ ├── add(A, B) ─────────────────── O(nnz) merge-sort add                       │   │
│  │ ├── scale(alpha, A) ───────────── O(nnz) direct scaling                      │   │
│  │ └── sparse_transpose(A) ───────── O(nnz) sparse transpose                     │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐   │
│  │ src/sparse/gpu_kernels.mojo (66 lines)                                        │   │
│  │ ├── spmv_kernel ────────────────── One thread per row                         │   │
│  │ └── batch_spmv_kernel ──────────── grid=(nrows, B), one block per batch      │   │
│  └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Direct File Links: Layer 0 → 1

| MAX Kernel | Source File | Target File |
|------------|-------------|-------------|
| `kernels.linalg.matmul` | [src/engines/nais/nais_net.mojo](src/engines/nais/nais_net.mojo) | src/engines/nais/ |
| `kernels.linalg.matmul` | [src/numerics/nn/stable_linear.mojo](src/numerics/nn/stable_linear.mojo) | src/numerics/nn/ |
| `kernels.nn.rfft` | [src/engines/nais/volterra.mojo](src/engines/nais/volterra.mojo) | src/engines/nais/ |
| `kernels.nn.irfft` | [src/engines/nais/volterra.mojo](src/engines/nais/volterra.mojo) | src/engines/nais/ |

---

## 3. Layer 1 → Layer 2: Sparse Math → Numerics

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: Sparse Math                                               │
│                                                                                        │
│  src/sparse/                                                                          │
│  ├── [csr.mojo](src/sparse/csr.mojo) ───────────────────────────────────────────┐    │
│  ├── [coo.mojo](src/sparse/coo.mojo) ───────────────────────────────────────────┤    │
│  ├── [ops.mojo](src/sparse/ops.mojo) ───► Kron, spgemm, add, scale            │    │
│  └── [gpu_kernels.mojo](src/sparse/gpu_kernels.mojo)                            │    │
│                                                                                    │    │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │    │
│  │ src/numerics/bspline/basis.mojo (188 lines)                                │ │    │
│  │                                                                             │ │    │
│  │ uses:                                                                       │ │    │
│  │ ├── src/sparse/coo.mojo ──► COOMatrix for collocation matrix               │ │    │
│  │ └── src/sparse/csr.mojo ──► CSRMatrix for sparse basis representation     │ │    │
│  │                                                                             │ │    │
│  │ dependencies:                                                               │ │    │
│  │ ├── src/numerics/bspline/knots.mojo (GenerateKnots)                       │ │    │
│  │ └── src/numerics/utils.mojo (zeros)                                        │ │    │
│  └────────────────────────────────────────────────────────────────────────────┘ │    │
│                                    │                                                 │    │
│                                    ▼                                                 │    │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │    │
│  │ src/numerics/bspline/tensor_product.mojo                                  │ │    │
│  │                                                                             │ │    │
│  │ uses:                                                                       │ │    │
│  │ ├── src/numerics/bspline/basis.mojo ──► BSplineBasis (1D)                │ │    │
│  │ └── src/sparse/ops.mojo ──────────► kron for 2D basis construction       │ │    │
│  └────────────────────────────────────────────────────────────────────────────┘ │    │
│                                                                                    │    │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │    │
│  │ src/numerics/bspline/recombination.mojo                                   │ │    │
│  │ └── uses: src/numerics/bspline/basis.mojo                                │ │    │
│  └────────────────────────────────────────────────────────────────────────────┘ │    │
│                                                                                    └────┘
│                                    │
│                                    ▼
│  ┌────────────────────────────────────────────────────────────────────────────────┐
│  │                          LAYER 2: Domain Numerics                              │
│  │                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ src/numerics/ode/radau.mojo (377 lines)                                     │ │
│  │                                                                              │ │
│  │ struct RadauIIA[System: ODESystem] — 3-stage implicit RK, order 5         │ │
│  │ struct BackwardEuler[System: ODESystem] — Simple stiff fallback           │ │
│  │                                                                              │ │
│  │ uses:                                                                       │ │
│  │ ├── src/numerics/ode/types.mojo ──► ODESystem trait, ODESolution         │ │
│  │ ├── src/numerics/linalg.mojo ────► lu_solve for Newton iteration          │ │
│  │ └── src/numerics/utils.mojo ─────► zeros, copy_vec, swap_rows             │ │
│  │                                                                              │ │
│  │ ⭐ KEY: FPESparseSystem.rhs() calls CSRMatrix.spmv_into() for O(nnz)     │ │
│  │    ODE evaluation instead of O(n²) dense matvec                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                                 │
│                                    ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ src/numerics/ode/rk45.mojo                                                │ │
│  │ └── uses: src/numerics/ode/types.mojo, src/numerics/linalg.mojo          │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ src/numerics/optim/osqp.mojo (81 lines)                                   │ │
│  │                                                                              │ │
│  │ struct OSQP — ADMM-based QP solver for NNLS                               │ │
│  │                                                                              │ │
│  │ uses:                                                                       │ │
│  │ ├── src/numerics/linalg.mojo ────► lu_solve                               │ │
│  │ └── src/sparse/csr.mojo ────────► Sparse constraint matrix               │ │
│  │                                                                              │ │
│  │ ⭐ Used by: src/engines/fpe/initial_cond.mojo (initial condition QP)     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                                 │
│                                    ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ src/numerics/optim/lm.mojo (100 lines)                                   │ │
│  │                                                                              │ │
│  │ struct LevenbergMarquardt — Nonlinear least squares                        │ │
│  │                                                                              │ │
│  │ uses:                                                                       │ │
│  │ ├── kernels.linalg.qr_factorization ──► QR for least-squares step          │ │
│  │ └── src/numerics/linalg.mojo ─────────► lu_solve                          │ │
│  │                                                                              │ │
│  │ ⭐ Used by: src/engines/calibrator/calibrator.mojo                        │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │ src/numerics/nn/autograd.mojo (149 lines)                                  │ │
│  │                                                                              │ │
│  │ struct Tape — Reverse-mode autodiff tape                                    │ │
│  │ ├── record_value(x) ──► Input value                                       │ │
│  │ ├── record_add(a, b)                                                     │ │
│  │ ├── record_mul(a, b)                                                     │ │
│  │ ├── record_sin(x)                                                        │ │
│  │ ├── record_linear(W, b, x) ──► W@x + b                                  │ │
│  │ └── backward(loss_idx) ──► Gradient backpropagation                        │ │
│  │                                                                              │ │
│  │ ⭐ Used by: src/engines/nais/nais_net.mojo (forward_tracked)              │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Direct File Links: Layer 1 → 2

| Source File | Target File | Dependency |
|-------------|-------------|------------|
| [src/sparse/csr.mojo](src/sparse/csr.mojo) | [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | `FPESparseSystem.rhs()` calls `spmv_into()` |
| [src/sparse/ops.mojo](src/sparse/ops.mojo) | [src/engines/fpe/galerkin.mojo](src/engines/fpe/galerkin.mojo) | `kron`, `spgemm`, `add`, `scale` |
| [src/sparse/csr.mojo](src/sparse/csr.mojo) | [src/numerics/bspline/basis.mojo](src/numerics/bspline/basis.mojo) | `CSRMatrix` for sparse collocation |
| [src/sparse/coo.mojo](src/sparse/coo.mojo) | [src/numerics/bspline/basis.mojo](src/numerics/bspline/basis.mojo) | `COOMatrix` for basis assembly |
| [src/sparse/ops.mojo](src/sparse/ops.mojo) | [src/numerics/bspline/tensor_product.mojo](src/numerics/bspline/tensor_product.mojo) | `kron` for 2D basis |
| [src/numerics/linalg.mojo](src/numerics/linalg.mojo) | [src/numerics/ode/radau.mojo](src/numerics/ode/radau.mojo) | `lu_solve` for Newton iteration |
| [src/numerics/linalg.mojo](src/numerics/linalg.mojo) | [src/numerics/optim/lm.mojo](src/numerics/optim/lm.mojo) | `lu_solve` for LM step |
| [src/numerics/linalg.mojo](src/numerics/linalg.mojo) | [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | `lu_solve` for M⁻¹K computation |

---

## 4. Layer 2 → Layer 3: Numerics → Engines

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 2: Domain Numerics                                     │
│                                                                                        │
│  src/numerics/                                                                        │
│  ├── bspline/                                                                         │
│  │   ├── [basis.mojo](src/numerics/bspline/basis.mojo) ──► BSplineBasis[degree]     │
│  │   │   └── _de_boor_cox_simd() ──► SIMD vectorized evaluation                     │
│  │   ├── [knots.mojo](src/numerics/bspline/knots.mojo) ──► GenerateKnots            │
│  │   ├── [tensor_product.mojo](src/numerics/bspline/tensor_product.mojo)            │
│  │   └── [recombination.mojo](src/numerics/bspline/recombination.mojo)              │
│  │                                                                                   │
│  ├── ode/                                                                             │
│  │   ├── [radau.mojo](src/numerics/ode/radau.mojo) ──► RadauIIA, BackwardEuler    │
│  │   ├── [rk45.mojo](src/numerics/ode/rk45.mojo) ──► RungeKutta45                 │
│  │   └── [types.mojo](src/numerics/ode/types.mojo) ──► ODESystem trait              │
│  │                                                                                   │
│  ├── optim/                                                                           │
│  │   ├── [osqp.mojo](src/numerics/optim/osqp.mojo) ──► OSQP QP solver              │
│  │   └── [lm.mojo](src/numerics/optim/lm.mojo) ──► LevenbergMarquardt            │
│  │                                                                                   │
│  ├── nn/                                                                              │
│  │   ├── [autograd.mojo](src/numerics/nn/autograd.mojo) ──► Tape, GradientTape     │
│  │   ├── [stable_linear.mojo](src/numerics/nn/stable_linear.mojo) ──► StableLinear  │
│  │   └── [adam.mojo](src/numerics/nn/adam.mojo) ──► Adam optimizer                 │
│  │                                                                                   │
│  └── [utils.mojo](src/numerics/utils.mojo) ──► zeros, copy_vec, linspace          │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                      LAYER 3: Engines                                          │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                        │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│  FPE ENGINE  ──────────────────────────────────────────────────────────────────────  │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/fpe/solver.mojo (236 lines)                                    │     │
│  │                                                                             │     │
│  │ struct FPESolver[B: Int]                                                    │     │
│  │                                                                             │     │
│  │ ┌──────────────────────────────────────────────────────────────────────┐ │     │
│  │ │                         solve(params)                                  │ │     │
│  │ │                                                                      │ │     │
│  │ │   ┌───────────────────────────────────────────────────────────────┐  │ │     │
│  │ │   │  Step 1: GalerkinAssembler[B].mass_matrix(domain)            │  │ │     │
│  │ │   │  Input: FPEDomain                                             │  │ │     │
│  │ │   │  Output: CSRMatrix[DType.float64]                             │  │ │     │
│  │ │   │  File: src/engines/fpe/galerkin.mojo                        │  │ │     │
│  │ │   └───────────────────────────────────────────────────────────────┘  │ │     │
│  │ │                              │                                    │ │     │
│  │ │                              ▼                                    │ │     │
│  │ │   ┌───────────────────────────────────────────────────────────────┐  │ │     │
│  │ │   │  Step 2: GalerkinAssembler[B].stiffness_matrix(domain, params)│ │ │     │
│  │ │   │  Input: FPEDomain, HestonParams                               │  │ │     │
│  │ │   │  Output: CSRMatrix[DType.float64]                             │  │ │     │
│  │ │   └───────────────────────────────────────────────────────────────┘  │ │     │
│  │ │                              │                                    │ │     │
│  │ │                              ▼                                    │ │     │
│  │ │   ┌───────────────────────────────────────────────────────────────┐  │ │     │
│  │ │   │  Step 3: InitialCondition[B].compute(domain, params)         │  │ │     │
│  │ │   │  Input: FPEDomain, HestonParams                               │  │ │     │
│  │ │   │  Output: List[Float64] (q0)                                  │  │ │     │
│  │ │   │  File: src/engines/fpe/initial_cond.mojo                    │  │ │     │
│  │ │   └───────────────────────────────────────────────────────────────┘  │ │     │
│  │ │                              │                                    │ │     │
│  │ │                              ▼                                    │ │     │
│  │ │   ┌───────────────────────────────────────────────────────────────┐  │ │     │
│  │ │   │  Step 4: ODE Integration                                     │  │ │     │
│  │ │   │                                                              │  │ │     │
│  │ │   │  Comptime dispatch:                                          │  │ │     │
│  │ │   │  ┌───────────────────────────────────────────────────────┐  │  │ │     │
│  │ │   │  │ if B == 1:                                            │  │  │ │     │
│  │ │   │  │    _integrate_cpu_sparse(M, K, q0, t_eval)            │  │  │ │     │
│  │ │   │  │    └─► RadauIIA[FPESparseSystem]                      │  │  │ │     │
│  │ │   │  │         └─► src/numerics/ode/radau.mojo               │  │  │ │     │
│  │ │   │  │                                                      │  │  │ │     │
│  │ │   │  │  else if has_accelerator():                          │  │  │ │     │
│  │ │   │  │    _solve_gpu_batch(M, K, q0, t_eval)                │  │  │ │     │
│  │ │   │  │    └─► GPUFullChainExecutor[B]                        │  │  │ │     │
│  │ │   │  │         └─► src/engines/fpe/gpu/executor.mojo        │  │  │ │     │
│  │ │   │  │                                                      │  │  │ │     │
│  │ │   │  │  else:                                               │  │  │ │     │
│  │ │   │  │    _solve_cpu_parallel(M, K, q0, t_eval)             │  │  │ │     │
│  │ │   │  │    └─► RadauIIA + parallelize[]                       │  │  │ │     │
│  │ │   │  └───────────────────────────────────────────────────────┘  │  │ │     │
│  │ │   │  Output: ODESolution.y (coeff trajectories)               │  │ │     │
│  │ │   └───────────────────────────────────────────────────────────────┘  │ │     │
│  │ │                              │                                    │ │     │
│  │ │                              ▼                                    │ │     │
│  │ │   ┌───────────────────────────────────────────────────────────────┐  │ │     │
│  │ │   │  Step 5: PDFComputer[B].compute(basis, sol)                  │  │ │     │
│  │ │   │  Input: TensorProductBasis, ODESolution                     │  │ │     │
│  │ │   │  Output: PDFGrid (2D probability density)                  │  │ │     │
│  │ │   │  File: src/engines/fpe/pdf.mojo                            │  │ │     │
│  │ │   └───────────────────────────────────────────────────────────────┘  │ │     │
│  │ └──────────────────────────────────────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/fpe/galerkin.mojo (158 lines)                                   │     │
│  │                                                                              │     │
│  │ struct GalerkinAssembler[B: Int]                                             │     │
│  │                                                                              │     │
│  │ mass_matrix(domain) ──► M = ΦᵀWΦ                                            │     │
│  │   └─► uses: src/numerics/bspline/tensor_product.mojo (Φ)                   │     │
│  │   └─► uses: src/sparse/ops.mojo (spgemm, add, scale)                       │     │
│  │                                                                              │     │
│  │ stiffness_matrix(domain, params) ──► K = drift + diffusion                   │     │
│  │   └─► uses: src/numerics/bspline/tensor_product.mojo (Φ, ∂Φ/∂S, ∂Φ/∂V)    │     │
│  │   └─► uses: src/sparse/ops.mojo (kron, spgemm, add, scale)                 │     │
│  │   └─► uses: src/sparse/csr.mojo (CSRMatrix)                                │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/fpe/initial_cond.mojo                                          │     │
│  │                                                                              │     │
│  │ InitialCondition[B].compute(domain, params) ──► q0                         │     │
│  │   └─► uses: src/numerics/optim/osqp.mojo (OSQP for NNLS)                  │     │
│  │   └─► uses: src/numerics/bspline/tensor_product.mojo                       │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│  NAIS ENGINE  ──────────────────────────────────────────────────────────────────────  │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/nais/nais_net.mojo (387 lines)                                 │     │
│  │                                                                              │     │
│  │ struct NaisNet                                                              │     │
│  │                                                                              │     │
│  │ forward(t, x) ──────────────────────────────────────────────────────────┐ │     │
│  │ │   └─► Layer 1: sin(W₁u + b₁)                                        │ │     │
│  │ │   └─► Layers 2-4: [StableLinear + skip + sin] × 3                  │ │     │
│  │ │   └─► Layer 5: u = W₅h + b₅                                          │ │     │
│  │ │   └─► Layer 6: φ = W₆h + b₆                                          │ │     │
│  │ │   └─► uses: src/numerics/nn/stable_linear.mojo (StableLinear)        │ │     │
│  │ └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │ forward_tracked(t, x, tape) ──────────────────────────────────────────┐ │     │
│  │ │   └─► Records operations on autodiff tape                              │ │     │
│  │ │   └─► uses: src/numerics/nn/autograd.mojo (Tape)                     │ │     │
│  │ │   └─► uses: kernels.linalg.matmul (for record_linear)               │ │     │
│  │ └───────────────────────────────────────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/nais/volterra.mojo (139 lines)                                 │     │
│  │                                                                              │     │
│  │ struct VolterraProcess[B: Int]                                             │     │
│  │                                                                              │     │
│  │ generate(W) ──► O(N²) direct convolution                                   │     │
│  │ generate_fft(W) ──► O(N log N) FFT convolution                            │     │
│  │   └─► uses: kernels.nn.rfft                                               │     │
│  │   └─► uses: kernels.nn.irfft                                              │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/nais/fbsde.mojo                                                │     │
│  │                                                                              │     │
│  │ struct FBSDELoss[B: Int]                                                   │     │
│  │   └─► uses: src/engines/nais/nais_net.mojo (NaisNet)                      │     │
│  │   └─► uses: src/engines/nais/volterra.mojo (VolterraProcess)              │     │
│  │   └─► uses: src/engines/nais/variance.mojo (VarianceProcess)             │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│  CALIBRATOR  ───────────────────────────────────────────────────────────────────────  │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/calibrator/calibrator.mojo                                    │     │
│  │                                                                              │     │
│  │ Calibrator[B].run(market, init_params) ──► calibrated params              │     │
│  │   │                                                                          │     │
│  │   ├── Step 1: FPESolver[B].solve(domain, params) ──► PDF grids           │     │
│  │   │       └─► src/engines/fpe/solver.mojo                                 │     │
│  │   │                                                                          │     │
│  │   ├── Step 2: Pricer[B].price(pdf, requests) ──► model prices           │     │
│  │   │       └─► src/server/pricer.mojo                                       │     │
│  │   │                                                                          │     │
│  │   ├── Step 3: ObjectiveFunction[B].compute(prices, market) ──► loss      │     │
│  │   │       └─► src/engines/calibrator/objective.mojo                        │     │
│  │   │                                                                          │     │
│  │   └── Step 4: LevenbergMarquardt.step() ──► param update                  │     │
│  │           └─► src/numerics/optim/lm.mojo                                  │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Direct File Links: Layer 2 → 3

| Source File | Target File | Dependency |
|-------------|-------------|------------|
| [src/numerics/ode/radau.mojo](src/numerics/ode/radau.mojo) | [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | `RadauIIA[FPESparseSystem].solve()` |
| [src/numerics/bspline/tensor_product.mojo](src/numerics/bspline/tensor_product.mojo) | [src/engines/fpe/galerkin.mojo](src/engines/fpe/galerkin.mojo) | `TensorProductBasis` for M/K assembly |
| [src/numerics/optim/osqp.mojo](src/numerics/optim/osqp.mojo) | [src/engines/fpe/initial_cond.mojo](src/engines/fpe/initial_cond.mojo) | OSQP for NNLS |
| [src/numerics/nn/autograd.mojo](src/numerics/nn/autograd.mojo) | [src/engines/nais/nais_net.mojo](src/engines/nais/nais_net.mojo) | `Tape` for forward_tracked |
| [src/numerics/nn/stable_linear.mojo](src/numerics/nn/stable_linear.mojo) | [src/engines/nais/nais_net.mojo](src/engines/nais/nais_net.mojo) | `StableLinear` in each block |
| [src/numerics/optim/lm.mojo](src/numerics/optim/lm.mojo) | [src/engines/calibrator/calibrator.mojo](src/engines/calibrator/calibrator.mojo) | `LevenbergMarquardt.step()` |
| [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | [src/engines/fpe/gpu/executor.mojo](src/engines/fpe/gpu/executor.mojo) | `_solve_gpu_batch()` → `GPUFullChainExecutor` |

---

## 5. Layer 3 → Layer 4: Engines → Server

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 3: Engines                                             │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/fpe/solver.mojo (236 lines)                                    │     │
│  │                                                                              │     │
│  │ FPESolver[B].solve(domain, params, t_eval) ──────────────────────────────┐ │     │
│  │ │   └─► Output: List[List[Float64]] (solution trajectories)            │ │     │
│  │ │   └─► Used by: src/server/pricer.mojo (via PDFGrid)                  │ │     │
│  │ └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │ src/engines/fpe/pdf.mojo                                                   │     │
│  │   └─► PDFComputer[B].compute(basis, sol) ──► PDFGrid                     │     │
│  │       └─► Input to: src/server/pricing_engine.mojo (PDFCache)            │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                    │                                                   │
│                                    ▼                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/nais/inferencer.mojo                                          │     │
│  │                                                                              │     │
│  │ Inferencer[B].infer(t, S, V) ──► (price, delta, implied_vol)            │     │
│  │   └─► uses: src/engines/nais/nais_net.mojo (NaisNet.forward)             │     │
│  │   └─► Input to: src/server/vol_surface.mojo                               │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │ src/engines/calibrator/calibrator.mojo                                    │     │
│  │                                                                              │     │
│  │ Calibrator[B].calibrate(market, init_params) ──► HestonParams           │     │
│  │   └─► Output fed back to: FPESolver[B] for refined PDF                   │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │                          LAYER 4: Pricing Server                             │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/pricing_engine.mojo (42 lines)                             │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct PricingEngine                                                   │ │     │
│  │  │                                                                        │ │     │
│  │  │ price[B](requests) ───────────────────────────────────────────────┐  │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   Step 1: cache.get(param_hash) ──► Optional[PDFGrid]          │  │ │ │     │
│  │  │ │       └─► src/server/pdf_cache.mojo                             │  │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   if not cached:                                                   │  │ │ │     │
│  │  │ │       FPESolver[B].solve() ──► PDFGrid ──► cache.store()        │  │ │ │     │
│  │  │ │           └─► src/engines/fpe/solver.mojo                       │  │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   Step 2: Pricer[B].price(grid, requests) ──► PricingResult     │  │ │ │     │
│  │  │ │       └─► src/server/pricer.mojo                                 │  │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ └───────────────────────────────────────────────────────────────────┘  │ │     │
│  │  │                                                                        │ │     │
│  │  │ calibrate[B](market, init_params) ──────────────────────────────────┐  │ │     │
│  │  │ │   └─► Calibrator[B].calibrate()                                  │  │ │     │
│  │  │ │       └─► src/engines/calibrator/calibrator.mojo                 │  │ │     │
│  │  │ └───────────────────────────────────────────────────────────────────┘  │ │     │
│  │  │                                                                        │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/pdf_cache.mojo (161 lines)                                 │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct PDFCache                                                       │ │     │
│  │  │ ├── get(param_hash) ─────► Optional[PDFGrid]                         │ │     │
│  │  │ ├── store(param_hash, grid)                                           │ │     │
│  │  │ ├── save_to_disk(path)  ──► JSON serialization                       │ │     │
│  │  │ └── load_from_disk(path) ──► JSON deserialization                   │ │     │
│  │  │                                                                        │ │     │
│  │  │ PDFGrid:                                                               │ │     │
│  │  │ ├── s_points: List[Float64]   (spot price grid)                       │ │     │
│  │  │ ├── v_points: List[Float64]   (variance grid)                         │ │     │
│  │  │ └── pdf: List[List[Float64]]  (2D probability density)               │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/pricer.mojo (447 lines)                                    │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct Pricer[B: Int]                                                 │ │     │
│  │  │                                                                        │ │     │
│  │  │ price(grid, requests) ──────────────────────────────────────────────┐ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   Comptime dispatch:                                               │ │ │     │
│  │  │ │   ┌────────────────────────────────────────────────────────────┐   │ │ │ │     │
│  │  │ │   │ if B == 1:                                                │   │ │ │ │     │
│  │  │ │   │    _price_single(grid, requests)                         │   │ │ │ │     │
│  │  │ │   │    └─► Pre-compute trapezoidal weights                     │   │ │ │ │     │
│  │  │ │   │    └─► _integrate_payoff_fast() for each option          │   │ │ │ │     │
│  │  │ │   │                                                        │   │ │ │ │     │
│  │  │ │   │ else if has_accelerator():                               │   │ │ │     │
│  │  │ │   │    _price_gpu_batch(grid, requests)                      │   │ │ │ │     │
│  │  │ │   │    └─► payoff_integration_kernel (GPU)                    │   │ │ │ │     │
│  │  │ │   │    └─► src/server/gpu_pricing_kernels.mojo               │   │ │ │ │     │
│  │  │ │   │                                                        │   │ │ │ │     │
│  │  │ │   │ else:                                                   │   │ │ │ │     │
│  │  │ │   │    _price_cpu_parallel(grid, requests)                  │   │ │ │ │     │
│  │  │ │   │    └─► parallelize[] across options                     │   │ │ │ │     │
│  │  │ │   └────────────────────────────────────────────────────────────┘   │ │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   ┌────────────────────────────────────────────────────────────┐   │ │ │ │     │
│  │  │ │   │  _integrate_payoff_fast()                                 │   │ │ │ │     │
│  │  │ │   │  ┌──────────────────────────────────────────────────────┐ │   │ │ │ │     │
│  │  │ │   │  │ for i in range(n_s):                                │ │   │ │ │ │     │
│  │  │ │   │  │     payoff_val = _payoff_value(req, S[i])           │ │   │ │ │ │     │
│  │  │ │   │  │     payoff_ds = payoff_val * ds_weights[i]          │ │   │ │ │ │     │
│  │  │ │   │  │     # SIMD inner loop over V dimension:             │ │   │ │ │ │     │
│  │  │ │   │  │     v_sum = Σ pdf[i][j] * dv_weights[j]            │ │   │ │ │ │     │
│  │  │ │   │  │     price += payoff_ds * v_sum                     │ │   │ │ │ │     │
│  │  │ │   │  └──────────────────────────────────────────────────────┘ │   │ │ │ │     │
│  │  │ │   │  └─► src/server/payoffs.mojo (payoff functions)         │   │ │ │ │     │
│  │  │ │   └────────────────────────────────────────────────────────────┘   │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ │   ┌────────────────────────────────────────────────────────────┐   │ │ │ │     │
│  │  │ │   │  Greeks[B].compute_delta/gamma/vega()                    │   │ │ │ │     │
│  │  │ │   │  └─► src/server/greeks.mojo                             │   │ │ │ │     │
│  │  │ │   │  └─► src/server/interpolator.mojo (bicubic)             │   │ │ │ │     │
│  │  │ │   └────────────────────────────────────────────────────────────┘   │ │ │     │
│  │  │ │                                                                      │ │ │     │
│  │  │ └───────────────────────────────────────────────────────────────────┘ │ │     │
│  │  │                                                                        │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/payoffs.mojo (83 lines)                                   │ │     │
│  │  │                                                                        │ │     │
│  │  │ trait Payoff ─────────────────────────────────────────────────────┐ │ │     │
│  │  │ │   fn evaluate(S, K, barrier) -> Float64                         │ │ │     │
│  │  │ └────────────────────────────────────────────────────────────────┘ │ │     │
│  │  │                                                                        │ │     │
│  │  │ implementations:                                                      │ │     │
│  │  │ ├── BarrierUpAndOut  ──► max(S-K, 0) if S < barrier              │ │     │
│  │  │ ├── BarrierDownAndIn ──► max(S-K, 0) if min(S) <= barrier       │ │     │
│  │  │ ├── EuropeanCall      ──► max(S-K, 0)                            │ │     │
│  │  │ └── EuropeanPut       ──► max(K-S, 0)                            │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/greeks.mojo (135 lines)                                   │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct Greeks[B: Int]                                                │ │     │
│  │  │ ├── compute_delta()  ──► (P(S+h) - P(S-h)) / (2h)                 │ │     │
│  │  │ ├── compute_gamma()  ──► (P(S+h) - 2P(S) + P(S-h)) / h²          │ │     │
│  │  │ ├── compute_vega()   ──► (P(V+h) - P(V-h)) / (2h)                │ │     │
│  │  │ └── compute_theta()  ──► (P(t+dt) - P(t)) / dt                   │ │     │
│  │  │                                                                        │ │     │
│  │  │ uses: src/server/interpolator.mojo (bicubic interpolation)          │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/interpolator.mojo (187 lines)                             │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct Interpolator                                                   │ │     │
│  │  │ ├── bicubic_interp(s, v, grid) ──► pdf value at (s, v)             │ │     │
│  │  │ └── bilinear_interp(s, v, grid)  ──► faster linear interpolation    │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/vol_surface.mojo (20 lines)                               │ │     │
│  │  │                                                                        │ │     │
│  │  │ struct VolSurfaceGenerator                                            │ │     │
│  │  │ └── generate(nais_inferencer, strikes, expiries) ──► implied vols   │ │     │
│  │  │     └─► uses: src/engines/nais/inferencer.mojo                      │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │     │
│  │  │ src/server/gpu_pricing_kernels.mojo (77 lines)                        │ │     │
│  │  │                                                                        │ │     │
│  │  │ fn payoff_integration_kernel(...) ──► GPU kernel                      │ │     │
│  │  │     └─► grid_dim = n_options, block_dim = 256                        │ │     │
│  │  │     └─► One thread per option                                        │ │     │
│  │  │     └─► Uses METAL_VEC_LAYOUT or CUDA_VEC_LAYOUT                     │ │     │
│  │  └───────────────────────────────────────────────────────────────────────┘ │     │
│  │                                                                              │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Direct File Links: Layer 3 → 4

| Source File | Target File | Dependency |
|-------------|-------------|------------|
| [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | [src/server/pricing_engine.mojo](src/server/pricing_engine.mojo) | `FPESolver.solve()` → PDFGrid → Pricer |
| [src/engines/fpe/pdf.mojo](src/engines/fpe/pdf.mojo) | [src/server/pdf_cache.mojo](src/server/pdf_cache.mojo) | PDFGrid serialization |
| [src/engines/nais/inferencer.mojo](src/engines/nais/inferencer.mojo) | [src/server/vol_surface.mojo](src/server/vol_surface.mojo) | Implied vol surface generation |
| [src/engines/calibrator/calibrator.mojo](src/engines/calibrator/calibrator.mojo) | [src/server/pricing_engine.mojo](src/server/pricing_engine.mojo) | Calibrator result → PricingEngine |
| [src/server/pricer.mojo](src/server/pricer.mojo) | [src/server/payoffs.mojo](src/server/payoffs.mojo) | `_payoff_value()` for integration |
| [src/server/pricer.mojo](src/server/pricer.mojo) | [src/server/greeks.mojo](src/server/greeks.mojo) | Greeks computation |
| [src/server/greeks.mojo](src/server/greeks.mojo) | [src/server/interpolator.mojo](src/server/interpolator.mojo) | Bicubic interpolation for finite diff |
| [src/server/gpu_pricing_kernels.mojo](src/server/gpu_pricing_kernels.mojo) | [src/gpu_utils/host_utils.mojo](src/gpu_utils/host_utils.mojo) | `create_device_context()` |

---

## 6. Layer 4 → Layer 5: Server → Bindings

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 4: Pricing Server                                     │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐       │
│  │ src/server/pricing_engine.mojo (42 lines)                                    │       │
│  │                                                                              │       │
│  │ PricingEngine.price[B](requests) ─────────────────────────────────────────┐ │       │
│  │ │   └─► Output: List[PricingResult]                                       │ │       │
│  │ └───────────────────────────────────────────────────────────────────────┘ │       │
│  │                                                                              │       │
│  │ PricingEngine.calibrate[B](market, init_params) ──────────────────────────┐ │       │
│  │ │   └─► Output: HestonParamsBatch[B]                                    │ │       │
│  │ └───────────────────────────────────────────────────────────────────────┘ │       │
│  └────────────────────────────────────────────────────────────────────────────┘       │
│                                    │                                                   │
│  ════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐       │
│  │                          LAYER 5: Bindings                                  │       │
│  │                                                                              │       │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │       │
│  │  │ src/bindings/python_module.mojo                                       │ │       │
│  │  │                                                                        │ │       │
│  │  │ @export                                                               │ │       │
│  │  │ fn PyInit_fpe_engine() -> PythonObject                                 │ │       │
│  │  │                                                                        │ │       │
│  │  │ exported functions:                                                    │ │       │
│  │  │ ├── py_price_single(params) ──► price_single()                       │ │       │
│  │  │ │       └─► src/server/pricing_engine.mojo                           │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── py_price_batch(params, batch_size) ──► price_batch()             │ │       │
│  │  │ │       └─► src/server/pricing_engine.mojo                           │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── py_calibrate_batch(market, init_params) ──► calibrate_batch()   │ │       │
│  │  │ │       └─► src/engines/calibrator/calibrator.mojo                  │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── py_solve_fpe(params) ──► solve_fpe()                            │ │       │
│  │  │ │       └─► src/engines/fpe/solver.mojo                             │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── py_nais_train(config) ──► nais_train()                           │ │       │
│  │  │ │       └─► src/engines/nais/trainer.mojo                            │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ └── py_nais_infer(t, S, V) ──► nais_infer()                         │ │       │
│  │  │         └─► src/engines/nais/inferencer.mojo                        │ │       │
│  │  │                                                                        │ │       │
│  │  └───────────────────────────────────────────────────────────────────────┘ │       │
│  │                                                                              │       │
│  │  ┌───────────────────────────────────────────────────────────────────────┐ │       │
│  │  │ src/bindings/c_abi.mojo                                              │ │       │
│  │  │                                                                        │ │       │
│  │  │ @export extern "C" functions:                                         │ │       │
│  │  │ ├── fpe_init(config_path) ──► Initialize engine                       │ │       │
│  │  │ ├── fpe_destroy() ──► Cleanup                                         │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── fpe_price_single(...) ──► CPU single pricing                      │ │       │
│  │  │ │       └─► src/server/pricing_engine.mojo                           │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── fpe_price_batch(...) ──► GPU batch pricing                        │ │       │
│  │  │ │       └─► src/server/pricing_engine.mojo                           │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── fpe_calibrate(...) ──► Heston calibration                         │ │       │
│  │  │ │       └─► src/engines/calibrator/calibrator.mojo                   │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ ├── fpe_precompute_pdf(...) ──► Cache PDF grid                       │ │       │
│  │  │ │       └─► src/server/pdf_cache.mojo                               │ │       │
│  │  │ │                                                                      │ │       │
│  │  │ └── fpe_load_cache(path) ──► Load cached PDFs                        │ │       │
│  │  │         └─► src/server/pdf_cache.mojo                               │ │       │
│  │  │                                                                        │ │       │
│  │  └───────────────────────────────────────────────────────────────────────┘ │       │
│  │                                                                              │       │
│  └────────────────────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Direct File Links: Layer 4 → 5

| Source File | Target File | Dependency |
|-------------|-------------|------------|
| [src/server/pricing_engine.mojo](src/server/pricing_engine.mojo) | [src/bindings/python_module.mojo](src/bindings/python_module.mojo) | Python extension exports |
| [src/server/pricing_engine.mojo](src/server/pricing_engine.mojo) | [src/bindings/c_abi.mojo](src/bindings/c_abi.mojo) | C ABI exports |
| [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) | [src/bindings/python_module.mojo](src/bindings/python_module.mojo) | `solve_fpe()` export |
| [src/engines/nais/trainer.mojo](src/engines/nais/trainer.mojo) | [src/bindings/python_module.mojo](src/bindings/python_module.mojo) | `nais_train()` export |
| [src/engines/nais/inferencer.mojo](src/engines/nais/inferencer.mojo) | [src/bindings/python_module.mojo](src/bindings/python_module.mojo) | `nais_infer()` export |
| [src/engines/calibrator/calibrator.mojo](src/engines/calibrator/calibrator.mojo) | [src/bindings/c_abi.mojo](src/bindings/c_abi.mojo) | `fpe_calibrate()` export |

---

## 7. Complete Dependency Tree

```
FPE_option (root)
│
├── Layer 0: MAX AI Kernels (external)
│   ├── kernels.linalg.matmul
│   ├── kernels.linalg.gemv
│   ├── kernels.linalg.qr_factorization
│   ├── kernels.nn.rfft
│   ├── kernels.nn.irfft
│   └── kernels.nn.activations
│
├── Layer 1: Sparse Math
│   ├── [src/sparse/csr.mojo](src/sparse/csr.mojo) ───────────────────────────► used by 8 files
│   │   └── uses: (standalone)
│   ├── [src/sparse/coo.mojo](src/sparse/coo.mojo) ───────────────────────────► used by 4 files
│   │   └── uses: (standalone)
│   ├── [src/sparse/diag.mojo](src/sparse/diag.mojo) ────────────────────────► used by 2 files
│   │   └── uses: (standalone)
│   ├── [src/sparse/ops.mojo](src/sparse/ops.mojo) ───────────────────────────► used by 4 files
│   │   └── uses: src/sparse/csr.mojo, src/sparse/coo.mojo
│   └── [src/sparse/gpu_kernels.mojo](src/sparse/gpu_kernels.mojo) ───────────► used by 2 files
│       └── uses: src/gpu_utils/dtype.mojo
│
├── Layer 2: Domain Numerics
│   ├── [src/numerics/utils.mojo](src/numerics/utils.mojo) ──────────────────► used by 15 files
│   │   └── uses: (standalone)
│   ├── [src/numerics/linalg.mojo](src/numerics/linalg.mojo) ─────────────────► used by 6 files
│   │   └── uses: src/numerics/utils.mojo
│   │
│   ├── B-Spline Module
│   │   ├── [src/numerics/bspline/knots.mojo](src/numerics/bspline/knots.mojo) ──► used by 3 files
│   │   ├── [src/numerics/bspline/basis.mojo](src/numerics/bspline/basis.mojo) ──► used by 4 files
│   │   │   └── uses: src/sparse/coo.mojo, src/sparse/csr.mojo, src/numerics/utils.mojo
│   │   ├── [src/numerics/bspline/recombination.mojo](src/numerics/bspline/recombination.mojo)
│   │   │   └── uses: src/numerics/bspline/basis.mojo
│   │   └── [src/numerics/bspline/tensor_product.mojo](src/numerics/bspline/tensor_product.mojo)
│   │       └── uses: src/numerics/bspline/basis.mojo, src/sparse/ops.mojo
│   │
│   ├── ODE Module
│   │   ├── [src/numerics/ode/types.mojo](src/numerics/ode/types.mojo) ───────► used by 5 files
│   │   ├── [src/numerics/ode/rk45.mojo](src/numerics/ode/rk45.mojo) ───────► used by 1 file
│   │   │   └── uses: src/numerics/ode/types.mojo, src/numerics/linalg.mojo
│   │   └── [src/numerics/ode/radau.mojo](src/numerics/ode/radau.mojo) ──────► used by 3 files
│   │       └── uses: src/numerics/ode/types.mojo, src/numerics/linalg.mojo, src/numerics/utils.mojo
│   │
│   ├── Optimization Module
│   │   ├── [src/numerics/optim/osqp.mojo](src/numerics/optim/osqp.mojo) ────► used by 1 file
│   │   │   └── uses: src/numerics/linalg.mojo, src/sparse/csr.mojo
│   │   └── [src/numerics/optim/lm.mojo](src/numerics/optim/lm.mojo) ────────► used by 1 file
│   │       └── uses: src/numerics/linalg.mojo, (MAX kernels.linalg.qr)
│   │
│   └── Neural Network Module
│       ├── [src/numerics/nn/autograd.mojo](src/numerics/nn/autograd.mojo) ───► used by 2 files
│       ├── [src/numerics/nn/stable_linear.mojo](src/numerics/nn/stable_linear.mojo) ──► used by 2 files
│       │   └── uses: (MAX kernels.linalg.matmul)
│       └── [src/numerics/nn/adam.mojo](src/numerics/nn/adam.mojo) ──────────► used by 1 file
│
├── Layer 3: Engines
│   │
│   ├── FPE Engine
│   │   ├── [src/engines/fpe/heston_params.mojo](src/engines/fpe/heston_params.mojo) ──► used by 5 files
│   │   ├── [src/engines/fpe/domain.mojo](src/engines/fpe/domain.mojo) ──────────► used by 5 files
│   │   │   └── uses: src/numerics/bspline/tensor_product.mojo, src/numerics/bspline/knots.mojo
│   │   ├── [src/engines/fpe/galerkin.mojo](src/engines/fpe/galerkin.mojo) ─────► used by 2 files
│   │   │   └── uses: src/engines/fpe/domain.mojo, src/sparse/ops.mojo, src/sparse/csr.mojo
│   │   ├── [src/engines/fpe/initial_cond.mojo](src/engines/fpe/initial_cond.mojo) ──► used by 1 file
│   │   │   └── uses: src/engines/fpe/domain.mojo, src/numerics/optim/osqp.mojo
│   │   ├── [src/engines/fpe/pdf.mojo](src/engines/fpe/pdf.mojo) ──────────────► used by 2 files
│   │   │   └── uses: src/engines/fpe/domain.mojo
│   │   ├── [src/engines/fpe/solver.mojo](src/engines/fpe/solver.mojo) ────────► used by 4 files
│   │   │   └── uses: src/engines/fpe/domain.mojo, src/engines/fpe/galerkin.mojo,
│   │   │               src/engines/fpe/initial_cond.mojo, src/numerics/ode/radau.mojo,
│   │   │               src/numerics/linalg.mojo, src/sparse/csr.mojo
│   │   │
│   │   └── GPU FPE Kernels
│   │       ├── [src/engines/fpe/gpu/executor.mojo](src/engines/fpe/gpu/executor.mojo) ──► used by 1 file
│   │       ├── [src/engines/fpe/gpu/domain.mojo](src/engines/fpe/gpu/domain.mojo)
│   │       ├── [src/engines/fpe/gpu/matrix.mojo](src/engines/fpe/gpu/matrix.mojo)
│   │       ├── [src/engines/fpe/gpu/solver.mojo](src/engines/fpe/gpu/solver.mojo)
│   │       ├── [src/engines/fpe/gpu/integration.mojo](src/engines/fpe/gpu/integration.mojo)
│   │       └── [src/engines/fpe/gpu/calibration.mojo](src/engines/fpe/gpu/calibration.mojo)
│   │
│   ├── NAIS Engine
│   │   ├── [src/engines/nais/nais_net.mojo](src/engines/nais/nais_net.mojo) ──► used by 4 files
│   │   │   └── uses: src/numerics/nn/stable_linear.mojo, src/numerics/nn/autograd.mojo
│   │   ├── [src/engines/nais/volterra.mojo](src/engines/nais/volterra.mojo) ─► used by 2 files
│   │   │   └── uses: (MAX kernels.nn.rfft, irfft)
│   │   ├── [src/engines/nais/variance.mojo](src/engines/nais/variance.mojo) ─► used by 2 files
│   │   ├── [src/engines/nais/fbsde.mojo](src/engines/nais/fbsde.mojo) ───────► used by 2 files
│   │   │   └── uses: src/engines/nais/nais_net.mojo, src/engines/nais/volterra.mojo,
│   │   │               src/engines/nais/variance.mojo, src/numerics/nn/autograd.mojo
│   │   ├── [src/engines/nais/trainer.mojo](src/engines/nais/trainer.mojo) ───► used by 1 file
│   │   │   └── uses: src/engines/nais/nais_net.mojo, src/engines/nais/fbsde.mojo,
│   │   │               src/numerics/nn/adam.mojo, src/numerics/nn/autograd.mojo
│   │   ├── [src/engines/nais/inferencer.mojo](src/engines/nais/inferencer.mojo) ──► used by 2 files
│   │   │   └── uses: src/engines/nais/nais_net.mojo
│   │   ├── [src/engines/nais/gpu_trainer.mojo](src/engines/nais/gpu_trainer.mojo)
│   │   ├── [src/engines/nais/gpu_forward_kernels.mojo](src/engines/nais/gpu_forward_kernels.mojo)
│   │   └── [src/engines/nais/gpu_train_kernels.mojo](src/engines/nais/gpu_train_kernels.mojo)
│   │
│   └── Calibrator
│       ├── [src/engines/calibrator/objective.mojo](src/engines/calibrator/objective.mojo) ──► used by 1 file
│       └── [src/engines/calibrator/calibrator.mojo](src/engines/calibrator/calibrator.mojo) ──► used by 2 files
│           └── uses: src/engines/fpe/solver.mojo, src/engines/calibrator/objective.mojo,
│                       src/numerics/optim/lm.mojo
│
├── Layer 4: Pricing Server
│   ├── [src/server/pdf_cache.mojo](src/server/pdf_cache.mojo) ──────────────► used by 1 file
│   ├── [src/server/interpolator.mojo](src/server/interpolator.mojo) ─────────► used by 3 files
│   ├── [src/server/payoffs.mojo](src/server/payoffs.mojo) ─────────────────► used by 2 files
│   ├── [src/server/greeks.mojo](src/server/greeks.mojo) ───────────────────► used by 2 files
│   │   └── uses: src/server/interpolator.mojo
│   ├── [src/server/pricer.mojo](src/server/pricer.mojo) ──────────────────► used by 1 file
│   │   └── uses: src/server/pdf_cache.mojo, src/server/interpolator.mojo,
│   │               src/server/payoffs.mojo, src/server/greeks.mojo,
│   │               src/server/gpu_pricing_kernels.mojo, src/gpu_utils/host_utils.mojo
│   ├── [src/server/vol_surface.mojo](src/server/vol_surface.mojo) ────────► used by 0 files (entry)
│   │   └── uses: src/engines/nais/inferencer.mojo
│   ├── [src/server/pricing_engine.mojo](src/server/pricing_engine.mojo) ────► used by 2 files
│   │   └── uses: src/server/pdf_cache.mojo, src/server/greeks.mojo,
│   │               src/server/interpolator.mojo, src/server/pricer.mojo,
│   │               src/engines/fpe/solver.mojo, src/engines/calibrator/calibrator.mojo
│   └── [src/server/gpu_pricing_kernels.mojo](src/server/gpu_pricing_kernels.mojo)
│       └── uses: src/gpu_utils/dtype.mojo, src/gpu_utils/host_utils.mojo
│
├── Layer 5: Bindings
│   ├── [src/bindings/python_module.mojo](src/bindings/python_module.mojo) ───► used by 0 files (entry)
│   │   └── uses: src/server/pricing_engine.mojo, src/engines/fpe/solver.mojo,
│   │               src/engines/nais/trainer.mojo, src/engines/nais/inferencer.mojo
│   └── [src/bindings/c_abi.mojo](src/bindings/c_abi.mojo) ─────────────────► used by 0 files (entry)
│       └── uses: src/server/pricing_engine.mojo, src/engines/calibrator/calibrator.mojo,
│                   src/server/pdf_cache.mojo
│
└── GPU Utilities
    ├── [src/gpu_utils/detect.mojo](src/gpu_utils/detect.mojo) ──────────────► used by 2 files
    ├── [src/gpu_utils/dtype.mojo](src/gpu_utils/dtype.mojo) ─────────────────► used by 5 files
    └── [src/gpu_utils/host_utils.mojo](src/gpu_utils/host_utils.mojo) ────────► used by 2 files
        └── uses: src/gpu_utils/detect.mojo
```

---

## 8. Entry Points & Usage

### Python API Entry Points

```python
# src/bindings/python_module.mojo
from fpe_engine import price_single, price_batch, calibrate_batch
from fpe_engine import solve_fpe, nais_train, nais_infer

# Single pricing
result = price_single(
    S=100.0, K=100.0, T=1.0,
    payoff_type=0, barrier=110.0,
    params={'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7}
)

# Batch pricing
results = price_batch(requests, batch_size=1000)

# Calibration
calibrated_params = calibrate_batch(market_prices, strikes, expiries, init_params)
```

### C API Entry Points

```c
// src/bindings/c_abi.mojo
#include "fpe_engine.h"

// Initialize
fpe_init("/path/to/config");

// Single pricing
double price, delta, gamma, vega;
fpe_price_single(100.0, 100.0, 1.0, 0, 110.0, param_hash,
                  &price, &delta, &gamma, &vega);

// Batch pricing
fpe_price_batch(S_array, K_array, T_array, types, barriers, 
                count, param_hash, prices, deltas, gammas, vegas);

// Calibration
fpe_calibrate(market_prices, strikes, expiries, n_options,
              init_params, n_param_sets, out_params);

// Cleanup
fpe_destroy();
```

### Mojo API Entry Points

```mojo
// Direct Mojo usage

// 1. FPE Solver
from engines.fpe.solver import FPESolver
from engines.fpe.heston_params import HestonParams

var params = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, ...)
var solver = FPESolver[1](rtol=1e-6, atol=1e-8)
var pdf = solver.solve(domain, params, t_eval)

// 2. Pricing Engine
from server.pricing_engine import PricingEngine

var engine = PricingEngine()
var results = engine.price[1](requests)

// 3. Calibration
from engines.calibrator.calibrator import Calibrator

var calibrator = Calibrator[64]()
var calibrated = calibrator.calibrate(market, init_params)

// 4. NAIS Training
from engines.nais.trainer import Trainer
from engines.nais.nais_net import NaisNet

var model = NaisNet(in_dim=3, hidden=12, phi_dim=2)
var trainer = Trainer[1](model, lr=0.001)
trainer.train(config)
```

---

## Summary: Key Data Flow Paths

### Path 1: Single Pricing (CPU)
```
PricingEngine.price[1]()
  └─► PDFCache.get()
  └─► FPESolver[1].solve()
        ├─► GalerkinAssembler.mass_matrix()
        ├─► GalerkinAssembler.stiffness_matrix()
        ├─► InitialCondition.compute()
        ├─► RadauIIA[FPESparseSystem].solve()
        │     └─► CSRMatrix.spmv_into()  ← O(nnz) per rhs call
        └─► PDFComputer.compute()
  └─► Pricer[1].price()
        ├─► _integrate_payoff_fast()
        │     └─► Payoff.evaluate()
        ├─► Interpolator.bicubic_interp()
        └─► Greeks.compute_delta/gamma/vega()
```

### Path 2: Batch Pricing (GPU)
```
PricingEngine.price[1000]()
  └─► GPUFullChainExecutor[B].execute_batch_pricing()
        ├─► generate_knots_gpu_kernel()
        ├─► grid_gpu_kernel()
        ├─► basis_gpu_kernel()
        ├─► boundary_gpu_kernel()
        ├─► spmatrix_gpu_kernel()
        ├─► delta_gpu_kernel()
        ├─► initial_gpu_kernel()
        ├─► lu_gpu_kernel()
        ├─► radau5_gpu_kernel()
        └─► integrate_gpu_kernel()
  └─► payoff_integration_kernel()  ← GPU parallel over N options
```

### Path 3: Calibration
```
Calibrator[B].calibrate()
  └─► for iter in 1..max_iters:
        ├─► FPESolver[B].solve()  ← GPU batch FPE
        ├─► Pricer[B].price()     ← GPU batch pricing
        ├─► ObjectiveFunction.compute()
        └─► LevenbergMarquardt.step()
              └─► QR factorization (MAX kernels.linalg.qr)
```

---

*Generated with detailed module call graph and data pipeline documentation*
