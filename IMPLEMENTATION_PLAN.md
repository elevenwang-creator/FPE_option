# FPE Option Pricing Engine — Mojo Full Rewrite Implementation Plan (v3)

> **Project**: Production-grade exotic option pricing engine using Fokker-Planck PDE + NAIS-Net neural solver
> **Language**: Mojo v0.26.2+ with MAX AI Kernels
> **Architecture**: Unified parametric code — write once, deploy to CPU (batch=1) and GPU (batch=N)
> **Target**: Sub-ms single pricing, GPU batch pricing + calibration, Python + C++ bindings
> **Timeline**: 18–22 weeks (revised from 24 — using MAX Kernels saves ~30%)
> **Design Principle**: One algorithm, one codebase. `comptime` batch dimension + `has_accelerator()` selects CPU/GPU at compile time.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Three Runtime Modes](#2-three-runtime-modes)
3. [Architecture](#3-architecture)
4. [Layer 0: MAX AI Kernels (Pre-built)](#4-layer-0-max-ai-kernels-pre-built)
5. [Layer 1: Sparse Math Library (Custom)](#5-layer-1-sparse-math-library-custom)
6. [Layer 2: Domain Numerics](#6-layer-2-domain-numerics)
7. [Layer 3: Engines](#7-layer-3-engines)
8. [Layer 4: Pricing Server](#8-layer-4-pricing-server)
9. [Layer 5: Bindings](#9-layer-5-bindings)
10. [Phased Delivery Schedule](#10-phased-delivery-schedule)
11. [Performance Budget](#11-performance-budget)
12. [Risk Analysis](#12-risk-analysis)
13. [Testing Strategy](#13-testing-strategy)
14. [Project Structure](#14-project-structure)

---

## 1. Project Overview

### What We're Building

A production-grade option pricing engine reconstructing three Python codebases into high-performance Mojo, powered by MAX AI Kernels:

| Component | Source | Purpose | Runtime |
|---|---|---|---|
| **FPE Solver** | `FPE_Solver_Final_Version.py` (1,153 lines) | Heston model FPE via B-spline Galerkin | CPU single / GPU batch |
| **NAIS-Net** | `NAIS_rBM.py` (448 lines) | Neural FBSDE solver for rough Bergomi | Train: GPU, Infer: GPU |
| **Pricing Server** | `BarrierOptionPricing.ipynb` + new | Sub-ms single + GPU batch pricing | CPU + GPU |
| **Bindings** | New | Python extension + C ABI shared library | Both |

### Key Constraints

- **Sub-millisecond single pricing**: CPU path returns in <1ms per option (cached PDF lookup)
- **GPU batch pricing**: Price 1000+ options in parallel on GPU (<10ms per batch)
- **GPU batch calibration**: Calibrate N parameter sets simultaneously on GPU
- **GPU portability**: NVIDIA, AMD, Apple Silicon via Mojo's GPU abstraction
- **Full Mojo-native**: No Python/scipy/TF dependencies in production path
- **Dual bindings**: Python (research/backtest) + C++ (live trading)
- **Extensible payoffs**: Barrier first, framework supports European, Asian, lookback, etc.

### What Mojo + MAX Enables (vs. Python baseline)

| Bottleneck in Python | Mojo + MAX Advantage | Expected Speedup |
|---|---|---|
| De Boor-Cox recursion (Python loops) | `comptime for` unrolled + SIMD vectorization | 50–100× |
| Sparse matrix assembly (nested loops) | SIMD parallel assembly + zero-overhead structs | 30–80× |
| ODE integration (scipy.integrate) | Custom Radau with `comptime` Butcher tableaux | 20–50× |
| OSQP optimization (CVXPY overhead) | Native OSQP in Mojo, no Python dispatch | 10–30× |
| Dense matmul (numpy) | **MAX Kernels matmul** (cuBLAS/MKL-grade) | 80–120× |
| Volterra FFT (scipy.fftpack) | **MAX Kernels rfft/irfft** (GPU-optimized) | 100–500× |
| NN forward pass (TF overhead) | **MAX Kernels nn** (fused ops, zero overhead) | 50–200× |
| Monte Carlo paths (numpy random) | **std.random.philox** (GPU-parallel Philox PRNG) | 100–1000× |
| Batch pricing (Python serial loop) | **GPU kernel**: parallel payoff integration | 500–2000× |

---

## 2. Unified Compute Model — Write Once, Deploy Everywhere

Mojo's core philosophy: **one algorithm, one codebase, compile to CPU or GPU.**

Instead of separate `cpu_solver` / `gpu_solver`, every compute kernel is written ONCE
with a `comptime batch_size` parameter. The compiler generates optimal code for each target:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED FPE COMPUTE PIPELINE                              │
│                                                                              │
│  SAME CODE for all three modes. Only batch_size and target differ:           │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │  FPESolver[batch_size, target]                                       │   │
│  │                                                                       │   │
│  │  comptime if batch_size == 1:                                         │   │
│  │      # CPU path: SIMD vectorized, single-stream                       │   │
│  │      # vectorize[] across evaluation points                           │   │
│  │  else:                                                                │   │
│  │      comptime if has_accelerator():                                   │   │
│  │          # GPU path: parallelize across batch dimension               │   │
│  │          # ctx.enqueue_function[solve_kernel, solve_kernel](...)      │   │
│  │      else:                                                            │   │
│  │          # CPU fallback: parallelize[] across batch dimension         │   │
│  │                                                                       │   │
│  │  The algorithm (B-spline → Galerkin → ODE → PDF → Payoff → Greeks)   │   │
│  │  is IDENTICAL regardless of batch_size or target.                     │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Mode 1: FPESolver[1]        →  CPU single pricing, <1ms (cached PDF)       │
│  Mode 2: FPESolver[N]        →  GPU batch pricing, N=1000+                  │
│  Mode 3: Calibrator[B]       →  GPU batch calibration, B param sets         │
│           └─ uses FPESolver[B] internally                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The Parametric Pattern

Every compute-heavy struct is parametric on `batch_size`:

```mojo
struct FPESolver[batch_size: Int]:
    """Unified FPE solver. batch_size=1 → CPU, batch_size>1 → GPU."""

    def solve(self, params: HestonParamsBatch[batch_size]) -> PDFGridBatch[batch_size]:
        # Step 1: Build basis (shared across batch — done once)
        var basis = self.build_basis(params)

        # Step 2: Assemble M, K (parametric on batch_size)
        var M = self.assemble_mass[batch_size](basis)
        var K = self.assemble_stiffness[batch_size](basis, params)

        # Step 3: Initial condition
        var q0 = self.initial_condition[batch_size](basis, params)

        # Step 4: ODE integration — SAME algorithm, different parallelism
        comptime if batch_size == 1:
            var sol = self.ode.solve(FPESystem(M, K), q0)       # CPU RadauIIA
        else:
            comptime if has_accelerator():
                var sol = self.ode_gpu[batch_size](M, K, q0)    # GPU parallel
            else:
                var sol = self.ode_parallel[batch_size](M, K, q0) # CPU multi-thread

        # Step 5: PDF grid (same formula, batch-aware)
        return self.compute_pdf[batch_size](basis, sol)


struct Pricer[batch_size: Int]:
    """Unified pricer. batch_size=1 → CPU sub-ms, batch_size>1 → GPU batch."""

    def price(self, pdf: PDFGridBatch[batch_size],
              requests: RequestBatch[batch_size],
              payoffs: PayoffRegistry) -> ResultBatch[batch_size]:

        comptime if batch_size == 1:
            # CPU SIMD: interpolate + integrate + Greeks
            return self.price_single_simd(pdf, requests, payoffs)
        else:
            comptime if has_accelerator():
                # GPU: one thread per option
                return self.price_batch_gpu(pdf, requests, payoffs)
            else:
                # CPU fallback: parallelize across options
                return self.price_batch_cpu(pdf, requests, payoffs)
```

### Why This Is Better

| Aspect | v2 (separate files) | v3 (unified parametric) |
|---|---|---|
| **Code duplication** | Algorithm written twice | Algorithm written ONCE |
| **Bug surface** | Fix in CPU, forget GPU | Fix once, both targets updated |
| **Testing** | Test CPU and GPU separately | Test algorithm once, verify targets |
| **Mojo philosophy** | Python/CUDA mindset | True heterogeneous compute |
| **Maintainability** | 2× files to maintain | 1× files, `comptime` handles dispatch |

### Mode 1: CPU Single Pricing (HFT Path)

For sub-millisecond latency in live trading:

```
Input: (S, K, T, barrier_type, barrier_level, param_hash)
  │
  ▼
[1] Cache lookup: Dict[param_hash → PDFGrid]         ← O(1), <1μs
  │
  ▼
[2] Bicubic interpolation on S×V grid                 ← SIMD vectorized, <50μs
  │
  ▼
[3] Payoff integration: ∫∫ payoff(S) · PDF dS dV      ← Pre-computed quad weights, <100μs
  │
  ▼
[4] Greeks: ∂price/∂S, ∂²price/∂S², ∂price/∂σ        ← Finite diff on cached surface, <200μs
  │
  ▼
Output: (price, Δ, Γ, Vega, Θ)                        ← Total: <400μs
```

### Mode 2: GPU Batch Pricing (Risk / EOD)

For pricing thousands of options at once (different strikes, expiries, barriers):

```
Input: Batch of N pricing requests + shared HestonParams
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ GPU BATCH PRICING PIPELINE                                      │
│                                                                 │
│ [1] Build B-spline basis (once per param set)                   │
│     ├─ GenerateKnots → BSplineBasis → TensorProductBasis       │
│     └─ Assemble Mass(M) + Stiffness(K) sparse matrices         │
│                                                                 │
│ [2] Solve ODE: dq/dt = -M⁻¹Kq                                 │
│     ├─ Initial condition q₀ via QP                              │
│     └─ RadauIIA integration → q(t) at target expiries           │
│                                                                 │
│ [3] Compute PDF grid: pdf = Φ @ q(t)                            │
│                                                                 │
│ [4] GPU Kernel: Parallel payoff integration (N options)         │
│     ├─ Each thread: one (K, T, barrier) combination             │
│     ├─ Interpolate PDF at target (S, V, T)                      │
│     ├─ Evaluate payoff function                                 │
│     └─ Numerical integration via quadrature                     │
│                                                                 │
│ [5] GPU Kernel: Parallel Greeks (N × 4 finite diffs)            │
│     └─ Reuse PDF grid, shift (S±h, V±h)                        │
│                                                                 │
│ Output: N × (price, Δ, Γ, Vega, Θ)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key optimization**: Steps [1]–[3] are done ONCE per parameter set. Step [4]–[5] are embarrassingly parallel across N options → GPU kernel with one thread per option.

### Mode 3: GPU Batch Calibration

For fitting Heston parameters to market data:

```
Input: Market option prices + strikes + expiries + initial param guesses (B batches)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ GPU BATCH CALIBRATION PIPELINE                                  │
│                                                                 │
│ for iter in 1..max_iters:                                       │
│                                                                 │
│   [1] GPU Kernel: B parallel FPE solves                         │
│       ├─ Each thread-block: 1 parameter set                     │
│       ├─ Shared memory: B-spline basis for that block           │
│       └─ Output: B × PDF grids                                  │
│                                                                 │
│   [2] GPU Kernel: B × M parallel option prices                  │
│       ├─ For each param set, price all M market options          │
│       └─ Output: B × M model prices                             │
│                                                                 │
│   [3] GPU Kernel: Objective function                            │
│       ├─ loss_b = Σ_m (model_price - market_price)²             │
│       └─ Output: B loss values + Jacobian                       │
│                                                                 │
│   [4] GPU Kernel: Levenberg-Marquardt step                      │
│       ├─ J^T J + λI) δ = -J^T r                                │
│       ├─ Update: params += δ                                    │
│       └─ MAX Kernels matmul for J^T J                           │
│                                                                 │
│ Output: B × calibrated (κ, θ, σ, ρ, V₀)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture

### Revised Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: BINDINGS                                                  │
│  ┌──────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ Python Extension      │  │ C ABI Shared Library                │  │
│  │ (PyInit_fpe_engine)   │  │ (libfpe_engine.so/.dylib)           │  │
│  └──────────────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: PRICING SERVER                                            │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ PricingEngine                                                   ││
│  │ ├── Pricer[B] (unified: B=1→CPU SIMD, B=N→GPU parallel)       ││
│  │ ├── PDFCache (pre-computed grids, shared-memory / disk)        ││
│  │ ├── PayoffRegistry (extensible: barrier, European, Asian...)   ││
│  │ └── VolSurfaceGenerator (NAIS-Net implied vol output)          ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: ENGINES (all parametric on batch_size)                    │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐│
│  │ FPE Engine       │ │ NAIS Engine      │ │ Calibrator           ││
│  │ FPESolver[B]     │ │ ├─ NaisNet       │ │ uses FPESolver[B]   ││
│  │ ├─ GalerkinAs[B] │ │ ├─ FBSDESolver   │ │ + LevenbergMarquardt││
│  │ ├─ InitialCond   │ │ ├─ Volterra[B]   │ │ + ObjectiveFunc      ││
│  │ ├─ ODESolver[B]  │ │ ├─ Trainer[B]    │ │                      ││
│  │ └─ PDFComputer   │ │ └─ Inferencer[B] │ │                      ││
│  └──────────────────┘ └──────────────────┘ └──────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: DOMAIN NUMERICS                                           │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ B-Spline    │ │ ODE Solver   │ │ Optimizer│ │ NN Runtime     │  │
│  │ ├─ Knots    │ │ ├─ RK45      │ │ ├─ OSQP  │ │ ├─ StableLinear│  │
│  │ ├─ Basis    │ │ ├─ RadauIIA  │ │ └─ LM    │ │ ├─ AutoDiff    │  │
│  │ ├─ Recomb   │ │ └─ Adaptive  │ │          │ │ └─ Adam        │  │
│  │ └─ Tensor   │ │              │ │          │ │ (uses MAX for  │  │
│  │             │ │              │ │          │ │  matmul/activ)  │  │
│  └─────────────┘ └──────────────┘ └──────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: SPARSE MATH (Custom — only what MAX lacks)                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ CSRMatrix · COOMatrix · DiagMatrix · kron · spmv · spgemm   │   │
│  │ GPU SpMV kernel · GPU sparse assembly kernel                │   │
│  └──────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 0: MAX AI KERNELS + MOJO STDLIB (Pre-built, optimized)       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌─────────────┐  │
│  │ linalg   │ │ nn       │ │ layout   │ │random│ │ std         │  │
│  │ ·matmul  │ │ ·softmax │ │ ·Layout  │ │·Philo│ │ ·algorithm  │  │
│  │ ·gemv    │ │ ·activ   │ │ ·Layout  │ │ x    │ │  vectorize  │  │
│  │ ·bmm     │ │ ·norm    │ │  Tensor  │ │·randn│ │  parallelize│  │
│  │ ·qr      │ │ ·rfft    │ │ ·Tile    │ │      │ │ ·math       │  │
│  │ ·transp  │ │ ·irfft   │ │  Tensor  │ │      │ │ ·complex    │  │
│  │ ·vendor  │ │ ·conv    │ │          │ │      │ │ ·gpu        │  │
│  │  _blas   │ │          │ │          │ │      │ │ ·ffi        │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 0: MAX AI Kernels (Pre-built)

**No code to write.** These are production-optimized by Modular's kernel team.

### Linear Algebra (`kernels.linalg`)

| Component | Import | Use In Our System |
|---|---|---|
| `matmul` | `from kernels.linalg.matmul import ...` | NN forward/backward, Jacobian computation |
| `gemv` | `from kernels.linalg.gemv import ...` | Sparse-dense mixed ops, PDF computation |
| `bmm` | `from kernels.linalg.bmm import ...` | Batch pricing: N options × matrix ops |
| `grouped_matmul` | `from kernels.linalg.grouped_matmul import ...` | Batch calibration: B param sets × Jacobian |
| `qr_factorization` | `from kernels.linalg.qr_factorization import ...` | LM solver: QR for least-squares step |
| `transpose` | `from kernels.linalg.transpose import ...` | Matrix operations throughout |
| `vendor_blas` | `from kernels.linalg.vendor_blas import ...` | Fallback to cuBLAS/MKL when optimal |

### Neural Network (`kernels.nn`)

| Component | Import | Use In Our System |
|---|---|---|
| `rfft` / `irfft` | `from kernels.nn import ...` | Volterra process FFT convolution |
| `activations` | `from kernels.nn import ...` | NAIS-Net (sin, relu, etc.) |
| `softmax` | `from kernels.nn import ...` | If needed for attention-based pricing |
| `normalization` | `from kernels.nn import ...` | Layer normalization in NAIS-Net |

### Layout (`layout`)

| Component | Import | Use In Our System |
|---|---|---|
| `Layout` | `from layout import Layout` | All GPU tensor layout definitions |
| `LayoutTensor` | `from layout import LayoutTensor` | GPU kernel data passing |
| `TileTensor` | `from layout import TileTensor` | Tiled GPU memory access patterns |

### Standard Library

| Component | Import | Use In Our System |
|---|---|---|
| `Philox PRNG` | `from std.random import ...` | Monte Carlo path generation (GPU-parallel) |
| `vectorize` | `from std.algorithm import vectorize` | SIMD loops in B-spline eval, interpolation |
| `parallelize` | `from std.algorithm import parallelize` | Multi-core CPU pricing, assembly |
| `math` | `from std.math import ...` | exp, log, sqrt, erf, sin, cos |
| `complex` | `from std.complex import ...` | FFT intermediate values |
| `gpu` | `from std.gpu import ...` | GPU kernel primitives |
| `ffi` | `from std.ffi import ...` | C/C++ library calls if needed |

---

## 5. Layer 1: Sparse Math Library (Custom)

**The only Layer 1 code we write.** MAX Kernels focus on dense operations — sparse is our gap.

### 5.1 Sparse Matrix Formats

```mojo
@align(64)
struct CSRMatrix[dtype: DType](Copyable, Movable, Writable):
    """Compressed Sparse Row — primary format for assembled FPE matrices."""
    var data: List[Scalar[Self.dtype]]       # non-zero values
    var indices: List[Int]                    # column indices
    var indptr: List[Int]                     # row pointers
    var nrows: Int
    var ncols: Int
    var nnz: Int

    def spmv(self, x: Span[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]:
        """Sparse matrix-vector multiply. SIMD-vectorized row dot products."""
        ...

    def to_gpu(self, ctx: DeviceContext) -> GPUCSRMatrix[Self.dtype]:
        """Transfer to GPU DeviceBuffers for kernel use."""
        ...

struct COOMatrix[dtype: DType](Movable):
    """Coordinate format — used during assembly, then converted to CSR."""
    var row: List[Int]
    var col: List[Int]
    var data: List[Scalar[Self.dtype]]

    def to_csr(self) -> CSRMatrix[Self.dtype]:
        """Sort by row, compress to CSR. Uses std.algorithm for parallel sort."""
        ...

struct DiagMatrix[dtype: DType](Copyable, Movable):
    """Diagonal matrix — for quadrature weights."""
    var diag: List[Scalar[Self.dtype]]
```

### 5.2 Sparse Operations

| Operation | Description | Mojo Optimization |
|---|---|---|
| `spmv(A, x)` | Sparse matrix × dense vector | SIMD row dot products via `vectorize` |
| `kron(A, B)` | Kronecker product CSR × CSR → CSR | Key for tensor product basis |
| `spgemm(A, B)` | Sparse matrix × sparse matrix | For `basis.T @ W @ basis` |
| `spmm(A, D)` | Sparse × dense matrix | For batch pricing on GPU |

### 5.3 GPU Sparse Kernels

```mojo
def spmv_kernel(
    data: LayoutTensor[dtype, data_layout, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, idx_layout, MutAnyOrigin],
    indptr: LayoutTensor[DType.int32, ptr_layout, MutAnyOrigin],
    x: LayoutTensor[dtype, vec_layout, MutAnyOrigin],
    y: LayoutTensor[dtype, vec_layout, MutAnyOrigin],
    nrows: Int,
):
    """GPU SpMV: one thread per row, warp-level reduction for long rows."""
    var row = global_idx.x
    if row < nrows:
        var start = indptr[row]
        var end = indptr[row + 1]
        var sum: Scalar[dtype] = 0
        for j in range(Int(start), Int(end)):
            sum += rebind[Scalar[dtype]](data[j]) * rebind[Scalar[dtype]](x[indices[j]])
        y[row] = rebind[y.element_type](sum)

def batch_spmv_kernel(
    # B sparse matrices (shared structure) × B vectors → B result vectors
    data: LayoutTensor[dtype, ..., MutAnyOrigin],     # (B, nnz)
    indices: LayoutTensor[DType.int32, ..., MutAnyOrigin],  # shared
    indptr: LayoutTensor[DType.int32, ..., MutAnyOrigin],   # shared
    X: LayoutTensor[dtype, ..., MutAnyOrigin],        # (B, ncols)
    Y: LayoutTensor[dtype, ..., MutAnyOrigin],        # (B, nrows)
    nrows: Int,
):
    """Batch SpMV for GPU batch pricing/calibration.
    grid_dim = (nrows, B), each thread-block handles one (row, batch) pair."""
    var row = global_idx.x
    var batch = global_idx.y
    if row < nrows:
        # ... accumulate row dot product for this batch
        ...
```

---

## 6. Layer 2: Domain Numerics

### 6.1 B-Spline Module (`numerics/bspline/`)

Unchanged from v1 plan. Direct reconstruction of `BSplineBasis`, `RecombinationBasis`, `MultivariateBSpline`.

| Component | Python Original | Mojo Optimization |
|---|---|---|
| `GenerateKnots` | `GenerateKnots` class | SIMD `vectorize`, `comptime` Chebyshev |
| `BSplineBasis[degree]` | De Boor-Cox with `@lru_cache` | `comptime for` unrolled recursion, SIMD batch |
| `RecombinationBasis` | Sparse recombination matrix | Pre-computed `comptime` recombination |
| `TensorProductBasis` | `sp.kron` of 1D bases | Fused Kronecker via custom `kron()` |

### 6.2 ODE Integrator Module (`numerics/ode/`)

| Solver | Use Case | Notes |
|---|---|---|
| `RungeKutta45` | Non-stiff, adaptive | Dormand-Prince at `comptime` |
| `RadauIIA` | Stiff FPE system | LU solve per step (small system, use in-place) |

**Key**: ODE RHS uses sparse `spmv` (Layer 1) for `M⁻¹Kq`.

```mojo
trait ODESystem:
    def rhs(self, t: Float64, y: Span[Float64], mut dydt: Span[Float64]): ...

struct FPESystem(ODESystem):
    var M_inv_K: CSRMatrix[DType.float64]   # pre-computed -M⁻¹K

    def rhs(self, t: Float64, y: Span[Float64], mut dydt: Span[Float64]):
        # dydt = -M⁻¹K @ y  →  sparse matrix-vector multiply
        self.M_inv_K.spmv_into(y, dydt)
```

### 6.3 Convex Optimizer (`numerics/optim/`)

| Component | Description |
|---|---|
| `OSQP` | ADMM-based QP for initial condition (non-negative constraint) |
| `LevenbergMarquardt` | For Heston calibration; uses **MAX Kernels `qr_factorization`** |
| `ProjectedGradient` | Simpler alternative for non-negative QP |

### 6.4 Neural Network Runtime (`numerics/nn/`)

Builds ON TOP of MAX Kernels — we write only domain-specific layers:

| Component | Implementation | MAX Kernels Used |
|---|---|---|
| `StableLinear` | Custom (weight constraint R^TR norm) | `kernels.linalg.matmul` for W×x |
| `NaisLinear` | Thin wrapper | `kernels.linalg.matmul` + bias add |
| `AutoDiff` | Custom reverse-mode tape | `kernels.linalg.matmul` for Jacobian |
| `Adam` | Custom optimizer | `kernels.linalg.gemv` for param update |
| Activations | Direct call | `kernels.nn.activations` (sin, relu) |
| FFT conv | Direct call | `kernels.nn.rfft` + `kernels.nn.irfft` |

---

## 7. Layer 3: Engines

### 7.1 FPE Engine (`engines/fpe/`) — Unified Parametric Design

**One solver, parameterized by `batch_size`.** `comptime` dispatch handles CPU vs GPU.

| Component | Responsibility |
|---|---|
| `HestonParams` | Validated parameter struct (`κ, θ, σ, ρ, r, T, S₀, V₀`) with Feller check |
| `HestonParamsBatch[B]` | `B` parameter sets — `B=1` for single, `B=N` for batch |
| `FPEDomain` | Grid generation: knots (S × V), B-spline basis, recombination |
| `GalerkinAssembler[B]` | Mass `M` + stiffness `K` assembly — batch-aware |
| `InitialCondition[B]` | Delta approx + OSQP — batch-aware |
| `FPESolver[B]` | **Unified ODE solver**: `comptime` selects CPU SIMD or GPU parallel |
| `PDFComputer[B]` | `pdf = Φ @ q(t)` — batch-aware reshape |

**Unified architecture — same code, different `B`:**

```mojo
struct FPESolver[batch_size: Int]:
    var domain: FPEDomain
    var ode: RadauIIA

    def solve(self, params: HestonParamsBatch[batch_size]
             ) -> PDFGridBatch[batch_size]:
        # All steps are batch-aware. Same algorithm for B=1 or B=1000.
        var basis = self.domain.build_basis()                       # shared
        var M = GalerkinAssembler[batch_size].mass(basis)           # (B, n, n) sparse
        var K = GalerkinAssembler[batch_size].stiffness(basis, params)
        var q0 = InitialCondition[batch_size].compute(basis, params)

        # comptime dispatch: CPU single-stream or GPU parallel
        comptime if batch_size == 1:
            var sol = self.ode.solve(FPESystem(M, K), q0)
        else:
            comptime if has_accelerator():
                # GPU: one thread-block per batch element
                var sol = self.solve_gpu(M, K, q0)
            else:
                # CPU fallback: parallelize across batch
                var sol = self.solve_parallel(M, K, q0)

        return PDFComputer[batch_size].compute(basis, sol)
```

**All three modes use the SAME code:**

```
Mode 1: FPESolver[1].solve(single_params)         → 1 PDF grid (CPU, sub-ms from cache)
Mode 2: FPESolver[1].solve(params) + Pricer[N]    → N option prices (GPU parallel payoff)
Mode 3: Calibrator uses FPESolver[B].solve(B_params) → B PDF grids per LM iteration
```

**Key insight**: Mode 2 (batch pricing) separates into:
- FPE solve once (batch=1 per param set) → PDF grid
- Pricing is embarrassingly parallel across N options → `Pricer[N]` on GPU

### 7.2 NAIS Engine (`engines/nais/`)

| Component | Responsibility | MAX Kernels Used |
|---|---|---|
| `NaisNet` | Network: Linear + StableLinear + skip + sin | `matmul`, `activations` |
| `VolterraProcess` | Fractional BM via hybrid scheme | `rfft`, `irfft` |
| `VarianceProcess` | Rough Bergomi: `ε(t)·exp(η·X̃ - 0.5η²t^{2H})` | `std.math.exp` |
| `FBSDELoss` | Forward-backward SDE loss | `matmul`, `gemv` |
| `Trainer` | GPU training: Adam + gradient tape | `matmul`, `Adam` |
| `Inferencer` | GPU inference: (t,S,V) → (price, φ, Du) | `matmul`, `activations` |

### 7.3 GPU Calibrator (`engines/calibrator/`)

| Component | Responsibility | MAX Kernels Used |
|---|---|---|
| `CalibrationTarget` | Market prices + strikes + expiries | — |
| `ObjectiveFunction` | Σ(model_price - market_price)² | `gemv` |
| `LevenbergMarquardt` | (J^TJ + λI)δ = -J^Tr | `matmul`, `qr_factorization` |
| `BatchCalibrator` | Orchestrate B parallel calibrations | `grouped_matmul`, `bmm` |

---

## 8. Layer 4: Pricing Server

```mojo
struct PricingEngine:
    var pdf_cache: PDFCache
    var payoff_registry: PayoffRegistry
    var nais_inferencer: Optional[Inferencer]
    var gpu_ctx: Optional[DeviceContext]

    def price[batch_size: Int](self, reqs: RequestBatch[batch_size]
                              ) -> ResultBatch[batch_size]:
        """Unified pricing entry point.
        batch_size=1 → CPU sub-ms (cached PDF lookup).
        batch_size=N → GPU parallel (N options simultaneously).
        comptime dispatch — no runtime branching."""

        # Step 1: Get PDF (from cache or solve on-demand)
        var pdf = self.get_or_solve_pdf(reqs.params)

        # Step 2: Price — Pricer[batch_size] handles CPU/GPU dispatch
        return Pricer[batch_size].price(pdf, reqs, self.payoff_registry)

    def calibrate[batch_size: Int](self, market: MarketData,
                                   init_params: HestonParamsBatch[batch_size]
                                  ) -> HestonParamsBatch[batch_size]:
        """Unified calibration. batch_size=B param sets in parallel."""
        return Calibrator[batch_size].run(market, init_params)


# Usage — the SAME function, different compile-time batch:
var engine = PricingEngine(...)
var single = engine.price[1](one_request)            # CPU, <1ms
var batch  = engine.price[1000](thousand_requests)    # GPU, <10ms
var params = engine.calibrate[64](market, guesses)    # GPU, 64 param sets
```

### Payoff Registry (unchanged)

```mojo
trait Payoff:
    def evaluate(self, S: Span[Float64], params: PayoffParams) -> List[Float64]: ...
    def name(self) -> StaticString: ...

struct BarrierUpAndOut(Payoff): ...
struct BarrierDownAndIn(Payoff): ...
struct EuropeanCall(Payoff): ...
struct EuropeanPut(Payoff): ...
# Extensible: implement Payoff trait for new exotic types
```

---

## 9. Layer 5: Bindings

### 9.1 Python Extension Module

```mojo
@export
fn PyInit_fpe_engine() -> PythonObject:
    var m = PythonModuleBuilder("fpe_engine")

    # Mode 1: CPU single pricing
    m.def_function[py_price_single]("price_single")

    # Mode 2: GPU batch pricing
    m.def_function[py_price_batch]("price_batch")

    # Mode 3: GPU calibration
    m.def_function[py_calibrate_batch]("calibrate_batch")

    # FPE solver
    m.def_function[py_solve_fpe]("solve_fpe")
    m.def_function[py_solve_fpe_batch]("solve_fpe_batch")

    # NAIS functions
    m.def_function[py_nais_train]("nais_train")
    m.def_function[py_nais_infer]("nais_infer")
    m.def_function[py_nais_vol_surface]("nais_vol_surface")

    return m.finalize()
```

### 9.2 C ABI Shared Library

```cpp
extern "C" {
    // Lifecycle
    int32_t fpe_init(const char* config_path);     // Init engine + GPU
    void    fpe_destroy();

    // Mode 1: CPU single pricing (<1ms)
    int32_t fpe_price_single(
        double S, double K, double T,
        int32_t payoff_type, double barrier,
        uint64_t param_hash,
        double* price, double* delta, double* gamma, double* vega
    );

    // Mode 2: GPU batch pricing
    int32_t fpe_price_batch(
        const double* S, const double* K, const double* T,
        const int32_t* payoff_types, const double* barriers,
        int32_t count, uint64_t param_hash,
        double* prices, double* deltas, double* gammas, double* vegas
    );

    // Mode 3: GPU calibration
    int32_t fpe_calibrate(
        const double* market_prices, const double* strikes,
        const double* expiries, int32_t n_options,
        const double* init_params, int32_t n_param_sets,
        double* out_params
    );

    // Cache management
    int32_t fpe_precompute_pdf(const double* params, const char* cache_path);
    int32_t fpe_load_cache(const char* cache_path);
}
```

---

## 10. Phased Delivery Schedule

### Phase 1: Foundation (Weeks 1–3)

| Week | Deliverable | Validation |
|---|---|---|
| 1 | Project setup: pixi, mojoproject.toml, directory structure | `mojo build` succeeds |
| 1 | MAX Kernels integration test: matmul, gemv, rfft | Verify MAX imports work |
| 1–2 | `sparse/`: CSRMatrix, COOMatrix, DiagMatrix | Unit tests match scipy.sparse |
| 2–3 | `sparse/`: kron, spmv, spgemm + GPU SpMV kernel | Benchmark vs scipy.sparse |

### Phase 2: B-Spline + ODE (Weeks 4–7)

| Week | Deliverable | Validation |
|---|---|---|
| 4–5 | `numerics/bspline/`: full module (knots, basis, recomb, tensor) | Output matches Python to 1e-10 |
| 5–6 | `numerics/ode/`: RK45, RadauIIA | Solve test ODEs, match scipy |
| 6–7 | `numerics/optim/`: OSQP + LevenbergMarquardt (using MAX qr) | Match CVXPY output |

### Phase 3: FPE Engine + All 3 Modes (Weeks 8–12)

| Week | Deliverable | Validation |
|---|---|---|
| 8–9 | FPE Engine: Galerkin assembler (M, K) + initial condition | Matrix entries match Python 1e-8 |
| 9–10 | **Mode 1**: CPU single FPE solve + PDFCache + SinglePricer | Sub-ms on cached PDF |
| 10–11 | **Mode 2**: GPU batch pricing kernel (N parallel options) | 1000 options <10ms |
| 11–12 | **Mode 3**: GPU batch calibration pipeline | Converge on test market data |

### Phase 4: NAIS Engine (Weeks 13–17)

| Week | Deliverable | Validation |
|---|---|---|
| 13–14 | `numerics/nn/`: StableLinear + AutoDiff (using MAX matmul) | Forward/backward match TF |
| 14–15 | NAIS-Net: architecture + Volterra (using MAX rfft/irfft) | Forward pass matches TF |
| 15–16 | FBSDE loss + GPU training loop | Loss converges |
| 16–17 | Inference + Greeks + vol surface generation | Match Python predict() |

### Phase 5: Bindings + Production (Weeks 18–22)

| Week | Deliverable | Validation |
|---|---|---|
| 18–19 | Python extension module (all 3 modes) | Python end-to-end test |
| 19–20 | C ABI shared library + C++ header + example | C++ program prices options |
| 20–21 | Performance optimization + benchmarking | Meet all latency targets |
| 21–22 | Integration tests, documentation, packaging | Production-ready release |

---

## 11. Performance Budget

### Mode 1: CPU Single Pricing (<1ms)

| Step | Budget | Approach |
|---|---|---|
| Cache lookup | <1μs | Dict hash O(1) |
| PDF interpolation | <50μs | SIMD bicubic via `vectorize` |
| Payoff integration | <100μs | Pre-computed quad weights, SIMD dot |
| Greeks (finite diff) | <200μs | 4 interpolations (S±h, V±h) |
| **Total** | **<400μs** | **Well within 1ms** |

### Mode 2: GPU Batch Pricing

| Metric | Target |
|---|---|
| Batch size | 1,000–10,000 options |
| FPE solve (per param set) | <100ms (GPU SpMV + ODE) |
| Payoff integration (N options) | <1ms (1 thread per option) |
| Greeks (N × 4 finite diffs) | <5ms |
| **Total (shared params)** | **<110ms for 10,000 options** |
| **Amortized per option** | **~11μs** |

### Mode 3: GPU Batch Calibration

| Metric | Target |
|---|---|
| Batch size | 64–256 parameter sets |
| Per-iteration (B FPE + pricing + LM) | <500ms |
| Convergence (50 iterations) | <25s |
| MAX Kernels speedup (matmul/qr) | 5–10× vs custom implementation |

### Offline FPE Solve (single, CPU)

| Step | Python | Mojo+MAX Target | Speedup |
|---|---|---|---|
| B-spline basis | ~5s | ~50ms | 100× |
| M/K assembly | ~3s | ~30ms | 100× |
| Initial cond (OSQP) | ~10s | ~200ms | 50× |
| ODE (Radau) | ~30s | ~500ms | 60× |
| **Total** | **~48s** | **~800ms** | **60×** |

---

## 12. Risk Analysis

| Risk | Severity | Mitigation |
|---|---|---|
| MAX Kernels API stability | Medium | Pin to v26.2; wrap behind thin adapters |
| Sparse matrix correctness | Medium | Bit-exact vs scipy at every operation |
| ODE solver numerical stability | High | Same Butcher tableaux as scipy; step-by-step validation |
| GPU batch SpMV performance | Medium | Start with simple row-per-thread; optimize to warp-reduction if needed |
| Autograd complexity | High | Start with manual backprop for NAIS-Net; general autograd Phase 4 |
| Cross-platform GPU | Medium | Test NVIDIA + Apple Silicon; Mojo abstracts backends |
| MAX Kernels rfft precision | Low | Validate FFT output vs scipy.fftpack reference |
| Batch pricing GPU memory | Medium | Stream batches if >10K options; monitor DeviceBuffer allocation |

---

## 13. Testing Strategy

### Reference Data Generation (Python → .npz)

```python
# tests/reference/generate_reference.py
# Run BEFORE any Mojo implementation

# B-spline references
np.savez("ref_bspline.npz", basis=basis.toarray(), deriv=deriv.toarray(), knots=knots)

# FPE references
np.savez("ref_mass_matrix.npz", M=M.toarray())
np.savez("ref_stiffness_matrix.npz", K=K.toarray())
np.savez("ref_initial_cond.npz", q0=q0)
np.savez("ref_ode_solution.npz", t=sol.t, y=sol.y)
np.savez("ref_pdf_grid.npz", pdf=pdf_2d)

# NAIS-Net references
np.savez("ref_nais_forward.npz", input=x, output_u=u, output_phi=phi)
np.savez("ref_volterra.npz", t=t, W=W, tilde_X=tilde_X)
np.savez("ref_nais_weights.npz", **{k: v.numpy() for k, v in model.weights})

# Pricing references
np.savez("ref_barrier_price.npz", price=price, delta=delta, gamma=gamma)
```

### Unit Tests (per module)

```mojo
from std.testing import assert_equal, assert_true, TestSuite

def test_csr_spmv() raises:
    var A = CSRMatrix[DType.float64].from_dense(...)
    var x = [1.0, 2.0, 3.0]
    var y = A.spmv(Span(x))
    assert_float_close(y[0], 14.0, atol=1e-12)

def test_max_matmul_integration() raises:
    # Verify MAX Kernels matmul produces correct results
    ...

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

### Benchmark Tests

```mojo
from std.benchmark import Bench, BenchConfig, BenchId, Bencher

# SAME Pricer, different batch_size — Mojo compiles both variants
@parameter
@always_inline
def bench_price_1(mut b: Bencher) capturing raises:
    b.iter[engine.price[1]](single_request)

@parameter
@always_inline
def bench_price_1000(mut b: Bencher) capturing raises:
    @parameter
    @always_inline
    def launch(ctx: DeviceContext) raises:
        engine.price[1000](batch_requests)
    b.iter_custom[launch](ctx)

var bench = Bench(BenchConfig(max_iters=10000))
bench.bench_function[bench_price_1](BenchId("price[1]_cpu"))
bench.bench_function[bench_price_1000](BenchId("price[1000]_gpu"))
# Same code path, different comptime instantiation
```

---

## 14. Project Structure

```
fpe_option/
├── pixi.toml
├── mojoproject.toml
├── IMPLEMENTATION_PLAN.md
├── Mojo_Language_Advantages.md
│
├── src/
│   ├── sparse/                            # Layer 1: Custom sparse math
│   │   ├── csr.mojo                      # CSRMatrix struct + spmv
│   │   ├── coo.mojo                      # COOMatrix struct + to_csr
│   │   ├── diag.mojo                     # DiagMatrix struct
│   │   ├── ops.mojo                      # kron, spgemm, spmm
│   │   ├── gpu_kernels.mojo              # spmv_kernel, batch_spmv_kernel
│   │   └── __init__.mojo
│   │
│   ├── numerics/                          # Layer 2: Domain numerics
│   │   ├── bspline/
│   │   │   ├── knots.mojo                # GenerateKnots
│   │   │   ├── basis.mojo                # BSplineBasis (SIMD + comptime)
│   │   │   ├── recombination.mojo        # RecombinationBasis
│   │   │   ├── tensor_product.mojo       # MultivariateBSpline (uses sparse kron)
│   │   │   └── __init__.mojo
│   │   ├── ode/
│   │   │   ├── types.mojo                # ODESystem trait, ODESolution
│   │   │   ├── rk45.mojo                 # Dormand-Prince (comptime Butcher)
│   │   │   ├── radau.mojo                # RadauIIA stiff solver
│   │   │   └── __init__.mojo
│   │   ├── optim/
│   │   │   ├── osqp.mojo                 # ADMM-based QP solver
│   │   │   ├── lm.mojo                   # Levenberg-Marquardt (uses MAX qr)
│   │   │   └── __init__.mojo
│   │   └── nn/
│   │       ├── stable_linear.mojo        # NAIS StableLinear (uses MAX matmul)
│   │       ├── autograd.mojo             # Reverse-mode autodiff tape
│   │       ├── adam.mojo                  # Adam optimizer
│   │       └── __init__.mojo
│   │
│   ├── engines/                           # Layer 3: Engines (all parametric on batch_size)
│   │   ├── fpe/
│   │   │   ├── heston_params.mojo        # HestonParams + HestonParamsBatch[B]
│   │   │   ├── domain.mojo               # FPEDomain (grid, basis — shared)
│   │   │   ├── galerkin.mojo             # GalerkinAssembler[B] (batch-aware M, K)
│   │   │   ├── initial_cond.mojo         # InitialCondition[B] (batch-aware)
│   │   │   ├── solver.mojo               # FPESolver[B] — UNIFIED: comptime CPU/GPU
│   │   │   ├── pdf.mojo                  # PDFComputer[B] — batch-aware
│   │   │   └── __init__.mojo
│   │   ├── nais/
│   │   │   ├── nais_net.mojo             # NaisNet (uses MAX matmul/activ)
│   │   │   ├── volterra.mojo             # VolterraProcess[B] (uses MAX rfft/irfft)
│   │   │   ├── variance.mojo             # VarianceProcess[B]
│   │   │   ├── fbsde.mojo               # FBSDELoss[B]
│   │   │   ├── trainer.mojo              # Trainer[B] — unified train loop
│   │   │   ├── inferencer.mojo           # Inferencer[B] — unified inference
│   │   │   └── __init__.mojo
│   │   └── calibrator/
│   │       ├── objective.mojo            # ObjectiveFunction[B]
│   │       ├── calibrator.mojo           # Calibrator[B] — uses FPESolver[B]
│   │       └── __init__.mojo
│   │
│   ├── server/                            # Layer 4: Pricing server
│   │   ├── pricing_engine.mojo           # PricingEngine — unified price[B]()
│   │   ├── pricer.mojo                   # Pricer[B] — UNIFIED: comptime CPU/GPU
│   │   ├── pdf_cache.mojo                # PDF grid cache
│   │   ├── interpolator.mojo             # SIMD bicubic interpolation
│   │   ├── payoffs.mojo                  # Payoff trait + implementations
│   │   ├── greeks.mojo                   # Greeks[B] — batch-aware
│   │   ├── vol_surface.mojo              # Implied vol surface
│   │   └── __init__.mojo
│   │
│   └── bindings/                          # Layer 5: Bindings
│       ├── python_module.mojo             # PyInit_fpe_engine
│       ├── c_abi.mojo                     # @export C functions
│       └── __init__.mojo
│
├── tests/
│   ├── test_sparse.mojo
│   ├── test_bspline.mojo
│   ├── test_ode.mojo
│   ├── test_optim.mojo
│   ├── test_nn.mojo
│   ├── test_fpe_engine.mojo
│   ├── test_nais_engine.mojo
│   ├── test_pricing_server.mojo
│   ├── test_gpu_batch.mojo
│   └── reference/
│       ├── generate_reference.py
│       └── data/                          # .npz reference files
│
├── benchmarks/
│   ├── bench_sparse_ops.mojo
│   ├── bench_bspline.mojo
│   ├── bench_fpe_solve.mojo
│   ├── bench_single_pricing.mojo
│   ├── bench_gpu_batch_pricing.mojo
│   └── bench_nais_inference.mojo
│
├── examples/
│   ├── single_price.mojo                  # Mode 1 example
│   ├── batch_price.mojo                   # Mode 2 example
│   ├── calibrate.mojo                     # Mode 3 example
│   └── nais_train_infer.mojo
│
├── python/
│   ├── fpe_engine/__init__.py
│   └── examples/
│       ├── backtest.py
│       └── research.ipynb
│
└── cpp/
    ├── include/fpe_engine.h
    └── examples/live_trading.cpp
```

---

## Summary of v1 → v2 → v3 Evolution

| Aspect | v1 | v2 | v3 (Current) |
|---|---|---|---|
| **Layer 1** | 17 files from scratch | 5 files sparse only | 5 files sparse only |
| **Dense ops** | Custom SIMD | MAX Kernels | MAX Kernels |
| **FFT** | Custom Radix-2 | MAX `rfft`/`irfft` | MAX `rfft`/`irfft` |
| **Random** | Custom MT19937 | `std.random.philox` | `std.random.philox` |
| **FPE solver** | CPU only | `cpu_solver` + `gpu_batch_solver` (separate) | **`FPESolver[B]` unified** |
| **Pricer** | CPU only | `single_pricer` + `batch_pricer` (separate) | **`Pricer[B]` unified** |
| **Design** | Python mindset | CUDA mindset (CPU≠GPU) | **Mojo mindset (write once)** |
| **Files** | ~50 | ~40 | **~36** |

### Key Architectural Decisions (v3)

| Decision | Rationale |
|---|---|
| **`comptime batch_size` parametric design** | Write once, deploy to CPU (B=1) or GPU (B=N). Mojo's core value proposition. |
| **`comptime if has_accelerator()`** | Compile-time GPU/CPU dispatch — zero runtime overhead, no code duplication |
| **MAX Kernels as Layer 0** | Production-optimized by Modular; no reason to rewrite GEMM/FFT |
| **Custom sparse only** | MAX focuses on dense AI ops; FPE requires sparse CSR/COO/kron |
| **`Calibrator[B]` reuses `FPESolver[B]`** | Calibration is just FPE solve in a loop — same parametric solver |
| **Unified `Pricer[B]`** | B=1 → CPU SIMD interp, B=1000 → GPU one-thread-per-option. Same algorithm. |
| **`HestonParamsBatch[B]`** | Type-safe batch dimension — compiler catches mismatched batch sizes |

### The Mojo Philosophy in Practice

```mojo
# THIS is the Mojo way:
var result_1    = engine.price[1](req)          # compiles to CPU SIMD
var result_1000 = engine.price[1000](batch)     # compiles to GPU kernel
var params_64   = engine.calibrate[64](market)  # compiles to GPU kernel

# NOT this (v2 anti-pattern):
var result = engine.price_single(req)           # ❌ separate CPU code
var result = engine.price_batch(batch)           # ❌ separate GPU code
```

---

*v3 revised 2026-03-31. Unified parametric design: write once, comptime dispatch to CPU/GPU. MAX AI Kernels as optimized foundation.*
