# FPE Engine — Learnings & Conventions

## 2026-03-31 Session ses_2bcd21f22ffeF2bi27TYZ6AyJz

### Mojo v0.26.2 Syntax Rules (from mojo-syntax skill)
- `def` only (no `fn`), add `raises` explicitly when needed
- `comptime` replaces `alias` and `@parameter`
- Argument conventions: `read` (default), `mut`, `var`, `out`, `deinit`, `ref`
- `@fieldwise_init` generates constructors from fields
- Self-qualify struct params: always `Self.T` not bare `T`
- No `let` — use `var` for all variables
- Collection literals: `[1,2,3]` for List, `{"k": v}` for Dict
- `comptime if has_accelerator()` for GPU/CPU dispatch at compile time

### Architecture Decisions
- **Unified parametric design**: `FPESolver[B]`, `Pricer[B]` — batch_size is comptime param
- `batch_size=1` → CPU SIMD path via `vectorize`
- `batch_size>1` + GPU → `ctx.enqueue_function[kernel, kernel](...)`
- `batch_size>1` + no GPU → `parallelize` across batch
- **MAX AI Kernels** used for: matmul, gemv, bmm, grouped_matmul, qr_factorization, rfft, irfft, activations
- **Custom sparse only**: CSR, COO, Diag, kron, spmv, spgemm — MAX has no sparse support
- **Philox PRNG** (`std.random.philox`) for GPU-parallel Monte Carlo
- **LayoutTensor** from `layout` package for all GPU kernel data

### Python Source Files (reference)
- `FPE_Solver_Final_Version.py` — 1,153 lines, Heston FPE via B-spline Galerkin
- `NAIS_rBM.py` — 448 lines, NAIS-Net FBSDE for rough Bergomi
- `BarrierOptionPricing.ipynb` — barrier option pricing workflow

### Key Python Dependencies → Mojo Replacements
- `scipy.sparse` → custom `src/sparse/` (CSR/COO/Diag/ops)
- `scipy.integrate.solve_ivp` → `src/numerics/ode/radau.mojo`
- `scipy.fftpack.fft/ifft` → MAX `kernels.nn.rfft`/`irfft`
- `cvxpy` / OSQP → `src/numerics/optim/osqp.mojo`
- `numpy.random.normal` → `std.random.philox`
- `tensorflow` / Keras → MAX `kernels.linalg.matmul` + custom autograd
- `numpy.linalg` → MAX `kernels.linalg.*`

### FPE Math Summary
- Heston model: dS = rS dt + √V S dW₁, dV = κ(θ-V)dt + σ√V dW₂, corr(dW₁,dW₂)=ρ
- FPE for joint density p(S,V,t): ∂p/∂t = L*p (Fokker-Planck operator)
- Galerkin: expand p ≈ Σ qᵢ(t) φᵢ(S,V), get ODE: M dq/dt = -K q
- Initial condition: q₀ via OSQP (non-negative QP)
- Barrier option price: ∫∫ payoff(S) · p(S,V,T) dS dV

### NAIS-Net Math Summary
- Rough Bergomi: dS = S(r dt + √V dW₁), V(t) = ε(t)·exp(η·X̃(t) - 0.5η²t^{2H})
- X̃(t) = Volterra process via hybrid scheme (FFT convolution)
- NAIS-Net: skip connections + sin activation + StableLinear weight constraint
- FBSDE: Y(t) = u(t,X(t)), Z = σ·∇u, loss = Σ|Y(tₙ₊₁) - Ỹ(tₙ₊₁)|²

## 2026-03-31 Sparse module implementation notes

- Mojo package import style in this repo uses package-qualified imports (`from sparse.csr import ...`), not relative imports (`from .csr ...`).
- Returning `List[...]` and structs containing `List` requires explicit move on return (`return out^`) to satisfy ownership rules.
- Avoid copying nested list rows from function args (`var row = dense[i]`) unless using `.copy()`; direct indexed access avoids implicit-copy errors.
- `mojo build` expects an executable with `main`; for library-only sparse modules use `mojo package -I src -o /tmp/sparse.mojopkg src/sparse` as compilation verification.
- CSR/COO conversion path: accumulate in COO, stable sort by `(row, col)`, coalesce duplicate coordinates, then emit CSR `indptr/indices/data`.

## 2026-03-31 — test_max_integration.mojo smoke test

### File created
- `tests/test_max_integration.mojo` — verifies MAX stdlib + layout imports compile

### Import patterns confirmed (Mojo v0.26.2)
- `from std.algorithm import vectorize, parallelize`
- `from std.math import sqrt, exp, log, sin`
- `from std.random import randn` → usage: `randn[DType.float64]()`
- `from std.sys import has_accelerator`
- `from std.testing import assert_true, TestSuite`
- `from layout import Layout, LayoutTensor` (import only; LayoutTensor needs backing buffer to construct)
- GPU imports (`from std.gpu import global_idx`, `from std.gpu.host import DeviceContext`) guarded inside `comptime if has_accelerator():`

### vectorize pattern
```mojo
def inner[width: Int](i: Int):
    var v = SIMD[DType.float32, width](Float32(i))
    _ = v * v
vectorize[inner, 8](16)
```

### TestSuite entry point pattern
```mojo
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

### Gotchas
- `randn` returns a single scalar — discard with `_ = randn[DType.float64]()` to avoid unused-var warning
- LayoutTensor requires a backing buffer; smoke test only confirms the import resolves
- Pre-existing LSP errors in Python reference files (numpy/scipy/tensorflow not installed) are unrelated

## 2026-03-31 — bspline module implementation notes

### Mojo ownership conventions hit in this task
- `List`-holding constructors must take owned args (`var knots: List[...]`) and call sites must transfer (`knots^`) or explicitly copy (`knots.copy()`).
- Structs containing `List` are not implicitly copyable at call sites; pass with `^` into constructors like `TensorProductBasis(... basis_s=bs^, basis_v=bv^)`.

### Command/packaging details
- `mojo run` include-path order matters in this workspace: use `mojo run -I src tests/...` (placing `-I` before the file path).
- Nested numerics bspline package compiles via `mojo package -I src -o /tmp/bspline.mojopkg src/numerics/bspline` without touching other numerics modules.

### BSpline implementation decisions
- De Boor-Cox implemented with a `comptime for` over degree and a runtime work buffer for stable recursive unrolling.
- Recombination matrix built in COO then converted to CSR so duplicate corner entries naturally coalesce for Neumann corner additions.

## generate_reference.py — key observations (2026-03-31)

### Source file APIs confirmed:
- `GenerateKnots(n, p, method, center, boundary, mean, std, cheby_num=13)` — `.generate_knots()` returns knot vector; `x[1:]` skips the duplicate zero
- `BSplineBasis(degree, knots)` — `.basis_function(u)` and `.first_derivative(u)` return sparse CSR matrices
- `RecombinationBasis(degree, knots, conditions)` — same interface
- `HestonSolver(degrees, knots_list, conditions_list, params)` — properties: `.mass_matrix`, `.stiffness_matrix`, `.s_points`, `.v_points`, `.nodes_weights`
- `HestonSolver.delta_approx(sigma)` — returns 2D numpy array (n_s, n_v); Gaussian approx of delta
- `HestonSolver.q_initial(sigma)` — OSQP solve; returns 1D coefficient vector
- `HestonSolver.fpe_solver(sigma0, time=None)` — returns `[pdf_2d, t]` where `pdf_2d.shape = (n_s, n_v, n_t)`
- `FBSNN(Xi, T, M, N, D, H, eta, pho, r, layers)` — `.volterra()` → `(t, W, tilde_X)`; `.variance()` → `(W_s, var)`

### Barrier option approach:
- No dedicated `barrier_price` method in `HestonSolver`
- Must compute from FPE output: integrate `marginal_s = pdf_T @ v_weights`, then `∫ (S-K)⁺ · p(S) dS`
- Delta/gamma via finite differences over three S0 offsets: (59.5, 60.0, 60.5)

### `HestonSolver` params dict defaults include `'T'`; must pass explicitly for short maturities

### Script design:
- All sections wrapped in `try/except` — failure in one does not block others
- `T` parameter added to params for `fpe_solver` sections to avoid KeyError

## 2026-03-31 — test_sparse.mojo

### File created
- `tests/test_sparse.mojo` — 228 lines, 6 test functions for the sparse module

### Test construction patterns for Mojo structs without Copyable
- `COOMatrix` is `Movable` only (not `Copyable`). Call `coo.to_csr()` using `read self` convention — does NOT consume the COO. Return a `var csr = coo.to_csr()`.
- `DiagMatrix` is `Copyable + Movable`. Direct field assignment works: `D.diag[0] = Float64(2.0)`.
- Nested `List[List[Scalar[DType.float64]]]`: build each inner row as a named var, then move-append with `^`. Inline nested list literals are not reliable for nested generic types.

### Type safety in tests
- `COOMatrix.append(r, c, v)`: the third arg must be `Scalar[dtype]`. Use explicit `Float64(1.0)` to avoid ambiguity with FloatLiteral when dtype parameter is in scope.
- `CSRMatrix.from_dense` accepts `List[List[Scalar[dtype]]]` — bracket literals with annotated var type coerce correctly.

### Verified hand-computed expected values
- spmv: A=[[1,0,2],[0,3,0],[4,0,5]], x=[1,2,3] → y=[7,6,19]
- coo→csr: 5 entries → indptr=[0,2,3,5], indices=[0,2,1,0,2]
- diag_vec_mul: [2,3,4]⊙[1,2,3] = [2,6,12]
- kron([[1,2],[3,4]], [[0,5],[6,7]]) = 4×4 block: row0=[0,5,0,10], row1=[6,7,12,14], row2=[0,15,0,20], row3=[18,21,24,28]
- spgemm([[1,2],[3,4]], [[5,6],[7,8]]) = [[19,22],[43,50]]
- spmm([[1,0],[0,2]], [[3,4],[5,6]]) = [[3,4],[10,12]]

### assert_float_close helper
- Defined with `atol: Float64 = 1e-10` default
- Uses `String(b)` for error messages — works because Float64 (SIMD[DType.float64, 1]) implements Writable
- Zero-entry check (diff=0.0 < 1e-10) passes correctly

## 2026-03-31 — ODE module implementation notes

- New ODE package files: `src/numerics/ode/types.mojo`, `rk45.mojo`, `radau.mojo`; `__init__.mojo` now re-exports `ODESystem`, `ODESolution`, `RungeKutta45`, `BackwardEuler`, `RadauIIA`.
- Mojo trait method declarations require bodies (`...`) and correct raises order: `def f(...) raises -> T:` not `-> T raises:`.
- Parametric struct method signatures must self-qualify parameters (`Self.System`), including nested generic construction (`BackwardEuler[Self.System]`).
- For non-`ImplicitlyCopyable` nested containers like `List[List[Float64]]`, swap rows by element-wise scalar swaps to avoid implicit-copy violations.
- Project test execution needs include path: `pixi run mojo -I src tests/test_ode.mojo`.
- Current pixi `build` task (`mojo build src/`) is invalid for package dirs; compile verification works with `pixi run mojo package -I src -o /tmp/ode.mojopkg src/numerics/ode`.

## 2026-03-31 — optim module implementation notes

- Added `src/numerics/optim/osqp.mojo` using projected-gradient NNLS (`min 0.5||Ac-b||², c>=0`) with auto step-size fallback `1/(max diag(AᵀA)+1e-10)`.
- Added `src/numerics/optim/lm.mojo` with Levenberg-Marquardt normal-equation step `(JᵀJ + λI)δ = -Jᵀr` and local LU solver copied from ODE pattern.
- Added `tests/test_optim.mojo` with three deterministic checks: NNLS exact `[1,1]`, NNLS projection to zero, LM linear fit recovering slope/intercept `[2,1]`.
- Build verification for package modules in this repo should use `pixi run mojo package -I src -o /tmp/<name>.mojopkg src/<pkg>`; this task passed for `src/numerics/optim`.

## 2026-03-31 — FPE engine core implementation notes

- New core files created under `src/engines/fpe/`: `heston_params.mojo`, `domain.mojo`, `galerkin.mojo`, `initial_cond.mojo`, `solver.mojo`, `pdf.mojo`; `__init__.mojo` now re-exports all FPE symbols.
- Mojo generic self-qualification is mandatory in structs with type params (`Self.B` instead of `B`) for assertions, constructors, and `comptime if` checks.
- Tensor-product quadrature/matrix assembly currently uses dense helper transforms around sparse primitives (`spgemm`, `kron`) to keep implementation simple and robust for small-grid tests.
- `BackwardEuler` in this repo ignores exact `t_eval` sampling; `FPESolver` currently returns solver internal states and applies a non-negative projection/renormalization post-step to enforce probability constraints in tests.
- Compilation verification for this module works via `pixi run mojo package -I src -o /tmp/fpe.mojopkg src/engines/fpe`.
- Pipeline test command: `pixi run mojo -I src tests/test_fpe_engine.mojo`.

## 2026-03-31 — pricing server implementation notes

- Added server modules: `pdf_cache.mojo`, `interpolator.mojo`, `payoffs.mojo`, `greeks.mojo`, `pricer.mojo`, `pricing_engine.mojo`, plus re-exports in `src/server/__init__.mojo`.
- Empty structs used as value fields/temporary objects need explicit `__init__(out self)` constructors in this workspace style (e.g., `Interpolator`, payoff structs).
- `Dict` insert of non-ImplicitlyCopyable values requires ownership transfer (`grid^`); store APIs should accept `var` for move semantics.
- `Dict.get(key)` cleanly returns `Optional[V]` and avoids raising index access in non-`raises` methods.
- For structs with non-implicit-copy fields, pass explicit copies when constructing other structs (`self.interpolator.copy()`).
- Pricing server tests execute with `pixi run mojo -I src tests/test_pricing_server.mojo`; package build verification with `pixi run mojo package -I src -o /tmp/server.mojopkg src/server`.

## 2026-03-31 — calibrator + NAIS engine implementation notes

- New calibrator module files: `src/engines/calibrator/objective.mojo`, `src/engines/calibrator/calibrator.mojo`, and re-exports in `src/engines/calibrator/__init__.mojo`.
- `ObjectiveFunction` requires an explicit owned-arg constructor (`var List[...]`) because list fields are non-ImplicitlyCopyable.
- Added `PDFComputer.__init__(out self)` to allow direct construction (`PDFComputer[B]()`), matching workspace patterns for zero-field structs.
- Closure-heavy APIs hit ownership friction with captured non-ImplicitlyCopyable state; calibrator LM loop is implemented directly in `Calibrator.calibrate` while still using LM hyperparameters (`lambda_init/up/down`, `tol`, `max_iter`).
- New NN primitives in `src/numerics/nn/`: `StableLinear`, `GradientTape`, `Adam`; `__init__.mojo` re-exports all three.
- For tuple returns that include `List`, tuple element extraction often needs explicit `.copy()` to satisfy ownership (`out[1].copy()`).
- Avoid naming collisions with comptime params (e.g., struct param `B`) in method args/locals; rename Brownian path variables to avoid `invalid redefinition of 'B'`.
- NAIS engine files added: `nais_net.mojo`, `volterra.mojo`, `variance.mojo`, `fbsde.mojo`, `trainer.mojo`, `inferencer.mojo`, and re-exports in `src/engines/nais/__init__.mojo`.
- Verification commands used for this batch:
  - `pixi run mojo package -I src -o /tmp/calibrator.mojopkg src/engines/calibrator`
  - `pixi run mojo package -I src -o /tmp/nn.mojopkg src/numerics/nn`
  - `pixi run mojo package -I src -o /tmp/nais.mojopkg src/engines/nais`
  - `pixi run mojo -I src tests/test_calibrator.mojo`
  - `pixi run mojo -I src tests/test_nais_engine.mojo`

## 2026-03-31 — Phase 5 bindings/production artifacts notes

- Mojo package compilation in this workspace rejects module-level mutable globals (`var _engine = ...`) in bindings; keep exported bindings stateless or pass state via explicit handles.
- `PythonModuleBuilder.def_function[...]` currently fails inference for high-arity exports in this environment; packing parameters into one `PythonObject` dictionary avoids arity limits.
- Ownership rules apply heavily in bindings/tests: move non-ImplicitlyCopyable values into structs/containers (`req^`, `grid^`, list fields with `^`).
- For quick binding demos/benchmarks, seeding a synthetic uniform `PDFGrid` is a practical way to exercise `PricingEngine.price[1]` without adding new engine APIs.
- Verification command set that passed for this phase: `pixi run mojo package -I src -o /tmp/bindings.mojopkg src/bindings`, `pixi run mojo -I src tests/test_bindings.mojo`, `pixi run mojo -I src benchmarks/bench_pricing.mojo`, `clang++ -I cpp/include -c cpp/examples/live_trading.cpp -o /tmp/live_trading.o`.

## 2026-03-31 — LM callable trait fix

- Mojo v0.26.2 `solve` callables are safer as traits than `fn(...) capturing[_]` types for free-function wrappers.
- Empty wrapper structs need explicit `__init__(out self)` before `ResidualLine()` / `JacobianLine()` instantiation works.
- `tests/test_optim.mojo` must import trait names directly when using them in struct conformance lists.
- No `.mojo` LSP server is configured in this workspace, so diagnostics tooling cannot validate Mojo files here; use `pixi run mojo -I src tests/test_optim.mojo` for verification instead.
