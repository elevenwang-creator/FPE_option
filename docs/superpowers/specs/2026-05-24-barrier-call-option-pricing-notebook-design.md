# Barrier Call Option Pricing Notebook Design

## Summary

Create an end-user Jupyter notebook demonstrating the FPEPricer stepwise API for pricing barrier and European call options under the Heston stochastic volatility model. The notebook uses plotnine for 2D visualizations and matplotlib for 3D surface plots, with a layered structure serving both practitioners (quick results) and developers (pipeline internals).

## Motivation

The FPEPricer stepwise API is complete and tested, but there is no example notebook showing end-users how to use it. The existing reference notebook (`Barrier_Call_Option_Pricing.ipynb`) uses the old Python-only solver with matplotlib. We need a new notebook that:

1. Uses the Mojo-backed `FPEPricer` class (fast, stateful, lazy-cached)
2. Uses plotnine for beautiful 2D plots (per project constraint)
3. Includes 3D surface plots for PDF and 2D basis (matplotlib `plot_surface`)
4. Serves both audiences: quick-start for practitioners, stepwise deep-dive for developers
5. Demonstrates two option types: European call and down-and-out barrier call

## Target File

`python/examples/Barrier_Call_Option_Pricing.ipynb`

## Audience

**Both — layered:**

- Layer 1 (practitioners): Quick-start cell, pricing results, Greeks tables
- Layer 2 (developers): Stepwise API deep-dive, intermediate results, convergence analysis

## Option Types

1. **European call** — vanilla, no barrier
2. **Down-and-out barrier call** — barrier=50, same Heston parameters as the reference notebook

## Heston Parameters (from reference notebook Table 1)

| Parameter | Value |
|-----------|-------|
| S0 | 60.0 |
| V0 | 0.1 |
| r | 0.1 |
| T | 0.6 |
| kappa | 1.2 |
| theta | 0.05 |
| sigma (eta) | 0.35 |
| rho | -0.4 |
| Barrier | 50.0 |

## Notebook Sections

### Section 1: Quick Start

Single cell showing the fastest way to get a price:

```python
from fpe_engine import FPEPricer

with FPEPricer(S0=60, V0=0.1, T=0.6, r=0.1, n_s=16, n_v=16) as p:
    result = p.price(100.0)
    print(f"Price: {result.prices[0]:.6f}")
    print(f"Delta: {result.deltas[0]:.6f}")
```

No visualization. Just output. Gives practitioners immediate value. Note: `price()` returns a `PriceResult` dataclass with `prices`, `deltas`, `gammas`, `vegas`, and `success` fields.

### Section 2: Introduction

Markdown cells with:

- Heston model SDEs (LaTeX)
- Fokker-Planck equation (LaTeX)
- Variational formulation overview
- Numerical procedure summary (B-splines, Radau IIA ODE solver)
- Reference: Stoykov (2024) paper link

Adapted from reference notebook's introduction cells.

### Section 3: Parameters & Setup

- Heston parameter table (markdown)
- Create two `FPEPricer` instances (both as context managers for resource cleanup):
  - `pricer_eu`: `FPEPricer(..., option_type="european_call", barrier=0.0)`
  - `pricer_barrier`: `FPEPricer(..., option_type="down_and_out_call", barrier=50.0)`
- Both at n_s=16, n_v=16 (fast for interactive use)
- All FPEPricer instances must use `with` context managers or explicit `.close()` to prevent native resource leaks

### Section 4: Knots

- Call `knots()` on both pricers
- **2D plot**: plotnine scatter plot of knot points (s vs v), colored by pricer type
- Brief markdown explaining non-uniform knot generation and concentration near initial condition

### Section 5: Basis Functions

- Call `basis_1d()` and `basis_2d()`
- **Data extraction for plotting**: `Basis1DResult` contains sparse CSR matrices (`Bs`, `dBs`, `Bv`, `dBv`). Each matrix has shape `(n_eval_points, n_basis_functions)` where `n_eval_points` = number of Gauss-Legendre quadrature nodes per dimension. To plot individual basis functions, convert a column slice to dense: `Bs[:, i].toarray().flatten()` gives the i-th basis function evaluated at all quadrature nodes. For the evaluation grid, use a linear grid over the domain: `s_grid = np.linspace(knots.s[0], knots.s[-1], Bs.shape[0])`, similarly for v. Grid dimensions `n_s_eval` and `n_v_eval` for 2D reshaping are available from `pdf().shape`.
- **2D plots** (plotnine): Line plots of Bs values and dBs derivatives over evaluation grid. Same for Bv/dBv. Select a subset of basis functions (e.g., every 3rd) to avoid clutter. Faceted by derivative order (value vs derivative).
- **3D plot** (matplotlib `plot_surface`): Single 2D tensor-product basis function. Extract a column from `basis_2d()` (which is a CSR matrix of shape `(n_s_eval * n_v_eval, n_total_basis)`). Reshape the column accounting for Kronecker ordering: `col.reshape(n_v_eval, n_s_eval).T` to get `(n_s_eval, n_v_eval)` grid. Grid dimensions from `pdf().shape`. Create meshgrid from s/v evaluation points and plot as 3D surface. Shows the tensor product structure clearly.
- Markdown: boundary conditions (Dirichlet at s=0 for absorbing, Neumann at v=0 for reflecting)

### Section 6: Initial Condition

- Call `initial_condition()` to get q0 coefficients (1D array of length `n_total_basis`)
- Reconstruct the 2D delta approximation: `delta_flat = basis_2d @ q0` yields a 1D array of length `n_s_eval * n_v_eval`. Reshape to 2D accounting for Kronecker row ordering: `kron(Bs, Bv)` stores rows with s varying fastest, so use `delta_2d = delta_flat.reshape(n_v_eval, n_s_eval).T` for correct `(s, v)` orientation. Grid dimensions `n_s_eval` and `n_v_eval` are obtained from `pdf().shape` (which is already 2D) — i.e., `n_s_eval, n_v_eval = pdf().shape`.
- **2D plot** (plotnine): Heatmap of the delta function approximation in (s, v) space. Use `plotnine.geom_tile` with the reshaped 2D array flattened into a pandas DataFrame with columns `s`, `v`, `density`.
- **3D plot** (matplotlib `plot_surface`): Same delta approximation as a 3D surface using meshgrid from s/v grids
- Markdown: explains delta approximation via narrow normal distribution at (S0, V0)

### Section 7: Solve & PDF

- Call `pdf()` to get the PDF at maturity — returns a **2D** `np.ndarray` of shape `(n_s_eval, n_v_eval)`. No reshaping needed.
- **2D plot** (plotnine): Heatmap of PDF at maturity in (s, v) space. Flatten the 2D array into a pandas DataFrame with columns `s`, `v`, `density`, then use `geom_tile`. The s/v grids are derived from `knots().s` boundaries: `s_grid = np.linspace(knots.s[0], knots.s[-1], pdf.shape[0])`, similarly for v.
- **3D plot** (matplotlib `plot_surface`): Same PDF as a 3D surface using meshgrid from s/v grids
- **2D plots** (plotnine): Marginal distributions — integrate PDF over v (sum along axis=1 weighted by v-spacing) to get marginal over s, and over s (sum along axis=0 weighted by s-spacing) to get marginal over v. Line plots.
- Compare European vs barrier PDFs side-by-side (faceted)

### Section 8: Pricing

- Call `payoff_price(K)` for strikes [65, 70, 75, 80, 85, 90, 95, 100]. This uses the cached PDF (computing it first if necessary) and integrates the payoff function — it reuses the stepwise pipeline's cached results. In contrast, `price()` runs the full pipeline from scratch independently and is used when you don't need intermediate results.
- Monte Carlo reference values from the reference notebook
- **2D plot** (plotnine): Grouped bar chart — FPE price vs Monte Carlo, faceted by option type
- **2D plot** (plotnine): Relative error bar chart (|FPE - MC| / MC * 100%)
- Display price table as pandas DataFrame

### Section 9: Convergence Study

- Price at grid sizes n = [8, 12, 16, 20, 24, 28, 32] where `n_s = n_v = n` for each point
- Use `price()` for each grid size (fresh `FPEPricer` each time, with `with` context manager). Must pass `option_type` explicitly for each: one loop for `"european_call"` and one for `"down_and_out_call"` with `barrier=50`.
- **2D plot** (plotnine): Line plot of price vs n for K=100, with horizontal dashed line at Monte Carlo reference. Two lines (european + barrier).
- Markdown: explains that small n gives negative prices (under-resolution), convergence is monotonic at larger n

### Section 10: Greeks

- Call `greeks(K)` for a range of strikes
- **2D plots** (plotnine): Three line plots — Delta vs K, Gamma vs K, Vega vs K. Each as a separate faceted panel. Two lines per plot (european + barrier).
- Markdown: bump-based finite differences, note computational cost at high n

## Visualization Strategy

| Plot Type | Library | Use Case |
|-----------|---------|----------|
| Scatter, line, bar, heatmap | plotnine | All 2D visualizations |
| 3D surface | matplotlib `plot_surface` | 2D basis function, initial condition, PDF |

### plotnine Theme

Consistent theme across all 2D plots:

```python
from plotnine import theme_minimal, element_text

THEME = theme_minimal() + theme(
    plot_title=element_text(size=14, weight="bold"),
    axis_title=element_text(size=11),
    legend_title=element_text(size=11),
)
```

Color palette: `scale_color_brewer(type="qual", palette="Set1")` for categorical, `scale_fill_distiller(palette="YlOrRd")` for heatmaps.

### matplotlib 3D Style

```python
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(S, V, Z, cmap="turbo", edgecolor="none", alpha=0.95)
fig.colorbar(surf, shrink=0.5)
```

Matches the reference notebook's 3D style.

## Dependencies

- `fpe_engine` (with `_fpe_native` Mojo extension)
- `numpy`
- `scipy`
- `pandas`
- `plotnine`
- `matplotlib` (for 3D only)

## Constraints

- No matplotlib for 2D plots — plotnine only
- Grid sizes in the notebook stay small (8-32) for interactive runtimes
- Greeks are slow at n>=20 (bump-based FD) — use n=16 for the Greeks section
- The notebook must run end-to-end with `pixi run jupyter`
- All `FPEPricer` instances must use context managers (`with` blocks) for resource cleanup
- `num_insert` defaults to 50 (no need to override in the notebook)
