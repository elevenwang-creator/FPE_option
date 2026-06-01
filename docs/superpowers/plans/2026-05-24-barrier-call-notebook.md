# Barrier Call Option Pricing Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an end-user Jupyter notebook demonstrating the FPEPricer stepwise API with plotnine 2D visualizations and matplotlib 3D surface plots.

**Architecture:** Single notebook file following the 10-section design spec. Each section is a mix of markdown (math background, explanations) and code cells (API calls + visualizations). Two FPEPricer instances (european + barrier) used throughout.

**Tech Stack:** fpe_engine, numpy, scipy, pandas, plotnine (2D), matplotlib (3D only)

---

## File Structure

- Create: `python/examples/Barrier_Call_Option_Pricing.ipynb` — the entire notebook
- Reference: `python/fpe_engine/pricer.py` — API being demonstrated
- Reference: `Barrier_Call_Option_Pricing.ipynb` (root) — old reference notebook for math content

---

### Task 1: Create notebook skeleton with metadata and imports

**Files:**
- Create: `python/examples/Barrier_Call_Option_Pricing.ipynb`

- [ ] **Step 1: Create the notebook JSON with kernel metadata, imports cell, and theme setup cell**

The notebook is a JSON file with cells array. Create:

1. **Markdown cell**: Title + authors (styled like reference notebook)
2. **Code cell**: Imports — `fpe_engine`, `numpy`, `scipy.sparse`, `pandas`, `plotnine.*`, `matplotlib.pyplot`, `mpl_toolkits.mplot3d`
3. **Code cell**: Plotnine theme setup (`THEME` constant) + matplotlib 3D style helper

Imports cell:
```python
import numpy as np
import pandas as pd
from scipy import sparse
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_tile, geom_bar,
    geom_hline, facet_wrap, labs, theme_minimal, theme, element_text,
    scale_color_brewer, scale_fill_distiller, scale_x_continuous,
    scale_y_continuous, lims, coord_fixed, guide_legend, guides,
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpe_engine import FPEPricer, KnotsResult, Basis1DResult, GreeksResult, PriceResult
```

Theme cell:
```python
THEME = theme_minimal() + theme(
    plot_title=element_text(size=14, weight="bold"),
    axis_title=element_text(size=11),
    legend_title=element_text(size=11),
    figure_size=(10, 6),
)

def surface_3d(s_grid, v_grid, Z, title="", s_label="s", v_label="v"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    S, V = np.meshgrid(s_grid, v_grid)
    surf = ax.plot_surface(S, V, Z.T, cmap="turbo", edgecolor="none", alpha=0.95)
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel(s_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    return fig
```

- [ ] **Step 2: Verify notebook loads in Jupyter**

Run: `pixi run jupyter nbconvert --to notebook python/examples/Barrier_Call_Option_Pricing.ipynb --output test.ipynb 2>&1 | head -5`

Expected: No JSON parse errors

---

### Task 2: Section 1 (Quick Start) + Section 2 (Introduction) + Section 3 (Parameters)

- [ ] **Step 1: Add Quick Start section**

Markdown: "## Quick Start"
Code cell:
```python
with FPEPricer(S0=60, V0=0.1, T=0.6, r=0.1, n_s=16, n_v=16) as p:
    result = p.price(100.0)
    print(f"European call price at K=100: {result.prices[0]:.6f}")
    print(f"Delta: {result.deltas[0]:.6f}")
```

- [ ] **Step 2: Add Introduction markdown cells**

Copy/adapt math content from reference notebook: Heston SDEs, FPE equation, variational formulation, numerical procedure summary. Include Stoykov (2024) reference link.

- [ ] **Step 3: Add Parameters section**

Markdown: parameter table.
Code cell creating both pricers:
```python
HESTON = dict(kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.6, S0=60.0, V0=0.1)

pricer_eu = FPEPricer(**HESTON, n_s=16, n_v=16, option_type="european_call", barrier=0.0)
pricer_bar = FPEPricer(**HESTON, n_s=16, n_v=16, option_type="down_and_out_call", barrier=50.0)
```

---

### Task 3: Section 4 (Knots) + Section 5 (Basis Functions)

- [ ] **Step 1: Add Knots section**

Code cell:
```python
knots_eu = pricer_eu.knots()
knots_bar = pricer_bar.knots()

df_knots = pd.concat([
    pd.DataFrame({"s": knots_eu.s, "v": np.full_like(knots_eu.s, knots_eu.v[len(knots_eu.v)//2]), "type": "European"}),
    pd.DataFrame({"s": knots_bar.s, "v": np.full_like(knots_bar.s, knots_bar.v[len(knots_bar.v)//2]), "type": "Barrier"}),
])

(ggplot(df_knots, aes("s", "v", color="type"))
 + geom_point(size=2, alpha=0.7)
 + labs(title="Non-uniform Knot Points", x="Asset Price (s)", y="Variance (v)")
 + scale_color_brewer(type="qual", palette="Set1")
 + THEME
)
```

- [ ] **Step 2: Add Basis Functions section — 1D line plots**

Code cell extracting basis columns and plotting with plotnine. Select every 3rd basis function to avoid clutter.

- [ ] **Step 3: Add Basis Functions section — 3D tensor product surface**

Code cell using `surface_3d()` helper. Extract a column from `basis_2d()`, reshape with `.reshape(n_v, n_s).T`.

---

### Task 4: Section 6 (Initial Condition) + Section 7 (Solve & PDF)

- [ ] **Step 1: Add Initial Condition section — heatmap + 3D surface**

Reconstruct delta: `delta_flat = basis_2d @ q0`, reshape `.reshape(n_v, n_s).T`, flatten to DataFrame for `geom_tile`, then `surface_3d()`.

- [ ] **Step 2: Add Solve & PDF section — heatmap + 3D surface**

`pdf()` is already 2D. Flatten to DataFrame for `geom_tile`. Then `surface_3d()`.

- [ ] **Step 3: Add marginal distribution plots**

Integrate PDF along each axis, plot with plotnine line plots. Faceted by option type.

---

### Task 5: Section 8 (Pricing) + Section 9 (Convergence)

- [ ] **Step 1: Add Pricing section**

`payoff_price()` for multiple strikes. Bar chart vs Monte Carlo. Relative error bar chart. Price table as DataFrame.

Monte Carlo reference: `[4.716, 2.939, 1.729, 0.973, 0.531, 0.285, 0.152, 0.0805]`

- [ ] **Step 2: Add Convergence section**

Loop over n values, fresh `FPEPricer` each time with context manager, collect prices. Line plot with plotnine. Horizontal dashed line at MC reference.

---

### Task 6: Section 10 (Greeks) + Final cleanup

- [ ] **Step 1: Add Greeks section**

`greeks(K)` for a range of strikes. Three plotnine line plots: delta, gamma, vega vs K. Two lines each (european + barrier).

- [ ] **Step 2: Add cleanup cell — close both pricers**

```python
pricer_eu.close()
pricer_bar.close()
print("Done.")
```

- [ ] **Step 3: Verify notebook JSON is valid**

Run: `python -c "import json; json.load(open('python/examples/Barrier_Call_Option_Pricing.ipynb'))"` 

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add python/examples/Barrier_Call_Option_Pricing.ipynb
git commit -m "feat: add stepwise API example notebook with plotnine visualizations"
```
