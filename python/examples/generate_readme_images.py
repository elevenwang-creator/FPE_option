#!/usr/bin/env python3
"""
Generate figures from the FPE engine for embedding in README.md.

Saves PNGs to docs/images/ for GitHub rendering.
"""

import numpy as np
import polars as pl
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpe_engine import Compute, price

import os, sys
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'images')
os.makedirs(OUT, exist_ok=True)

plt.style.use('ggplot')
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                      "axes.titlesize": 13, "axes.labelsize": 11})

HESTON = dict(kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.6, S0=60.0, V0=0.1, s_min=0.0, s_max=150.0)
PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33"]

print("Building pricers (n_s=38, n_v=38)...")
p_eu = Compute(**HESTON, n_s=38, n_v=38, option_type='european_call', barrier=0.0)
p_bar = Compute(**HESTON, n_s=38, n_v=38, option_type='down_and_out_call', barrier=50.0)

# === 1. Knot Grid Points ===
print("1/9  Knot Grid Points")
knots_eu = p_eu.knots; knots_bar = p_bar.knots
df_knots = pl.concat([
    pl.DataFrame({"s": knots_eu.s, "type": "European"}),
    pl.DataFrame({"s": knots_bar.s, "type": "Barrier"}),
])
df_knots_v = pl.concat([
    pl.DataFrame({"v": knots_eu.v, "type": "European"}),
    pl.DataFrame({"v": knots_bar.v, "type": "Barrier"}),
])
grid_df = df_knots.join(df_knots_v, how="cross")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for t, c in [("European", "#E41A1C"), ("Barrier", "#377EB8")]:
    s = grid_df.filter(pl.col("type") == t)
    ax1.scatter(s["s"], s["v"], alpha=0.6, s=15, color=c, label=t)
    ax2.scatter(s["s"], s["v"], alpha=0.6, s=15, color=c, label=t)
ax1.set_xlabel("Asset Price (s)"); ax1.set_ylabel("Variance (v)")
ax1.set_title("Knot Grid Points — European"); ax1.legend()
ax2.set_xlabel("Asset Price (s)"); ax2.set_ylabel("Variance (v)")
ax2.set_title("Knot Grid Points — Barrier"); ax2.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "knot_grid_points.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 2. 1D B-Spline Basis Functions ===
print("2/9  B-Spline Basis Functions (s-direction)")
gp = p_eu.grid_points
s_grid = gp.s; v_grid = gp.v
basis = p_eu.basis_1d

step = 2
fig, ax = plt.subplots(figsize=(10, 5))
n_bs = basis.Bs.shape[1]
for i in range(0, n_bs, step):
    vals = basis.Bs[:, i].toarray().flatten()
    ax.plot(s_grid, vals, label=f"Bs{i}", lw=1.5)
ax.set_xlabel("Asset Price (s)"); ax.set_ylabel("B(s)")
ax.set_title("1D B-Spline Basis Functions (s-direction)")
ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bspline_basis_s.png"), dpi=120, bbox_inches='tight')
plt.close()

print("3/9  B-Spline Basis Functions (v-direction)")
fig, ax = plt.subplots(figsize=(10, 5))
n_bv = basis.Bv.shape[1]
for i in range(0, n_bv, step):
    vals = basis.Bv[:, i].toarray().flatten()
    ax.plot(v_grid, vals, label=f"Bv{i}", lw=1.5)
ax.set_xlabel("Variance (v)"); ax.set_ylabel("B(v)")
ax.set_title("1D B-Spline Basis Functions (v-direction)")
ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bspline_basis_v.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 3. 2D B-Spline Basis Function ===
print("4/9  2D B-Spline Basis Function")
basis_2d = p_eu.basis_2d
s_idx_range = np.argmin(np.abs(s_grid - 120.0))
v_idx_range = np.argmin(np.abs(v_grid - 0.3))
col_idx = basis_2d.shape[1] // 2
two_d = basis_2d[:, col_idx].toarray().reshape(len(s_grid), len(v_grid))

S, V = np.meshgrid(s_grid[:s_idx_range], v_grid[:v_idx_range], indexing="ij")
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(S, V, two_d[:s_idx_range, :v_idx_range],
                       cmap="turbo", linewidth=0, antialiased=True, alpha=0.95)
ax.set_xlabel("Asset Price (s)"); ax.set_ylabel("Variance (v)"); ax.set_zlabel("Value")
ax.set_title("2D Tensor-Product B-Spline Basis Function")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bspline_2d_basis.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 4. Initial Condition ===
print("5/9  Initial Condition")
q0 = p_eu.initial_condition
delta_flat = basis_2d @ q0
delta_2d = np.array(delta_flat).reshape(len(s_grid), len(v_grid))
marginal_s = delta_2d @ gp.v_weights

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(s_grid, marginal_s, lw=2.5, color=PALETTE[0])
ax.set_xlim(55, 65)
ax.set_xlabel("Asset Price (s)"); ax.set_ylabel("Marginal Density")
ax.set_title("Initial Condition — s-direction Marginal Distribution")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "initial_condition.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 5. Terminal PDF (2D heatmap) ===
print("6/9  Terminal PDF Heatmaps")
pdf_eu = p_eu.pdf; pdf_bar = p_bar.pdf
gp_eu = p_eu.grid_points; gp_bar = p_bar.grid_points
s_idx_eu = np.argmin(np.abs(gp_eu.s - 100))
v_idx = np.argmin(np.abs(gp_eu.v - 0.3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
im1 = ax1.pcolormesh(gp_eu.s, gp_eu.v, pdf_eu.T, shading="auto", cmap="plasma")
ax1.set_xlabel("Asset Price (s)"); ax1.set_ylabel("Variance (v)")
ax1.set_title("European Call — Terminal PDF")
ax1.set_xlim(20, 100); ax1.set_ylim(0.0, 0.3)
fig.colorbar(im1, ax=ax1)

im2 = ax2.pcolormesh(gp_bar.s, gp_bar.v, pdf_bar.T, shading="auto", cmap="plasma")
ax2.set_xlabel("Asset Price (s)"); ax2.set_ylabel("Variance (v)")
ax2.set_title("Barrier Call — Terminal PDF")
ax2.set_xlim(50, 100); ax2.set_ylim(0.0, 0.3)
fig.colorbar(im2, ax=ax2)
fig.suptitle("Terminal PDF at Maturity (t=T)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "terminal_pdf_heatmap.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 6. Terminal PDF (3D) ===
print("7/9  Terminal PDF 3D")
s_idx = np.argmin(np.abs(gp_eu.s - 120.0))
v_idx = np.argmin(np.abs(gp_eu.v - 0.3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={"projection": "3d"})
S_eu, V_eu = np.meshgrid(gp_eu.s[:s_idx], gp_eu.v[:v_idx], indexing="ij")
surf1 = ax1.plot_surface(S_eu, V_eu, pdf_eu[:s_idx, :v_idx],
                         cmap="turbo", linewidth=0, antialiased=True, alpha=0.95)
ax1.set_xlabel("s"); ax1.set_ylabel("v")
ax1.set_title("European Call")

S_bar, V_bar = np.meshgrid(gp_bar.s[:s_idx], gp_bar.v[:v_idx], indexing="ij")
surf2 = ax2.plot_surface(S_bar, V_bar, pdf_bar[:s_idx, :v_idx],
                         cmap="turbo", linewidth=0, antialiased=True, alpha=0.95)
ax2.set_xlabel("s"); ax2.set_ylabel("v"); ax2.set_zlabel("Value")
ax2.set_title("Barrier Call")
fig.suptitle("Terminal PDF at Maturity (3D)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "terminal_pdf_3d.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 7. Marginal Distributions ===
print("8/9  Marginal Distributions")
s_weights = p_eu.grid_points.s_weights
v_weights = p_eu.grid_points.v_weights

df_marg_s = pl.concat([
    pl.DataFrame({"s": gp_eu.s, "density": pdf_eu @ v_weights, "type": "European"}),
    pl.DataFrame({"s": gp_bar.s, "density": pdf_bar @ v_weights, "type": "Barrier"}),
])
df_marg_v = pl.concat([
    pl.DataFrame({"v": gp_eu.v, "density": pdf_eu.T @ s_weights, "type": "European"}),
    pl.DataFrame({"v": gp_bar.v, "density": pdf_bar.T @ s_weights, "type": "Barrier"}),
])

fig, (axs, axv) = plt.subplots(1, 2, figsize=(14, 5))
for t, c in [("European", "#E41A1C"), ("Barrier", "#377EB8")]:
    s = df_marg_s.filter(pl.col("type") == t)
    axs.plot(s["s"], s["density"], label=t, color=c, lw=2.5)
axs.set_xlabel("Asset Price (s)"); axs.set_ylabel("Density")
axs.set_title("Marginal Distribution of Asset Price"); axs.legend()

for t, c in [("European", "#E41A1C"), ("Barrier", "#377EB8")]:
    s = df_marg_v.filter(pl.col("type") == t)
    axv.plot(s["v"], s["density"], label=t, color=c, lw=2.5)
axv.set_xlabel("Variance (v)"); axv.set_ylabel("Density")
axv.set_title("Marginal Distribution of Variance"); axv.legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "marginal_distributions.png"), dpi=120, bbox_inches='tight')
plt.close()

# === 8. Pricing Table ===
print("9/9  Pricing comparison data")
result_eu = price(**HESTON, n_s=38, n_v=38, K=[80, 90, 100], option_type='european_call')
result_bar = price(**HESTON, n_s=38, n_v=38, K=[80, 90, 100], barrier=50.0, option_type='down_and_out_call')

print("\nEuropean call prices:")
for i, K in enumerate([80, 90, 100]):
    print(f"  K={K}: {result_eu.prices[i]:.6f} (Δ={result_eu.deltas[i]:.6f}, Γ={result_eu.gammas[i]:.6f}, ν={result_eu.vegas[i]:.6f})")
print("Barrier call prices (Down-and-Out, B=50):")
for i, K in enumerate([80, 90, 100]):
    print(f"  K={K}: {result_bar.prices[i]:.6f}")

print(f"\nImages saved to {OUT}/")
