"""
FPE Solver Comparison: Python vs Mojo
=====================================
Comprehensive error analysis and performance comparison.

Heston Parameters (common):
  kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.6, S0=60, V0=0.1
  S_range=[50,150], V_range=[1e-4,1.0]

Monte Carlo reference (barrier down-and-out call, barrier=50):
  K=65: 4.716, K=70: 2.939, K=75: 1.729, K=80: 0.973
  K=85: 0.531, K=90: 0.285, K=95: 0.152, K=100: 0.0805
"""
import json
import numpy as np

mc_prices = {65: 4.716, 70: 2.939, 75: 1.729, 80: 0.973,
             85: 0.531, 90: 0.285, 95: 0.152, 100: 0.0805}

# Python results (from compare_fpe.py output)
py_n21 = {
    'label': 'Py n=21 uniform',
    'system_size': 225,
    'ode_time': 7.832, 'fpe_time': 0.783, 'total_time': 10.580,
    'ode_steps': 22,
    'pdf_int_0': 1.0, 'pdf_int_T': 0.6246,
    'e_s_0': 60.276, 'e_s_T': 71.211,
    'call_prices': {65: 4.717, 70: 3.060, 75: 1.913, 80: 1.168,
                    85: 0.703, 90: 0.421, 95: 0.252, 100: 0.151},
}
py_n18 = {
    'label': 'Py n=18 non-uniform',
    'system_size': 576,
    'ode_time': 8.502, 'fpe_time': 2.395, 'total_time': 12.523,
    'ode_steps': 76,
    'pdf_int_0': 1.0, 'pdf_int_T': 0.6314,
    'e_s_0': 60.000, 'e_s_T': 70.654,
    'call_prices': {65: 4.435, 70: 2.765, 75: 1.628, 80: 0.917,
                    85: 0.500, 90: 0.268, 95: 0.144, 100: 0.078},
}

# Mojo results (from single_price.mojo output)
mojo_n21 = {
    'label': 'Mojo n=21 uniform',
    'system_size': 576,
    'ode_time': 360.0, 'total_time': 374.0,
    'ode_steps': 36,
    'pdf_int_0': 1.0, 'pdf_int_T': 1.0,
    'e_s_0': 62.868, 'e_s_T': 71.174,
    'call_prices': {65: 8.793, 70: 6.483, 75: 4.711, 80: 3.391,
                    85: 2.459, 90: 1.798, 95: 1.307, 100: 0.950},
}
mojo_n11 = {
    'label': 'Mojo n=11 uniform',
    'system_size': 196,
    'ode_time': 15.0, 'total_time': 17.5,
    'ode_steps': 43,
    'pdf_int_0': 1.0, 'pdf_int_T': 1.0,
    'e_s_0': 71.245, 'e_s_T': 78.411,
    'call_prices': {65: 14.919, 70: 12.089, 75: 9.259, 80: 7.491,
                    85: 5.962, 90: 4.432, 95: 3.094, 100: 2.495},
}

S0, r, T = 60.0, 0.1, 0.6
expected_e_s_T = S0 * np.exp(r * T)

print("=" * 80)
print("  FPE SOLVER COMPARISON: Python vs Mojo")
print("  Heston: kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.6")
print("=" * 80)

# === 1. FPE Solution Quality ===
print("\n" + "=" * 80)
print("  1. FPE SOLUTION QUALITY")
print("=" * 80)
print(f"\n  {'Metric':<30} {'Py n=21':>12} {'Py n=18':>12} {'Mojo n=21':>12} {'Mojo n=11':>12}")
print("-" * 80)
print(f"  {'System size':<30} {py_n21['system_size']:>12} {py_n18['system_size']:>12} {mojo_n21['system_size']:>12} {mojo_n11['system_size']:>12}")
print(f"  {'ODE steps':<30} {py_n21['ode_steps']:>12} {py_n18['ode_steps']:>12} {mojo_n21['ode_steps']:>12} {mojo_n11['ode_steps']:>12}")
print(f"  {'PDF integral t=0':<30} {py_n21['pdf_int_0']:>12.6f} {py_n18['pdf_int_0']:>12.6f} {mojo_n21['pdf_int_0']:>12.6f} {mojo_n11['pdf_int_0']:>12.6f}")
print(f"  {'PDF integral t=T':<30} {py_n21['pdf_int_T']:>12.6f} {py_n18['pdf_int_T']:>12.6f} {mojo_n21['pdf_int_T']:>12.6f} {mojo_n11['pdf_int_T']:>12.6f}")
print(f"  {'Mass loss (%)':<30} {(1-py_n21['pdf_int_T'])*100:>11.2f}% {(1-py_n18['pdf_int_T'])*100:>11.2f}% {(1-mojo_n21['pdf_int_T'])*100:>11.2f}% {(1-mojo_n11['pdf_int_T'])*100:>11.2f}%")
print()
print(f"  {'E[S] at t=0':<30} {py_n21['e_s_0']:>12.3f} {py_n18['e_s_0']:>12.3f} {mojo_n21['e_s_0']:>12.3f} {mojo_n11['e_s_0']:>12.3f}")
print(f"  {'E[S] at t=0 error':<30} {py_n21['e_s_0']-S0:>12.3f} {py_n18['e_s_0']-S0:>12.3f} {mojo_n21['e_s_0']-S0:>12.3f} {mojo_n11['e_s_0']-S0:>12.3f}")
print(f"  {'E[S] at t=T':<30} {py_n21['e_s_T']:>12.3f} {py_n18['e_s_T']:>12.3f} {mojo_n21['e_s_T']:>12.3f} {mojo_n11['e_s_T']:>12.3f}")
print(f"  {'E[S] at t=T error':<30} {py_n21['e_s_T']-expected_e_s_T:>12.3f} {py_n18['e_s_T']-expected_e_s_T:>12.3f} {mojo_n21['e_s_T']-expected_e_s_T:>12.3f} {mojo_n11['e_s_T']-expected_e_s_T:>12.3f}")
print(f"  {'Expected E[S] at t=T':<30} {expected_e_s_T:>12.3f}")

# === 2. Option Price Comparison ===
print("\n" + "=" * 80)
print("  2. VANILLA CALL OPTION PRICES (discounted)")
print("=" * 80)
strikes = [65, 70, 75, 80, 85, 90, 95, 100]
print(f"\n  {'Strike':>6} {'MC(ref)':>10} {'Py n=21':>10} {'Py n=18':>10} {'Mojo n=21':>10} {'Mojo n=11':>10}")
print("-" * 70)
for K in strikes:
    mc = mc_prices[K]
    p21 = py_n21['call_prices'][K]
    p18 = py_n18['call_prices'][K]
    m21 = mojo_n21['call_prices'][K]
    m11 = mojo_n11['call_prices'][K]
    print(f"  {K:>6} {mc:>10.4f} {p21:>10.4f} {p18:>10.4f} {m21:>10.4f} {m11:>10.4f}")

# === 3. Error Analysis vs MC ===
print("\n" + "=" * 80)
print("  3. ERROR ANALYSIS vs Monte Carlo Reference")
print("  (Note: MC prices are for barrier down-and-out call, barrier=50)")
print("  (FPE prices are for vanilla call - not directly comparable)")
print("=" * 80)
print(f"\n  {'Strike':>6} {'Py n=21 err':>12} {'Py n=18 err':>12} {'Mojo n=21 err':>14} {'Mojo n=11 err':>14}")
print("-" * 70)
py21_errs, py18_errs, mj21_errs, mj11_errs = [], [], [], []
for K in strikes:
    mc = mc_prices[K]
    p21 = py_n21['call_prices'][K]
    p18 = py_n18['call_prices'][K]
    m21 = mojo_n21['call_prices'][K]
    m11 = mojo_n11['call_prices'][K]
    e21 = abs(p21 - mc) / mc * 100
    e18 = abs(p18 - mc) / mc * 100
    em21 = abs(m21 - mc) / mc * 100
    em11 = abs(m11 - mc) / mc * 100
    py21_errs.append(e21)
    py18_errs.append(e18)
    mj21_errs.append(em21)
    mj11_errs.append(em11)
    print(f"  {K:>6} {e21:>11.2f}% {e18:>11.2f}% {em21:>13.2f}% {em11:>13.2f}%")

print(f"\n  {'Mean rel err':>14} {np.mean(py21_errs):>11.2f}% {np.mean(py18_errs):>11.2f}% {np.mean(mj21_errs):>13.2f}% {np.mean(mj11_errs):>13.2f}%")

# === 4. Normalized comparison (adjust for mass loss) ===
print("\n" + "=" * 80)
print("  4. NORMALIZED CALL PRICES (adjusted for PDF integral)")
print("  Python prices divided by pdf_int_T to compensate mass loss")
print("=" * 80)
print(f"\n  {'Strike':>6} {'MC(ref)':>10} {'Py n=21 norm':>13} {'Py n=18 norm':>13} {'Mojo n=21':>10} {'Mojo n=11':>10}")
print("-" * 75)
for K in strikes:
    mc = mc_prices[K]
    p21n = py_n21['call_prices'][K] / py_n21['pdf_int_T']
    p18n = py_n18['call_prices'][K] / py_n18['pdf_int_T']
    m21 = mojo_n21['call_prices'][K]
    m11 = mojo_n11['call_prices'][K]
    print(f"  {K:>6} {mc:>10.4f} {p21n:>13.4f} {p18n:>13.4f} {m21:>10.4f} {m11:>10.4f}")

# === 5. Performance Analysis ===
print("\n" + "=" * 80)
print("  5. PERFORMANCE ANALYSIS")
print("=" * 80)
print(f"\n  {'Metric':<30} {'Py n=21':>12} {'Py n=18':>12} {'Mojo n=21':>12} {'Mojo n=11':>12}")
print("-" * 80)
print(f"  {'System size':<30} {py_n21['system_size']:>12} {py_n18['system_size']:>12} {mojo_n21['system_size']:>12} {mojo_n11['system_size']:>12}")
print(f"  {'ODE time (s)':<30} {py_n21['ode_time']:>12.2f} {py_n18['ode_time']:>12.2f} {mojo_n21['ode_time']:>12.2f} {mojo_n11['ode_time']:>12.2f}")
print(f"  {'Total time (s)':<30} {py_n21['total_time']:>12.2f} {py_n18['total_time']:>12.2f} {mojo_n21['total_time']:>12.2f} {mojo_n11['total_time']:>12.2f}")
print(f"  {'ODE steps':<30} {py_n21['ode_steps']:>12} {py_n18['ode_steps']:>12} {mojo_n21['ode_steps']:>12} {mojo_n11['ode_steps']:>12}")

# Speedup for comparable system sizes
py_576_time = py_n18['total_time']
mojo_576_time = mojo_n21['total_time']
print(f"\n  Same system size (576x576):")
print(f"    Python (n=18 non-uniform): {py_576_time:.2f}s")
print(f"    Mojo   (n=21 uniform):     {mojo_576_time:.2f}s")
print(f"    Python is {mojo_576_time/py_576_time:.1f}x FASTER than Mojo")

# === 6. Key Findings ===
print("\n" + "=" * 80)
print("  6. KEY FINDINGS & DIAGNOSIS")
print("=" * 80)
print("""
  A. MASS CONSERVATION:
     - Mojo: PDF integral preserved at 1.0 (excellent, error < 1e-13)
     - Python: PDF integral drops to ~0.63 (37% mass loss!)
     - Python's fpe_solver clips negative values, breaking mass conservation
     - Mojo's RADAU5 preserves mass by not clipping

  B. E[S] DRIFT (common issue):
     - Both implementations show E[S] drift at t=T
     - Expected: E[S] = S0*exp(rT) = 63.71
     - Python n=21: E[S] = 71.21 (error +11.8%)
     - Mojo  n=21: E[S] = 71.17 (error +11.7%)
     - Root cause: boundary effects + insufficient resolution near S0
     - Python n=18 non-uniform: E[S] = 70.65 (better, +10.9%)

  C. INITIAL CONDITION QUALITY:
     - Python n=18 non-uniform: E[S] at t=0 = 60.00 (perfect!)
     - Python n=21 uniform:     E[S] at t=0 = 60.28 (good)
     - Mojo  n=21 uniform:      E[S] at t=0 = 62.87 (bias +4.8%)
     - Mojo  n=11 uniform:      E[S] at t=0 = 71.24 (bias +18.7%)
     - Mojo's NNLS projection biases E[S] upward with uniform knots

  D. OPTION PRICING:
     - Python prices are closer to MC (barrier) reference
     - But comparison is vanilla vs barrier (not apples-to-apples)
     - After normalizing for mass loss, Python and Mojo n=21 prices
       are more similar but both overestimate (due to E[S] drift)

  E. PERFORMANCE:
     - Python is ~30x faster for same system size (576x576)
     - Python: scipy Radau with dense Jacobian + splu
     - Mojo: custom RADAU5 with SparseLU (currently slow)
     - Mojo SparseLU needs optimization for larger systems

  F. RECOMMENDATIONS:
     1. Implement non-uniform knots in Mojo (concentrate near S0)
     2. Implement RecombinationBasis for boundary conditions
     3. Optimize SparseLU factorization for performance
     4. Fix initial condition to enforce E[S] = S0 constraint
     5. Add barrier option pricing for direct MC comparison
""")
