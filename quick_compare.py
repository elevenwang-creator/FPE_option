"""Quick comparison: Python FPE with n=18, sigma0=0.1"""
import sys, json
import numpy as np
from time import perf_counter

sys.path.insert(0, '/Users/knight/Agent/FPE_option')
from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

params = {
    'kappa': 1.2, 'theta': 0.05, 'sigma': 0.35, 'rho': -0.4,
    'r': 0.1, 'T': 0.6, 'S0': 60.0, 'V0': 0.1,
    'S_range': (50.0, 150.0), 'V_range': (0.0, 1.0),
}

n = 18
d = 3
normalize_S0 = (params['S0'] - params['S_range'][0]) / (params['S_range'][1] - params['S_range'][0])

s = GenerateKnots(n, d, method='non-uniform', center=round(normalize_S0, 3),
                  boundary=params['S_range'], mean=params['S0'], std=0.1).generate_knots()
s = s[1:]

v = GenerateKnots(n, d, method='non-uniform', center=0.1,
                  boundary=params['V_range'], mean=0.1, std=0.001).generate_knots()

degrees = [d, d]
knots_lists = [s, v]
conditions_list = [('dirichlet', 'newmann'), ('newmann', 'newmann')]

t0 = perf_counter()
fpe = HestonSolver(degrees, knots_lists, conditions_list, params=params)
t1 = perf_counter()
print(f"Init time: {t1-t0:.3f}s")
print(f"System size: {fpe.mass_matrix.shape}")
print(f"s_points: {len(fpe.s_points)}, v_points: {len(fpe.v_points)}")
print(f"s_points range: [{fpe.s_points.min():.2f}, {fpe.s_points.max():.2f}]")
print(f"v_points range: [{fpe.v_points.min():.4f}, {fpe.v_points.max():.4f}]")

solution = fpe.qt_ode(sigma0=0.1)
q0 = solution.y[:, 0]
q_T = solution.y[:, -1]
m = fpe.basis_integral

mass_0 = np.dot(m, q0)
mass_T = np.dot(m, q_T)
print(f"PDF integral at t=0: {mass_0:.10f}")
print(f"PDF integral at t=T: {mass_T:.10f}")

pdf_result = fpe.fpe_solver(sigma0=0.1)
pdf_2d, t_arr = pdf_result
pdf_init = pdf_2d[:, :, 0]
pdf_final = pdf_2d[:, :, -1]

sw = fpe.nodes_weights['s_weights']
vw = fpe.nodes_weights['v_weights']

e_s_0 = np.einsum('ij,i,j->', pdf_init * fpe.s_points[:, None], sw, vw) / np.einsum('ij,i,j->', pdf_init, sw, vw)
e_s_T = np.einsum('ij,i,j->', pdf_final * fpe.s_points[:, None], sw, vw) / np.einsum('ij,i,j->', pdf_final, sw, vw)
print(f"E[S] at t=0: {e_s_0:.6f} (should be S0={params['S0']})")
print(f"E[S] at t=T: {e_s_T:.6f} (should be S0*exp(rT)={params['S0']*np.exp(params['r']*params['T']):.6f})")

r = params['r']
T_val = params['T']
discount = np.exp(-r * T_val)
strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]

print(f"\nVanilla Call Prices:")
for K in strikes:
    marginal_v = np.einsum('ij,j->i', pdf_final, vw)
    payoff = np.maximum(fpe.s_points - K, 0.0)
    price = discount * np.dot(marginal_v * payoff, sw)
    print(f"  K={K:.0f}: {price:.6f}")
