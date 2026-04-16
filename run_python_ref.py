import sys, numpy as np
sys.path.insert(0, '.')
from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

kappa, theta, sigma, rho = 1.2, 0.05, 0.35, -0.4
r, T, S0, V0 = 0.1, 0.6, 60.0, 0.1

n_s, n_v = 10, 10
p = 5
gk_s = GenerateKnots(n_s, p, method='uniform', boundary=(0, 1))
gk_v = GenerateKnots(n_v, p, method='uniform', boundary=(0, 1))
knots_s = gk_s.generate_knots()
knots_v = gk_v.generate_knots()

degrees = [p, p]
knots_list = [knots_s, knots_v]
conditions_list = [('dirichlet', 'newmann'), ('dirichlet', 'newmann')]

params = {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho,
          'r': r, 'T': T, 'S0': S0, 'V0': V0}

solver = HestonSolver(degrees, knots_list, conditions_list, params)
solution = solver.fpe_solver(0.1)

print("Python FPE results (n_s=10, n_v=10):")
for K_strike in [65, 70, 75, 80, 85, 90, 95, 100]:
    price = solver.price_vanilla_call(K_strike)
    print(f"  K={K_strike}: {price:.6f}")
