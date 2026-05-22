import numpy as np
#import scipy.sparse as sp
from docs.python_reference.FPE_Solver_Final_Version import HestonSolver, GenerateKnots
from time import perf_counter

start_time = perf_counter()
params = {
    'V_range': (0.0, 1.0),
    'S_range': (50.0, 150.0),
    'kappa': 1.20,
    'theta': 0.05,
    'sigma': 0.35,
    'rho': -0.4,
    'r': 0.1,
    'T': 0.6,
    'V0': 0.1,
    'S0': 60.0,
}

degrees = [3, 3]
n_x = 38
n_v = 38

center_s = (params['S0'] - params['S_range'][0]) / (params['S_range'][1] - params['S_range'][0])
x_gen = GenerateKnots(n_x, 3, method='non-uniform', center=center_s,
                   boundary=(params['S_range'][0], params['S_range'][1]), mean=params['S0'], std=0.1)
x = x_gen.generate_knots()
#center_v = (params['V0'] - params['V_range'][0]) / (params['V_range'][1] - params['V_range'][0])
v_gen = GenerateKnots(n_v, 3, method='non-uniform', center=0.1,
                   boundary=(params['V_range'][0], params['V_range'][1]), mean=params['V0'], std=0.001)
v = v_gen.generate_knots()

print(f"Python x (before x[1:]): len={len(x)}")
print(f"Python v: len={len(v)}")

x = x[1:]
print(f"Python x (after x[1:]): len={len(x)}")

knots_list = [x, v]
conditions_list = []

fpe = HestonSolver(degrees, knots_list, conditions_list, params=params)

print(f"\nnum_basis_s: {len(x) - 3 - 1}")
print(f"num_basis_v: {len(v) - 3 - 1}")

time_matrix = perf_counter()
M = fpe.mass_matrix
K = fpe.stiffness_matrix
time_after_matrix = perf_counter()
print(f"Assembling Galerkin matrices time: {time_after_matrix - time_matrix:.6f}")
#q0 = fpe.q_initial(0.1)
#print(f"\nq0 len: {len(q0)}")
#print(f"q0 sum: {q0.sum():.10f}")

time_eval = np.array([0, 1, 10, 30, 216]) / 360
pdf_result = fpe.fpe_solver(0.1, time=None)
if pdf_result is not None:
    pdf_2d, t_arr = pdf_result
    pdf_T = pdf_2d[:, :, -1]
    pdf_0 = pdf_2d[:, :, 0]
    
    marginal_s = pdf_T @ fpe.nodes_weights['v_weights']
    prices = fpe.s_points
    E_S = np.dot(marginal_s * prices, fpe.nodes_weights['s_weights'])
    print(f"\nE[S] at T: {E_S:.10f}")
    print(f"Expected E[S]: {params['S0'] * np.exp(params['r'] * params['T']):.10f}")
    
    pdf_integral = pdf_T @ fpe.nodes_weights['v_weights'] @ fpe.nodes_weights['s_weights']
    print(f"PDF integral at T: {pdf_integral:.10f}")
    
    marginal_s_0 = pdf_0 @ fpe.nodes_weights['v_weights']
    E_S_0 = np.dot(marginal_s_0 * prices, fpe.nodes_weights['s_weights'])
    print(f"E[S] at t=0: {E_S_0:.10f}")
    
    pdf_integral_0 = pdf_0 @ fpe.nodes_weights['v_weights'] @ fpe.nodes_weights['s_weights']
    print(f"PDF integral at t=0: {pdf_integral_0:.10f}")

    print("Vanilla Call Prices:")
    for K_strike in [65, 70, 75, 80, 85, 90, 95, 100]:
        payoff = np.maximum(prices - K_strike, 0.0)
        call_price = np.exp(-params['r'] * params['T']) * np.dot(marginal_s * payoff, fpe.nodes_weights['s_weights'])
        print(f"  K={K_strike}: {call_price:.10f}")

end_time = perf_counter()
print(f"Python time: {end_time - start_time:.6f}")
