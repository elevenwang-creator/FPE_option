"""
Compare Python FPE solver results with Mojo results.
Runs HestonSolver from FPE_Solver_Final_Version.py with two parameter sets:
  1) Matched to current Mojo params (r=0.1, n=21 uniform)
  2) Notebook params (r=0.1, n=18 non-uniform)
Outputs: q(T) stats, PDF integral, E[S], option prices, timing.
"""
import sys, json
import numpy as np
from time import perf_counter

sys.path.insert(0, '/Users/knight/Agent/FPE_option')
from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

def run_fpe_comparison(params, knots_config, sigma0, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    degrees = knots_config['degrees']
    knots_lists = knots_config['knots_lists']
    conditions_list = knots_config['conditions_list']

    t0 = perf_counter()
    fpe = HestonSolver(degrees, knots_lists, conditions_list, params=params)
    t1 = perf_counter()
    print(f"  Init time: {t1-t0:.3f}s")
    print(f"  System size: {fpe.mass_matrix.shape}")

    t2 = perf_counter()
    solution = fpe.qt_ode(sigma0=sigma0)
    t3 = perf_counter()
    ode_time = t3 - t2
    print(f"  ODE solve time: {ode_time:.3f}s")
    print(f"  ODE steps: {len(solution.t)}")

    q_T = solution.y[:, -1]
    q_0 = solution.y[:, 0]
    m = fpe.basis_integral
    mass_int_T = np.dot(m, q_T)
    mass_int_0 = np.dot(m, q_0)
    print(f"  PDF integral at t=0 (m^T q0): {mass_int_0:.10f}")
    print(f"  PDF integral at t=T (m^T qT): {mass_int_T:.10f}")

    t4 = perf_counter()
    pdf_result = fpe.fpe_solver(sigma0=sigma0)
    t5 = perf_counter()
    fpe_time = t5 - t4
    print(f"  FPE solver time: {fpe_time:.3f}s")

    pdf_2d, t_arr = pdf_result
    pdf_init = pdf_2d[:, :, 0]
    pdf_final = pdf_2d[:, :, -1]

    s_weights = fpe.nodes_weights['s_weights']
    v_weights = fpe.nodes_weights['v_weights']

    pdf_int_0 = np.einsum('ij,i,j->', pdf_init, s_weights, v_weights)
    pdf_int_T = np.einsum('ij,i,j->', pdf_final, s_weights, v_weights)
    print(f"  PDF integral at t=0 (numerical): {pdf_int_0:.10f}")
    print(f"  PDF integral at t=T (numerical): {pdf_int_T:.10f}")

    e_s_0 = np.einsum('ij,i,j->', pdf_init * fpe.s_points[:, None], s_weights, v_weights) / pdf_int_0
    e_s_T = np.einsum('ij,i,j->', pdf_final * fpe.s_points[:, None], s_weights, v_weights) / pdf_int_T
    r = params.get('r', 0.1)
    T_val = params.get('T', 0.6)
    S0 = params.get('S0', 60.0)
    print(f"  E[S] at t=0: {e_s_0:.6f} (should be S0={S0})")
    print(f"  E[S] at t=T: {e_s_T:.6f} (should be S0*exp(rT)={S0*np.exp(r*T_val):.6f})")

    tau = T_val
    strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    prices_s = fpe.s_points

    print(f"\n  Vanilla Call Prices (r={r}, T={T_val}):")
    print(f"  {'Strike':>8}  {'Price':>12}")
    call_prices = {}
    for K in strikes:
        payoff = np.maximum(prices_s - K, 0.0)
        marginal_v = np.einsum('ij,j->i', pdf_final, v_weights)
        price = np.exp(-r * tau) * np.dot(marginal_v * payoff, s_weights)
        call_prices[K] = price
        print(f"  {K:8.1f}  {price:12.6f}")

    total_time = t5 - t0
    print(f"\n  Total time: {total_time:.3f}s")

    return {
        'label': label,
        'ode_time': ode_time,
        'fpe_time': fpe_time,
        'total_time': total_time,
        'ode_steps': len(solution.t),
        'mass_integral_0': float(mass_int_0),
        'mass_integral_T': float(mass_int_T),
        'pdf_integral_0': float(pdf_int_0),
        'pdf_integral_T': float(pdf_int_T),
        'e_s_0': float(e_s_0),
        'e_s_T': float(e_s_T),
        'call_prices': {str(k): float(v) for k, v in call_prices.items()},
        'system_size': fpe.mass_matrix.shape[0],
    }


if __name__ == '__main__':
    results = []

    # --- Config 1: Match Mojo params (r=0.1, uniform knots, n=21) ---
    params1 = {
        'kappa': 1.2, 'theta': 0.05, 'sigma': 0.35, 'rho': -0.4,
        'r': 0.1, 'T': 0.6, 'S0': 60.0, 'V0': 0.1,
        'S_range': (50.0, 150.0), 'V_range': (1e-4, 1.0),
    }
    n1 = 21
    d = 3
    x1 = GenerateKnots(n1, d, method='uniform', boundary=(50.0, 150.0)).generate_knots()
    v1 = GenerateKnots(n1, d, method='uniform', boundary=(1e-4, 1.0)).generate_knots()
    knots1 = {
        'degrees': [d, d],
        'knots_lists': [x1, v1],
        'conditions_list': [['dirichlet', 'newmann'], ['newmann', 'newmann']],
    }
    res1 = run_fpe_comparison(params1, knots1, sigma0=2.0, label="Python (r=0.1, n=21 uniform, sigma0=2)")
    results.append(res1)

    # --- Config 2: Notebook params (r=0.1, non-uniform knots, n=18) ---
    params2 = {
        'kappa': 1.2, 'theta': 0.05, 'sigma': 0.35, 'rho': -0.4,
        'r': 0.1, 'T': 0.6, 'S0': 60.0, 'V0': 0.1,
        'S_range': (50.0, 150.0), 'V_range': (0.0, 1.0),
    }
    n2 = 18
    x2 = GenerateKnots(n2, d, method='non-uniform', center=0.1,
                        boundary=(50.0, 150.0), mean=60.0, std=0.1).generate_knots()
    v2 = GenerateKnots(n2, d, method='non-uniform', center=0.1,
                        boundary=(0.0, 1.0), mean=0.1, std=0.001).generate_knots()
    knots2 = {
        'degrees': [d, d],
        'knots_lists': [x2, v2],
        'conditions_list': [['dirichlet', 'newmann'], ['newmann', 'newmann']],
    }
    res2 = run_fpe_comparison(params2, knots2, sigma0=0.1, label="Python (r=0.1, n=18 non-uniform, sigma0=0.1)")
    results.append(res2)

    with open('/Users/knight/Agent/FPE_option/python_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to python_results.json")
