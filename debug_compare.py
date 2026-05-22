import numpy as np
import scipy.sparse as sp
from .docs.python_reference.FPE_Solver_Final_Version import HestonSolver, GenerateKnots

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
n_x = 42
n_v = 42

x_gen = GenerateKnots(n_x, 3, method='non-uniform', center=0.1,
                   boundary=(50.0, 150.0), mean=60.0, std=0.1)
x = x_gen.generate_knots()
v_gen = GenerateKnots(n_v, 3, method='non-uniform', center=0.1,
                   boundary=(0.0, 1.0), mean=0.1, std=0.001)
v = v_gen.generate_knots()

print(f"Python x (before x[1:]): len={len(x)}, first={x[0]:.10f}, last={x[-1]:.10f}")
print(f"Python v: len={len(v)}, first={v[0]:.10f}, last={v[-1]:.10f}")

x = x[1:]

print(f"Python x (after x[1:]): len={len(x)}, first={x[0]:.10f}, last={x[-1]:.10f}")

knots_list = [x, v]
conditions_list = []

fpe = HestonSolver(degrees, knots_list, conditions_list, params=params)

M = fpe.mass_matrix
K = fpe.stiffness_matrix

print("=== Matrix Statistics ===")
print(f"M shape: {M.shape}, nnz: {M.nnz}")
print(f"K shape: {K.shape}, nnz: {K.nnz}")
print(f"M diagonal sum: {M.diagonal().sum():.10f}")
print(f"K diagonal sum: {K.diagonal().sum():.10f}")
print(f"M Frobenius norm: {sp.linalg.norm(M):.10f}")
print(f"K Frobenius norm: {sp.linalg.norm(K):.10f}")
print(f"M trace: {M.diagonal().sum():.10f}")
print(f"K trace: {K.diagonal().sum():.10f}")

M_diag = M.diagonal()
D = sp.diags(np.sqrt(M_diag))
Dinv = sp.diags(1.0 / np.sqrt(M_diag))

M_s = Dinv @ M @ Dinv
K_s = Dinv @ K @ Dinv

print(f"\nM_scaled diagonal sum: {M_s.diagonal().sum():.10f}")
print(f"K_scaled diagonal sum: {K_s.diagonal().sum():.10f}")
print(f"M_scaled Frobenius norm: {sp.linalg.norm(M_s):.10f}")
print(f"K_scaled Frobenius norm: {sp.linalg.norm(K_s):.10f}")

print(f"\nnum_basis_s: {len(x) - 3 - 1}")
print(f"num_basis_v: {len(v) - 3 - 1}")
print(f"s_knots len: {len(x)}")
print(f"v_knots len: {len(v)}")
print(f"s_points len: {len(fpe.s_points)}")
print(f"v_points len: {len(fpe.v_points)}")

print(f"\ns_knots[:5]: {x[:5]}")
print(f"s_knots[-5:]: {x[-5:]}")
print(f"v_knots[:5]: {v[:5]}")
print(f"v_knots[-5:]: {v[-5:]}")

print(f"\ns_points[:5]: {fpe.s_points[:5]}")
print(f"s_points[-5:]: {fpe.s_points[-5:]}")
print(f"v_points[:5]: {fpe.v_points[:5]}")
print(f"v_points[-5:]: {fpe.v_points[-5:]}")

print(f"\nK[0,:5] nonzero: {K[0,:5].toarray()}")
print(f"K[-1,-5:] nonzero: {K[-1,-5:].toarray()}")
print(f"M[0,:5] nonzero: {M[0,:5].toarray()}")
print(f"M[-1,-5:] nonzero: {M[-1,-5:].toarray()}")

K_row_sums = np.array(K.sum(axis=1)).flatten()
M_row_sums = np.array(M.sum(axis=1)).flatten()
print(f"\nK row sums - min: {K_row_sums.min():.10f}, max: {K_row_sums.max():.10f}, mean: {K_row_sums.mean():.10f}")
print(f"M row sums - min: {M_row_sums.min():.10f}, max: {M_row_sums.max():.10f}, mean: {M_row_sums.mean():.10f}")

K_col_sums = np.array(K.sum(axis=0)).flatten()
M_col_sums = np.array(M.sum(axis=0)).flatten()
print(f"K col sums - min: {K_col_sums.min():.10f}, max: {K_col_sums.max():.10f}, mean: {K_col_sums.mean():.10f}")
print(f"M col sums - min: {M_col_sums.min():.10f}, max: {M_col_sums.max():.10f}, mean: {M_col_sums.mean():.10f}")

q0 = fpe.q_initial(0.1)
print(f"\nq0 len: {len(q0)}")
print(f"q0 sum: {q0.sum():.10f}")
print(f"q0[:5]: {q0[:5]}")

q0_s = D @ q0
print(f"q0_scaled sum: {q0_s.sum():.10f}")
print(f"q0_scaled[:5]: {q0_s[:5]}")

from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu, LinearOperator

M_lu = splu(M_s.tocsc())
M_x = LinearOperator(M_s.shape, lambda x: M_lu.solve(x))

def constrained_rhs(t, q):
    Kq = -K_s @ q
    dqdt = M_x.matvec(Kq)
    return dqdt

jacobian = -M_x.matmat(K_s.toarray())

solver_options = {
    'method': 'Radau',
    'atol': 1e-6,
    'rtol': 1e-4,
    'max_step': 0.1,
    'first_step': 1e-6,
    'jac': jacobian,
    't_eval': [0.0, params['T']],
}

solution = solve_ivp(constrained_rhs, (0.0, params['T']), q0_s, **solver_options)
q_T_s = solution.y[:, -1]
q_T = Dinv @ q_T_s

print(f"\nODE solution success: {solution.success}")
print(f"ODE steps: {len(solution.t)}")

pdf = fpe.fpe_solver(0.1, time=[0.0, params['T']])
if pdf is not None:
    pdf_T = pdf[:, :, -1]
    marginal_s = pdf_T @ fpe.nodes_weights['v_weights']
    prices = fpe.s_points
    E_S = np.dot(marginal_s, prices)
    print(f"\nE[S] at T: {E_S:.10f}")
    print(f"Expected E[S]: {params['S0'] * np.exp(params['r'] * params['T']):.10f}")
    
    pdf_integral = np.sum(pdf_T * fpe.integ_weights.toarray())
    print(f"PDF integral at T: {pdf_integral:.10f}")
