import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

M_data = np.loadtxt('dump_M.mtx', skiprows=2)
K_data = np.loadtxt('dump_K.mtx', skiprows=2)
q0 = np.loadtxt('dump_q0.txt')

n = 196

rows_M = M_data[:, 0].astype(int) - 1
cols_M = M_data[:, 1].astype(int) - 1
vals_M = M_data[:, 2]
M = coo_matrix((vals_M, (rows_M, cols_M)), shape=(n, n)).tocsc()

rows_K = K_data[:, 0].astype(int) - 1
cols_K = K_data[:, 1].astype(int) - 1
vals_K = K_data[:, 2]
K = coo_matrix((vals_K, (rows_K, cols_K)), shape=(n, n)).tocsc()

print(f'M shape: {M.shape}, nnz: {M.nnz}')
print(f'K shape: {K.shape}, nnz: {K.nnz}')
print(f'q0 shape: {q0.shape}, sum: {q0.sum():.6f}')

M_lu = splu(M)
J = -M_lu.solve(K.toarray())
print(f'J shape: {J.shape}, nnz: {np.count_nonzero(J)}')

def rhs(t, q):
    return M_lu.solve(-K @ q)

T_end = 0.6
sol = solve_ivp(rhs, [0, T_end], q0, method='Radau', jac=J,
                atol=1e-6, rtol=1e-4, max_step=0.1, first_step=1e-6)
print(f'scipy steps: {len(sol.t)}')
print(f'scipy q(T) sum: {sol.y[:,-1].sum():.6f}')
print(f'scipy q(T) max: {np.max(np.abs(sol.y[:,-1])):.6f}')

# Load domain info for PDF computation
with open('dump_domain.txt', 'r') as f:
    n_s = int(f.readline().strip())
    s_points = []
    s_weights = []
    for _ in range(n_s):
        parts = f.readline().strip().split()
        s_points.append(float(parts[0]))
        s_weights.append(float(parts[1]))
    n_v = int(f.readline().strip())
    v_points = []
    v_weights = []
    for _ in range(n_v):
        parts = f.readline().strip().split()
        v_points.append(float(parts[0]))
        v_weights.append(float(parts[1]))

s_points = np.array(s_points)
s_weights = np.array(s_weights)
v_points = np.array(v_points)
v_weights = np.array(v_weights)

print(f'\nDomain: {n_s} S-points, {n_v} V-points')
print(f'S range: [{s_points[0]:.2f}, {s_points[-1]:.2f}]')
print(f'V range: [{v_points[0]:.6f}, {v_points[-1]:.6f}]')

# Simple PDF: q(T) reshaped and integrated
# The PDF is p(S,V) = sum_i q_i * phi_i(S,V)
# For a quick check, just compute the integral using the mass matrix:
# integral(p) = q^T M 1 where 1 is the vector of ones
ones = np.ones(n)
pdf_integral_mass = sol.y[:,-1] @ (M @ ones)
print(f'\nPDF integral (via M*ones): {pdf_integral_mass:.6f}')

# Also check: q0^T M 1
pdf_integral_q0 = q0 @ (M @ ones)
print(f'q0 integral (via M*ones): {pdf_integral_q0:.6f}')

# Check conservation: d/dt(q^T M 1) = (M dq/dt)^T 1 = (-K q)^T 1 = -q^T K^T 1
K_ones = K.T @ ones
print(f'K^T * 1 min/max: {K_ones.min():.6e} / {K_ones.max():.6e}')
print(f'Sum of K^T * 1: {K_ones.sum():.6e}')

# If K^T * 1 ≈ 0, then probability is conserved
# Check row sums of K
K_row_sums = np.array(K.sum(axis=1)).flatten()
print(f'K row sums min/max: {K_row_sums.min():.6e} / {K_row_sums.max():.6e}')

# Pricing
K_strike = 65.0
S0 = 60.0; r = 0.05; T = 0.6
call_payoff = np.maximum(s_points - K_strike, 0)
put_payoff = np.maximum(K_strike - s_points, 0)

# Simple integration assuming q maps to a grid
# This is approximate - the real PDF needs basis evaluation
# But we can check put-call parity with the mass matrix approach
# C - P = S0 - K*exp(-rT)
intrinsic = S0 - K_strike * np.exp(-r * T)
print(f'\nPut-Call Parity reference: S0 - K*exp(-rT) = {intrinsic:.6f}')

# Step-by-step comparison: first step
h = 1e-6
q_first = sol.y[:,0]
print(f'\nFirst step comparison:')
print(f'scipy q after step 0: max={np.max(np.abs(sol.y[:,1])):.10f}')
print(f'Mojo  q after step 0: need to check')

# Print scipy step sizes
print(f'\nscipy step sizes:')
for i in range(min(5, len(sol.t)-1)):
    print(f'  step {i}: h={sol.t[i+1]-sol.t[i]:.6e}')
