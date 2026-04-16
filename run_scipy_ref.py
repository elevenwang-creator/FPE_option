import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu
import scipy.sparse as sp
from scipy.special import roots_legendre
from scipy.stats import multivariate_normal

class HestonParams:
    def __init__(self, kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.05, T=0.6, S0=60.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0):
        self.kappa=kappa; self.theta=theta; self.sigma=sigma; self.rho=rho
        self.r=r; self.T=T; self.S0=S0; self.V0=V0
        self.S_min=S_min; self.S_max=S_max; self.V_min=V_min; self.V_max=V_max

params = HestonParams()

# Use the full Python reference
import sys
sys.path.insert(0, '.')
from FPE_Solver_Final_Version import *

domain = FPEDomain(params, n_s=11, n_v=11)
assembler = GalerkinAssembler(domain, params)
M, K = assembler.assemble()
q0 = assembler.initial_condition(sigma0=10.0)

M_scaled, K_scaled, D = assembler.scale_matrices(M, K)
M_x = splu(M_scaled.tocsc())
jacobian = - M_x.matmat(K_scaled.toarray())

def constrained_rhs(t, q):
    return M_x.matvec(-K_scaled @ q)

sol = solve_ivp(constrained_rhs, [0, params.T], q0, method='Radau', jac=jacobian, atol=1e-6, rtol=1e-4, max_step=0.1, first_step=1e-6)
print(f'scipy steps: {len(sol.t)}')
print(f'scipy q(T) sum: {sum(sol.y[:,-1]):.6f}')

Phi = assembler.eval_basis()
q_final = sol.y[:,-1]
pdf_vals = Phi @ q_final
pdf_grid = pdf_vals.reshape(domain.n_s_quad, domain.n_v_quad)
pdf_integral = np.sum(pdf_grid * np.outer(domain.s_weights, domain.v_weights))
print(f'scipy PDF integral: {pdf_integral:.6f}')

K_strike = 65.0
call_payoff = np.maximum(domain.s_points - K_strike, 0)
put_payoff = np.maximum(K_strike - domain.s_points, 0)
call_price = 0.0
put_price = 0.0
for i in range(domain.n_s_quad):
    for j in range(domain.n_v_quad):
        call_price += call_payoff[i] * pdf_grid[i,j] * domain.s_weights[i] * domain.v_weights[j]
        put_price += put_payoff[i] * pdf_grid[i,j] * domain.s_weights[i] * domain.v_weights[j]
print(f'scipy Call K={K_strike}: {call_price:.6f}')
print(f'scipy Put  K={K_strike}: {put_price:.6f}')
parity = call_price - put_price
intrinsic = params.S0 - K_strike * np.exp(-params.r * params.T)
print(f'scipy C-P = {parity:.6f}, S0-K*exp(-rT) = {intrinsic:.6f}, error = {abs(parity-intrinsic):.6f}')
