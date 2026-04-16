import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp

M_data = np.loadtxt('dump_M.mtx', skiprows=2)
K_data = np.loadtxt('dump_K.mtx', skiprows=2)
q0_s10 = np.loadtxt('dump_q0.txt')
q0_default = np.loadtxt('dump_q0_default.txt')
n = 196
M = coo_matrix((M_data[:,2], (M_data[:,0].astype(int)-1, M_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()
K = coo_matrix((K_data[:,2], (K_data[:,0].astype(int)-1, K_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()

M_lu = splu(M)
J = -M_lu.solve(K.toarray())

# Test with BOTH q0s
for label, q0 in [("sigma0=10.0", q0_s10), ("sigma0=2.0(default)", q0_default)]:
    sol = solve_ivp(lambda t,q: M_lu.solve(-K@q), [0, 0.6], q0, method='Radau', jac=J,
                    atol=1e-6, rtol=1e-4, max_step=0.1, first_step=1e-6)
    qT = sol.y[:,-1]
    
    # PDF integral via mass matrix
    ones = np.ones(n)
    pdf_mass_q0 = q0 @ (M @ ones)
    pdf_mass_qT = qT @ (M @ ones)
    
    # PDF integral via basis functions (approximate: q^T * m where m = integral of basis)
    # m[j] = sum_i(Phi[i,j] * w_i), but we can compute it as M * ones / (quadrature scaling)
    # Actually, let's just check q_sum and mass integral
    print(f'\n{label}:')
    print(f'  q0 sum={q0.sum():.6f}, q(T) sum={qT.sum():.6f}')
    print(f'  PDF mass q0={pdf_mass_q0:.6f}, PDF mass qT={pdf_mass_qT:.6f}')
    print(f'  scipy steps={len(sol.t)}')
