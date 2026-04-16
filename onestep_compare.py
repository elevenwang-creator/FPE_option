import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

M_data = np.loadtxt('dump_M.mtx', skiprows=2)
K_data = np.loadtxt('dump_K.mtx', skiprows=2)
q0 = np.loadtxt('dump_q0.txt')
n = 196

M = coo_matrix((M_data[:,2], (M_data[:,0].astype(int)-1, M_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()
K = coo_matrix((K_data[:,2], (K_data[:,0].astype(int)-1, K_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()

M_lu = splu(M)
J = -M_lu.solve(K.toarray())

# scipy reference: one step
sol = solve_ivp(lambda t,q: M_lu.solve(-K@q), [0, 1e-6], q0, method='Radau', jac=J,
                atol=1e-6, rtol=1e-4, max_step=0.1, first_step=1e-6, dense_output=True)
q_scipy = sol.y[:,-1]
print(f'scipy q(1e-6) sum={q_scipy.sum():.10f} max={np.max(np.abs(q_scipy)):.10f}')

# Our Schur approach: one step manually
sqrt6 = np.sqrt(6.0)
b1 = (16-sqrt6)/36; b2 = (16+sqrt6)/36; b3 = 1.0/9

Q = np.array([
    [0.13866510875190752, 0.04627814930949071, 0.98925745916384644],
    [-0.22964124235174019, -0.97017888655183304, 0.07757466016809490],
    [-0.96334671195056865, 0.23793121061671349, 0.12390258911134427]
])
T = np.array([
    [0.16255558520216112, 0.51074439865233900, -0.47719467969124402],
    [-0.06697332890760048, 0.16255558520216112, -0.28529656780973917],
    [0.0, 0.0, 0.27488882959567795]
])

t22 = T[2,2]; t00=T[0,0]; t01=T[0,1]; t02=T[0,2]
t10=T[1,0]; t11=T[1,1]; t12=T[1,2]
qt0 = Q[0,0]+Q[1,0]+Q[2,0]
qt1 = Q[0,1]+Q[1,1]+Q[2,1]
qt2 = Q[0,2]+Q[1,2]+Q[2,2]

h = 1e-6

# Diagonal scaling
M_diag = np.array(M.todense()).diagonal()
D = np.sqrt(np.abs(M_diag))
D[D < 1e-14] = 1.0
Dinv = 1.0/D

# Scaled matrices
M_s = M.toarray() * np.outer(Dinv, Dinv)
K_s = K.toarray() * np.outer(Dinv, Dinv)
y = D * q0  # scaled state

# Real block: (M_s + h*t22*K_s)*z2 = -qt2*K_s*y
A_real = M_s + h*t22*K_s
w = K_s @ y
rhs_real = -qt2 * w
z2 = np.linalg.solve(A_real, rhs_real)

# Complex block
w2 = K_s @ z2
rhs_complex = np.zeros(2*n)
rhs_complex[:n] = -qt0*w - h*t02*w2
rhs_complex[n:] = -qt1*w - h*t12*w2

A_complex = np.zeros((2*n, 2*n))
A_complex[:n,:n] = M_s + h*t00*K_s
A_complex[:n,n:] = h*t01*K_s
A_complex[n:,:n] = h*t10*K_s
A_complex[n:,n:] = M_s + h*t11*K_s

z_01 = np.linalg.solve(A_complex, rhs_complex)

# Recover k = Q*z
z0 = z_01[:n]; z1 = z_01[n:]
k1 = Q[0,0]*z0 + Q[0,1]*z1 + Q[0,2]*z2
k2 = Q[1,0]*z0 + Q[1,1]*z1 + Q[1,2]*z2
k3 = Q[2,0]*z0 + Q[2,1]*z1 + Q[2,2]*z2

y_new_s = y + h*(b1*k1 + b2*k2 + b3*k3)
y_new = Dinv * y_new_s  # unscale

print(f'Schur q(1e-6) sum={y_new.sum():.10f} max={np.max(np.abs(y_new)):.10f}')

# Direct solve (no scaling, no Schur) - ground truth
A_mat = np.array([
    [(88-7*sqrt6)/360, (296-169*sqrt6)/1800, (-2+3*sqrt6)/225],
    [(296+169*sqrt6)/1800, (88+7*sqrt6)/360, (-2-3*sqrt6)/225],
    [(16-sqrt6)/36, (16+sqrt6)/36, 1.0/9.0]
])

# Direct: (I - h*(A⊗J))*K_vec = 1⊗(J*y0)
f0 = J @ q0
K_vec = np.linalg.solve(np.eye(3*n) - h*np.kron(A, J), np.kron(np.ones(3), f0))
k1_d = K_vec[:n]; k2_d = K_vec[n:2*n]; k3_d = K_vec[2*n:]
y_new_direct = q0 + h*(b1*k1_d + b2*k2_d + b3*k3_d)
print(f'Direct q(1e-6) sum={y_new_direct.sum():.10f} max={np.max(np.abs(y_new_direct)):.10f}')

# Compare
print(f'\nSchur vs Direct max diff: {np.max(np.abs(y_new - y_new_direct)):.6e}')
print(f'scipy vs Direct max diff: {np.max(np.abs(q_scipy - y_new_direct)):.6e}')

# Check: is the Schur approach giving the same as Direct?
if np.max(np.abs(y_new - y_new_direct)) > 1e-10:
    print('\n*** BUG: Schur approach differs from Direct! ***')
    # Debug: check each step
    print(f'z2 max: {np.max(np.abs(z2)):.6e}')
    print(f'z0 max: {np.max(np.abs(z0)):.6e}')
    print(f'z1 max: {np.max(np.abs(z1)):.6e}')
    
    # Check real block residual
    res_real = A_real @ z2 - rhs_real
    print(f'Real block residual: {np.max(np.abs(res_real)):.6e}')
    
    # Check complex block residual
    res_complex = A_complex @ z_01 - rhs_complex
    print(f'Complex block residual: {np.max(np.abs(res_complex)):.6e}')
    
    # Check k recovery: K_vec = Q ⊗ I * z_vec
    z_vec = np.concatenate([z0, z1, z2])
    K_vec_schur = np.kron(Q, np.eye(n)) @ z_vec
    k1_s = K_vec_schur[:n]; k2_s = K_vec_schur[n:2*n]; k3_s = K_vec_schur[2*n:]
    print(f'k1 Schur vs Direct max diff: {np.max(np.abs(k1_s - k1_d)):.6e}')
    print(f'k2 Schur vs Direct max diff: {np.max(np.abs(k2_s - k2_d)):.6e}')
    print(f'k3 Schur vs Direct max diff: {np.max(np.abs(k3_s - k3_d)):.6e}')
    
    # Check: does z = Q^T * K_vec?
    z_from_k = np.kron(Q.T, np.eye(n)) @ K_vec  # z = Q^T ⊗ I * K_vec
    z_vec_schur = np.concatenate([z0, z1, z2])
    print(f'z from Q^T*K_vec vs solved z max diff: {np.max(np.abs(z_from_k - z_vec_schur)):.6e}')
    
    # Check: does (I - h*T⊗J)*z = Q^T * (1⊗f0)?
    z_rhs = np.kron(Q.T, np.eye(n)) @ np.kron(np.ones(3), f0)
    z_lhs = z_vec_schur - h * np.kron(T, J) @ z_vec_schur
    print(f'Schur system residual: {np.max(np.abs(z_lhs - z_rhs)):.6e}')
    
    # Check scaled vs unscaled
    print(f'\nScaled y max: {np.max(np.abs(y)):.6e}')
    print(f'Unscaled q0 max: {np.max(np.abs(q0)):.6e}')
    print(f'D*y0 vs y: {np.max(np.abs(D*q0 - y)):.6e}')
else:
    print('\nSchur approach matches Direct. Bug is elsewhere.')
