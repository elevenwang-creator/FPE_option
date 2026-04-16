import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp

M_data = np.loadtxt('dump_M.mtx', skiprows=2)
K_data = np.loadtxt('dump_K.mtx', skiprows=2)
q0 = np.loadtxt('dump_q0.txt')
n = 196
M = coo_matrix((M_data[:,2], (M_data[:,0].astype(int)-1, M_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()
K = coo_matrix((K_data[:,2], (K_data[:,0].astype(int)-1, K_data[:,1].astype(int)-1)), shape=(n,n)).tocsc()

sqrt6 = np.sqrt(6.0)
b1 = (16-sqrt6)/36; b2 = (16+sqrt6)/36; b3 = 1.0/9
E0 = (-13-7*sqrt6)/3; E1 = (-13+7*sqrt6)/3; E2 = -1.0/3
d1 = -1.0/3 - sqrt6/2; d2 = -1.0/3 + sqrt6/2; d3 = -1.0/3

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
t22=T[2,2]; t00=T[0,0]; t01=T[0,1]; t02=T[0,2]; t10=T[1,0]; t11=T[1,1]; t12=T[1,2]
qt0 = Q[0,0]+Q[1,0]+Q[2,0]; qt1 = Q[0,1]+Q[1,1]+Q[2,1]; qt2 = Q[0,2]+Q[1,2]+Q[2,2]

# Diagonal scaling
M_dense = M.toarray(); K_dense = K.toarray()
M_diag = M_dense.diagonal()
D = np.sqrt(np.abs(M_diag)); D[D<1e-14]=1.0; Dinv = 1.0/D
M_s = M_dense * np.outer(Dinv, Dinv)
K_s = K_dense * np.outer(Dinv, Dinv)

def schur_radau_step(y_s, h, M_s, K_s, n):
    w = K_s @ y_s
    A_real = M_s + h*t22*K_s
    rhs_real = -qt2 * w
    z2 = np.linalg.solve(A_real, rhs_real)
    
    w2 = K_s @ z2
    rhs_c = np.zeros(2*n)
    rhs_c[:n] = -qt0*w - h*t02*w2
    rhs_c[n:] = -qt1*w - h*t12*w2
    
    A_c = np.zeros((2*n,2*n))
    A_c[:n,:n] = M_s + h*t00*K_s; A_c[:n,n:] = h*t01*K_s
    A_c[n:,:n] = h*t10*K_s; A_c[n:,n:] = M_s + h*t11*K_s
    z01 = np.linalg.solve(A_c, rhs_c)
    
    z0=z01[:n]; z1=z01[n:]
    k1 = Q[0,0]*z0+Q[0,1]*z1+Q[0,2]*z2
    k2 = Q[1,0]*z0+Q[1,1]*z1+Q[1,2]*z2
    k3 = Q[2,0]*z0+Q[2,1]*z1+Q[2,2]*z2
    
    y_new_s = y_s + h*(b1*k1+b2*k2+b3*k3)
    
    # Error estimation (using d coefficients - same as Mojo)
    ZE = d1*k1+d2*k2+d3*k3
    M_ZE = M_s @ ZE
    rhs_err = h*t22*(-w+M_ZE)
    error_s = np.linalg.solve(A_real, rhs_err)
    
    err_norm_sq = 0.0
    for kk in range(n):
        err_orig = Dinv[kk]*error_s[kk]
        y_orig = Dinv[kk]*y_s[kk]
        y_new_orig = Dinv[kk]*y_new_s[kk]
        sc = 1e-6 + 1e-4*max(abs(y_orig),abs(y_new_orig))
        ratio = err_orig/sc
        err_norm_sq += ratio*ratio
    err_norm = np.sqrt(err_norm_sq/n)
    
    return y_new_s, k1, k2, k3, err_norm

# Error estimation with E coefficients (correct)
def schur_radau_step_E(y_s, h, M_s, K_s, n):
    w = K_s @ y_s
    A_real = M_s + h*t22*K_s
    rhs_real = -qt2 * w
    z2 = np.linalg.solve(A_real, rhs_real)
    
    w2 = K_s @ z2
    rhs_c = np.zeros(2*n)
    rhs_c[:n] = -qt0*w - h*t02*w2
    rhs_c[n:] = -qt1*w - h*t12*w2
    
    A_c = np.zeros((2*n,2*n))
    A_c[:n,:n] = M_s + h*t00*K_s; A_c[:n,n:] = h*t01*K_s
    A_c[n:,:n] = h*t10*K_s; A_c[n:,n:] = M_s + h*t11*K_s
    z01 = np.linalg.solve(A_c, rhs_c)
    
    z0=z01[:n]; z1=z01[n:]
    k1 = Q[0,0]*z0+Q[0,1]*z1+Q[0,2]*z2
    k2 = Q[1,0]*z0+Q[1,1]*z1+Q[1,2]*z2
    k3 = Q[2,0]*z0+Q[2,1]*z1+Q[2,2]*z2
    
    y_new_s = y_s + h*(b1*k1+b2*k2+b3*k3)
    
    # Error estimation using E coefficients (correct)
    ZE = E0*k1+E1*k2+E2*k3
    M_ZE = M_s @ ZE
    rhs_err = h*t22*(-w+M_ZE)
    error_s = np.linalg.solve(A_real, rhs_err)
    
    err_norm_sq = 0.0
    for kk in range(n):
        err_orig = Dinv[kk]*error_s[kk]
        y_orig = Dinv[kk]*y_s[kk]
        y_new_orig = Dinv[kk]*y_new_s[kk]
        sc = 1e-6 + 1e-4*max(abs(y_orig),abs(y_new_orig))
        ratio = err_orig/sc
        err_norm_sq += ratio*ratio
    err_norm = np.sqrt(err_norm_sq/n)
    
    return y_new_s, err_norm

# Simple embedded error (no linear solve)
def simple_error(y_s, k1, k2, k3, h):
    err = h*(E0*k1+E1*k2+E2*k3)
    err_norm_sq = 0.0
    for kk in range(n):
        y_orig = Dinv[kk]*y_s[kk]
        y_new_orig = Dinv[kk]*(y_s[kk]+h*(b1*k1[kk]+b2*k2[kk]+b3*k3[kk]))
        sc = 1e-6 + 1e-4*max(abs(y_orig),abs(y_new_orig))
        ratio = Dinv[kk]*err[kk]/sc
        err_norm_sq += ratio*ratio
    return np.sqrt(err_norm_sq/n)

# Run full integration with d-based error (Mojo approach)
y_s = D * q0
t = 0.0; T_end = 0.6
h = 1e-6; h_old = -1.0; err_old = -1.0
safety = 0.9; min_factor = 0.2; max_factor = 10.0
n_steps = 0
t_vals_d = [0.0]; q_vals_d = [q0.copy()]

while t < T_end - 1e-14:
    if t+h > T_end: h = T_end-t
    y_new_s, k1, k2, k3, err_norm = schur_radau_step(y_s, h, M_s, K_s, n)
    
    if err_norm <= 1.0:
        t += h; y_s = y_new_s; n_steps += 1
        q_orig = Dinv * y_s
        t_vals_d.append(t); q_vals_d.append(q_orig.copy())
        
        # PI controller (same as Mojo)
        if h_old > 0 and err_old > 0 and err_norm > 0:
            multiplier = h/h_old * (err_old/err_norm)**0.25
        else:
            multiplier = 1.0
        if err_norm > 0:
            factor = min(1.0, multiplier) * err_norm**(-0.25)
        else:
            factor = 1.0
        factor = min(max_factor, safety*factor)
        factor = max(min_factor, factor)
        h_old = h; err_old = err_norm
        h *= factor
        h = min(h, 0.1)
    else:
        if err_norm > 0:
            factor = max(min_factor, safety*err_norm**(-0.25))
        else:
            factor = min_factor
        h *= factor

q_final_d = Dinv * y_s
print(f'd-based error: {n_steps} steps, q(T) sum={q_final_d.sum():.6f}')

# Run with E-based error (correct)
y_s = D * q0
t = 0.0; h = 1e-6; h_old = -1.0; err_old = -1.0; n_steps = 0

while t < T_end - 1e-14:
    if t+h > T_end: h = T_end-t
    y_new_s, err_norm = schur_radau_step_E(y_s, h, M_s, K_s, n)
    
    if err_norm <= 1.0:
        t += h; y_s = y_new_s; n_steps += 1
        if h_old > 0 and err_old > 0 and err_norm > 0:
            multiplier = h/h_old * (err_old/err_norm)**0.25
        else:
            multiplier = 1.0
        if err_norm > 0:
            factor = min(1.0, multiplier) * err_norm**(-0.25)
        else:
            factor = 1.0
        factor = min(max_factor, safety*factor)
        factor = max(min_factor, factor)
        h_old = h; err_old = err_norm
        h *= factor
        h = min(h, 0.1)
    else:
        if err_norm > 0:
            factor = max(min_factor, safety*err_norm**(-0.25))
        else:
            factor = min_factor
        h *= factor

q_final_E = Dinv * y_s
print(f'E-based error: {n_steps} steps, q(T) sum={q_final_E.sum():.6f}')

# Run with simple embedded error
y_s = D * q0
t = 0.0; h = 1e-6; h_old = -1.0; err_old = -1.0; n_steps = 0

while t < T_end - 1e-14:
    if t+h > T_end: h = T_end-t
    y_new_s, k1, k2, k3, _ = schur_radau_step(y_s, h, M_s, K_s, n)
    err_norm = simple_error(y_s, k1, k2, k3, h)
    
    if err_norm <= 1.0:
        t += h; y_s = y_new_s; n_steps += 1
        if h_old > 0 and err_old > 0 and err_norm > 0:
            multiplier = h/h_old * (err_old/err_norm)**0.25
        else:
            multiplier = 1.0
        if err_norm > 0:
            factor = min(1.0, multiplier) * err_norm**(-0.25)
        else:
            factor = 1.0
        factor = min(max_factor, safety*factor)
        factor = max(min_factor, factor)
        h_old = h; err_old = err_norm
        h *= factor
        h = min(h, 0.1)
    else:
        if err_norm > 0:
            factor = max(min_factor, safety*err_norm**(-0.25))
        else:
            factor = min_factor
        h *= factor

q_final_simple = Dinv * y_s
print(f'Simple error: {n_steps} steps, q(T) sum={q_final_simple.sum():.6f}')

# scipy reference
M_lu = splu(M)
J = -M_lu.solve(K.toarray())
sol = solve_ivp(lambda t,q: M_lu.solve(-K@q), [0, T_end], q0, method='Radau', jac=J,
                atol=1e-6, rtol=1e-4, max_step=0.1, first_step=1e-6)
print(f'scipy: {len(sol.t)} steps, q(T) sum={sol.y[:,-1].sum():.6f}')

print(f'\nDiff vs scipy:')
print(f'  d-based: {np.max(np.abs(q_final_d - sol.y[:,-1])):.6e}')
print(f'  E-based: {np.max(np.abs(q_final_E - sol.y[:,-1])):.6e}')
print(f'  Simple:  {np.max(np.abs(q_final_simple - sol.y[:,-1])):.6e}')
