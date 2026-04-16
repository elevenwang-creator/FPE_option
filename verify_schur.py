import numpy as np
S6 = np.sqrt(6.0)

A = np.array([
    [(88-7*S6)/360, (296-169*S6)/1800, (-2+3*S6)/225],
    [(296+169*S6)/1800, (88+7*S6)/360, (-2-3*S6)/225],
    [(16-S6)/36, (16+S6)/36, 1.0/9.0]
])

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

A_reconstructed = Q @ T @ Q.T
print('A - Q*T*Q^T max error:', np.max(np.abs(A - A_reconstructed)))
print('Q*Q^T - I max error:', np.max(np.abs(Q @ Q.T - np.eye(3))))
print('Q^T*Q - I max error:', np.max(np.abs(Q.T @ Q - np.eye(3))))

ones = np.ones(3)
print('Q^T * 1:', Q.T @ ones)
print('1^T * Q:', ones @ Q)

# Also verify scipy's T and TI
T_scipy = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI_scipy = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# scipy uses eigendecomposition: A = T_scipy * diag(MU) * TI_scipy
MU_REAL = 3 + 3**(2/3) - 3**(1/3)
MU_COMPLEX = (3 + 0.5*(3**(1/3) - 3**(2/3))) - 0.5j*(3**(5/6) + 3**(7/6))
print('\nMU_REAL:', MU_REAL)
print('1/MU_REAL = T_22?', 1.0/MU_REAL, 'vs T_22:', T[2,2])
print('Match?', np.isclose(1.0/MU_REAL, T[2,2]))

# Verify: A = T_scipy * diag(MU) * TI_scipy
MU_diag = np.diag([MU_REAL, MU_COMPLEX, np.conj(MU_COMPLEX)])
A_scipy = T_scipy @ MU_diag @ TI_scipy
print('\nA - T*diag(MU)*TI max error:', np.max(np.abs(A - A_scipy.real)))

# Key: scipy's system matrix is MU_REAL/h * I - J
# Our system matrix is (M + h*T_22*K)
# Since 1/MU_REAL = T_22, we have MU_REAL = 1/T_22
# So MU_REAL/h * I - J = (1/(h*T_22)) * I + M^{-1}*K
# Multiply by M: (1/(h*T_22))*M + K = (M + h*T_22*K)/(h*T_22)
# So scipy's system = our system / (h*T_22)
# This means they're proportional - same solution for linear systems!
print('\n=== SIGN CHECK ===')
# For M*dq/dt = -K*q, the RHS in scipy's formulation:
# f = -M^{-1}*K*y, J = -M^{-1}*K
# Newton iteration: (MU_REAL/h * I - J) * delta = f + ...
# = (1/(h*T_22) * I + M^{-1}*K) * delta = -M^{-1}*K*y + ...
# Multiply by M: (M/(h*T_22) + K) * delta = -K*y + ...
# = (M + h*T_22*K)/(h*T_22) * delta = -K*y + ...
# So (M + h*T_22*K) * delta = h*T_22 * (-K*y + ...)
print('Sign convention: M*dq/dt = -K*q => f = -M^{-1}*K*y, J = -M^{-1}*K')
