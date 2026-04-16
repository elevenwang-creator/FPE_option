import numpy as np
from scipy.linalg import schur

sqrt6 = np.sqrt(6.0)
A = np.array([
    [(88 - 7*sqrt6)/360, (296 - 169*sqrt6)/1800, (-2 + 3*sqrt6)/225],
    [(296 + 169*sqrt6)/1800, (88 + 7*sqrt6)/360, (-2 - 3*sqrt6)/225],
    [(16 - sqrt6)/36, (16 + sqrt6)/36, 1.0/9.0]
])

print("A =")
print(A)
print()

T, Q = schur(A, output='real')
print("T (real Schur form) =")
print(repr(T))
print()
print("Q (orthogonal) =")
print(repr(Q))
print()

eigvals = np.linalg.eigvals(A)
print("Eigenvalues of A:")
print(eigvals)
print()

print("Q @ Q.T (should be I):")
print(Q @ Q.T)
print()

print("--- Mojo constants ---")
print("// T (real Schur form of Butcher matrix A)")
for i in range(3):
    for j in range(3):
        print(f"var t_{i}{j}: Float64 = {T[i,j]:.17g}")

print()
print("// Q (orthogonal Schur vectors)")
for i in range(3):
    for j in range(3):
        print(f"var q_{i}{j}: Float64 = {Q[i,j]:.17g}")

print()
print(f"// Real eigenvalue: T[0,0] = {T[0,0]:.17g}")
print(f"// 2x2 block eigenvalues: {eigvals[1]:.17g}, {eigvals[2]:.17g}")
