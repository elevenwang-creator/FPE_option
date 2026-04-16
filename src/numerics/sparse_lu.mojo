"""Sparse LU factorization with left-looking column-wise algorithm.

Correct implementation using:
- Left-looking factorization: for each column k, apply all previous
  columns' contributions (L[:,j]*U[j,k]) before extracting U and L entries
- Inverse permutation (pinv) applied during scatter to ensure W is in
  the permuted row order, matching the L entries from previous columns
- Column-wise forward/backward substitution matching CSC storage
- Partial pivoting for numerical stability

Accepts CSCMatrix input for efficient column-wise access.
Use csr_to_csc() to convert CSR matrices before factorizing.

Storage: L and U in CSC format (column-wise):
  Lp[k]:Lp[k+1] = entries in column k of L (row indices in Lj, values in Lx)
  Up[k]:Up[k+1] = entries in column k of U (row indices in Uj, values in Ux)
"""

from numerics.utils import FixedSizeVector, abs_f64, zeros
from sparse.csc import CSCMatrix
from std.sys import simd_width_of


comptime SIMD_WIDTH = simd_width_of[DType.float64]()


struct SparseLU:
    var Lp: List[Int]
    var Lj: List[Int]
    var Lx: List[Float64]
    var Up: List[Int]
    var Uj: List[Int]
    var Ux: List[Float64]
    var perm: List[Int]
    var diag_vals: List[Float64]
    var n: Int

    def __init__(out self, n: Int):
        self.n = n
        self.Lp = List[Int]()
        self.Lj = List[Int]()
        self.Lx = List[Float64]()
        self.Up = List[Int]()
        self.Uj = List[Int]()
        self.Ux = List[Float64]()
        self.perm = List[Int]()
        self.diag_vals = List[Float64]()
        for _ in range(n + 1):
            self.Lp.append(0)
            self.Up.append(0)
        for i in range(n):
            self.perm.append(i)
            self.diag_vals.append(1.0)

    def factorize(mut self, A: CSCMatrix) raises:
        var n = self.n

        var W: List[Float64] = []
        for _ in range(n):
            W.append(0.0)

        var perm: List[Int] = []
        var pinv: List[Int] = []
        for i in range(n):
            perm.append(i)
            pinv.append(i)

        var L_col_start: List[Int] = []
        var L_col_nnz: List[Int] = []
        var U_col_start: List[Int] = []
        var U_col_nnz: List[Int] = []
        for _ in range(n):
            L_col_start.append(0)
            L_col_nnz.append(0)
            U_col_start.append(0)
            U_col_nnz.append(0)

        var Lj_all: List[Int] = []
        var Lx_all: List[Float64] = []
        var Uj_all: List[Int] = []
        var Ux_all: List[Float64] = []

        var nnzL = 0
        var nnzU = 0

        for k in range(n):
            L_col_start[k] = nnzL
            U_col_start[k] = nnzU

            for p in range(A.colptr[k], A.colptr[k + 1]):
                var row = A.indices[p]
                W[pinv[row]] = A.data[p]

            for j in range(k):
                var u_jk = W[j]
                if abs_f64(u_jk) < 1e-14:
                    continue

                Uj_all.append(j)
                Ux_all.append(u_jk)
                nnzU += 1
                U_col_nnz[k] += 1

                var l_start = L_col_start[j]
                var l_end = l_start + L_col_nnz[j]
                for p in range(l_start, l_end):
                    var row = Lj_all[p]
                    W[row] = W[row] - Lx_all[p] * u_jk

                W[j] = 0.0

            var piv_val = W[k]
            var piv_row = k
            for i in range(k + 1, n):
                if abs_f64(W[i]) > abs_f64(piv_val):
                    piv_val = W[i]
                    piv_row = i

            if abs_f64(piv_val) < 1e-14:
                for p in range(A.colptr[k], A.colptr[k + 1]):
                    var row = A.indices[p]
                    W[pinv[row]] = 0.0
                for j in range(k):
                    var l_start = L_col_start[j]
                    var l_end = l_start + L_col_nnz[j]
                    for p in range(l_start, l_end):
                        W[Lj_all[p]] = 0.0
                continue

            if piv_row != k:
                var tmp = W[k]
                W[k] = W[piv_row]
                W[piv_row] = tmp

                for j in range(k):
                    var l_start = L_col_start[j]
                    var l_end = l_start + L_col_nnz[j]
                    var l_k_val: Float64 = 0.0
                    var l_piv_val: Float64 = 0.0
                    var l_k_idx: Int = -1
                    var l_piv_idx: Int = -1
                    for p in range(l_start, l_end):
                        if Lj_all[p] == k:
                            l_k_val = Lx_all[p]
                            l_k_idx = p
                        if Lj_all[p] == piv_row:
                            l_piv_val = Lx_all[p]
                            l_piv_idx = p
                    if l_k_idx >= 0 and l_piv_idx >= 0:
                        Lx_all[l_k_idx] = l_piv_val
                        Lx_all[l_piv_idx] = l_k_val
                    elif l_k_idx >= 0:
                        Lj_all[l_k_idx] = piv_row
                    elif l_piv_idx >= 0:
                        Lj_all[l_piv_idx] = k

                var old_perm_k = perm[k]
                var old_perm_piv = perm[piv_row]
                perm[k] = old_perm_piv
                perm[piv_row] = old_perm_k
                pinv[old_perm_k] = piv_row
                pinv[old_perm_piv] = k

            Uj_all.append(k)
            Ux_all.append(W[k])
            nnzU += 1
            U_col_nnz[k] += 1

            for i in range(k + 1, n):
                if abs_f64(W[i]) > 1e-14:
                    Lj_all.append(i)
                    Lx_all.append(W[i] / W[k])
                    nnzL += 1
                    L_col_nnz[k] += 1

            for p in range(A.colptr[k], A.colptr[k + 1]):
                var row = A.indices[p]
                W[pinv[row]] = 0.0
            for j in range(k):
                var l_start = L_col_start[j]
                var l_end = l_start + L_col_nnz[j]
                for p in range(l_start, l_end):
                    W[Lj_all[p]] = 0.0
            for idx in range(L_col_nnz[k]):
                W[Lj_all[L_col_start[k] + idx]] = 0.0
            W[k] = 0.0

        var Lp: List[Int] = [0]
        var Up: List[Int] = [0]
        for k in range(n):
            Lp.append(Lp[k] + L_col_nnz[k])
            Up.append(Up[k] + U_col_nnz[k])

        self.diag_vals = List[Float64]()
        for k in range(n):
            var diag: Float64 = 1.0
            var u_start = Up[k]
            var u_end = Up[k + 1]
            for p in range(u_start, u_end):
                if Uj_all[p] == k:
                    diag = Ux_all[p]
                    break
            self.diag_vals.append(diag)

        self.Lp = Lp^
        self.Lj = Lj_all^
        self.Lx = Lx_all^
        self.Up = Up^
        self.Uj = Uj_all^
        self.Ux = Ux_all^
        self.perm = perm^

    def solve(self, b: List[Float64]) -> List[Float64]:
        var n = self.n
        var y = zeros(n)

        for i in range(n):
            y[i] = b[self.perm[i]]

        for k in range(n):
            var l_start = self.Lp[k]
            var l_end = self.Lp[k + 1]
            var y_k = y[k]
            for p in range(l_start, l_end):
                var i = self.Lj[p]
                y[i] = y[i] - self.Lx[p] * y_k

        var x = zeros(n)
        for i in range(n):
            x[i] = y[i]

        for k in range(n - 1, -1, -1):
            var u_start = self.Up[k]
            var u_end = self.Up[k + 1]
            var diag = self.diag_vals[k]
            x[k] = x[k] / diag
            var x_k = x[k]
            for p in range(u_start, u_end):
                var j = self.Uj[p]
                if j < k:
                    x[j] = x[j] - self.Ux[p] * x_k

        return x^

    def solve_inplace(mut self, mut b: FixedSizeVector, mut work: FixedSizeVector):
        var n = self.n

        for i in range(n):
            work[i] = b[self.perm[i]]

        for k in range(n):
            var l_start = self.Lp[k]
            var l_end = self.Lp[k + 1]
            var y_k = work[k]
            for p in range(l_start, l_end):
                var row = self.Lj[p]
                work[row] = work[row] - self.Lx[p] * y_k

        b.copy_from_fixed(work)

        for k in range(n - 1, -1, -1):
            var u_start = self.Up[k]
            var u_end = self.Up[k + 1]
            var diag = self.diag_vals[k]
            b[k] = b[k] / diag
            var b_k = b[k]
            for p in range(u_start, u_end):
                var j = self.Uj[p]
                if j < k:
                    b[j] = b[j] - self.Ux[p] * b_k
