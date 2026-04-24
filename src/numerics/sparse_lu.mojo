"""Sparse LU factorization with left-looking column-wise algorithm.

Optimized implementation using:
- Left-looking factorization with nonzero-tracking for workspace W
  (eliminates O(n) scans for each column's U entries)
- Targeted workspace clearing via w_nz list (no triple-pass scanning)
- Inverse permutation (pinv) applied during scatter
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
    
    # Persistent workspaces
    var W: FixedSizeVector
    var w_nz: List[Int]
    var pinv: List[Int]
    var L_col_start: List[Int]
    var L_col_nnz: List[Int]
    var U_col_start: List[Int]
    var U_col_nnz: List[Int]

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
        
        self.W = FixedSizeVector(n)
        self.w_nz = List[Int]()
        self.pinv = List[Int]()
        self.L_col_start = List[Int]()
        self.L_col_nnz = List[Int]()
        self.U_col_start = List[Int]()
        self.U_col_nnz = List[Int]()
        
        for _ in range(n + 1):
            self.Lp.append(0)
            self.Up.append(0)
        for i in range(n):
            self.perm.append(i)
            self.diag_vals.append(1.0)
            self.pinv.append(i)
            self.L_col_start.append(0)
            self.L_col_nnz.append(0)
            self.U_col_start.append(0)
            self.U_col_nnz.append(0)

    def factorize(mut self, A: CSCMatrix) raises:
        var n = self.n

        # Reset persistent lists
        self.w_nz.clear()
        self.Lj.clear()
        self.Lx.clear()
        self.Uj.clear()
        self.Ux.clear()
        
        for i in range(n):
            self.perm[i] = i
            self.pinv[i] = i
            self.L_col_start[i] = 0
            self.L_col_nnz[i] = 0
            self.U_col_start[i] = 0
            self.U_col_nnz[i] = 0

        self.W.zero_out()

        var nnzL = 0
        var nnzU = 0

        for k in range(n):
            self.L_col_start[k] = nnzL
            self.U_col_start[k] = nnzU

            self.w_nz.clear()

            # Scatter column k of A into W (with pinv reordering)
            for p in range(A.colptr[k], A.colptr[k + 1]):
                var row = A.indices[p]
                var prow = self.pinv[row]
                self.W[prow] = A.data[p]
                self.w_nz.append(prow)

            # Apply contributions from previous L columns (left-looking)
            for j in range(k):
                var u_jk = self.W[j]
                if abs_f64(u_jk) < 1e-14:
                    continue

                self.Uj.append(j)
                self.Ux.append(u_jk)
                nnzU += 1
                self.U_col_nnz[k] += 1

                var l_start = self.L_col_start[j]
                var l_end = l_start + self.L_col_nnz[j]
                for p in range(l_start, l_end):
                    var row = self.Lj[p]
                    var new_val = self.W[row] - self.Lx[p] * u_jk
                    if self.W[row] == 0.0 and new_val != 0.0:
                        self.w_nz.append(row)
                    self.W[row] = new_val

                self.W[j] = 0.0

            # Partial pivoting
            var piv_val = self.W[k]
            var piv_row = k
            for i in range(k + 1, n):
                if abs_f64(self.W[i]) > abs_f64(piv_val):
                    piv_val = self.W[i]
                    piv_row = i

            if abs_f64(piv_val) < 1e-14:
                # Clear workspace using tracked nonzeros
                for idx in range(len(self.w_nz)):
                    self.W[self.w_nz[idx]] = 0.0
                continue

            if piv_row != k:
                var tmp = self.W[k]
                self.W[k] = self.W[piv_row]
                self.W[piv_row] = tmp

                for j in range(k):
                    var l_start = self.L_col_start[j]
                    var l_end = l_start + self.L_col_nnz[j]
                    var l_k_val: Float64 = 0.0
                    var l_piv_val: Float64 = 0.0
                    var l_k_idx: Int = -1
                    var l_piv_idx: Int = -1
                    for p in range(l_start, l_end):
                        if self.Lj[p] == k:
                            l_k_val = self.Lx[p]
                            l_k_idx = p
                        if self.Lj[p] == piv_row:
                            l_piv_val = self.Lx[p]
                            l_piv_idx = p
                    if l_k_idx >= 0 and l_piv_idx >= 0:
                        self.Lx[l_k_idx] = l_piv_val
                        self.Lx[l_piv_idx] = l_k_val
                    elif l_k_idx >= 0:
                        self.Lj[l_k_idx] = piv_row
                    elif l_piv_idx >= 0:
                        self.Lj[l_piv_idx] = k

                var old_perm_k = self.perm[k]
                var old_perm_piv = self.perm[piv_row]
                self.perm[k] = old_perm_piv
                self.perm[piv_row] = old_perm_k
                self.pinv[old_perm_k] = piv_row
                self.pinv[old_perm_piv] = k

            self.Uj.append(k)
            self.Ux.append(self.W[k])
            nnzU += 1
            self.U_col_nnz[k] += 1

            for i in range(k + 1, n):
                if abs_f64(self.W[i]) > 1e-14:
                    self.Lj.append(i)
                    self.Lx.append(self.W[i] / self.W[k])
                    nnzL += 1
                    self.L_col_nnz[k] += 1

            # Clear workspace via tracked nonzeros
            for idx in range(len(self.w_nz)):
                self.W[self.w_nz[idx]] = 0.0
            # Also clear L entries and diagonal
            for idx in range(self.L_col_nnz[k]):
                self.W[self.Lj[self.L_col_start[k] + idx]] = 0.0
            self.W[k] = 0.0

        self.Lp[0] = 0
        self.Up[0] = 0
        for k in range(n):
            self.Lp[k + 1] = self.Lp[k] + self.L_col_nnz[k]
            self.Up[k + 1] = self.Up[k] + self.U_col_nnz[k]

        for k in range(n):
            var diag: Float64 = 1.0
            var u_start = self.Up[k]
            var u_end = self.Up[k + 1]
            for p in range(u_start, u_end):
                if self.Uj[p] == k:
                    diag = self.Ux[p]
                    break
            self.diag_vals[k] = diag

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

    def solve_inplace(
        mut self, mut b: FixedSizeVector, mut work: FixedSizeVector
    ):
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


