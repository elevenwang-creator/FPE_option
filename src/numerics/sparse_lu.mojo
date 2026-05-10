"""Sparse LU factorization with left-looking column-wise algorithm.

Optimized implementation using:
- FixedSizeVector workspace with memset_zero clearing
- SIMD abs() instead of branching abs_f64 for pivoting search
- List.reserve() pre-allocation for workspace tracking
- In-place forward+backward substitution (no intermediate copy)
- vectorize for SIMD pivoting search
- @always_inline on all hot-path methods
- memcpy for zero-copy vector transfer

Accepts CSCMatrix input for efficient column-wise access.
Use CSRMatrix.to_csc() to convert CSR matrices before factorizing.

Storage: L and U in CSC format (column-wise):
Lp[k]:Lp[k+1] = entries in column k of L (row indices in Lj, values in Lx)
Up[k]:Up[k+1] = entries in column k of U (row indices in Uj, values in Ux)
"""

from numerics.utils import FixedSizeVector
from sparse.csc import CSCMatrix
from std.algorithm.backend.vectorize import vectorize
from std.memory import UnsafePointer, alloc, memcpy, memset_zero
from std.math import abs
from std.sys import simd_width_of

comptime SIMD_WIDTH = simd_width_of[DType.float64]()


struct SparseLU(Movable):
    var Lp: List[Int]
    var Lj: List[Int]
    var Lx: List[Float64]
    var Up: List[Int]
    var Uj: List[Int]
    var Ux: List[Float64]
    var perm: List[Int]
    var diag_vals: List[Float64]
    var n: Int

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

    @always_inline
    def factorize(mut self, A: CSCMatrix) raises:
        var n = self.n

        self.w_nz.clear()
        self.Lj.clear()
        self.Lx.clear()
        self.Uj.clear()
        self.Ux.clear()

        self.Lj.reserve(n * n // 4)
        self.Lx.reserve(n * n // 4)
        self.Uj.reserve(n * n // 4)
        self.Ux.reserve(n * n // 4)
        self.w_nz.reserve(n)

        for i in range(n):
            self.perm[i] = i
            self.pinv[i] = i
            self.L_col_start[i] = 0
            self.L_col_nnz[i] = 0
            self.U_col_start[i] = 0
            self.U_col_nnz[i] = 0

        self.W.zero_out()

        var W_ptr = self.W.ptr()
        var nnzL = 0
        var nnzU = 0

        for k in range(n):
            self.L_col_start[k] = nnzL
            self.U_col_start[k] = nnzU

            self.w_nz.clear()

            var cp_start = A.colptr[k]
            var cp_end = A.colptr[k + 1]
            for p in range(cp_start, cp_end):
                var row = A.indices[p]
                var prow = self.pinv[row]
                W_ptr[prow] = A.data[p]
                self.w_nz.append(prow)

            for j in range(k):
                var u_jk = W_ptr[j]
                if abs(u_jk) < 1e-14:
                    continue

                self.Uj.append(j)
                self.Ux.append(u_jk)
                nnzU += 1
                self.U_col_nnz[k] += 1

                var l_start = self.L_col_start[j]
                var l_end = l_start + self.L_col_nnz[j]
                for p in range(l_start, l_end):
                    var row = self.Lj[p]
                    var old_val = W_ptr[row]
                    var new_val = old_val - self.Lx[p] * u_jk
                    if old_val == 0.0 and new_val != 0.0:
                        self.w_nz.append(row)
                    W_ptr[row] = new_val

                W_ptr[j] = 0.0

            var piv_val = W_ptr[k]
            var piv_row = k

            var remain = n - k - 1
            if remain > 0:
                def find_pivot[width: Int](offset: Int) {mut piv_val, mut piv_row, k, W_ptr}:
                    for w in range(width):
                        var idx = k + 1 + offset + w
                        var v = W_ptr[idx]
                        if abs(v) > abs(piv_val):
                            piv_val = v
                            piv_row = idx

                vectorize[SIMD_WIDTH](remain, find_pivot)

            if abs(piv_val) < 1e-14:
                var wnz_len = len(self.w_nz)
                for idx in range(wnz_len):
                    W_ptr[self.w_nz[idx]] = 0.0
                continue

            if piv_row != k:
                var tmp = W_ptr[k]
                W_ptr[k] = W_ptr[piv_row]
                W_ptr[piv_row] = tmp

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
                self.perm.swap_elements(k, piv_row)
                self.pinv[old_perm_k] = piv_row
                self.pinv[old_perm_piv] = k

            self.Uj.append(k)
            self.Ux.append(W_ptr[k])
            nnzU += 1
            self.U_col_nnz[k] += 1

            var inv_diag = 1.0 / W_ptr[k]
            for i in range(k + 1, n):
                if abs(W_ptr[i]) > 1e-14:
                    var l_val = W_ptr[i] * inv_diag
                    self.Lj.append(i)
                    self.Lx.append(l_val)
                    nnzL += 1
                    self.L_col_nnz[k] += 1

            var wnz_len2 = len(self.w_nz)
            for idx in range(wnz_len2):
                W_ptr[self.w_nz[idx]] = 0.0
            var l_col_nnz_k = self.L_col_nnz[k]
            var l_col_start_k = self.L_col_start[k]
            for idx in range(l_col_nnz_k):
                W_ptr[self.Lj[l_col_start_k + idx]] = 0.0
            W_ptr[k] = 0.0

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
        var y = FixedSizeVector(n)
        var y_ptr = y.ptr()

        for i in range(n):
            y_ptr[i] = b[self.perm[i]]

        for k in range(n):
            var l_start = self.Lp[k]
            var l_end = self.Lp[k + 1]
            var y_k = y_ptr[k]
            for p in range(l_start, l_end):
                var i = self.Lj[p]
                y_ptr[i] = y_ptr[i] - self.Lx[p] * y_k

        var x = FixedSizeVector(n)
        var x_ptr = x.ptr()
        memcpy(dest=x_ptr, src=y_ptr, count=n)

        for k in range(n - 1, -1, -1):
            var u_start = self.Up[k]
            var u_end = self.Up[k + 1]
            var diag = self.diag_vals[k]
            x_ptr[k] = x_ptr[k] / diag
            var x_k = x_ptr[k]
            for p in range(u_start, u_end):
                var j = self.Uj[p]
                if j < k:
                    x_ptr[j] = x_ptr[j] - self.Ux[p] * x_k

        return x.to_list()

    @always_inline
    def solve_inplace(
        mut self, mut b: FixedSizeVector, mut work: FixedSizeVector
    ):
        var n = self.n
        var work_ptr = work.ptr()
        var b_ptr = b.ptr()

        for i in range(n):
            work_ptr[i] = b_ptr[self.perm[i]]

        for k in range(n):
            var l_start = self.Lp[k]
            var l_end = self.Lp[k + 1]
            var y_k = work_ptr[k]
            for p in range(l_start, l_end):
                var row = self.Lj[p]
                work_ptr[row] = work_ptr[row] - self.Lx[p] * y_k

        memcpy(dest=b_ptr, src=work_ptr, count=n)

        for k in range(n - 1, -1, -1):
            var u_start = self.Up[k]
            var u_end = self.Up[k + 1]
            var diag = self.diag_vals[k]
            b_ptr[k] = b_ptr[k] / diag
            var b_k = b_ptr[k]
            for p in range(u_start, u_end):
                var j = self.Uj[p]
                if j < k:
                    b_ptr[j] = b_ptr[j] - self.Ux[p] * b_k
