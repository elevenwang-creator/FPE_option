"""Debug test for RADAU5 Schur decomposition approach.

Simplified version without diagonal scaling, with debug output.
"""

from numerics.ode.types import ODESolution, ODESystem
from numerics.utils import abs_f64, max_f64, min_f64, zeros, copy_vec
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.ops import add, scale
from std.math import exp, log, sqrt, abs, min, max
from std.sys import simd_width_of
from std.algorithm import parallelize

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from engines.fpe.solver import FPESolver


comptime SIMD_WIDTH = simd_width_of[DType.float64]()


def main() raises:
    print("=== RADAU5 Debug Test ===")
    print()

    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var domain = FPEDomain[3, 3](params, n_s=11, n_v=11)

    var assembler = GalerkinAssembler[3]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)
    var n = M.nrows
    print("n_dof = ", n)

    print("M nrows=", M.nrows, " nnz=", len(M.data))
    print("K nrows=", K.nrows, " nnz=", len(K.data))
    print()

    print("[Step 1] Factorize M (CSC)...")
    var M_csc = csr_to_csc(M)
    var m_lu = SparseLU(n)
    m_lu.factorize(M_csc)
    print("  Done!")

    print("[Step 2] Compute J = -M^(-1) @ K...")
    var K_csc = csr_to_csc(scale(-1.0, K))

    var J_all_indices: List[List[Int]] = []
    var J_all_data: List[List[Float64]] = []
    var J_all_nnz: List[Int] = []
    for _ in range(n):
        J_all_indices.append([])
        J_all_data.append([])
        J_all_nnz.append(0)

    for col in range(n):
        var rhs = zeros(n)
        for p in range(K_csc.indptr[col], K_csc.indptr[col + 1]):
            var row = K_csc.row[p]
            rhs[row] = K_csc.data[p]

        var x = m_lu.solve(rhs)

        var nnz = 0
        for i in range(n):
            if abs_f64(x[i]) > 1e-14:
                J_all_indices[col].append(i)
                J_all_data[col].append(x[i])
                nnz += 1
        J_all_nnz[col] = nnz

    var J_indptr = [0]
    var J_indices: List[Int] = []
    var J_data: List[Float64] = []
    for col in range(n):
        for i in range(len(J_all_indices[col])):
            J_indices.append(J_all_indices[col][i])
            J_data.append(J_all_data[col][i])
        J_indptr.append(J_indptr[col] + J_all_nnz[col])

    var J = CSRMatrix[DType.float64](n, n)
    J.indptr = J_indptr^
    J.indices = J_indices^
    J.data = J_data^
    J.nrows = n
    J.ncols = n
    print("  J nnz = ", len(J.data))

    var j_max: Float64 = 0.0
    for i in range(len(J.data)):
        if abs_f64(J.data[i]) > j_max:
            j_max = abs_f64(J.data[i])
    print("  J max element = ", j_max)
    print()

    print("[Step 3] Test Schur decomposition approach...")
    var sqrt6 = sqrt(6.0)
    var t_00: Float64 = 0.16255558520216112
    var t_01: Float64 = 0.51074439865923390
    var t_02: Float64 = -0.47719467969124402
    var t_10: Float64 = -0.06697332890760048
    var t_11: Float64 = 0.16255558520216112
    var t_12: Float64 = -0.28529656780973917
    var t_22: Float64 = 0.27488882959567795

    var q_00: Float64 = 0.13866510875190752
    var q_01: Float64 = 0.04627814930949071
    var q_02: Float64 = 0.98925745916384644
    var q_10: Float64 = -0.22964124235174019
    var q_11: Float64 = -0.97017888655183304
    var q_12: Float64 = 0.07757466016809490
    var q_20: Float64 = -0.96334671195056865
    var q_21: Float64 = 0.23793121061671349
    var q_22: Float64 = 0.12390258911134427

    var h = 0.03

    print("  Building (I - h*t22*J) system, n=", n, "...")
    var coeff_real = -h * t_22
    var sys_real_indptr = [0]
    var sys_real_indices: List[Int] = []
    var sys_real_data: List[Float64] = []
    for row in range(n):
        var diag_val: Float64 = 1.0
        var off_indices: List[Int] = []
        var off_data: List[Float64] = []
        var j_start = J.indptr[row]
        var j_end = J.indptr[row + 1]
        for p in range(j_start, j_end):
            var col = J.indices[p]
            if col == row:
                diag_val = 1.0 + coeff_real * J.data[p]
            else:
                off_indices.append(col)
                off_data.append(coeff_real * J.data[p])
        var diag_inserted = False
        var k = 0
        while k < len(off_indices):
            if not diag_inserted and off_indices[k] > row:
                sys_real_indices.append(row)
                sys_real_data.append(diag_val)
                diag_inserted = True
            sys_real_indices.append(off_indices[k])
            sys_real_data.append(off_data[k])
            k += 1
        if not diag_inserted:
            sys_real_indices.append(row)
            sys_real_data.append(diag_val)
        sys_real_indptr.append(sys_real_indptr[row] + len(off_indices) + 1)

    var sys_real = CSRMatrix[DType.float64](n, n)
    sys_real.indptr = sys_real_indptr^
    sys_real.indices = sys_real_indices^
    sys_real.data = sys_real_data^
    sys_real.nrows = n
    sys_real.ncols = n

    var sys_real_csc = csr_to_csc(sys_real)
    print("  Factorizing (I - h*t22*J)...")
    var lu_real = SparseLU(n)
    lu_real.factorize(sys_real_csc)
    print("  Done! nnz(L)=", len(lu_real.Lx), " nnz(U)=", len(lu_real.Ux))

    print("  Building 2n system (2n=", 2*n, ")...")
    var n2 = 2 * n
    var sys_2n_indptr = [0]
    var sys_2n_indices: List[Int] = []
    var sys_2n_data: List[Float64] = []

    for block_row in range(2):
        var t_row_0 = t_00 if block_row == 0 else t_10
        var t_row_1 = t_01 if block_row == 0 else t_11
        for row in range(n):
            var global_row = block_row * n + row
            var diag_val: Float64 = 1.0
            var off_indices: List[Int] = []
            var off_data: List[Float64] = []
            for block_col in range(2):
                var coeff = -h * (t_row_0 if block_col == 0 else t_row_1)
                var j_start = J.indptr[row]
                var j_end = J.indptr[row + 1]
                for p in range(j_start, j_end):
                    var col = J.indices[p]
                    var global_col = block_col * n + col
                    if global_col == global_row:
                        diag_val = 1.0 + coeff * J.data[p]
                    else:
                        off_indices.append(global_col)
                        off_data.append(coeff * J.data[p])
            var diag_inserted = False
            var k = 0
            while k < len(off_indices):
                if not diag_inserted and off_indices[k] > global_row:
                    sys_2n_indices.append(global_row)
                    sys_2n_data.append(diag_val)
                    diag_inserted = True
                sys_2n_indices.append(off_indices[k])
                sys_2n_data.append(off_data[k])
                k += 1
            if not diag_inserted:
                sys_2n_indices.append(global_row)
                sys_2n_data.append(diag_val)
            sys_2n_indptr.append(sys_2n_indptr[global_row] + len(off_indices) + 1)

    var sys_2n = CSRMatrix[DType.float64](n2, n2)
    sys_2n.indptr = sys_2n_indptr^
    sys_2n.indices = sys_2n_indices^
    sys_2n.data = sys_2n_data^
    sys_2n.nrows = n2
    sys_2n.ncols = n2

    var sys_2n_csc = csr_to_csc(sys_2n)
    print("  Factorizing 2n system...")
    var lu_complex = SparseLU(n2)
    lu_complex.factorize(sys_2n_csc)
    print("  Done! nnz(L)=", len(lu_complex.Lx), " nnz(U)=", len(lu_complex.Ux))
    print()

    print("[Step 4] Test one RADAU5 step...")
    var ic = InitialCondition[1]()
    var q0 = ic.compute(domain, params, sigma0=10.0)
    var y = copy_vec(q0)

    var f0 = J.spmv(y)
    var f_max: Float64 = 0.0
    for i in range(n):
        if abs_f64(f0[i]) > f_max:
            f_max = abs_f64(f0[i])
    print("  ||f0||_inf = ", f_max)

    var qt_col_sum_0 = q_00 + q_10 + q_20
    var qt_col_sum_1 = q_01 + q_11 + q_21
    var qt_col_sum_2 = q_02 + q_12 + q_22

    var f0_s = zeros(n)
    var f1_s = zeros(n)
    var f2_s = zeros(n)
    for k in range(n):
        f0_s[k] = qt_col_sum_0 * f0[k]
        f1_s[k] = qt_col_sum_1 * f0[k]
        f2_s[k] = qt_col_sum_2 * f0[k]

    print("  Solving for z2 (real eigenvalue block)...")
    var z2 = lu_real.solve(f2_s)
    var z2_max: Float64 = 0.0
    for i in range(n):
        if abs_f64(z2[i]) > z2_max:
            z2_max = abs_f64(z2[i])
    print("  ||z2||_inf = ", z2_max)

    print("  Computing J*z2...")
    var Jz2 = J.spmv(z2)

    print("  Solving for z0,z1 (complex eigenvalue block)...")
    var rhs_2n = zeros(2 * n)
    for k in range(n):
        rhs_2n[k] = f0_s[k] + h * t_02 * Jz2[k]
        rhs_2n[n + k] = f1_s[k] + h * t_12 * Jz2[k]

    var z_01 = lu_complex.solve(rhs_2n)
    var z01_max: Float64 = 0.0
    for i in range(2 * n):
        if abs_f64(z_01[i]) > z01_max:
            z01_max = abs_f64(z_01[i])
    print("  ||z01||_inf = ", z01_max)

    var z0 = zeros(n)
    var z1 = zeros(n)
    for k in range(n):
        z0[k] = z_01[k]
        z1[k] = z_01[n + k]

    print("  Computing stage vectors k1, k2, k3...")
    var k1 = zeros(n)
    var k2 = zeros(n)
    var k3 = zeros(n)
    for kk in range(n):
        k1[kk] = q_00 * z0[kk] + q_01 * z1[kk] + q_02 * z2[kk]
        k2[kk] = q_10 * z0[kk] + q_11 * z1[kk] + q_12 * z2[kk]
        k3[kk] = q_20 * z0[kk] + q_21 * z1[kk] + q_22 * z2[kk]

    var k1_max: Float64 = 0.0
    var k2_max: Float64 = 0.0
    var k3_max: Float64 = 0.0
    for i in range(n):
        if abs_f64(k1[i]) > k1_max:
            k1_max = abs_f64(k1[i])
        if abs_f64(k2[i]) > k2_max:
            k2_max = abs_f64(k2[i])
        if abs_f64(k3[i]) > k3_max:
            k3_max = abs_f64(k3[i])
    print("  ||k1||_inf = ", k1_max)
    print("  ||k2||_inf = ", k2_max)
    print("  ||k3||_inf = ", k3_max)

    var b1: Float64 = (16.0 - sqrt6) / 36.0
    var b2: Float64 = (16.0 + sqrt6) / 36.0
    var b3: Float64 = 1.0 / 9.0

    var y_new = zeros(n)
    for i in range(n):
        y_new[i] = y[i] + h * (b1 * k1[i] + b2 * k2[i] + b3 * k3[i])

    var y_new_max: Float64 = 0.0
    var y_new_sum: Float64 = 0.0
    for i in range(n):
        if abs_f64(y_new[i]) > y_new_max:
            y_new_max = abs_f64(y_new[i])
        y_new_sum = y_new_sum + y_new[i]
    print("  ||y_new||_inf = ", y_new_max)
    print("  y_new sum = ", y_new_sum)

    var e1: Float64 = -2.0 / 9.0
    var e3: Float64 = -2.0 / 9.0
    var e2: Float64 = -(e1 + e3)
    var err_norm = 0.0
    for i in range(n):
        var err_i = abs_f64(h * (e1 * k1[i] + e2 * k2[i] + e3 * k3[i]))
        var sc = 1e-6 + 1e-4 * max_f64(abs_f64(y[i]), abs_f64(y_new[i]))
        var ratio = err_i / sc
        if ratio > err_norm:
            err_norm = ratio
    print("  error norm = ", err_norm)
    print()

    print("[Step 5] Verify: solve (I - h*A⊗J)*k = f directly (3n system)...")
    var a11: Float64 = (88.0 - 7.0 * sqrt6) / 360.0
    var a12: Float64 = (296.0 - 169.0 * sqrt6) / 1800.0
    var a13: Float64 = (-2.0 + 3.0 * sqrt6) / 225.0
    var a21: Float64 = (296.0 + 169.0 * sqrt6) / 1800.0
    var a22: Float64 = (88.0 + 7.0 * sqrt6) / 360.0
    var a23: Float64 = (-2.0 - 3.0 * sqrt6) / 225.0
    var a31: Float64 = (16.0 - sqrt6) / 36.0
    var a32: Float64 = (16.0 + sqrt6) / 36.0
    var a33: Float64 = 1.0 / 9.0

    var n3 = 3 * n
    var a_vals = [a11, a12, a13, a21, a22, a23, a31, a32, a33]

    var sys3_indptr = [0]
    var sys3_indices: List[Int] = []
    var sys3_data: List[Float64] = []

    for block_row in range(3):
        for row in range(n):
            var global_row = block_row * n + row
            var diag_val: Float64 = 1.0
            var off_indices: List[Int] = []
            var off_data: List[Float64] = []
            for block_col in range(3):
                var a_val = a_vals[block_row * 3 + block_col]
                var coeff = -h * a_val
                var j_start = J.indptr[row]
                var j_end = J.indptr[row + 1]
                for p in range(j_start, j_end):
                    var col = J.indices[p]
                    var global_col = block_col * n + col
                    if global_col == global_row:
                        diag_val = 1.0 + coeff * J.data[p]
                    else:
                        off_indices.append(global_col)
                        off_data.append(coeff * J.data[p])
            var diag_inserted = False
            var k = 0
            while k < len(off_indices):
                if not diag_inserted and off_indices[k] > global_row:
                    sys3_indices.append(global_row)
                    sys3_data.append(diag_val)
                    diag_inserted = True
                sys3_indices.append(off_indices[k])
                sys3_data.append(off_data[k])
                k += 1
            if not diag_inserted:
                sys3_indices.append(global_row)
                sys3_data.append(diag_val)
            sys3_indptr.append(sys3_indptr[global_row] + len(off_indices) + 1)

    var sys3 = CSRMatrix[DType.float64](n3, n3)
    sys3.indptr = sys3_indptr^
    sys3.indices = sys3_indices^
    sys3.data = sys3_data^
    sys3.nrows = n3
    sys3.ncols = n3

    var sys3_csc = csr_to_csc(sys3)
    print("  Factorizing 3n system (3n=", n3, ")...")
    var lu3 = SparseLU(n3)
    lu3.factorize(sys3_csc)
    print("  Done!")

    var rhs3 = zeros(3 * n)
    for i in range(n):
        rhs3[i] = f0[i]
        rhs3[n + i] = f0[i]
        rhs3[2 * n + i] = f0[i]

    print("  Solving 3n system...")
    var k_vec = lu3.solve(rhs3)

    var k1_direct = zeros(n)
    var k2_direct = zeros(n)
    var k3_direct = zeros(n)
    for i in range(n):
        k1_direct[i] = k_vec[i]
        k2_direct[i] = k_vec[n + i]
        k3_direct[i] = k_vec[2 * n + i]

    var k1_diff: Float64 = 0.0
    var k2_diff: Float64 = 0.0
    var k3_diff: Float64 = 0.0
    for i in range(n):
        k1_diff = k1_diff + abs_f64(k1[i] - k1_direct[i])
        k2_diff = k2_diff + abs_f64(k2[i] - k2_direct[i])
        k3_diff = k3_diff + abs_f64(k3[i] - k3_direct[i])
    print("  ||k1_schur - k1_direct|| = ", k1_diff)
    print("  ||k2_schur - k2_direct|| = ", k2_diff)
    print("  ||k3_schur - k3_direct|| = ", k3_diff)
    print()

    print("=== Debug test complete ===")
