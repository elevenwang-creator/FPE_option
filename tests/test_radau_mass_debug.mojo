"""Debug test: verify mass matrix formulation of RADAU5.

Tests:
1. Diagonal scaling produces M_s with unit diagonal
2. System matrix (M_s + h*gamma*K_s) is correct
3. SparseLU gives correct solution on system matrices
4. Full RADAU5 step gives correct stage derivatives
"""

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from numerics.ode.radau import RadauSparseLinearSolver, LinearODESystem
from numerics.sparse_lu import SparseLU
from numerics.utils import zeros, copy_vec, abs_f64, max_f64
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from std.math import sqrt, abs


struct TestSystem(LinearODESystem):
    var M_mat: CSRMatrix[DType.float64]
    var K_mat: CSRMatrix[DType.float64]

    def __init__(out self, M: CSRMatrix[DType.float64], K: CSRMatrix[DType.float64]):
        self.M_mat = M
        self.K_mat = K

    def get_M(self) -> CSRMatrix[DType.float64]:
        return self.M_mat.copy()

    def get_K(self) -> CSRMatrix[DType.float64]:
        return self.K_mat.copy()


def main() raises:
    print("=" * 60)
    print("  RADAU5 Mass Matrix Formulation Debug Test")
    print("=" * 60)

    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )

    var domain = FPEDomain[3, 3](params, n_s=11, n_v=11)
    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)
    var n = M.nrows
    print("System size: " + String(n))
    print("M nnz: " + String(len(M.data)) + ", K nnz: " + String(len(K.data)))

    var D_diag: List[Float64] = []
    var Dinv_diag: List[Float64] = []
    for i in range(n):
        var m_ii: Float64 = 0.0
        for p in range(M.indptr[i], M.indptr[i + 1]):
            if M.indices[p] == i:
                m_ii = M.data[p]
                break
        var d = sqrt(abs_f64(m_ii))
        if d < 1e-14:
            d = 1.0
        D_diag.append(d)
        Dinv_diag.append(1.0 / d)

    var M_s_indptr: List[Int] = []
    for i in range(len(M.indptr)):
        M_s_indptr.append(M.indptr[i])
    var M_s_indices: List[Int] = []
    var M_s_data: List[Float64] = []
    for i in range(n):
        for p in range(M.indptr[i], M.indptr[i + 1]):
            var j = M.indices[p]
            M_s_indices.append(j)
            M_s_data.append(M.data[p] * Dinv_diag[i] * Dinv_diag[j])
    var M_s = CSRMatrix[DType.float64](n, n)
    M_s.indptr = M_s_indptr^
    M_s.indices = M_s_indices^
    M_s.data = M_s_data^
    M_s.nrows = n
    M_s.ncols = n

    var K_s_indptr: List[Int] = []
    for i in range(len(K.indptr)):
        K_s_indptr.append(K.indptr[i])
    var K_s_indices: List[Int] = []
    var K_s_data: List[Float64] = []
    for i in range(n):
        for p in range(K.indptr[i], K.indptr[i + 1]):
            var j = K.indices[p]
            K_s_indices.append(j)
            K_s_data.append(K.data[p] * Dinv_diag[i] * Dinv_diag[j])
    var K_s = CSRMatrix[DType.float64](n, n)
    K_s.indptr = K_s_indptr^
    K_s.indices = K_s_indices^
    K_s.data = K_s_data^
    K_s.nrows = n
    K_s.ncols = n

    print()
    print("[Test 1] Diagonal scaling: M_s should have unit diagonal")
    var max_diag_err = 0.0
    for i in range(n):
        var m_s_ii: Float64 = 0.0
        for p in range(M_s.indptr[i], M_s.indptr[i + 1]):
            if M_s.indices[p] == i:
                m_s_ii = M_s.data[p]
                break
        var err = abs_f64(m_s_ii - 1.0)
        if err > max_diag_err:
            max_diag_err = err
    print("  Max |M_s[i,i] - 1| = " + String(max_diag_err))
    print("  Result: " + ("PASS" if max_diag_err < 1e-12 else "FAIL"))

    print()
    print("[Test 2] SparseLU on M_s (should be well-conditioned)")
    var M_s_csc = csr_to_csc(M_s)
    var lu_m = SparseLU(n)
    lu_m.factorize(M_s_csc)

    var e_n = zeros(n)
    e_n[0] = 1.0
    var x_m = lu_m.solve(e_n)
    var residual_m = M_s.spmv(x_m)
    var res_norm_m = 0.0
    var x_norm_m = 0.0
    for i in range(n):
        res_norm_m += (residual_m[i] - e_n[i]) * (residual_m[i] - e_n[i])
        x_norm_m += x_m[i] * x_m[i]
    res_norm_m = sqrt(res_norm_m)
    x_norm_m = sqrt(x_norm_m)
    print("  ||M_s*x - e1|| / ||x|| = " + String(res_norm_m / x_norm_m))
    print("  Result: " + ("PASS" if res_norm_m / x_norm_m < 1e-10 else "FAIL"))

    print()
    print("[Test 3] Build system matrix (M_s + h*t22*K_s) and solve")
    var t_22: Float64 = 0.27488882959567795
    var h: Float64 = 0.01

    var h_gamma = h * t_22
    var sys_indptr: List[Int] = [0]
    var sys_indices: List[Int] = []
    var sys_data: List[Float64] = []
    for i in range(n):
        var m_p = M_s.indptr[i]
        var m_end = M_s.indptr[i + 1]
        var k_p = K_s.indptr[i]
        var k_end = K_s.indptr[i + 1]
        while m_p < m_end or k_p < k_end:
            var m_col = M_s.indices[m_p] if m_p < m_end else n + 1
            var k_col = K_s.indices[k_p] if k_p < k_end else n + 1
            if m_col == k_col:
                sys_indices.append(m_col)
                sys_data.append(M_s.data[m_p] + h_gamma * K_s.data[k_p])
                m_p += 1
                k_p += 1
            elif m_col < k_col:
                sys_indices.append(m_col)
                sys_data.append(M_s.data[m_p])
                m_p += 1
            else:
                sys_indices.append(k_col)
                sys_data.append(h_gamma * K_s.data[k_p])
                k_p += 1
        sys_indptr.append(len(sys_indices))

    var A_real = CSRMatrix[DType.float64](n, n)
    A_real.indptr = sys_indptr^
    A_real.indices = sys_indices^
    A_real.data = sys_data^
    A_real.nrows = n
    A_real.ncols = n
    print("  System matrix nnz: " + String(len(A_real.data)))

    var A_real_csc = csr_to_csc(A_real)
    var lu_real = SparseLU(n)
    lu_real.factorize(A_real_csc)

    var b_test = zeros(n)
    for i in range(n):
        b_test[i] = 1.0 / Float64(n)
    var x_test = lu_real.solve(b_test)

    var ax_test = A_real.spmv(x_test)
    var res_norm_r = 0.0
    var b_norm_r = 0.0
    for i in range(n):
        res_norm_r += (ax_test[i] - b_test[i]) * (ax_test[i] - b_test[i])
        b_norm_r += b_test[i] * b_test[i]
    res_norm_r = sqrt(res_norm_r)
    b_norm_r = sqrt(b_norm_r)
    print("  ||A*x - b|| / ||b|| = " + String(res_norm_r / b_norm_r))
    print("  Result: " + ("PASS" if res_norm_r / b_norm_r < 1e-8 else "FAIL"))

    print()
    print("[Test 4] Verify mass matrix formulation vs J-based formulation")
    var J = compute_jacobian_via_lu(M_s, K_s, n)
    print("  J computed, nnz: " + String(len(J.data)))

    var y0: List[Float64] = []
    for i in range(n):
        y0.append(D_diag[i] * 0.01)
    var y = copy_vec(y0)

    var f0_J = J.spmv(y)
    var w_K = K_s.spmv(y)

    var qt_col_sum_2: Float64 = 0.98925745916384644 + 0.07757466016809490 + 0.12390258911134427

    var rhs_J = zeros(n)
    var rhs_mass = zeros(n)
    for k in range(n):
        rhs_J[k] = qt_col_sum_2 * f0_J[k]
        rhs_mass[k] = -qt_col_sum_2 * w_K[k]

    var I_minus_hgJ_indptr: List[Int] = [0]
    var I_minus_hgJ_indices: List[Int] = []
    var I_minus_hgJ_data: List[Float64] = []
    for i in range(n):
        var has_diag = False
        for p in range(J.indptr[i], J.indptr[i + 1]):
            var j = J.indices[p]
            if j == i:
                I_minus_hgJ_indices.append(j)
                I_minus_hgJ_data.append(1.0 - h * t_22 * J.data[p])
                has_diag = True
            else:
                I_minus_hgJ_indices.append(j)
                I_minus_hgJ_data.append(-h * t_22 * J.data[p])
        if not has_diag:
            I_minus_hgJ_indices.append(i)
            I_minus_hgJ_data.append(1.0)
        I_minus_hgJ_indptr.append(len(I_minus_hgJ_indices))

    var I_hgJ = CSRMatrix[DType.float64](n, n)
    I_hgJ.indptr = I_minus_hgJ_indptr^
    I_hgJ.indices = I_minus_hgJ_indices^
    I_hgJ.data = I_minus_hgJ_data^
    I_hgJ.nrows = n
    I_hgJ.ncols = n

    var I_hgJ_csc = csr_to_csc(I_hgJ)
    var lu_J = SparseLU(n)
    lu_J.factorize(I_hgJ_csc)

    var z2_J = lu_J.solve(rhs_J)
    var z2_mass = lu_real.solve(rhs_mass)

    var diff_norm = 0.0
    var z2_norm = 0.0
    for i in range(n):
        var d = z2_J[i] - z2_mass[i]
        diff_norm += d * d
        z2_norm += z2_J[i] * z2_J[i]
    diff_norm = sqrt(diff_norm)
    z2_norm = sqrt(z2_norm)
    print("  ||z2_J - z2_mass|| / ||z2_J|| = " + String(diff_norm / z2_norm if z2_norm > 0 else diff_norm))
    print("  z2_J[0:3] = " + String(z2_J[0]) + ", " + String(z2_J[1]) + ", " + String(z2_J[2]))
    print("  z2_mass[0:3] = " + String(z2_mass[0]) + ", " + String(z2_mass[1]) + ", " + String(z2_mass[2]))
    print("  Result: " + ("PASS" if diff_norm / z2_norm < 1e-6 else "FAIL"))

    print()
    print("[Test 5] Check RHS equivalence: M_s * rhs_J vs rhs_mass")
    var M_rhs_J = M_s.spmv(rhs_J)
    var rhs_diff = 0.0
    var rhs_norm = 0.0
    for i in range(n):
        var d = M_rhs_J[i] - rhs_mass[i]
        rhs_diff += d * d
        rhs_norm += rhs_mass[i] * rhs_mass[i]
    rhs_diff = sqrt(rhs_diff)
    rhs_norm = sqrt(rhs_norm)
    print("  ||M_s*rhs_J - rhs_mass|| / ||rhs_mass|| = " + String(rhs_diff / rhs_norm if rhs_norm > 0 else rhs_diff))
    print("  Result: " + ("PASS" if rhs_diff / rhs_norm < 1e-8 else "FAIL"))

    print()
    print("[Test 6] Run RADAU5 solver for a few steps")
    var system = TestSystem(M, K)
    var solver = RadauSparseLinearSolver[TestSystem](rtol=1e-4, atol=1e-6, max_step=0.01)
    var y0_list: List[Float64] = []
    for i in range(n):
        y0_list.append(0.01)
    var sol = solver.solve(system, (0.0, 0.001), y0_list)
    print("  Success: " + String(sol.success))
    print("  Message: " + String(sol.message))
    if sol.success and len(sol.t) > 1:
        print("  Steps: " + String(len(sol.t) - 1))
        print("  y_final[0:3]: " + String(sol.y[len(sol.y)-1][0]) + ", " + String(sol.y[len(sol.y)-1][1]) + ", " + String(sol.y[len(sol.y)-1][2]))

    print()
    print("=" * 60)
    print("  Debug test complete")
    print("=" * 60)


def compute_jacobian_via_lu(
    M_s: CSRMatrix[DType.float64],
    K_s: CSRMatrix[DType.float64],
    n: Int,
) -> CSRMatrix[DType.float64]:
    var M_s_csc = csr_to_csc(M_s)
    var lu = SparseLU(n)
    lu.factorize(M_s_csc)

    var J_indptr: List[Int] = [0]
    var J_indices: List[Int] = []
    var J_data: List[Float64] = []

    for col in range(n):
        var k_col = zeros(n)
        for p in range(K_s.indptr[col], K_s.indptr[col + 1]):
            k_col[K_s.indices[p]] = K_s.data[p]

        var j_col = lu.solve(k_col)
        for i in range(n):
            if abs_f64(j_col[i]) > 1e-15:
                J_indices.append(i)
                J_data.append(-j_col[i])
        J_indptr.append(len(J_indices))

    var J = CSRMatrix[DType.float64](n, n)
    J.indptr = J_indptr^
    J.indices = J_indices^
    J.data = J_data^
    J.nrows = n
    J.ncols = n
    return J^
