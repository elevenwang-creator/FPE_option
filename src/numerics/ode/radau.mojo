"""Stiff ODE solver: RadauIIA (order 5, s=3) for linear systems.

Faithful reimplementation of Hairer's Fortran 77 RADAU5 (RADCOR).
For linear M*y' = -K*y, the simplified Newton iteration uses a diagonal
approximation of the Runge-Kutta matrix A (via eigenvalues of A^{-1}),
which does NOT converge in one step. Multiple iterations (up to NIT=7)
are needed, as in the Fortran RADCOR code.

System matrices are h-independent (following Fortran DECOMR/DECOMC):
  Real:    E1 = M*(U1/H) + K   =>  E1_h = U1*M + H*K  (solve E1_h*x = H*b)
  Complex: E2 = (ALPH+i*BETA)*M/H + K  =>  2Nx2N real system

Performance optimizations:
  - Pre-allocated FixedSizeVector work buffers (zero heap allocation in hot loop)
  - In-place spmv via CSRMatrix.spmv_fixed
  - In-place LU solve via SparseLU.solve_inplace
  - SIMD-vectorized element-wise operations via FixedSizeVector methods
  - Fused Newton RHS construction with SIMD

Reference: Hairer & Wanner, "Solving ODEs II", Ch. IV, Sec. 8
           Fortran source: RADAU5 / RADCOR / dc_lapack by E. Hairer, G. Wanner
"""

from numerics.ode.types import ODESolution
from numerics.utils import (
    FixedSizeVector, abs_f64, max_f64, min_f64, zeros, copy_vec, pow_pos,
)
from numerics.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix, csr_to_csc
from sparse.ops import add, scale
from std.math import sqrt, abs, min, max
from std.sys import simd_width_of


comptime SIMD_WIDTH = simd_width_of[DType.float64]()

comptime SQRT6: Float64 = 2.449489742783178

comptime C1: Float64 = (4.0 - SQRT6) / 10.0
comptime C2: Float64 = (4.0 + SQRT6) / 10.0
comptime C1M1: Float64 = C1 - 1.0
comptime C2M1: Float64 = C2 - 1.0
comptime C1MC2: Float64 = C1 - C2

comptime DD1: Float64 = (-13.0 - 7.0 * SQRT6) / 3.0
comptime DD2: Float64 = (-13.0 + 7.0 * SQRT6) / 3.0
comptime DD3: Float64 = -1.0 / 3.0

comptime U1: Float64 = 3.6378165692476072
comptime ALPH: Float64 = 2.6753213032678365
comptime BETA: Float64 = 3.0493570545593676

comptime T11: Float64 = 9.1232394870892942792e-02
comptime T12: Float64 = -1.4125529502095420843e-01
comptime T13: Float64 = -3.0029194105147424492e-02
comptime T21: Float64 = 2.4171793270710701896e-01
comptime T22: Float64 = 2.0412935229379993199e-01
comptime T23: Float64 = 3.8294211275726193779e-01
comptime T31: Float64 = 9.6604818261509293619e-01

comptime TI11: Float64 = 4.3255798900631553510e+00
comptime TI12: Float64 = 3.3919925181580986954e-01
comptime TI13: Float64 = 5.4177705399358748719e-01
comptime TI21: Float64 = -4.1787185915519047273e+00
comptime TI22: Float64 = -3.2768282076106238708e-01
comptime TI23: Float64 = 4.7662355450055045196e-01
comptime TI31: Float64 = -5.0287263494578687595e-01
comptime TI32: Float64 = 2.5719269498556054292e+00
comptime TI33: Float64 = -5.9603920482822492497e-01


trait LinearODESystem:
    def get_M(self) -> CSRMatrix:
        ...

    def get_K(self) -> CSRMatrix:
        ...


struct RadauSparseLinearSolver[System: LinearODESystem]:
    var rtol: Float64
    var atol: Float64
    var max_step: Float64
    var first_step: Float64

    def __init__(
        out self,
        rtol: Float64 = 1e-6,
        atol: Float64 = 1e-8,
        max_step: Float64 = 0.0,
        first_step: Float64 = 0.0,
    ):
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.first_step = first_step

    def solve(
        self,
        system: Self.System,
        t_span: Tuple[Float64, Float64],
        y0: List[Float64],
        t_eval: Optional[List[Float64]] = None,
    ) raises -> ODESolution:
        var M = system.get_M()
        var K = system.get_K()
        var n = len(y0)
        if n != M.nrows or n != M.ncols:
            return ODESolution([], [], False, "RadauSparseLinear: M dimension mismatch")
        if n != K.nrows or n != K.ncols:
            return ODESolution([], [], False, "RadauSparseLinear: K dimension mismatch")

        var t0 = t_span[0]
        var t1 = t_span[1]
        var posneg = 1.0
        if t1 < t0:
            posneg = -1.0

        var t_values: List[Float64] = [t0]
        var y_values: List[List[Float64]] = []
        y_values.append(copy_vec(y0))

        var y = FixedSizeVector(n)
        y.copy_from(y0)
        var t = t0

        var uround: Float64 = 1e-16
        var nit: Int = 7
        var safety: Float64 = 0.9
        var fac1: Float64 = 0.2
        var fac2: Float64 = 8.0
        var quot1: Float64 = 1.0
        var quot2: Float64 = 1.2

        var rtol_work = 0.1 * self.rtol ** (2.0 / 3.0)
        var atol_work = rtol_work * (self.atol / self.rtol)

        var fnewt = max_f64(
            10.0 * uround / rtol_work, min_f64(0.03, rtol_work ** 0.5)
        )

        var scal = FixedSizeVector(n)
        scal.update_scal(atol_work, rtol_work, y)

        var h: Float64
        if self.first_step > 0.0:
            h = self.first_step
        else:
            var k0_list = K.spmv(y0)
            var dnf = 0.0
            for k in range(n):
                dnf += (k0_list[k] / scal[k]) ** 2
            dnf = sqrt(dnf / Float64(n))
            if dnf <= 1e-10:
                h = max_f64(1e-6, abs_f64(t1 - t0) * 1e-3)
            else:
                h = 0.01 / dnf
        if self.max_step > 0.0:
            h = min_f64(h, self.max_step)
        h = min_f64(h, abs_f64(t1 - t0))
        h = posneg * h

        var n_steps: Int = 0
        var n_accepted: Int = 0
        var n_rejected: Int = 0
        var h_old: Float64 = h
        var reject = False
        var first = True
        var faccon: Float64 = 1.0
        var theta: Float64 = 0.0
        var hacc: Float64 = 0.0
        var erracc: Float64 = 1e-2
        var cfac = safety * Float64(1 + 2 * nit)

        var h_lu: Float64 = 0.0
        var lu_real = SparseLU(n)
        var lu_complex = SparseLU(2 * n)

        var w = FixedSizeVector(n)
        var Z1 = FixedSizeVector(n)
        var Z2 = FixedSizeVector(n)
        var Z3 = FixedSizeVector(n)
        var F1 = FixedSizeVector(n)
        var F2 = FixedSizeVector(n)
        var F3 = FixedSizeVector(n)
        var KZ1 = FixedSizeVector(n)
        var KZ2 = FixedSizeVector(n)
        var KZ3 = FixedSizeVector(n)
        var MF1 = FixedSizeVector(n)
        var MF2 = FixedSizeVector(n)
        var MF3 = FixedSizeVector(n)
        var rhs_real = FixedSizeVector(n)
        var rhs_complex = FixedSizeVector(2 * n)
        var dF1 = FixedSizeVector(n)
        var dF2_dF3 = FixedSizeVector(2 * n)
        var work_n = FixedSizeVector(n)
        var work_2n = FixedSizeVector(2 * n)
        var CONT = FixedSizeVector(n)
        var M_CONT = FixedSizeVector(n)
        var rhs_err = FixedSizeVector(n)
        var error_vec = FixedSizeVector(n)
        var scal_err = FixedSizeVector(n)

        while posneg * (t1 - t) > uround * max_f64(abs_f64(t), abs_f64(t1)):
            if n_steps > 100000:
                return ODESolution(
                    t_values^,
                    y_values^,
                    False,
                    "RadauSparseLinear: max steps exceeded",
                )

            if posneg * (t + 1.01 * h - t1) > 0.0:
                h = t1 - t

            var h_abs = abs_f64(h)
            if h_abs < 1e-14:
                return ODESolution(
                    t_values^,
                    y_values^,
                    False,
                    "RadauSparseLinear: step size underflow",
                )

            var need_lu: Bool
            if h_lu == 0.0:
                need_lu = True
            else:
                var h_ratio = abs_f64(h / h_lu)
                need_lu = h_ratio < quot1 or h_ratio > quot2
            if need_lu:
                var E1_h = self._build_real_system(M, K, h, n)
                var E1_csc = csr_to_csc(E1_h)
                lu_real = SparseLU(n)
                lu_real.factorize(E1_csc)

                var E2_h = self._build_complex_system(M, K, h, n)
                var E2_csc = csr_to_csc(E2_h)
                lu_complex = SparseLU(2 * n)
                lu_complex.factorize(E2_csc)

                h_lu = h

            scal.update_scal(atol_work, rtol_work, y)

            K.spmv_fixed(y, w)

            if first:
                Z1.zero_out()
                Z2.zero_out()
                Z3.zero_out()
                F1.zero_out()
                F2.zero_out()
                F3.zero_out()
            else:
                Z1.zero_out()
                Z2.zero_out()
                Z3.zero_out()
                F1.zero_out()
                F2.zero_out()
                F3.zero_out()

            var newt: Int = 0
            faccon = max_f64(faccon, uround) ** 0.8
            var theta_loc = abs_f64(theta)
            var dynold: Float64 = 0.0
            var thqold: Float64 = 0.0
            var converged = False
            var newt_fail = False

            while newt < nit:
                K.spmv_fixed(Z1, KZ1)
                K.spmv_fixed(Z2, KZ2)
                K.spmv_fixed(Z3, KZ3)

                M.spmv_fixed(F1, MF1)
                M.spmv_fixed(F2, MF2)
                M.spmv_fixed(F3, MF3)

                self._build_newton_rhs(
                    rhs_real, rhs_complex, w, KZ1, KZ2, KZ3, MF1, MF2, MF3, h, n
                )

                dF1.copy_from_fixed(rhs_real)
                lu_real.solve_inplace(dF1, work_n)

                dF2_dF3.copy_from_fixed(rhs_complex)
                lu_complex.solve_inplace(dF2_dF3, work_2n)

                newt += 1

                var dyno_sq = 0.0
                comptime width = SIMD_WIDTH
                var k = 0
                while k + width <= n:
                    var s_scal = SIMD[DType.float64, width]()
                    var s_dF1 = SIMD[DType.float64, width]()
                    var s_dF2 = SIMD[DType.float64, width]()
                    var s_dF3 = SIMD[DType.float64, width]()
                    for j in range(width):
                        s_scal[j] = scal[k + j]
                        s_dF1[j] = dF1[k + j]
                        s_dF2[j] = dF2_dF3[k + j]
                        s_dF3[j] = dF2_dF3[n + k + j]
                    var r1 = s_dF1 / s_scal
                    var r2 = s_dF2 / s_scal
                    var r3 = s_dF3 / s_scal
                    dyno_sq += (r1 * r1 + r2 * r2 + r3 * r3).reduce_add()
                    k += width
                while k < n:
                    var s = scal[k]
                    dyno_sq += (dF1[k] / s) ** 2 + (dF2_dF3[k] / s) ** 2 + (
                        dF2_dF3[n + k] / s
                    ) ** 2
                    k += 1
                var dyno = sqrt(dyno_sq / Float64(3 * n))

                if newt > 1 and newt < nit:
                    var thq = dyno / max_f64(dynold, uround)
                    if newt == 2:
                        theta_loc = thq
                    else:
                        theta_loc = sqrt(thq * thqold)
                    thqold = thq
                    if theta_loc < 0.99:
                        faccon = theta_loc / (1.0 - theta_loc)
                        var dyth = (
                            faccon
                            * dyno
                            * pow_pos(theta_loc, Float64(nit - 1 - newt))
                            / fnewt
                        )
                        if dyth >= 1.0:
                            var qnewt = max_f64(1e-4, min_f64(20.0, dyth))
                            var hhfac = 0.8 * qnewt ** (
                                -1.0 / Float64(4 + nit - 1 - newt)
                            )
                            h = hhfac * h
                            newt_fail = True
                            break
                    else:
                        newt_fail = True
                        break

                dynold = max_f64(dyno, uround)

                F1.addassign(dF1)
                F2.addassign_offset(dF2_dF3, 0)
                F3.addassign_offset(dF2_dF3, n)

                Z1.lin_comb_3(T11, F1, T12, F2, T13, F3)
                Z2.lin_comb_3(T21, F1, T22, F2, T23, F3)
                Z3.lin_comb_2(T31, F1, 1.0, F2)

                if faccon * dyno <= fnewt:
                    converged = True
                    break

            if newt_fail or not converged:
                reject = True
                if first:
                    h = h * 0.1
                else:
                    h = h * 0.5
                if abs_f64(h) < 1e-14:
                    return ODESolution(
                        t_values^,
                        y_values^,
                        False,
                        "RadauSparseLinear: step size underflow in Newton",
                    )
                continue

            theta = theta_loc

            CONT.lin_comb_3(DD1, Z1, DD2, Z2, DD3, Z3)

            M.spmv_fixed(CONT, M_CONT)
            rhs_err.sub_scaled(M_CONT, h, w)

            error_vec.copy_from_fixed(rhs_err)
            lu_real.solve_inplace(error_vec, work_n)

            scal_err.update_scal(atol_work, rtol_work, y)
            var err_norm_sq = error_vec.scaled_norm_sq(scal_err)
            var err_norm = sqrt(err_norm_sq / Float64(n))
            if err_norm < 1e-10:
                err_norm = 1e-10

            if err_norm >= 1.0 and (first or reject):
                var y_trial = FixedSizeVector(n)
                y_trial.add_from(y, error_vec)
                K.spmv_fixed(y_trial, w)
                rhs_err.sub_scaled(M_CONT, h, w)

                error_vec.copy_from_fixed(rhs_err)
                lu_real.solve_inplace(error_vec, work_n)

                err_norm_sq = error_vec.scaled_norm_sq(scal_err)
                err_norm = sqrt(err_norm_sq / Float64(n))
                if err_norm < 1e-10:
                    err_norm = 1e-10

            var fac = min_f64(safety, cfac / Float64(newt + 2 * nit))
            var quot = max_f64(
                1.0 / fac2, min_f64(1.0 / fac1, err_norm ** 0.25 / fac)
            )
            var h_new = h / quot

            if err_norm < 1.0:
                first = False
                n_accepted += 1
                t = t + h
                y.addassign(Z3)

                n_steps += 1

                t_values.append(t)
                y_values.append(y.to_list())

                if n_accepted > 1:
                    var facgus = (
                        (hacc / h)
                        * (err_norm ** 2 / erracc) ** 0.25
                        / safety
                    )
                    facgus = max_f64(1.0 / fac2, min_f64(1.0 / fac1, facgus))
                    quot = max_f64(quot, facgus)
                    h_new = h / quot

                hacc = h
                erracc = max_f64(1e-2, err_norm)
                h_old = h

                h_new = posneg * min_f64(abs_f64(h_new), abs_f64(t1 - t))
                if self.max_step > 0.0:
                    h_new = posneg * min_f64(abs_f64(h_new), self.max_step)
                if reject:
                    h_new = posneg * min_f64(abs_f64(h_new), abs_f64(h))
                reject = False
                h = h_new
            else:
                reject = True
                if first:
                    h = h * 0.1
                else:
                    h = h_new
                if n_accepted >= 1:
                    n_rejected += 1

        return ODESolution(
            t_values^,
            y_values^,
            True,
            "RadauSparseLinear: " + String(n_steps) + " steps",
        )

    def _build_newton_rhs(
        self,
        mut rhs_real: FixedSizeVector,
        mut rhs_complex: FixedSizeVector,
        w: FixedSizeVector,
        KZ1: FixedSizeVector,
        KZ2: FixedSizeVector,
        KZ3: FixedSizeVector,
        MF1: FixedSizeVector,
        MF2: FixedSizeVector,
        MF3: FixedSizeVector,
        h: Float64,
        n: Int,
    ):
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= n:
            var sw = SIMD[DType.float64, width]()
            var sKZ1 = SIMD[DType.float64, width]()
            var sKZ2 = SIMD[DType.float64, width]()
            var sKZ3 = SIMD[DType.float64, width]()
            for k in range(width):
                sw[k] = w.ptr()[i + k]
                sKZ1[k] = KZ1.ptr()[i + k]
                sKZ2[k] = KZ2.ptr()[i + k]
                sKZ3[k] = KZ3.ptr()[i + k]

            var sf1 = -sw - sKZ1
            var sf2 = -sw - sKZ2
            var sf3 = -sw - sKZ3

            var sW1 = TI11 * sf1 + TI12 * sf2 + TI13 * sf3
            var sW2 = TI21 * sf1 + TI22 * sf2 + TI23 * sf3
            var sW3 = TI31 * sf1 + TI32 * sf2 + TI33 * sf3

            var sMF1 = SIMD[DType.float64, width]()
            var sMF2 = SIMD[DType.float64, width]()
            var sMF3 = SIMD[DType.float64, width]()
            for k in range(width):
                sMF1[k] = MF1.ptr()[i + k]
                sMF2[k] = MF2.ptr()[i + k]
                sMF3[k] = MF3.ptr()[i + k]

            var s_rhs_real = h * sW1 - U1 * sMF1
            var s_rhs_cx = h * sW2 - ALPH * sMF2 + BETA * sMF3
            var s_rhs_cx2 = h * sW3 - ALPH * sMF3 - BETA * sMF2

            for k in range(width):
                rhs_real.ptr()[i + k] = s_rhs_real[k]
                rhs_complex.ptr()[i + k] = s_rhs_cx[k]
                rhs_complex.ptr()[n + i + k] = s_rhs_cx2[k]
            i += width

        while i < n:
            var f1_k = -w[i] - KZ1[i]
            var f2_k = -w[i] - KZ2[i]
            var f3_k = -w[i] - KZ3[i]
            var W1_k = TI11 * f1_k + TI12 * f2_k + TI13 * f3_k
            var W2_k = TI21 * f1_k + TI22 * f2_k + TI23 * f3_k
            var W3_k = TI31 * f1_k + TI32 * f2_k + TI33 * f3_k
            rhs_real[i] = h * W1_k - U1 * MF1[i]
            rhs_complex[i] = h * W2_k - ALPH * MF2[i] + BETA * MF3[i]
            rhs_complex[n + i] = h * W3_k - ALPH * MF3[i] - BETA * MF2[i]
            i += 1

    def _build_real_system(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
    ) -> CSRMatrix:
        return add(scale(U1, M), scale(h, K))

    def _build_complex_system(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
    ) -> CSRMatrix:
        var n2 = 2 * n

        var row_nnz = alloc[Int](n2)
        for i in range(n):
            var count = 0
            var m_p = M.indptr[i]
            var m_end = M.indptr[i + 1]
            var k_p = K.indptr[i]
            var k_end = K.indptr[i + 1]
            while m_p < m_end and k_p < k_end:
                count += 1
                if M.indices[m_p] < K.indices[k_p]:
                    m_p += 1
                elif M.indices[m_p] > K.indices[k_p]:
                    k_p += 1
                else:
                    m_p += 1
                    k_p += 1
            count += (m_end - m_p) + (k_end - k_p)
            count += M.indptr[i + 1] - M.indptr[i]
            row_nnz[i] = count

        for i in range(n):
            var count = M.indptr[i + 1] - M.indptr[i]
            var m_p = M.indptr[i]
            var m_end = M.indptr[i + 1]
            var k_p = K.indptr[i]
            var k_end = K.indptr[i + 1]
            while m_p < m_end and k_p < k_end:
                count += 1
                if M.indices[m_p] < K.indices[k_p]:
                    m_p += 1
                elif M.indices[m_p] > K.indices[k_p]:
                    k_p += 1
                else:
                    m_p += 1
                    k_p += 1
            count += (m_end - m_p) + (k_end - k_p)
            row_nnz[n + i] = count

        var total_nnz = 0
        for i in range(n2):
            total_nnz += row_nnz[i]

        var result = CSRMatrix(n2, n2, total_nnz)
        result.indptr[0] = 0
        for i in range(n2):
            result.indptr[i + 1] = result.indptr[i] + row_nnz[i]

        var dest = 0

        for i in range(n):
            var m_p = M.indptr[i]
            var m_end = M.indptr[i + 1]
            var k_p = K.indptr[i]
            var k_end = K.indptr[i + 1]

            while m_p < m_end and k_p < k_end:
                var m_col = M.indices[m_p]
                var k_col = K.indices[k_p]
                if m_col == k_col:
                    result.data[dest] = ALPH * M.data[m_p] + h * K.data[k_p]
                    result.indices[dest] = m_col
                    dest += 1
                    m_p += 1
                    k_p += 1
                elif m_col < k_col:
                    result.data[dest] = ALPH * M.data[m_p]
                    result.indices[dest] = m_col
                    dest += 1
                    m_p += 1
                else:
                    result.data[dest] = h * K.data[k_p]
                    result.indices[dest] = k_col
                    dest += 1
                    k_p += 1

            while m_p < m_end:
                result.data[dest] = ALPH * M.data[m_p]
                result.indices[dest] = M.indices[m_p]
                dest += 1
                m_p += 1
            while k_p < k_end:
                result.data[dest] = h * K.data[k_p]
                result.indices[dest] = K.indices[k_p]
                dest += 1
                k_p += 1

            for p in range(M.indptr[i], M.indptr[i + 1]):
                result.data[dest] = -BETA * M.data[p]
                result.indices[dest] = n + M.indices[p]
                dest += 1

        for i in range(n):
            for p in range(M.indptr[i], M.indptr[i + 1]):
                result.data[dest] = BETA * M.data[p]
                result.indices[dest] = M.indices[p]
                dest += 1

            var m_p = M.indptr[i]
            var m_end = M.indptr[i + 1]
            var k_p = K.indptr[i]
            var k_end = K.indptr[i + 1]

            while m_p < m_end and k_p < k_end:
                var m_col = M.indices[m_p]
                var k_col = K.indices[k_p]
                if m_col == k_col:
                    result.data[dest] = ALPH * M.data[m_p] + h * K.data[k_p]
                    result.indices[dest] = n + m_col
                    dest += 1
                    m_p += 1
                    k_p += 1
                elif m_col < k_col:
                    result.data[dest] = ALPH * M.data[m_p]
                    result.indices[dest] = n + m_col
                    dest += 1
                    m_p += 1
                else:
                    result.data[dest] = h * K.data[k_p]
                    result.indices[dest] = n + k_col
                    dest += 1
                    k_p += 1

            while m_p < m_end:
                result.data[dest] = ALPH * M.data[m_p]
                result.indices[dest] = n + M.indices[m_p]
                dest += 1
                m_p += 1
            while k_p < k_end:
                result.data[dest] = h * K.data[k_p]
                result.indices[dest] = n + K.indices[k_p]
                dest += 1
                k_p += 1

        row_nnz.free()
        return result^
