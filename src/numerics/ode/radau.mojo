"""Stiff ODE solver: RadauIIA (order 5, s=3) for linear systems.

Faithful reimplementation of Hairer's Fortran 77 RADAU5 (RADCOR).
For linear M*y' = -K*y, the simplified Newton iteration uses a diagonal
approximation of the Runge-Kutta matrix A (via eigenvalues of A^{-1}),
which does NOT converge in one step. Multiple iterations (up to NIT=6)
are needed, as in the Fortran RADCOR code.

System matrices are h-independent (following Fortran DECOMR/DECOMC):
    Real: E1 = M*(U1/H) + K => E1_h = U1*M + H*K (solve E1_h*x = H*b)
    Complex: E2 = (ALPH+i*BETA)*M/H + K => 2Nx2N real system

GMRES optimization: replaces 2n x 2n complex LU factorization with
left-preconditioned GMRES on the equivalent 2n real system, using
block-diagonal preconditioner P = diag(E1, E1). This avoids the O(n^2)
memory and O(n * nnz_per_row * fill-in) factorization cost of the 2n system.

Performance optimizations:
- Pre-allocated FixedSizeVector work buffers (zero heap allocation in hot loop)
- Fused triple SpMV: iterate K/M once per Newton iter (vs 6 separate passes)
- In-place LU solve via SparseLU.solve_inplace
- Proper SIMD vector loads via UnsafePointer.load[width=W]()
- Fused Newton RHS construction with SIMD
- Z-extrapolation from CONT polynomial for warm Newton starts
- Fused real system construction (no temporary CSR allocations)
- GMRES for complex system: avoids 2n LU, uses n-length vectors only
- GMRES uses real inner products (E2 is a real 2n x 2n matrix)

Reference: Hairer & Wanner, "Solving ODEs II", Ch. IV, Sec. 8
Fortran source: RADAU5 / RADCOR / dc_lapack by E. Hairer, G. Wanner
"""

from numerics.ode.types import ODESolution
from numerics.utils import (
    FixedSizeVector,
    pow_pos,
)
from numerics.utils.sparse_lu import SparseLU
from sparse.csr import CSRMatrix
from sparse.csc import CSCMatrix
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

comptime TI11: Float64 = 4.3255798900631553510e00
comptime TI12: Float64 = 3.3919925181580986954e-01
comptime TI13: Float64 = 5.4177705399358748719e-01
comptime TI21: Float64 = -4.1787185915519047273e00
comptime TI22: Float64 = -3.2768282076106238708e-01
comptime TI23: Float64 = 4.7662355450055045196e-01
comptime TI31: Float64 = -5.0287263494578687595e-01
comptime TI32: Float64 = 2.5719269498556054292e00
comptime TI33: Float64 = -5.9603920482822492497e-01

comptime GMRES_MAX_KRYLOV: Int = 10
comptime GMRES_TOL: Float64 = 1e-10


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
            return ODESolution(
                [], [], False, "RadauSparseLinear: M dimension mismatch"
            )
        if n != K.nrows or n != K.ncols:
            return ODESolution(
                [], [], False, "RadauSparseLinear: K dimension mismatch"
            )

        var t0 = t_span[0]
        var t1 = t_span[1]
        var posneg = 1.0
        if t1 < t0:
            posneg = -1.0

        var t_values: List[Float64] = [t0]
        var y_values: List[List[Float64]] = []
        y_values.append(y0.copy())

        var y = FixedSizeVector(n)
        y.copy_from(y0)
        var t = t0
        var t_eval_idx: Int = 0

        if t_eval is not None:
            if t_eval_idx < len(t_eval.value()):
                if abs(t_eval.value()[t_eval_idx] - t0) <= 1e-15:
                    t_eval_idx += 1

        var uround: Float64 = 1e-16
        var nit: Int = 6
        var safety: Float64 = 0.9
        var fac1: Float64 = 0.2
        var fac2: Float64 = 10.0
        var quot1: Float64 = 1.0
        var quot2: Float64 = 1.2

        var rtol_work = 0.1 * self.rtol ** (2.0 / 3.0)
        var atol_work = rtol_work * (self.atol / self.rtol)

        var fnewt = max(
            10.0 * uround / rtol_work, min(0.03, sqrt(rtol_work))
        )

        var scal = FixedSizeVector(n)
        scal.update_scal(atol_work, rtol_work, y)

        var h: Float64
        if self.first_step > 0.0:
            h = self.first_step
        else:
            var k0_list = K.spmv_new(y0)
            var dnf = 0.0
            for k in range(n):
                dnf += (k0_list[k] / scal[k]) ** 2
            dnf = sqrt(dnf / Float64(n))
            if dnf <= 1e-10:
                h = max(1e-6, abs(t1 - t0) * 1e-3)
            else:
                h = 0.01 / dnf
            if self.max_step > 0.0:
                h = min(h, self.max_step)
        h = min(h, abs(t1 - t0))
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
        var rhs_re = FixedSizeVector(n)
        var rhs_im = FixedSizeVector(n)
        var dF1 = FixedSizeVector(n)
        var dF2 = FixedSizeVector(n)
        var dF3 = FixedSizeVector(n)
        var work_n = FixedSizeVector(n)
        var CONT = FixedSizeVector(4 * n)
        var CONT_ERR = FixedSizeVector(n)
        var M_CONT = FixedSizeVector(n)
        var rhs_err = FixedSizeVector(n)
        var error_vec = FixedSizeVector(n)
        var scal_err = FixedSizeVector(n)

        # GMRES work arrays
        var gmres_V_re = List[FixedSizeVector]()
        var gmres_V_im = List[FixedSizeVector]()
        for _ in range(GMRES_MAX_KRYLOV + 1):
            gmres_V_re.append(FixedSizeVector(n))
            gmres_V_im.append(FixedSizeVector(n))

        var gmres_H = List[List[Float64]]()
        for _ in range(GMRES_MAX_KRYLOV + 1):
            var row = List[Float64](length=GMRES_MAX_KRYLOV, fill=0.0)
            gmres_H.append(row^)

        var gmres_cs = List[Float64](length=GMRES_MAX_KRYLOV + 1, fill=0.0)
        var gmres_sn = List[Float64](length=GMRES_MAX_KRYLOV + 1, fill=0.0)
        var gmres_s = List[Float64](length=GMRES_MAX_KRYLOV + 1, fill=0.0)

        var gmres_w_re = FixedSizeVector(n)
        var gmres_w_im = FixedSizeVector(n)
        var gmres_tmp_re = FixedSizeVector(n)
        var gmres_tmp_im = FixedSizeVector(n)
        var gmres_tmp2 = FixedSizeVector(n)

        # Pre-compute merged sparsity pattern for E1 and E_diag
        var E1_cached = self._build_real_system(M, K, 1.0, n)
        var E1_csc_cached = E1_cached.to_csc()
        var E_diag_cached = self._build_diag_system(M, K, 1.0, n)
        var E_diag_csc_cached = E_diag_cached.to_csc()

        while posneg * (t1 - t) > uround * max(abs(t), abs(t1)):
            if n_steps > 100000:
                return ODESolution(
                    t_values^,
                    y_values^,
                    False,
                    "RadauSparseLinear: max steps exceeded",
                )

            if posneg * (t + 1.01 * h - t1) > 0.0:
                h = t1 - t

            if t_eval is not None:
                if t_eval_idx < len(t_eval.value()):
                    var te = t_eval.value()[t_eval_idx]
                    if posneg * (t + 1.01 * h - te) > 0.0 and posneg * (te - t) > 0.0:
                        h = te - t

            var h_abs = abs(h)
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
                var h_ratio = abs(h / h_lu)
                need_lu = h_ratio < quot1 or h_ratio > quot2
            if need_lu:
                self._update_real_data(M, K, h, n, E1_cached, E1_csc_cached)
                lu_real.factorize(E1_csc_cached)

                self._update_diag_data(
                    M, K, h, n, E_diag_cached, E_diag_csc_cached
                )

                h_lu = h

            scal.update_scal(atol_work, rtol_work, y)

            K.spmv(y, w)

            if first or reject:
                Z1.zero_out()
                Z2.zero_out()
                Z3.zero_out()
                F1.zero_out()
                F2.zero_out()
                F3.zero_out()
            else:
                var ratio = h / h_old
                var c1r = C1 * ratio
                var c2r = C2 * ratio
                var c3r = ratio
                for k_idx in range(n):
                    var ak1 = CONT[k_idx + n]
                    var ak2 = CONT[k_idx + 2 * n]
                    var ak3 = CONT[k_idx + 3 * n]
                    Z1[k_idx] = c1r * (
                        ak1 + (c1r - C2M1) * (ak2 + (c1r - C1M1) * ak3)
                    )
                    Z2[k_idx] = c2r * (
                        ak1 + (c2r - C2M1) * (ak2 + (c2r - C1M1) * ak3)
                    )
                    Z3[k_idx] = c3r * (
                        ak1 + (c3r - C2M1) * (ak2 + (c3r - C1M1) * ak3)
                    )
                F1.lin_comb_3(TI11, Z1, TI12, Z2, TI13, Z3)
                F2.lin_comb_3(TI21, Z1, TI22, Z2, TI23, Z3)
                F3.lin_comb_3(TI31, Z1, TI32, Z2, TI33, Z3)

            var newt: Int = 0
            faccon = max(faccon, uround) ** 0.8
            var theta_loc = abs(theta)
            var dynold: Float64 = 0.0
            var thqold: Float64 = 0.0
            var converged = False
            var newt_fail = False

            while newt < nit:
                K.spmv_triple(Z1, Z2, Z3, KZ1, KZ2, KZ3)
                M.spmv_triple(F1, F2, F3, MF1, MF2, MF3)

                self._build_newton_rhs(
                    rhs_real,
                    rhs_re,
                    rhs_im,
                    w,
                    KZ1,
                    KZ2,
                    KZ3,
                    MF1,
                    MF2,
                    MF3,
                    h,
                    n,
                )

                dF1.copy_from_fixed(rhs_real)
                lu_real.solve_inplace(dF1, work_n)

                self._solve_complex_gmres(
                    M,
                    E_diag_cached,
                    lu_real,
                    rhs_re,
                    rhs_im,
                    dF2,
                    dF3,
                    work_n,
                    n,
                    gmres_V_re,
                    gmres_V_im,
                    gmres_H,
                    gmres_cs,
                    gmres_sn,
                    gmres_s,
                    gmres_w_re,
                    gmres_w_im,
                    gmres_tmp_re,
                    gmres_tmp_im,
                    gmres_tmp2,
                )

                newt += 1

                # Convergence norm with SIMD
                var dyno_sq = 0.0
                comptime width = SIMD_WIDTH
                var k = 0
                while k + width <= n:
                    var s_scal = (scal.ptr() + k).load[width=width]()
                    var s_dF1 = (dF1.ptr() + k).load[width=width]()
                    var s_dF2 = (dF2.ptr() + k).load[width=width]()
                    var s_dF3 = (dF3.ptr() + k).load[width=width]()
                    var r1 = s_dF1 / s_scal
                    var r2 = s_dF2 / s_scal
                    var r3 = s_dF3 / s_scal
                    dyno_sq += (r1 * r1 + r2 * r2 + r3 * r3).reduce_add()
                    k += width
                while k < n:
                    var s = scal[k]
                    dyno_sq += (
                        (dF1[k] / s) ** 2
                        + (dF2[k] / s) ** 2
                        + (dF3[k] / s) ** 2
                    )
                    k += 1
                var dyno = sqrt(dyno_sq / Float64(3 * n))

                if newt > 1 and newt < nit:
                    var thq = dyno / max(dynold, uround)
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
                            var qnewt = max(1e-4, min(20.0, dyth))
                            var hhfac = 0.8 * qnewt ** (
                                -1.0 / Float64(4 + nit - 1 - newt)
                            )
                            h = hhfac * h
                            newt_fail = True
                            break
                    else:
                        newt_fail = True
                        break

                dynold = max(dyno, uround)

                F1.addassign(dF1)
                F2.addassign(dF2)
                F3.addassign(dF3)

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
                if abs(h) < 1e-14:
                    return ODESolution(
                        t_values^,
                        y_values^,
                        False,
                        "RadauSparseLinear: step size underflow in Newton",
                    )
                continue

            theta = theta_loc

            for k_idx in range(n):
                var z2i = Z2[k_idx]
                var z1i = Z1[k_idx]
                CONT[k_idx] = y[k_idx] + Z3[k_idx]
                CONT[k_idx + n] = (z2i - Z3[k_idx]) / C2M1
                var ak = (z1i - z2i) / C1MC2
                var acont3 = z1i / C1
                acont3 = (ak - acont3) / C2
                CONT[k_idx + 2 * n] = (ak - CONT[k_idx + n]) / C1M1
                CONT[k_idx + 3 * n] = CONT[k_idx + 2 * n] - acont3
            CONT_ERR.lin_comb_3(DD1, Z1, DD2, Z2, DD3, Z3)

            M.spmv(CONT_ERR, M_CONT)
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
                K.spmv(y_trial, w)
                rhs_err.sub_scaled(M_CONT, h, w)

                error_vec.copy_from_fixed(rhs_err)
                lu_real.solve_inplace(error_vec, work_n)

                err_norm_sq = error_vec.scaled_norm_sq(scal_err)
                err_norm = sqrt(err_norm_sq / Float64(n))
                if err_norm < 1e-10:
                    err_norm = 1e-10

            var fac = min(safety, cfac / Float64(newt + 2 * nit))
            var quot = max(
                1.0 / fac2, min(1.0 / fac1, sqrt(sqrt(err_norm)) / fac)
            )
            var h_new = h / quot

            if err_norm < 1.0:
                first = False
                n_accepted += 1
                var t_old = t
                t = t + h
                y.addassign(Z3)

                n_steps += 1

                if t_eval is None:
                    t_values.append(t)
                    y_values.append(y.to_list())
                else:
                    while t_eval_idx < len(t_eval.value()):
                        var te = t_eval.value()[t_eval_idx]
                        if posneg * (te - t) > uround * max(abs(t), abs(te)):
                            break
                        if abs(te - t) <= uround * max(abs(t), abs(te)):
                            t_values.append(te)
                            y_values.append(y.to_list())
                        else:
                            var s_val = (te - t_old) / h_old
                            var y_interp: List[Float64] = []
                            for k_idx in range(n):
                                y_interp.append(contr5(k_idx, s_val, CONT, n))
                            t_values.append(te)
                            y_values.append(y_interp^)
                        t_eval_idx += 1

                if n_accepted > 1:
                    var facgus = (
                        (hacc / h) * sqrt(sqrt(err_norm**2 / erracc)) / safety
                    )
                    facgus = max(1.0 / fac2, min(1.0 / fac1, facgus))
                    quot = max(quot, facgus)
                    h_new = h / quot
                hacc = h
                erracc = max(1e-2, err_norm)
                h_old = h

                h_new = posneg * min(abs(h_new), abs(t1 - t))
                if self.max_step > 0.0:
                    h_new = posneg * min(abs(h_new), self.max_step)
                if reject:
                    h_new = posneg * min(abs(h_new), abs(h))
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
        mut rhs_re: FixedSizeVector,
        mut rhs_im: FixedSizeVector,
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
            var sw = (w.ptr() + i).load[width=width]()
            var sKZ1 = (KZ1.ptr() + i).load[width=width]()
            var sKZ2 = (KZ2.ptr() + i).load[width=width]()
            var sKZ3 = (KZ3.ptr() + i).load[width=width]()

            var neg_sw = -sw
            var sf1 = neg_sw - sKZ1
            var sf2 = neg_sw - sKZ2
            var sf3 = neg_sw - sKZ3

            var sW1 = TI11 * sf1 + TI12 * sf2 + TI13 * sf3
            var sW2 = TI21 * sf1 + TI22 * sf2 + TI23 * sf3
            var sW3 = TI31 * sf1 + TI32 * sf2 + TI33 * sf3

            var sMF1 = (MF1.ptr() + i).load[width=width]()
            var sMF2 = (MF2.ptr() + i).load[width=width]()
            var sMF3 = (MF3.ptr() + i).load[width=width]()

            var s_rhs_real = h * sW1 - U1 * sMF1
            var s_rhs_re = h * sW2 - ALPH * sMF2 + BETA * sMF3
            var s_rhs_im = h * sW3 - ALPH * sMF3 - BETA * sMF2

            (rhs_real.ptr() + i).store[width=width](s_rhs_real)
            (rhs_re.ptr() + i).store[width=width](s_rhs_re)
            (rhs_im.ptr() + i).store[width=width](s_rhs_im)
            i += width

        while i < n:
            var f1_k = -w[i] - KZ1[i]
            var f2_k = -w[i] - KZ2[i]
            var f3_k = -w[i] - KZ3[i]
            var W1_k = TI11 * f1_k + TI12 * f2_k + TI13 * f3_k
            var W2_k = TI21 * f1_k + TI22 * f2_k + TI23 * f3_k
            var W3_k = TI31 * f1_k + TI32 * f2_k + TI33 * f3_k
            rhs_real[i] = h * W1_k - U1 * MF1[i]
            rhs_re[i] = h * W2_k - ALPH * MF2[i] + BETA * MF3[i]
            rhs_im[i] = h * W3_k - ALPH * MF3[i] - BETA * MF2[i]
            i += 1

    def _build_real_system(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
    ) -> CSRMatrix:
        # E1_h = U1*M + h*K via single-pass sorted merge
        var row_nnz = alloc[Int](n)
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
            row_nnz[i] = count

        var total_nnz = 0
        for i in range(n):
            total_nnz += row_nnz[i]

        var result = CSRMatrix(n, n, total_nnz)
        result.indptr[0] = 0
        for i in range(n):
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
                    result.data[dest] = U1 * M.data[m_p] + h * K.data[k_p]
                    result.indices[dest] = m_col
                    dest += 1
                    m_p += 1
                    k_p += 1
                elif m_col < k_col:
                    result.data[dest] = U1 * M.data[m_p]
                    result.indices[dest] = m_col
                    dest += 1
                    m_p += 1
                else:
                    result.data[dest] = h * K.data[k_p]
                    result.indices[dest] = k_col
                    dest += 1
                    k_p += 1
            while m_p < m_end:
                result.data[dest] = U1 * M.data[m_p]
                result.indices[dest] = M.indices[m_p]
                dest += 1
                m_p += 1
            while k_p < k_end:
                result.data[dest] = h * K.data[k_p]
                result.indices[dest] = K.indices[k_p]
                dest += 1
                k_p += 1

        row_nnz.free()
        return result^

    def _update_real_data(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
        mut E1: CSRMatrix,
        mut E1_csc: CSCMatrix,
    ):
        """Numeric-only update of E1 = U1*M + h*K. No structural rebuild."""
        # Update CSR data[]
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
                    E1.data[dest] = U1 * M.data[m_p] + h * K.data[k_p]
                    dest += 1
                    m_p += 1
                    k_p += 1
                elif m_col < k_col:
                    E1.data[dest] = U1 * M.data[m_p]
                    dest += 1
                    m_p += 1
                else:
                    E1.data[dest] = h * K.data[k_p]
                    dest += 1
                    k_p += 1
            while m_p < m_end:
                E1.data[dest] = U1 * M.data[m_p]
                dest += 1
                m_p += 1
            while k_p < k_end:
                E1.data[dest] = h * K.data[k_p]
                dest += 1
                k_p += 1
        # Scatter CSR data into CSC data (same positions, same pattern)
        for i in range(n):
            for p in range(E1.indptr[i], E1.indptr[i + 1]):
                var j = E1.indices[p]
                for q in range(E1_csc.colptr[j], E1_csc.colptr[j + 1]):
                    if E1_csc.indices[q] == i:
                        E1_csc.data[q] = E1.data[p]
                        break

    def _build_diag_system(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
    ) -> CSRMatrix:
        """Build E_diag = ALPH*M + h*K (n x n diagonal block of complex system).
        """
        var row_nnz = alloc[Int](n)
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
            row_nnz[i] = count

        var total_nnz = 0
        for i in range(n):
            total_nnz += row_nnz[i]

        var result = CSRMatrix(n, n, total_nnz)
        result.indptr[0] = 0
        for i in range(n):
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

        row_nnz.free()
        return result^

    def _update_diag_data(
        self,
        M: CSRMatrix,
        K: CSRMatrix,
        h: Float64,
        n: Int,
        mut E_diag: CSRMatrix,
        mut E_diag_csc: CSCMatrix,
    ):
        """Numeric-only update of E_diag = ALPH*M + h*K. No structural rebuild.
        """
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
                    E_diag.data[dest] = ALPH * M.data[m_p] + h * K.data[k_p]
                    dest += 1
                    m_p += 1
                    k_p += 1
                elif m_col < k_col:
                    E_diag.data[dest] = ALPH * M.data[m_p]
                    dest += 1
                    m_p += 1
                else:
                    E_diag.data[dest] = h * K.data[k_p]
                    dest += 1
                    k_p += 1
            while m_p < m_end:
                E_diag.data[dest] = ALPH * M.data[m_p]
                dest += 1
                m_p += 1
            while k_p < k_end:
                E_diag.data[dest] = h * K.data[k_p]
                dest += 1
                k_p += 1
        for i in range(n):
            for p in range(E_diag.indptr[i], E_diag.indptr[i + 1]):
                var j = E_diag.indices[p]
                for q in range(E_diag_csc.colptr[j], E_diag_csc.colptr[j + 1]):
                    if E_diag_csc.indices[q] == i:
                        E_diag_csc.data[q] = E_diag.data[p]
                        break

    def _solve_complex_gmres(
        self,
        M: CSRMatrix,
        E_diag: CSRMatrix,
        mut lu_real: SparseLU,
        rhs_re: FixedSizeVector,
        rhs_im: FixedSizeVector,
        mut dF2: FixedSizeVector,
        mut dF3: FixedSizeVector,
        mut work_n: FixedSizeVector,
        n: Int,
        mut gmres_V_re: List[FixedSizeVector],
        mut gmres_V_im: List[FixedSizeVector],
        mut gmres_H: List[List[Float64]],
        mut gmres_cs: List[Float64],
        mut gmres_sn: List[Float64],
        mut gmres_s: List[Float64],
        mut gmres_w_re: FixedSizeVector,
        mut gmres_w_im: FixedSizeVector,
        mut gmres_tmp_re: FixedSizeVector,
        mut gmres_tmp_im: FixedSizeVector,
        mut gmres_tmp2: FixedSizeVector,
    ):
        """Solve complex 2n system via left-preconditioned GMRES.

        E2 = [E_diag, -BETA*M; +BETA*M, E_diag]
        where E_diag = ALPH*M + h*K

        Preconditioner P = block-diag(lu_real, lu_real)
        Solves P^{-1} E2 x = P^{-1} b via GMRES.

        IMPORTANT: E2 is a REAL 2n x 2n matrix. GMRES uses standard
        real dot products: <w,v> = w_re . v_re + w_im . v_im.
        """
        # Apply preconditioner to RHS: b_tilde = P^{-1} b
        gmres_V_re[0].copy_from_fixed(rhs_re)
        gmres_V_im[0].copy_from_fixed(rhs_im)
        lu_real.solve_inplace(gmres_V_re[0], work_n)
        lu_real.solve_inplace(gmres_V_im[0], work_n)

        # Compute initial residual norm: ||b_tilde|| = sqrt(||b_re||^2 + ||b_im||^2)
        var beta_sq = 0.0
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= n:
            var vr = (gmres_V_re[0].ptr() + i).load[width=width]()
            var vi = (gmres_V_im[0].ptr() + i).load[width=width]()
            beta_sq += (vr * vr + vi * vi).reduce_add()
            i += width
        while i < n:
            beta_sq += gmres_V_re[0][i] ** 2 + gmres_V_im[0][i] ** 2
            i += 1
        var beta = sqrt(beta_sq)

        if beta < 1e-30:
            dF2.zero_out()
            dF3.zero_out()
            return

        # Normalize V[0]
        var inv_beta = 1.0 / beta
        i = 0
        while i + width <= n:
            var vr = (gmres_V_re[0].ptr() + i).load[width=width]()
            var vi = (gmres_V_im[0].ptr() + i).load[width=width]()
            (gmres_V_re[0].ptr() + i).store[width=width](vr * inv_beta)
            (gmres_V_im[0].ptr() + i).store[width=width](vi * inv_beta)
            i += width
        while i < n:
            gmres_V_re[0][i] = gmres_V_re[0][i] * inv_beta
            gmres_V_im[0][i] = gmres_V_im[0][i] * inv_beta
            i += 1

        # Initialize s = [beta, 0, 0, ...]
        gmres_s[0] = beta
        for k_idx in range(1, GMRES_MAX_KRYLOV + 1):
            gmres_s[k_idx] = 0.0

        var j = 0
        while j < GMRES_MAX_KRYLOV:
            # Compute w = P^{-1} E2 V[j]
            # E2 * [v_re; v_im] = [E_diag*v_re - BETA*M*v_im; +BETA*M*v_re + E_diag*v_im]
            # Step 1: E_diag * V_re[j] -> w_re (partial)
            E_diag.spmv(gmres_V_re[j], gmres_w_re)
            # Step 2: M * V_im[j] -> tmp_re
            M.spmv(gmres_V_im[j], gmres_tmp_re)
            # Step 3: w_re = E_diag*V_re[j] - BETA * M * V_im[j]
            i = 0
            while i + width <= n:
                var sw_re = (gmres_w_re.ptr() + i).load[width=width]()
                var st_re = (gmres_tmp_re.ptr() + i).load[width=width]()
                (gmres_w_re.ptr() + i).store[width=width](sw_re - BETA * st_re)
                i += width
            while i < n:
                gmres_w_re[i] = gmres_w_re[i] - BETA * gmres_tmp_re[i]
                i += 1

            # Step 4: E_diag * V_im[j] -> w_im (partial)
            E_diag.spmv(gmres_V_im[j], gmres_w_im)
            # Step 5: M * V_re[j] -> tmp_im
            M.spmv(gmres_V_re[j], gmres_tmp_im)
            # Step 6: w_im = E_diag*V_im[j] + BETA * M * V_re[j]
            i = 0
            while i + width <= n:
                var sw_im = (gmres_w_im.ptr() + i).load[width=width]()
                var st_im = (gmres_tmp_im.ptr() + i).load[width=width]()
                (gmres_w_im.ptr() + i).store[width=width](sw_im + BETA * st_im)
                i += width
            while i < n:
                gmres_w_im[i] = gmres_w_im[i] + BETA * gmres_tmp_im[i]
                i += 1

            # Apply preconditioner: w = P^{-1} * [w_re; w_im]
            lu_real.solve_inplace(gmres_w_re, work_n)
            lu_real.solve_inplace(gmres_w_im, work_n)

            # Modified Gram-Schmidt orthogonalization
            # Uses REAL inner product: <w, v> = w_re . v_re + w_im . v_im
            for k_idx in range(j + 1):
                # Compute real dot product h = <w, V[k]>
                var h_val = 0.0
                i = 0
                while i + width <= n:
                    var sw_re = (gmres_w_re.ptr() + i).load[width=width]()
                    var sw_im = (gmres_w_im.ptr() + i).load[width=width]()
                    var sv_re = (gmres_V_re[k_idx].ptr() + i).load[
                        width=width
                    ]()
                    var sv_im = (gmres_V_im[k_idx].ptr() + i).load[
                        width=width
                    ]()
                    h_val += (sw_re * sv_re + sw_im * sv_im).reduce_add()
                    i += width
                while i < n:
                    h_val += (
                        gmres_w_re[i] * gmres_V_re[k_idx][i]
                        + gmres_w_im[i] * gmres_V_im[k_idx][i]
                    )
                    i += 1

                gmres_H[k_idx][j] = h_val

                # w = w - h * V[k]  (real vector subtraction)
                i = 0
                while i + width <= n:
                    var sw_re = (gmres_w_re.ptr() + i).load[width=width]()
                    var sw_im = (gmres_w_im.ptr() + i).load[width=width]()
                    var sv_re = (gmres_V_re[k_idx].ptr() + i).load[
                        width=width
                    ]()
                    var sv_im = (gmres_V_im[k_idx].ptr() + i).load[
                        width=width
                    ]()
                    (gmres_w_re.ptr() + i).store[width=width](
                        sw_re - h_val * sv_re
                    )
                    (gmres_w_im.ptr() + i).store[width=width](
                        sw_im - h_val * sv_im
                    )
                    i += width
                while i < n:
                    gmres_w_re[i] = gmres_w_re[i] - h_val * gmres_V_re[k_idx][i]
                    gmres_w_im[i] = gmres_w_im[i] - h_val * gmres_V_im[k_idx][i]
                    i += 1

            # H[j+1][j] = ||w|| (real 2-norm)
            var wnorm_sq = 0.0
            i = 0
            while i + width <= n:
                var sw_re = (gmres_w_re.ptr() + i).load[width=width]()
                var sw_im = (gmres_w_im.ptr() + i).load[width=width]()
                wnorm_sq += (sw_re * sw_re + sw_im * sw_im).reduce_add()
                i += width
            while i < n:
                wnorm_sq += gmres_w_re[i] ** 2 + gmres_w_im[i] ** 2
                i += 1
            var wnorm = sqrt(wnorm_sq)

            gmres_H[j + 1][j] = wnorm

            # V[j+1] = w / ||w||
            if wnorm > 1e-30:
                var inv_wnorm = 1.0 / wnorm
                i = 0
                while i + width <= n:
                    var sw_re = (gmres_w_re.ptr() + i).load[width=width]()
                    var sw_im = (gmres_w_im.ptr() + i).load[width=width]()
                    (gmres_V_re[j + 1].ptr() + i).store[width=width](
                        sw_re * inv_wnorm
                    )
                    (gmres_V_im[j + 1].ptr() + i).store[width=width](
                        sw_im * inv_wnorm
                    )
                    i += width
                while i < n:
                    gmres_V_re[j + 1][i] = gmres_w_re[i] * inv_wnorm
                    gmres_V_im[j + 1][i] = gmres_w_im[i] * inv_wnorm
                    i += 1

            # Apply previous Givens rotations to new column of H
            for k_idx in range(j):
                var temp = (
                    gmres_cs[k_idx] * gmres_H[k_idx][j]
                    - gmres_sn[k_idx] * gmres_H[k_idx + 1][j]
                )
                gmres_H[k_idx + 1][j] = (
                    gmres_sn[k_idx] * gmres_H[k_idx][j]
                    + gmres_cs[k_idx] * gmres_H[k_idx + 1][j]
                )
                gmres_H[k_idx][j] = temp

            # Compute new Givens rotation for H[j][j], H[j+1][j]
            var r = sqrt(gmres_H[j][j] ** 2 + gmres_H[j + 1][j] ** 2)
            if r > 1e-30:
                gmres_cs[j] = gmres_H[j][j] / r
                gmres_sn[j] = -gmres_H[j + 1][j] / r
            else:
                gmres_cs[j] = 1.0
                gmres_sn[j] = 0.0

            # Apply new rotation to H column and s
            gmres_H[j][j] = (
                gmres_cs[j] * gmres_H[j][j] - gmres_sn[j] * gmres_H[j + 1][j]
            )
            gmres_H[j + 1][j] = 0.0

            var s_old = gmres_s[j]
            gmres_s[j] = gmres_cs[j] * s_old - gmres_sn[j] * gmres_s[j + 1]
            gmres_s[j + 1] = gmres_sn[j] * s_old + gmres_cs[j] * gmres_s[j + 1]

            # Check convergence
            var residual = abs(gmres_s[j + 1])
            if residual < GMRES_TOL * beta:
                j += 1
                break

            j += 1

        # Solve upper triangular system H*y = s (j x j)
        var y_sol = List[Float64](length=j, fill=0.0)
        for k_idx in range(j - 1, -1, -1):
            var sum_val = gmres_s[k_idx]
            for m in range(k_idx + 1, j):
                sum_val = sum_val - gmres_H[k_idx][m] * y_sol[m]
            if abs(gmres_H[k_idx][k_idx]) > 1e-30:
                y_sol[k_idx] = sum_val / gmres_H[k_idx][k_idx]

        # x = sum(y[j] * V[j]) -- compute dF2, dF3 from V basis
        dF2.zero_out()
        dF3.zero_out()
        for k_idx in range(j):
            var yk = y_sol[k_idx]
            i = 0
            while i + width <= n:
                var s_dF2 = (dF2.ptr() + i).load[width=width]()
                var s_dF3 = (dF3.ptr() + i).load[width=width]()
                var sv_r = (gmres_V_re[k_idx].ptr() + i).load[width=width]()
                var sv_i = (gmres_V_im[k_idx].ptr() + i).load[width=width]()
                (dF2.ptr() + i).store[width=width](s_dF2 + yk * sv_r)
                (dF3.ptr() + i).store[width=width](s_dF3 + yk * sv_i)
                i += width
            while i < n:
                dF2[i] = dF2[i] + yk * gmres_V_re[k_idx][i]
                dF3[i] = dF3[i] + yk * gmres_V_im[k_idx][i]
                i += 1


@always_inline
def contr5(i: Int, s: Float64, cont: FixedSizeVector, n: Int) -> Float64:
    return cont[i] + s * (cont[i + n] + (s - C2M1) * (cont[i + 2 * n] + (s - C1M1) * cont[i + 3 * n]))
