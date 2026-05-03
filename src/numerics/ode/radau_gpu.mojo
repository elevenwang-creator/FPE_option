"""GPU kernel for Radau IIA (3-stage, order 5) stiff ODE solver.

Implements the implicit Runge-Kutta method with:
- Comptime Butcher tableau constants
- Simplified Newton iteration via 3n x 3n block system
- LU factorization with partial pivoting on GPU (thread-0 per block)

One thread-block per batch element. Thread-0 handles sequential
LU solve; all threads cooperate on parallel matvec stages.

Workspace layout per batch element (9n^2 + 14n entries available):
offset 0: k1, k2, k3 (3n entries)
offset 3n: block system (9n^2 entries, 3n x 3n)
offset 3n + 9n^2: rhs (3n entries)
offset 3n + 9n^2 + 3n: f1, f2, f3 (3n entries)
"""

from std.gpu import block_idx, thread_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from gpu_utils.dtype import (
    METAL_DTYPE,
    METAL_VEC_LAYOUT,
    METAL_MAX_N,
    CUDA_DTYPE,
    CUDA_VEC_LAYOUT,
    CUDA_MAX_N,
)
from std.sys import has_apple_gpu_accelerator

comptime GPU_RADAU_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime GPU_RADAU_VEC = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime GPU_RADAU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N
comptime GPU_R_SCALAR = Scalar[GPU_RADAU_DTYPE]


def radau5_gpu_kernel(
    q_out: LayoutTensor[GPU_RADAU_DTYPE, GPU_RADAU_VEC, MutAnyOrigin],
    neg_M_inv_K: LayoutTensor[GPU_RADAU_DTYPE, GPU_RADAU_VEC, MutAnyOrigin],
    q_in: LayoutTensor[GPU_RADAU_DTYPE, GPU_RADAU_VEC, MutAnyOrigin],
    workspace: LayoutTensor[GPU_RADAU_DTYPE, GPU_RADAU_VEC, MutAnyOrigin],
    n: Int,
    n_steps: Int,
):
    """3-stage Radau IIA (order 5) GPU kernel for batch FPE time integration."""
    comptime c1: GPU_R_SCALAR = 0.15505102572168219018
    comptime c2: GPU_R_SCALAR = 0.64494897427831780982

    comptime a11: GPU_R_SCALAR = 0.11208445195653073473
    comptime a12: GPU_R_SCALAR = -0.04067796433082320083
    comptime a13: GPU_R_SCALAR = 0.02581692768754036853
    comptime a21: GPU_R_SCALAR = 0.23402839165419385511
    comptime a22: GPU_R_SCALAR = 0.20686796081962466466
    comptime a23: GPU_R_SCALAR = -0.04783262767800705297
    comptime a31: GPU_R_SCALAR = 0.21668178412381825484
    comptime a32: GPU_R_SCALAR = 0.40612326386737472080
    comptime a33: GPU_R_SCALAR = 0.18903606908424243706

    comptime b1: GPU_R_SCALAR = a31
    comptime b2: GPU_R_SCALAR = a32
    comptime b3: GPU_R_SCALAR = a33

    comptime newton_max_iter: Int = 12
    comptime newton_tol: GPU_R_SCALAR = 1e-8

    var b_blk = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var threads = Int(block_dim.x)

    var n3 = 3 * n
    var base_q = b_blk * n
    var base_A = b_blk * n * n
    var base_ws = b_blk * (9 * n * n + 14 * n)

    var off_k1 = 0
    var off_k2 = n
    var off_k3 = 2 * n
    var off_block = 3 * n
    var off_rhs = 3 * n + 9 * n * n
    var off_f = 3 * n + 9 * n * n + 3 * n

    var h_step: GPU_R_SCALAR = 1.0 / GPU_R_SCALAR(n_steps)

    # Copy initial condition to q_out
    var i = Int(tid)
    while i < n:
        q_out[base_q + i] = q_in[base_q + i]
        i += Int(threads)
    barrier()

    for _step in range(n_steps):
        # Initialize k1=k2=k3 = J @ y
        var i_k = Int(tid)
        while i_k < n:
            var f_val: GPU_R_SCALAR = 0.0
            for j in range(n):
                f_val += rebind[GPU_R_SCALAR](
                    neg_M_inv_K[base_A + i_k * n + j]
                ) * rebind[GPU_R_SCALAR](q_out[base_q + j])
            workspace[base_ws + off_k1 + i_k] = rebind[workspace.element_type](
                f_val
            )
            workspace[base_ws + off_k2 + i_k] = rebind[workspace.element_type](
                f_val
            )
            workspace[base_ws + off_k3 + i_k] = rebind[workspace.element_type](
                f_val
            )
            i_k += Int(threads)
        barrier()

        for _newton in range(newton_max_iter):
            # Compute y_stage = y + h * sum(a_{stage,j} * k_j), then f_stage = J @ y_stage
            for stage in range(3):
                var i_f = Int(tid)
                while i_f < n:
                    var y_val: GPU_R_SCALAR = rebind[GPU_R_SCALAR](
                        q_out[base_q + i_f]
                    )
                    if stage == 0:
                        y_val = y_val + h_step * (
                            a11
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k1 + i_f]
                            )
                            + a12
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k2 + i_f]
                            )
                            + a13
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k3 + i_f]
                            )
                        )
                    elif stage == 1:
                        y_val = y_val + h_step * (
                            a21
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k1 + i_f]
                            )
                            + a22
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k2 + i_f]
                            )
                            + a23
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k3 + i_f]
                            )
                        )
                    else:
                        y_val = y_val + h_step * (
                            a31
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k1 + i_f]
                            )
                            + a32
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k2 + i_f]
                            )
                            + a33
                            * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_k3 + i_f]
                            )
                        )
                    var f_val: GPU_R_SCALAR = 0.0
                    for j in range(n):
                        f_val += (
                            rebind[GPU_R_SCALAR](
                                neg_M_inv_K[base_A + i_f * n + j]
                            )
                            * y_val
                        )
                    workspace[base_ws + off_f + stage * n + i_f] = rebind[
                        workspace.element_type
                    ](f_val)
                    i_f += Int(threads)
                barrier()

            # Thread-0: build and solve 3n x 3n block system
            if tid == 0:
                # Build 3n x 3n block system and rhs, solve via LU
                for i_block in range(3):
                    for j_block in range(3):
                        var a_coeff: GPU_R_SCALAR = a11
                        if i_block == 0 and j_block == 1:
                            a_coeff = a12
                        elif i_block == 0 and j_block == 2:
                            a_coeff = a13
                        elif i_block == 1 and j_block == 0:
                            a_coeff = a21
                        elif i_block == 1 and j_block == 1:
                            a_coeff = a22
                        elif i_block == 1 and j_block == 2:
                            a_coeff = a23
                        elif i_block == 2 and j_block == 0:
                            a_coeff = a31
                        elif i_block == 2 and j_block == 1:
                            a_coeff = a32
                        elif i_block == 2 and j_block == 2:
                            a_coeff = a33
                        for ib in range(n):
                            for jb in range(n):
                                var Jij = rebind[GPU_R_SCALAR](
                                    neg_M_inv_K[base_A + ib * n + jb]
                                )
                                var val: GPU_R_SCALAR = (
                                    0.0 - h_step * a_coeff * Jij
                                )
                                if ib == jb and i_block == j_block:
                                    val = val + 1.0
                                workspace[
                                    base_ws
                                    + off_block
                                    + (i_block * n + ib) * n3
                                    + j_block * n
                                    + jb
                                ] = rebind[workspace.element_type](val)

                # rhs: f_stage - k_stage
                for s in range(3):
                    for ib in range(n):
                        var f_val = rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_f + s * n + ib]
                        )
                        var k_val = rebind[GPU_R_SCALAR](
                            workspace[base_ws + s * n + ib]
                        )
                        workspace[base_ws + off_rhs + s * n + ib] = rebind[
                            workspace.element_type
                        ](f_val - k_val)

                # LU factorization with partial pivoting
                for k in range(n3):
                    var pivot = k
                    var max_val: GPU_R_SCALAR = 0.0
                    for row in range(k, n3):
                        var v = rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_block + row * n3 + k]
                        )
                        if v < 0.0:
                            v = 0.0 - v
                        if v > max_val:
                            max_val = v
                            pivot = row
                    if pivot != k:
                        for j in range(n3):
                            var tmp = workspace[
                                base_ws + off_block + k * n3 + j
                            ]
                            workspace[
                                base_ws + off_block + k * n3 + j
                            ] = workspace[base_ws + off_block + pivot * n3 + j]
                            workspace[
                                base_ws + off_block + pivot * n3 + j
                            ] = tmp
                        var tmp_r = workspace[base_ws + off_rhs + k]
                        workspace[base_ws + off_rhs + k] = workspace[
                            base_ws + off_rhs + pivot
                        ]
                        workspace[base_ws + off_rhs + pivot] = tmp_r
                    var diag = rebind[GPU_R_SCALAR](
                        workspace[base_ws + off_block + k * n3 + k]
                    )
                    if diag != 0.0:
                        for row in range(k + 1, n3):
                            var factor = (
                                rebind[GPU_R_SCALAR](
                                    workspace[
                                        base_ws + off_block + row * n3 + k
                                    ]
                                )
                                / diag
                            )
                            for j in range(k + 1, n3):
                                var v = rebind[GPU_R_SCALAR](
                                    workspace[
                                        base_ws + off_block + row * n3 + j
                                    ]
                                )
                                v = v - factor * rebind[GPU_R_SCALAR](
                                    workspace[base_ws + off_block + k * n3 + j]
                                )
                                workspace[
                                    base_ws + off_block + row * n3 + j
                                ] = rebind[workspace.element_type](v)
                            var r_val = rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_rhs + row]
                            )
                            r_val = r_val - factor * rebind[GPU_R_SCALAR](
                                workspace[base_ws + off_rhs + k]
                            )
                            workspace[base_ws + off_rhs + row] = rebind[
                                workspace.element_type
                            ](r_val)
                            workspace[
                                base_ws + off_block + row * n3 + k
                            ] = rebind[workspace.element_type](0.0)

                # Back-substitution
                for rev in range(n3):
                    var ii = n3 - 1 - rev
                    var s_val = rebind[GPU_R_SCALAR](
                        workspace[base_ws + off_rhs + ii]
                    )
                    for j in range(ii + 1, n3):
                        s_val = s_val - rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_block + ii * n3 + j]
                        ) * rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_rhs + j]
                        )
                    var d = rebind[GPU_R_SCALAR](
                        workspace[base_ws + off_block + ii * n3 + ii]
                    )
                    if d != 0.0:
                        workspace[base_ws + off_rhs + ii] = rebind[
                            workspace.element_type
                        ](s_val / d)

                # Update k1, k2, k3: k_s += dk_s
                for ib in range(n):
                    workspace[base_ws + off_k1 + ib] = rebind[
                        workspace.element_type
                    ](
                        rebind[GPU_R_SCALAR](workspace[base_ws + off_k1 + ib])
                        + rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_rhs + ib]
                        )
                    )
                    workspace[base_ws + off_k2 + ib] = rebind[
                        workspace.element_type
                    ](
                        rebind[GPU_R_SCALAR](workspace[base_ws + off_k2 + ib])
                        + rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_rhs + n + ib]
                        )
                    )
                    workspace[base_ws + off_k3 + ib] = rebind[
                        workspace.element_type
                    ](
                        rebind[GPU_R_SCALAR](workspace[base_ws + off_k3 + ib])
                        + rebind[GPU_R_SCALAR](
                            workspace[base_ws + off_rhs + 2 * n + ib]
                        )
                    )

            barrier()

            # Accept step: y_{n+1} = y_n + h*(b1*k1 + b2*k2 + b3*k3)
            i = Int(tid)
            while i < n:
                var y_new = rebind[GPU_R_SCALAR](q_out[base_q + i]) + h_step * (
                    b1 * rebind[GPU_R_SCALAR](workspace[base_ws + off_k1 + i])
                    + b2 * rebind[GPU_R_SCALAR](workspace[base_ws + off_k2 + i])
                    + b3 * rebind[GPU_R_SCALAR](workspace[base_ws + off_k3 + i])
                )
                if y_new < 0.0:
                    y_new = 0.0
                q_out[base_q + i] = rebind[q_out.element_type](y_new)
                i += Int(threads)
            barrier()

    # Normalize final q to sum to 1 (thread-0 sequential)
    if tid == 0:
        var sum_q: GPU_R_SCALAR = 0.0
        for j in range(n):
            var val = rebind[GPU_R_SCALAR](q_out[base_q + j])
            if val < 0.0:
                val = 0.0
                q_out[base_q + j] = rebind[q_out.element_type](0.0)
            sum_q = sum_q + val
        if sum_q > 0.0:
            var inv_sum: GPU_R_SCALAR = 1.0 / sum_q
            for j in range(n):
                q_out[base_q + j] = rebind[q_out.element_type](
                    rebind[GPU_R_SCALAR](q_out[base_q + j]) * inv_sum
                )
