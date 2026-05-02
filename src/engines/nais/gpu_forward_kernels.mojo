"""GPU batch forward pass kernel for NAIS training.

Each thread handles one forward pass evaluation for a specific input.
Used to accelerate the O(n_params) forward passes in finite-difference
gradient computation.

Kernel constraints:
- nonraising (no exceptions on GPU)
- LayoutTensor for all parameters with concrete comptime layouts
- No dynamic memory allocation (no List, uses InlineArray)
- No print statements (Apple Silicon limitation)

The kernel reconstructs a simplified NAIS-Net forward pass from flattened
params using stack-allocated accumulators only.
"""

from std.gpu import block_idx, thread_idx, block_dim
from layout import Layout, LayoutTensor
from gpu_utils.dtype import GPU_DTYPE, GPU_VEC_LAYOUT, GPU_MAX_N
from std.math import sin


comptime FORWARD_VEC_LAYOUT = GPU_VEC_LAYOUT
comptime FORWARD_MAX_BATCH = GPU_MAX_N


def nais_forward_kernel[
    hidden: Int,
    phi_dim: Int,
](
    params: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    inputs: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    outputs: LayoutTensor[GPU_DTYPE, GPU_VEC_LAYOUT, MutAnyOrigin],
    n_params: Int,
    in_dim: Int,
    batch_size: Int,
):
    """Batch NAIS forward pass kernel using LayoutTensor.

    Each thread (block_idx.x) handles one batch element's forward pass.
    Network architecture: in_dim -> hidden -> (hidden x 3 residual blocks) -> 1 + phi_dim.

    Args:
        params: Flattened network weights (shared across batch).
        inputs: Input data [batch_size, in_dim + 1] row-major.
        outputs: Output data [batch_size, 1 + phi_dim] row-major.
        n_params: Total number of network parameters.
        in_dim: Input dimension (typically 3: t, x0, x1).
        batch_size: Number of batch elements.
    """
    var b = block_idx.x
    if Int(b) >= batch_size:
        return

    # Use block_idx.x mapping to absolute batch element
    var input_base = Int(b) * (in_dim + 1)

    # Thread identifier (for cooperative future expansions)
    var tid = thread_idx.x

    # Read input values
    var t_val = rebind[Scalar[GPU_DTYPE]](inputs[input_base])
    var x0 = rebind[Scalar[GPU_DTYPE]](inputs[input_base + 1])
    var x1 = rebind[Scalar[GPU_DTYPE]](inputs[input_base + 2])

    # Layer 1: sin(W1 @ [t, x0, x1] + b1) -> hidden
    # W1: [in_dim+1, hidden], b1: [hidden]
    var p_idx = 0
    # Stack-allocated hidden activations generically sized according to GPU_DTYPE
    var h: InlineArray[Scalar[GPU_DTYPE], hidden] = InlineArray[
        Scalar[GPU_DTYPE], hidden
    ]()

    for j in range(hidden):
        var acc: Scalar[GPU_DTYPE] = (
            rebind[Scalar[GPU_DTYPE]](params[p_idx]) * t_val
        )
        p_idx += 1
        acc = acc + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * x0
        p_idx += 1
        acc = acc + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * x1
        p_idx += 1
        acc = acc + rebind[Scalar[GPU_DTYPE]](params[p_idx])
        p_idx += 1
        h[j] = rebind[Scalar[GPU_DTYPE]](sin(Float64(acc)))

    # Residual blocks (3 blocks)
    for _blk in range(3):
        var skip: InlineArray[Scalar[GPU_DTYPE], hidden] = InlineArray[
            Scalar[GPU_DTYPE], hidden
        ]()
        for j in range(hidden):
            var acc_s: Scalar[GPU_DTYPE] = (
                rebind[Scalar[GPU_DTYPE]](params[p_idx]) * t_val
            )
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * x0
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * x1
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[GPU_DTYPE]](params[p_idx])
            p_idx += 1
            skip[j] = acc_s

        for j in range(hidden):
            var acc_b: Scalar[GPU_DTYPE] = 0.0
            for i in range(hidden):
                acc_b = (
                    acc_b + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * h[i]
                )
                p_idx += 1
            acc_b = acc_b + rebind[Scalar[GPU_DTYPE]](params[p_idx])
            p_idx += 1
            h[j] = h[j] + rebind[Scalar[GPU_DTYPE]](sin(Float64(acc_b + skip[j])))

    # Output u
    var u_out: Scalar[GPU_DTYPE] = 0.0
    for i in range(hidden):
        u_out = u_out + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * h[i]
        p_idx += 1
    u_out = u_out + rebind[Scalar[GPU_DTYPE]](params[p_idx])
    p_idx += 1

    var out_base = Int(b) * (1 + phi_dim)
    outputs[out_base] = rebind[outputs.element_type](u_out)

    for j in range(phi_dim):
        var phi: Scalar[GPU_DTYPE] = 0.0
        for i in range(hidden):
            phi = phi + rebind[Scalar[GPU_DTYPE]](params[p_idx]) * h[i]
            p_idx += 1
        phi = phi + rebind[Scalar[GPU_DTYPE]](params[p_idx])
        p_idx += 1
        outputs[out_base + 1 + j] = rebind[outputs.element_type](phi)
