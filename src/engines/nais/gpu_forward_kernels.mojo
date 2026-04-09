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
from gpu_utils.dtype import (
    METAL_DTYPE, METAL_VEC_LAYOUT, METAL_MAX_N,
    CUDA_DTYPE, CUDA_VEC_LAYOUT, CUDA_MAX_N,
)
from std.math import sin
from std.sys import has_apple_gpu_accelerator


comptime FORWARD_DTYPE = METAL_DTYPE if has_apple_gpu_accelerator() else CUDA_DTYPE
comptime FORWARD_VEC_LAYOUT = METAL_VEC_LAYOUT if has_apple_gpu_accelerator() else CUDA_VEC_LAYOUT
comptime FORWARD_MAX_BATCH = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N


def nais_forward_kernel[
    hidden: Int,
    phi_dim: Int,
](
    params: LayoutTensor[FORWARD_DTYPE, FORWARD_VEC_LAYOUT, MutAnyOrigin],
    inputs: LayoutTensor[FORWARD_DTYPE, FORWARD_VEC_LAYOUT, MutAnyOrigin],
    outputs: LayoutTensor[FORWARD_DTYPE, FORWARD_VEC_LAYOUT, MutAnyOrigin],
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
    var t_val = rebind[Scalar[FORWARD_DTYPE]](inputs[input_base])
    var x0 = rebind[Scalar[FORWARD_DTYPE]](inputs[input_base + 1])
    var x1 = rebind[Scalar[FORWARD_DTYPE]](inputs[input_base + 2])

    # Layer 1: sin(W1 @ [t, x0, x1] + b1) -> hidden
    # W1: [in_dim+1, hidden], b1: [hidden]
    var p_idx = 0
    # Stack-allocated hidden activations generically sized according to FORWARD_DTYPE
    var h: InlineArray[Scalar[FORWARD_DTYPE], hidden] = InlineArray[Scalar[FORWARD_DTYPE], hidden]()
    
    for j in range(hidden):
        var acc: Scalar[FORWARD_DTYPE] = rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * t_val
        p_idx += 1
        acc = acc + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * x0
        p_idx += 1
        acc = acc + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * x1
        p_idx += 1
        acc = acc + rebind[Scalar[FORWARD_DTYPE]](params[p_idx])
        p_idx += 1
        h[j] = rebind[Scalar[FORWARD_DTYPE]](sin(Float64(acc)))

    # Residual blocks (3 blocks)
    for _blk in range(3):
        var skip: InlineArray[Scalar[FORWARD_DTYPE], hidden] = InlineArray[Scalar[FORWARD_DTYPE], hidden]()
        for j in range(hidden):
            var acc_s: Scalar[FORWARD_DTYPE] = rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * t_val
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * x0
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * x1
            p_idx += 1
            acc_s = acc_s + rebind[Scalar[FORWARD_DTYPE]](params[p_idx])
            p_idx += 1
            skip[j] = acc_s

        for j in range(hidden):
            var acc_b: Scalar[FORWARD_DTYPE] = 0.0
            for i in range(hidden):
                acc_b = acc_b + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * h[i]
                p_idx += 1
            acc_b = acc_b + rebind[Scalar[FORWARD_DTYPE]](params[p_idx])
            p_idx += 1
            h[j] = h[j] + rebind[Scalar[FORWARD_DTYPE]](sin(Float64(acc_b + skip[j])))

    # Output u
    var u_out: Scalar[FORWARD_DTYPE] = 0.0
    for i in range(hidden):
        u_out = u_out + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * h[i]
        p_idx += 1
    u_out = u_out + rebind[Scalar[FORWARD_DTYPE]](params[p_idx])
    p_idx += 1

    var out_base = Int(b) * (1 + phi_dim)
    outputs[out_base] = rebind[outputs.element_type](u_out)

    for j in range(phi_dim):
        var phi: Scalar[FORWARD_DTYPE] = 0.0
        for i in range(hidden):
             phi = phi + rebind[Scalar[FORWARD_DTYPE]](params[p_idx]) * h[i]
             p_idx += 1
        phi = phi + rebind[Scalar[FORWARD_DTYPE]](params[p_idx])
        p_idx += 1
        outputs[out_base + 1 + j] = rebind[outputs.element_type](phi)