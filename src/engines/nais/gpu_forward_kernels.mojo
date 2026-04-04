"""GPU batch forward pass kernel for NAIS training.

Each thread handles one forward pass evaluation for a specific input.
Used to accelerate the O(n_params) forward passes in finite-difference gradient computation.

Kernel constraints:
- nonraising (no exceptions on GPU)
- UnsafePointer for all parameters
- No print statements (Apple Silicon limitation)
"""

from std.gpu import global_idx
from std.math import sin


def nais_forward_kernel(
    # Network parameters (flattened)
    params: UnsafePointer[Float64, MutAnyOrigin],
    # Input data: [batch_size, in_dim + 1] (time + state)
    inputs: UnsafePointer[Float64, MutAnyOrigin],
    # Output: [batch_size, 1 + phi_dim] (u + phi)
    outputs: UnsafePointer[Float64, MutAnyOrigin],
    # Architecture constants
    batch_size: Int,
    in_dim: Int,
    hidden: Int,
    phi_dim: Int,
):
    """Batch NAIS forward pass kernel.

    Each thread (global_idx.x) handles one batch element's forward pass.

    Layout:
    - params: flattened network weights (read-only)
    - inputs: [batch_size, in_dim + 1] row-major
    - outputs: [batch_size, 1 + phi_dim] row-major
    """
    var b = global_idx.x
    if b >= UInt(batch_size):
        return

    var idx = Int(b)
    var input_base = idx * (in_dim + 1)

    # Extract input for this batch element
    var t_val = inputs[input_base]
    var x0 = inputs[input_base + 1]
    var x1 = inputs[input_base + 2]

    # Build u_in = [t, x0, x1]
    var u_in: List[Float64] = [t_val, x0, x1]

    # Reconstruct network weights from flattened params
    # Layer 1: [in_dim+1, hidden] weights + [hidden] bias
    var p_idx = 0
    var l1_w: List[List[Float64]] = []
    var i = 0
    while i < in_dim + 1:
        var row: List[Float64] = []
        var j = 0
        while j < hidden:
            row.append(params[p_idx])
            p_idx += 1
            j += 1
        l1_w.append(row^)
        i += 1
    var l1_b: List[Float64] = []
    i = 0
    while i < hidden:
        l1_b.append(params[p_idx])
        p_idx += 1
        i += 1

    # Layer 1 forward: h = sin(W1 @ u_in + b1)
    var h: List[Float64] = []
    j = 0
    while j < hidden:
        var acc: Float64 = 0.0
        i = 0
        while i < in_dim + 1:
            acc += l1_w[i][j] * u_in[i]
            i += 1
        acc += l1_b[j]
        h.append(sin(acc))
        j += 1

    # Layers 2-4: residual blocks with skip connections
    var block_count = 3
    var blk = 0
    while blk < block_count:
        # Skip connection: W_skip @ u_in + b_skip
        var skip: List[Float64] = []
        j = 0
        while j < hidden:
            var acc: Float64 = 0.0
            i = 0
            while i < in_dim + 1:
                acc += params[p_idx] * u_in[i]
                p_idx += 1
                i += 1
            acc += params[p_idx]
            p_idx += 1
            skip.append(acc)
            j += 1

        # Block linear: W @ h + b
        var block_out: List[Float64] = []
        j = 0
        while j < hidden:
            var acc: Float64 = 0.0
            i = 0
            while i < hidden:
                acc += params[p_idx] * h[i]
                p_idx += 1
                i += 1
            acc += params[p_idx]
            p_idx += 1
            block_out.append(acc)
            j += 1

        # sin(skip + block_out)
        j = 0
        while j < hidden:
            block_out[j] = sin(skip[j] + block_out[j])
            j += 1

        # Residual: h = h + block_out
        j = 0
        while j < hidden:
            h[j] = h[j] + block_out[j]
            j += 1

        blk += 1

    # Layer 5: u = W5 @ h + b5 (output: 1 value)
    var u_out: Float64 = 0.0
    i = 0
    while i < hidden:
        u_out += params[p_idx] * h[i]
        p_idx += 1
        i += 1
    u_out += params[p_idx]
    p_idx += 1

    # Layer 6: phi = W6 @ h + b6 (output: phi_dim values)
    var phi: List[Float64] = []
    j = 0
    while j < phi_dim:
        var acc: Float64 = 0.0
        i = 0
        while i < hidden:
            acc += params[p_idx] * h[i]
            p_idx += 1
            i += 1
        acc += params[p_idx]
        p_idx += 1
        phi.append(acc)
        j += 1

    # Write output: [u, phi...]
    var out_base = idx * (1 + phi_dim)
    outputs[out_base] = u_out
    j = 0
    while j < phi_dim:
        outputs[out_base + 1 + j] = phi[j]
        j += 1
