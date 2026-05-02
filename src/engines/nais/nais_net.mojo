from numerics.utils import zeros, zeros_mat

from numerics.nn.stable_linear import StableLinear, make_stable_linear
from numerics.nn.autograd import Tape
from std.math import sin






def _make_weights(
    in_dim: Int, out_dim: Int, scale: Float64
) -> List[List[Float64]]:
    var W = zeros_mat(in_dim, out_dim)
    for i in range(in_dim):
        for j in range(out_dim):
            var k = ((i + 1) * 7 + (j + 1) * 5) % 17
            W[i][j] = scale * (Float64(k) - 8.0) / 8.0
    return W^


@always_inline
def _linear(
    W: List[List[Float64]], b: List[Float64], x: List[Float64]
) -> List[Float64]:
    """Linear transform: y = W @ x + b."""
    if len(W) == 0:
        var b_copy = List[Float64]()
        for i in range(len(b)):
            b_copy.append(b[i])
        return b_copy^
    var in_dim = len(W)
    var out_dim = len(b)

    var y = List[Float64]()
    for j in range(out_dim):
        var sum = 0.0
        for i in range(in_dim):
            if i < len(x):
                sum += W[i][j] * x[i]
        y.append(sum + b[j])
    return y^


@always_inline
def _sin_vec(x: List[Float64]) -> List[Float64]:
    var out = zeros(len(x))
    for i in range(len(x)):
        out[i] = sin(x[i])
    return out^


def _add_vec(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var out = zeros(len(a))
    for i in range(len(a)):
        out[i] = a[i] + b[i]
    return out^


@fieldwise_init
struct NaisNet(Copyable, Movable):
    """NAIS-Net architecture from NAIS_rBM.py."""

    var layer1: List[List[Float64]]
    var layer1_b: List[Float64]
    var layer2: StableLinear
    var layer2_input: List[List[Float64]]
    var layer2_input_b: List[Float64]
    var layer3: StableLinear
    var layer3_input: List[List[Float64]]
    var layer3_input_b: List[Float64]
    var layer4: StableLinear
    var layer4_input: List[List[Float64]]
    var layer4_input_b: List[Float64]
    var layer5: List[List[Float64]]
    var layer5_b: List[Float64]
    var layer6: List[List[Float64]]
    var layer6_b: List[Float64]

    def __init__(out self, in_dim: Int = 3, hidden: Int = 12, phi_dim: Int = 2):
        self.layer1 = _make_weights(in_dim, hidden, 0.08)
        self.layer1_b = zeros(hidden)

        self.layer2 = make_stable_linear(hidden, hidden)
        self.layer2_input = _make_weights(in_dim, hidden, 0.04)
        self.layer2_input_b = zeros(hidden)

        self.layer3 = make_stable_linear(hidden, hidden)
        self.layer3_input = _make_weights(in_dim, hidden, 0.04)
        self.layer3_input_b = zeros(hidden)

        self.layer4 = make_stable_linear(hidden, hidden)
        self.layer4_input = _make_weights(in_dim, hidden, 0.04)
        self.layer4_input_b = zeros(hidden)

        self.layer5 = _make_weights(hidden, 1, 0.06)
        self.layer5_b = zeros(1)

        self.layer6 = _make_weights(hidden, phi_dim, 0.06)
        self.layer6_b = zeros(phi_dim)

    def forward(
        self, t: Float64, x: List[Float64]
    ) -> Tuple[Float64, List[Float64]]:
        var u_in: List[Float64] = [t]
        for i in range(len(x)):
            u_in.append(x[i])

        var h = _sin_vec(_linear(self.layer1, self.layer1_b, u_in))

        var skip2 = _linear(self.layer2_input, self.layer2_input_b, u_in)
        var block2 = _sin_vec(_add_vec(skip2, self.layer2.forward(h)))
        h = _add_vec(h, block2)

        var skip3 = _linear(self.layer3_input, self.layer3_input_b, u_in)
        var block3 = _sin_vec(_add_vec(skip3, self.layer3.forward(h)))
        h = _add_vec(h, block3)

        var skip4 = _linear(self.layer4_input, self.layer4_input_b, u_in)
        var block4 = _sin_vec(_add_vec(skip4, self.layer4.forward(h)))
        h = _add_vec(h, block4)

        var u_vec = _linear(self.layer5, self.layer5_b, h)
        var phi = _linear(self.layer6, self.layer6_b, h)
        return (u_vec[0], phi^)

    def forward_tracked(
        self, t: Float64, x: List[Float64], mut tape: Tape,
        param_indices: List[Int] = [],
    ) raises -> Tuple[Int, List[Int]]:
        """Forward pass with operation recording on tape.
        If param_indices is non-empty, uses those indices for network weights.
        Otherwise, records new weight values on the tape.
        Returns (u_index, phi_indices) into tape.values.
        """
        var use_external_params = len(param_indices) > 0
        var p_idx = param_indices.copy()

        # Pre-compute offsets into p_idx matching _collect_param_indices layout:
        # layer1, layer2(W+b), layer3(W+b), layer4(W+b),
        # layer2_input, layer3_input, layer4_input, layer5, layer6
        var off_layer1 = 0
        var layer1_w_cols = len(self.layer1[0]) if len(self.layer1) > 0 else 0
        var off_layer2 = off_layer1 + len(self.layer1) * layer1_w_cols + len(self.layer1_b)
        var layer2_w_cols = len(self.layer2.W[0]) if len(self.layer2.W) > 0 else 0
        var off_layer3 = off_layer2 + len(self.layer2.W) * layer2_w_cols + len(self.layer2.b)
        var layer3_w_cols = len(self.layer3.W[0]) if len(self.layer3.W) > 0 else 0
        var off_layer4 = off_layer3 + len(self.layer3.W) * layer3_w_cols + len(self.layer3.b)
        var layer4_w_cols = len(self.layer4.W[0]) if len(self.layer4.W) > 0 else 0
        var off_layer2_input = off_layer4 + len(self.layer4.W) * layer4_w_cols + len(self.layer4.b)
        var l2i_cols = len(self.layer2_input[0]) if len(self.layer2_input) > 0 else 0
        var off_layer3_input = off_layer2_input + len(self.layer2_input) * l2i_cols + len(self.layer2_input_b)
        var l3i_cols = len(self.layer3_input[0]) if len(self.layer3_input) > 0 else 0
        var off_layer4_input = off_layer3_input + len(self.layer3_input) * l3i_cols + len(self.layer3_input_b)
        var l4i_cols = len(self.layer4_input[0]) if len(self.layer4_input) > 0 else 0
        var off_layer5 = off_layer4_input + len(self.layer4_input) * l4i_cols + len(self.layer4_input_b)
        var l5_cols = len(self.layer5[0]) if len(self.layer5) > 0 else 0
        var off_layer6 = off_layer5 + len(self.layer5) * l5_cols + len(self.layer5_b)

        # Record input values
        var t_idx = tape.record_value(t)
        var x_idx: List[Int] = []
        for i in range(len(x)):
            x_idx.append(tape.record_value(x[i]))

        # u_in = [t, x...]
        var u_in_idx: List[Int] = [t_idx]
        for i in range(len(x_idx)):
            u_in_idx.append(x_idx[i])

        # Layer 1: h = sin(W1 @ u_in + b1)
        var h_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer1, self.layer1, self.layer1_b, u_in_idx, tape)
            h_idx = result[0].copy()
        else:
            h_idx = self._linear_tracked_record_weights(self.layer1, self.layer1_b, u_in_idx, tape)
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_sin(h_idx[i])

        # Layer 2: skip2 = W2_in @ u_in + b2_in, block2 = sin(skip2 + layer2(h)), h = h + block2
        var skip2_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer2_input, self.layer2_input, self.layer2_input_b, u_in_idx, tape)
            skip2_idx = result[0].copy()
        else:
            skip2_idx = self._linear_tracked_record_weights(self.layer2_input, self.layer2_input_b, u_in_idx, tape)

        var block2_h_idx = self._stable_linear_forward_tracked(self.layer2, h_idx, tape, p_idx, off_layer2)
        var block2_idx: List[Int] = []
        for i in range(len(skip2_idx)):
            block2_idx.append(tape.record_add(skip2_idx[i], block2_h_idx[i]))
        for i in range(len(block2_idx)):
            block2_idx[i] = tape.record_sin(block2_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block2_idx[i])

        # Layer 3
        var skip3_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer3_input, self.layer3_input, self.layer3_input_b, u_in_idx, tape)
            skip3_idx = result[0].copy()
        else:
            skip3_idx = self._linear_tracked_record_weights(self.layer3_input, self.layer3_input_b, u_in_idx, tape)

        var block3_h_idx = self._stable_linear_forward_tracked(self.layer3, h_idx, tape, p_idx, off_layer3)
        var block3_idx: List[Int] = []
        for i in range(len(skip3_idx)):
            block3_idx.append(tape.record_add(skip3_idx[i], block3_h_idx[i]))
        for i in range(len(block3_idx)):
            block3_idx[i] = tape.record_sin(block3_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block3_idx[i])

        # Layer 4
        var skip4_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer4_input, self.layer4_input, self.layer4_input_b, u_in_idx, tape)
            skip4_idx = result[0].copy()
        else:
            skip4_idx = self._linear_tracked_record_weights(self.layer4_input, self.layer4_input_b, u_in_idx, tape)

        var block4_h_idx = self._stable_linear_forward_tracked(self.layer4, h_idx, tape, p_idx, off_layer4)
        var block4_idx: List[Int] = []
        for i in range(len(skip4_idx)):
            block4_idx.append(tape.record_add(skip4_idx[i], block4_h_idx[i]))
        for i in range(len(block4_idx)):
            block4_idx[i] = tape.record_sin(block4_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block4_idx[i])

        # Layer 5: u = W5 @ h + b5
        var u_out_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer5, self.layer5, self.layer5_b, h_idx, tape)
            u_out_idx = result[0].copy()
        else:
            u_out_idx = self._linear_tracked_record_weights(self.layer5, self.layer5_b, h_idx, tape)

        # Layer 6: phi = W6 @ h + b6
        var phi_out_idx: List[Int] = []
        if use_external_params:
            var result = self._linear_tracked_with_indices(p_idx, off_layer6, self.layer6, self.layer6_b, h_idx, tape)
            phi_out_idx = result[0].copy()
        else:
            phi_out_idx = self._linear_tracked_record_weights(self.layer6, self.layer6_b, h_idx, tape)

        return (u_out_idx[0], phi_out_idx^)

    def _count_params(self) -> Int:
        """Count total number of parameters in the network."""
        var count = 0
        # Layer 1
        for i in range(len(self.layer1)):
            count += len(self.layer1[i])
        count += len(self.layer1_b)
        # Layers 2-4
        for i in range(len(self.layer2.W)):
            count += len(self.layer2.W[i])
        count += len(self.layer2.b)
        for i in range(len(self.layer3.W)):
            count += len(self.layer3.W[i])
        count += len(self.layer3.b)
        for i in range(len(self.layer4.W)):
            count += len(self.layer4.W[i])
        count += len(self.layer4.b)
        # Skip connections
        for i in range(len(self.layer2_input)):
            count += len(self.layer2_input[i])
        count += len(self.layer2_input_b)
        for i in range(len(self.layer3_input)):
            count += len(self.layer3_input[i])
        count += len(self.layer3_input_b)
        for i in range(len(self.layer4_input)):
            count += len(self.layer4_input[i])
        count += len(self.layer4_input_b)
        # Output layers
        for i in range(len(self.layer5)):
            count += len(self.layer5[i])
        count += len(self.layer5_b)
        for i in range(len(self.layer6)):
            count += len(self.layer6[i])
        count += len(self.layer6_b)
        return count

    def _linear_tracked_with_indices(
        self, p_idx: List[Int], param_offset: Int,
        W: List[List[Float64]], b: List[Float64], x_idx: List[Int],
        mut tape: Tape,
    ) -> Tuple[List[Int], Int]:
        """Linear layer using pre-recorded parameter indices starting at param_offset.
        Returns (output_indices, updated_param_offset).
        """
        var in_dim = len(W)
        var out_dim = len(b)
        var y_out: List[Int] = []
        var offset = param_offset

        for j in range(out_dim):
            var sum_idx: Int = -1
            for i in range(in_dim):
                var w_idx = p_idx[offset]
                offset += 1
                var prod = tape.record_mul(w_idx, x_idx[i])
                if sum_idx == -1:
                    sum_idx = prod
                else:
                    sum_idx = tape.record_add(sum_idx, prod)
            var b_i = p_idx[offset]
            offset += 1
            y_out.append(tape.record_add(sum_idx, b_i))
        return (y_out^, offset)

    def _linear_tracked_record_weights(
        self,
        W: List[List[Float64]], b: List[Float64], x_idx: List[Int],
        mut tape: Tape,
    ) -> List[Int]:
        """Linear layer recording its own weights on tape."""
        var in_dim = len(W)
        var out_dim = len(b)
        var y_out: List[Int] = []

        for j in range(out_dim):
            var sum_idx: Int = -1
            for i in range(in_dim):
                var w_val = tape.record_value(W[i][j])
                var prod = tape.record_mul(w_val, x_idx[i])
                if sum_idx == -1:
                    sum_idx = prod
                else:
                    sum_idx = tape.record_add(sum_idx, prod)
            var b_val = tape.record_value(b[j])
            y_out.append(tape.record_add(sum_idx, b_val))
        return y_out^

    def _stable_linear_forward_tracked(
        self,
        layer: StableLinear,
        h_idx: List[Int],
        mut tape: Tape,
        p_idx: List[Int],
        param_offset: Int,
    ) raises -> List[Int]:
        """Record constrained linear layer forward pass on tape.
        If p_idx is non-empty, uses pre-recorded parameter indices starting at param_offset.
        Otherwise, records new weight values on tape.
        """
        var use_external = len(p_idx) > 0
        var W_c = layer._constrain_weight()
        if len(W_c) == 0:
            if use_external:
                var b_idx: List[Int] = []
                for i in range(len(layer.b)):
                    b_idx.append(p_idx[param_offset + i])
                return b_idx^
            else:
                var b_idx: List[Int] = []
                for i in range(len(layer.b)):
                    b_idx.append(tape.record_value(layer.b[i]))
                return b_idx^

        var in_dim = len(W_c)
        var out_dim = len(layer.b)

        if use_external:
            var W_idx: List[Int] = []
            var offset = param_offset
            for i in range(in_dim):
                for j in range(out_dim):
                    W_idx.append(p_idx[offset])
                    offset += 1
            var b_idx: List[Int] = []
            for i in range(out_dim):
                b_idx.append(p_idx[offset])
                offset += 1
            return tape.record_linear(W_idx, b_idx, h_idx)
        else:
            var W_idx: List[Int] = []
            for i in range(in_dim):
                for j in range(out_dim):
                    W_idx.append(tape.record_value(W_c[i][j]))

            var b_idx: List[Int] = []
            for i in range(out_dim):
                b_idx.append(tape.record_value(layer.b[i]))

            return tape.record_linear(W_idx, b_idx, h_idx)
