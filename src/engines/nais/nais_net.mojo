from numerics.utils import mat_vec_mul
from layout import TileTensor, coord
from layout.tile_layout import row_major

from numerics.nn.stable_linear import StableLinear, make_stable_linear
from numerics.nn.autograd import Tape
from std.math import sin
from std.memory import Span


def _make_weights(
    in_dim: Int, out_dim: Int, scale: Float64
) -> List[Float64]:
    var W_flat = List[Float64](length=out_dim * in_dim, fill=0.0)
    for i in range(in_dim):
        for j in range(out_dim):
            var k = ((i + 1) * 7 + (j + 1) * 5) % 17
            W_flat[j * in_dim + i] = scale * (Float64(k) - 8.0) / 8.0
    return W_flat^


@always_inline
def _linear(
    W_T_flat: List[Float64], b: List[Float64], x: List[Float64],
    in_dim: Int, out_dim: Int,
) -> List[Float64]:
    if out_dim == 0:
        return b.copy()
    var A = TileTensor(
        W_T_flat, row_major(coord[DType.int64]((out_dim, in_dim)))
    )
    var y = List[Float64](length=out_dim, fill=0.0)
    var y_span = Span[mut=True, Float64](y)
    mat_vec_mul(A, Span[Float64](x), y_span)
    for j in range(out_dim):
        y[j] += b[j]
    return y^


@always_inline
def _sin_vec(x: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=len(x), fill=0.0)
    for i in range(len(x)):
        out[i] = sin(x[i])
    return out^


@always_inline
def _add_vec(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=len(a), fill=0.0)
    for i in range(len(a)):
        out[i] = a[i] + b[i]
    return out^


@fieldwise_init
struct NaisNet(Copyable, Movable):
    var in_dim: Int
    var hidden: Int
    var phi_dim: Int
    var layer1_T_flat: List[Float64]
    var layer1_b: List[Float64]
    var layer2: StableLinear
    var layer2_input_T_flat: List[Float64]
    var layer2_input_b: List[Float64]
    var layer3: StableLinear
    var layer3_input_T_flat: List[Float64]
    var layer3_input_b: List[Float64]
    var layer4: StableLinear
    var layer4_input_T_flat: List[Float64]
    var layer4_input_b: List[Float64]
    var layer5_T_flat: List[Float64]
    var layer5_b: List[Float64]
    var layer6_T_flat: List[Float64]
    var layer6_b: List[Float64]

    def __init__(out self, in_dim: Int = 3, hidden: Int = 12, phi_dim: Int = 2):
        self.in_dim = in_dim
        self.hidden = hidden
        self.phi_dim = phi_dim
        self.layer1_T_flat = _make_weights(in_dim + 1, hidden, 0.08)
        self.layer1_b = List[Float64](length=hidden, fill=0.0)

        self.layer2 = make_stable_linear(hidden, hidden)
        self.layer2_input_T_flat = _make_weights(in_dim + 1, hidden, 0.04)
        self.layer2_input_b = List[Float64](length=hidden, fill=0.0)

        self.layer3 = make_stable_linear(hidden, hidden)
        self.layer3_input_T_flat = _make_weights(in_dim + 1, hidden, 0.04)
        self.layer3_input_b = List[Float64](length=hidden, fill=0.0)

        self.layer4 = make_stable_linear(hidden, hidden)
        self.layer4_input_T_flat = _make_weights(in_dim + 1, hidden, 0.04)
        self.layer4_input_b = List[Float64](length=hidden, fill=0.0)

        self.layer5_T_flat = _make_weights(hidden, 1, 0.06)
        self.layer5_b = List[Float64](length=1, fill=0.0)

        self.layer6_T_flat = _make_weights(hidden, phi_dim, 0.06)
        self.layer6_b = List[Float64](length=phi_dim, fill=0.0)

    def forward(
        self, t: Float64, x: List[Float64]
    ) -> Tuple[Float64, List[Float64]]:
        var u_in: List[Float64] = [t]
        for i in range(len(x)):
            u_in.append(x[i])
        var u_in_dim = self.in_dim + 1

        var h = _sin_vec(
            _linear(self.layer1_T_flat, self.layer1_b, u_in, u_in_dim, self.hidden)
        )

        var skip2 = _linear(
            self.layer2_input_T_flat, self.layer2_input_b, u_in, u_in_dim, self.hidden
        )
        var block2 = _sin_vec(_add_vec(skip2, self.layer2.forward(h)))
        h = _add_vec(h, block2)

        var skip3 = _linear(
            self.layer3_input_T_flat, self.layer3_input_b, u_in, u_in_dim, self.hidden
        )
        var block3 = _sin_vec(_add_vec(skip3, self.layer3.forward(h)))
        h = _add_vec(h, block3)

        var skip4 = _linear(
            self.layer4_input_T_flat, self.layer4_input_b, u_in, u_in_dim, self.hidden
        )
        var block4 = _sin_vec(_add_vec(skip4, self.layer4.forward(h)))
        h = _add_vec(h, block4)

        var u_vec = _linear(
            self.layer5_T_flat, self.layer5_b, h, self.hidden, 1
        )
        var phi = _linear(
            self.layer6_T_flat, self.layer6_b, h, self.hidden, self.phi_dim
        )
        return (u_vec[0], phi^)

    def forward_tracked(
        self,
        t: Float64,
        x: List[Float64],
        mut tape: Tape,
        param_indices: List[Int] = [],
    ) raises -> Tuple[Int, List[Int]]:
        var use_external_params = len(param_indices) > 0
        var p_idx = param_indices.copy()

        var u_in_dim = self.in_dim + 1
        var h = self.hidden
        var p = self.phi_dim

        var off_layer1 = 0
        var n_layer1 = u_in_dim * h
        var off_b1 = off_layer1 + n_layer1
        var off_layer2 = off_b1 + h

        var n_layer2 = self.layer2.in_features * self.layer2.out_features
        var off_b2 = off_layer2 + n_layer2
        var off_layer3 = off_b2 + self.layer2.out_features

        var n_layer3 = self.layer3.in_features * self.layer3.out_features
        var off_b3 = off_layer3 + n_layer3
        var off_layer4 = off_b3 + self.layer3.out_features

        var n_layer4 = self.layer4.in_features * self.layer4.out_features
        var off_b4 = off_layer4 + n_layer4
        var off_layer2_input = off_b4 + self.layer4.out_features

        var n_l2i = u_in_dim * h
        var off_b2i = off_layer2_input + n_l2i
        var off_layer3_input = off_b2i + h

        var n_l3i = u_in_dim * h
        var off_b3i = off_layer3_input + n_l3i
        var off_layer4_input = off_b3i + h

        var n_l4i = u_in_dim * h
        var off_b4i = off_layer4_input + n_l4i
        var off_layer5 = off_b4i + h

        var n_layer5 = h * 1
        var off_b5 = off_layer5 + n_layer5
        var off_layer6 = off_b5 + 1

        var t_idx = tape.record_value(t)
        var x_idx: List[Int] = []
        for i in range(len(x)):
            x_idx.append(tape.record_value(x[i]))

        var u_in_idx: List[Int] = [t_idx]
        for i in range(len(x_idx)):
            u_in_idx.append(x_idx[i])

        var h_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer1, self.layer1_T_flat, self.layer1_b,
                u_in_idx, u_in_dim, h, tape,
            )
            h_idx = result[0].copy()
        else:
            h_idx = self._linear_tracked_record_weights(
                self.layer1_T_flat, self.layer1_b,
                u_in_idx, u_in_dim, h, tape,
            )
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_sin(h_idx[i])

        var skip2_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer2_input, self.layer2_input_T_flat, self.layer2_input_b,
                u_in_idx, u_in_dim, h, tape,
            )
            skip2_idx = result[0].copy()
        else:
            skip2_idx = self._linear_tracked_record_weights(
                self.layer2_input_T_flat, self.layer2_input_b,
                u_in_idx, u_in_dim, h, tape,
            )

        var block2_h_idx = self._stable_linear_forward_tracked(
            self.layer2, h_idx, tape, p_idx, off_layer2
        )
        var block2_idx: List[Int] = []
        for i in range(len(skip2_idx)):
            block2_idx.append(tape.record_add(skip2_idx[i], block2_h_idx[i]))
        for i in range(len(block2_idx)):
            block2_idx[i] = tape.record_sin(block2_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block2_idx[i])

        var skip3_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer3_input, self.layer3_input_T_flat, self.layer3_input_b,
                u_in_idx, u_in_dim, h, tape,
            )
            skip3_idx = result[0].copy()
        else:
            skip3_idx = self._linear_tracked_record_weights(
                self.layer3_input_T_flat, self.layer3_input_b,
                u_in_idx, u_in_dim, h, tape,
            )

        var block3_h_idx = self._stable_linear_forward_tracked(
            self.layer3, h_idx, tape, p_idx, off_layer3
        )
        var block3_idx: List[Int] = []
        for i in range(len(skip3_idx)):
            block3_idx.append(tape.record_add(skip3_idx[i], block3_h_idx[i]))
        for i in range(len(block3_idx)):
            block3_idx[i] = tape.record_sin(block3_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block3_idx[i])

        var skip4_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer4_input, self.layer4_input_T_flat, self.layer4_input_b,
                u_in_idx, u_in_dim, h, tape,
            )
            skip4_idx = result[0].copy()
        else:
            skip4_idx = self._linear_tracked_record_weights(
                self.layer4_input_T_flat, self.layer4_input_b,
                u_in_idx, u_in_dim, h, tape,
            )

        var block4_h_idx = self._stable_linear_forward_tracked(
            self.layer4, h_idx, tape, p_idx, off_layer4
        )
        var block4_idx: List[Int] = []
        for i in range(len(skip4_idx)):
            block4_idx.append(tape.record_add(skip4_idx[i], block4_h_idx[i]))
        for i in range(len(block4_idx)):
            block4_idx[i] = tape.record_sin(block4_idx[i])
        for i in range(len(h_idx)):
            h_idx[i] = tape.record_add(h_idx[i], block4_idx[i])

        var u_out_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer5, self.layer5_T_flat, self.layer5_b,
                h_idx, h, 1, tape,
            )
            u_out_idx = result[0].copy()
        else:
            u_out_idx = self._linear_tracked_record_weights(
                self.layer5_T_flat, self.layer5_b,
                h_idx, h, 1, tape,
            )

        var phi_out_idx: List[Int]
        if use_external_params:
            var result = self._linear_tracked_with_indices(
                p_idx, off_layer6, self.layer6_T_flat, self.layer6_b,
                h_idx, h, p, tape,
            )
            phi_out_idx = result[0].copy()
        else:
            phi_out_idx = self._linear_tracked_record_weights(
                self.layer6_T_flat, self.layer6_b,
                h_idx, h, p, tape,
            )

        return (u_out_idx[0], phi_out_idx^)

    def _count_params(self) -> Int:
        var count = 0
        count += len(self.layer1_T_flat) + len(self.layer1_b)
        count += len(self.layer2.W_T_flat) + len(self.layer2.b)
        count += len(self.layer3.W_T_flat) + len(self.layer3.b)
        count += len(self.layer4.W_T_flat) + len(self.layer4.b)
        count += len(self.layer2_input_T_flat) + len(self.layer2_input_b)
        count += len(self.layer3_input_T_flat) + len(self.layer3_input_b)
        count += len(self.layer4_input_T_flat) + len(self.layer4_input_b)
        count += len(self.layer5_T_flat) + len(self.layer5_b)
        count += len(self.layer6_T_flat) + len(self.layer6_b)
        return count

    def _linear_tracked_with_indices(
        self,
        p_idx: List[Int],
        param_offset: Int,
        W_T_flat: List[Float64],
        b: List[Float64],
        x_idx: List[Int],
        in_dim: Int,
        out_dim: Int,
        mut tape: Tape,
    ) -> Tuple[List[Int], Int]:
        var y_out: List[Int] = []
        var offset = param_offset
        for _ in range(out_dim):
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
        W_T_flat: List[Float64],
        b: List[Float64],
        x_idx: List[Int],
        in_dim: Int,
        out_dim: Int,
        mut tape: Tape,
    ) -> List[Int]:
        var y_out: List[Int] = []
        for j in range(out_dim):
            var sum_idx: Int = -1
            for i in range(in_dim):
                var w_val = tape.record_value(W_T_flat[j * in_dim + i])
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
        var use_external = len(p_idx) > 0
        var W_c = layer._constrain_weight()
        var mf = layer.out_features
        if len(W_c) == 0:
            if use_external:
                var b_idx: List[Int] = []
                for i in range(mf):
                    b_idx.append(p_idx[param_offset + i])
                return b_idx^
            else:
                var b_idx: List[Int] = []
                for i in range(mf):
                    b_idx.append(tape.record_value(layer.b[i]))
                return b_idx^

        if use_external:
            var W_idx: List[Int] = []
            var offset = param_offset
            for _ in range(mf * mf):
                W_idx.append(p_idx[offset])
                offset += 1
            var b_idx: List[Int] = []
            for _ in range(mf):
                b_idx.append(p_idx[offset])
                offset += 1
            return tape.record_linear(W_idx, b_idx, h_idx)
        else:
            var W_idx: List[Int] = []
            for idx in range(mf * mf):
                W_idx.append(tape.record_value(W_c[idx]))
            var b_idx: List[Int] = []
            for i in range(mf):
                b_idx.append(tape.record_value(layer.b[i]))
            return tape.record_linear(W_idx, b_idx, h_idx)
