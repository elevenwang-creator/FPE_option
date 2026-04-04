from numerics.utils import max_f64

from engines.nais.nais_net import NaisNet
from numerics.nn.autograd import Tape
from std.math import sqrt




@fieldwise_init
struct FBSDEParams(Copyable, Movable):
    var Xi: List[Float64]
    var T: Float64
    var M: Int
    var N: Int
    var D: Int
    var H: Float64
    var eta: Float64
    var pho: Float64
    var r: Float64
    var epsilon_t: Float64


@fieldwise_init
struct FBSDELoss[B: Int]:
    """Forward-Backward SDE loss from NAIS_rBM.py loss_function()."""

    var pho: Float64
    var r: Float64
    var epsilon_t: Float64

    def compute(
        self,
        net: NaisNet,
        t: List[Float64],
        W: List[List[Float64]],
        BM: List[List[Float64]],
        Var: List[List[Float64]],
        Xi: List[Float64],
    ) -> Float64:
        """Compute FBSDE loss for one batch of trajectories."""
        var M = len(W)
        if M == 0 or len(t) <= 1:
            return 0.0

        var N = len(t) - 1
        var loss = 0.0

        for m in range(M):
            var x0 = Xi.copy()
            var out0 = net.forward(t[0], x0)
            var y0 = out0[0]
            var phi0 = out0[1].copy()

            for n in range(N):
                var dt = t[n + 1] - t[n]
                var dW = W[m][n + 1] - W[m][n]
                var dB = BM[m][n + 1] - BM[m][n]
                var var0 = max_f64(Var[m][n], 1e-12)

                var z = sqrt((1.0 - self.pho * self.pho) * var0) * phi0[0]
                var z_tilde = self.pho * sqrt(var0) * phi0[0] + phi0[0]

                var y1_tilde = y0 + self.r * y0 * dt + z * dB + z_tilde * dW

                var x1: List[Float64] = []
                x1.append(W[m][n + 1])
                if len(Xi) > 1:
                    x1.append(Var[m][n + 1])

                var out1 = net.forward(t[n + 1], x1)
                var y1 = out1[0]
                var phi1 = out1[1].copy()
                var diff = y1 - y1_tilde
                loss = loss + diff * diff

                y0 = y1
                phi0 = phi1^

            var terminal = max_f64(W[m][N] - Xi[0], 0.0)
            var t_diff = y0 - terminal
            loss = loss + 0.02 * t_diff * t_diff

        return loss / Float64(M)

    def compute_tracked(
        self,
        net: NaisNet,
        t: List[Float64],
        W: List[List[Float64]],
        BM: List[List[Float64]],
        Var: List[List[Float64]],
        Xi: List[Float64],
        mut tape: Tape,
    ) raises -> Int:
        """Compute FBSDE loss with operation recording on tape.
        Returns index of loss value in tape.values.
        """
        var M = len(W)
        if M == 0 or len(t) <= 1:
            return tape.record_value(0.0)

        var N = len(t) - 1
        var loss_idx = tape.record_value(0.0)

        for m in range(M):
            # Record Xi values
            var Xi_idx: List[Int] = []
            for i in range(len(Xi)):
                Xi_idx.append(tape.record_value(Xi[i]))

            # x0 = Xi
            var x0_idx = Xi_idx.copy()
            var out0 = net.forward_tracked(t[0], self._idx_to_values(tape, x0_idx), tape)
            var y0_idx = out0[0]
            var phi0_idx = out0[1].copy()

            for n in range(N):
                var dt = t[n + 1] - t[n]
                var dW = W[m][n + 1] - W[m][n]
                var dB = BM[m][n + 1] - BM[m][n]
                var var0 = max_f64(Var[m][n], 1e-12)

                var z = sqrt((1.0 - self.pho * self.pho) * var0) * self._get_value(tape, phi0_idx[0])
                var z_tilde = self.pho * sqrt(var0) * self._get_value(tape, phi0_idx[0]) + self._get_value(tape, phi0_idx[0])

                var y0_val = self._get_value(tape, y0_idx)
                var y1_tilde = y0_val + self.r * y0_val * dt + z * dB + z_tilde * dW

                var x1: List[Float64] = [W[m][n + 1]]
                if len(Xi) > 1:
                    x1.append(Var[m][n + 1])

                var out1 = net.forward_tracked(t[n + 1], x1, tape)
                var y1_idx = out1[0]
                var phi1_idx = out1[1].copy()

                var y1_val = self._get_value(tape, y1_idx)
                var diff = y1_val - y1_tilde
                var loss_inc = tape.record_value(diff * diff)
                loss_idx = tape.record_add(loss_idx, loss_inc)

                y0_idx = y1_idx
                phi0_idx = phi1_idx.copy()

            var terminal = max_f64(W[m][N] - Xi[0], 0.0)
            var y0_val = self._get_value(tape, y0_idx)
            var t_diff = y0_val - terminal
            var t_loss = tape.record_value(0.02 * t_diff * t_diff)
            loss_idx = tape.record_add(loss_idx, t_loss)

        # Divide by M
        var M_val = tape.record_value(Float64(M))
        loss_idx = tape.record_mul(loss_idx, tape.record_value(1.0 / Float64(M)))
        return loss_idx

    def _get_value(self, tape: Tape, idx: Int) -> Float64:
        return tape.values[idx]

    def _idx_to_values(self, tape: Tape, idx: List[Int]) -> List[Float64]:
        var out: List[Float64] = []
        for i in range(len(idx)):
            out.append(tape.values[idx[i]])
        return out^
