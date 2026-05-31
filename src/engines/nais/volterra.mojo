from numerics.utils import pow_pos, zeros_3d, zeros_mat

from std.math import exp, log, sqrt
from kernels.nn import rfft, irfft


def _complex_multiply(a: List[Float64], b: List[Float64]) -> List[Float64]:
    """Complex multiplication on interleaved real/imag rfft output.
    a and b are length (N/2+1)*2 with [re0, im0, re1, im1, ...].
    DC bin (index 0,1): re-only, multiply re parts.
    Nyquist bin (last pair): re-only, multiply re parts.
    Interior bins: (a_re*b_re - a_im*b_im, a_re*b_im + a_im*b_re).
    """
    var half_len = len(a) // 2
    var out: List[Float64] = []
    # DC bin: index 0 (re only, im=0)
    out.append(a[0] * b[0])
    out.append(a[1] * b[1])
    for k in range(1, half_len - 1):
        var ar = a[2 * k]
        var ai = a[2 * k + 1]
        var br = b[2 * k]
        var bi = b[2 * k + 1]
        out.append(ar * br - ai * bi)
        out.append(ar * bi + ai * br)
    # Nyquist bin: re only
    out.append(a[2 * (half_len - 1)] * b[2 * (half_len - 1)])
    out.append(a[2 * (half_len - 1) + 1] * b[2 * (half_len - 1) + 1])
    return out^


@fieldwise_init
struct VolterraProcess[B: Int]:
    """Fractional Brownian motion via hybrid scheme (direct convolution)."""

    var T: Float64
    var N: Int
    var D: Int
    var H: Float64

    def generate(
        self, W: List[List[List[Float64]]]
    ) -> List[List[List[Float64]]]:
        """Generate Volterra process X̃ from Brownian motion W."""
        var M = len(W)
        var out = zeros_3d(M, self.N + 1, self.D)
        if self.N <= 0:
            return out^

        var alpha = self.H - 0.5
        var dt = self.T / Float64(self.N)
        var sigma = sqrt(2.0 * alpha + 1.0)

        for m in range(M):
            for d in range(self.D):
                out[m][0][d] = W[m][0][d]

                var dW: List[Float64] = []
                for n in range(self.N):
                    dW.append(W[m][n + 1][d] - W[m][n][d])

                var kernel: List[Float64] = []
                kernel.append(pow_pos(dt * 0.5, alpha) * sigma)
                for k in range(1, self.N):
                    var kp1 = Float64(k + 1)
                    var kp2 = Float64(k + 2)
                    var y = (
                        pow_pos(kp2, alpha + 1.0) - pow_pos(kp1, alpha + 1.0)
                    ) / (alpha + 1.0)
                    var b = pow_pos(y, 1.0 / alpha)
                    kernel.append(pow_pos(b * dt, alpha) * sigma)

                for n in range(1, self.N + 1):
                    var acc = 0.0
                    for j in range(n):
                        acc = acc + dW[j] * kernel[n - 1 - j]
                    out[m][n][d] = acc

        return out^

    def generate_fft(
        self, W: List[List[List[Float64]]]
    ) -> List[List[List[Float64]]]:
        """Volterra process via FFT convolution: O(N log N) instead of O(N²)."""
        var M = len(W)
        var out = zeros_3d(M, self.N + 1, self.D)
        if self.N <= 0:
            return out^

        var alpha = self.H - 0.5
        var dt = self.T / Float64(self.N)
        var sigma = sqrt(2.0 * alpha + 1.0)

        # Build kernel once (N elements)
        var kernel: List[Float64] = []
        kernel.append(pow_pos(dt * 0.5, alpha) * sigma)
        for k in range(1, self.N):
            var kp1 = Float64(k + 1)
            var kp2 = Float64(k + 2)
            var y = (pow_pos(kp2, alpha + 1.0) - pow_pos(kp1, alpha + 1.0)) / (
                alpha + 1.0
            )
            var b = pow_pos(y, 1.0 / alpha)
            kernel.append(pow_pos(b * dt, alpha) * sigma)

        for m in range(M):
            for d in range(self.D):
                out[m][0][d] = W[m][0][d]

                var dW: List[Float64] = []
                for n in range(self.N):
                    dW.append(W[m][n + 1][d] - W[m][n][d])

                # Pad both to length 2N for circular convolution
                var padded_len = 2 * self.N
                var dW_padded = List[Float64](length=padded_len, fill=0.0)
                var k_padded = List[Float64](length=padded_len, fill=0.0)
                for i in range(self.N):
                    dW_padded[i] = dW[i]
                    k_padded[i] = kernel[i]

                # Forward FFT → multiply → inverse FFT
                var dW_freq = rfft(dW_padded)
                var k_freq = rfft(k_padded)
                var conv_freq = _complex_multiply(dW_freq, k_freq)
                var conv = irfft(conv_freq)

                for n in range(1, self.N + 1):
                    out[m][n][d] = conv[n - 1]

        return out^
