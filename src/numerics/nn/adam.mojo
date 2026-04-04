from numerics.utils import pow_pos

from std.math import exp, log




@fieldwise_init
struct Adam:
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var m: List[Float64]
    var v: List[Float64]
    var t: Int

    def __init__(
        out self,
        lr: Float64,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = []
        self.v = []
        self.t = 0

    def step(mut self, params: List[Float64], grads: List[Float64]) -> List[Float64]:
        """Adam update step."""
        if len(self.m) != len(params):
            self.m = []
            self.v = []
            for _ in range(len(params)):
                self.m.append(0.0)
                self.v.append(0.0)

        self.t = self.t + 1
        var t_float = Float64(self.t)
        var bc1 = 1.0 - pow_pos(self.beta1, t_float)
        var bc2 = 1.0 - pow_pos(self.beta2, t_float)
        if bc1 <= 1e-12:
            bc1 = 1e-12
        if bc2 <= 1e-12:
            bc2 = 1e-12

        var updated = params.copy()
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i]

            var m_hat = self.m[i] / bc1
            var v_hat = self.v[i] / bc2
            var denom = pow_pos(v_hat, 0.5) + self.eps
            updated[i] = updated[i] - self.lr * m_hat / denom

        return updated^
