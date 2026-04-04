from numerics.utils import abs_f64
from std.math import sin, cos


struct GradientTape:
    """Lightweight autodiff tape API using finite-difference reverse pass."""

    var epsilon: Float64

    def __init__(out self, epsilon: Float64 = 1e-6):
        self.epsilon = epsilon

    def gradients(
        self,
        loss_fn: fn(List[Float64]) raises -> Float64,
        params: List[Float64],
    ) raises -> List[Float64]:
        var grads: List[Float64] = []
        for _ in range(len(params)):
            grads.append(0.0)

        for i in range(len(params)):
            var eps = self.epsilon * (1.0 + abs_f64(params[i]))
            var plus = params.copy()
            var minus = params.copy()
            plus[i] = plus[i] + eps
            minus[i] = minus[i] - eps
            var lp = loss_fn(plus)
            var lm = loss_fn(minus)
            grads[i] = (lp - lm) / (2.0 * eps)

        return grads^


struct Variable:
    """Tracked value with gradient accumulation."""
    var value: Float64
    var grad: Float64
    var _tape_idx: Int

    def __init__(out self, value: Float64):
        self.value = value
        self.grad = 0.0
        self._tape_idx = -1


@fieldwise_init
struct TapeEntry(Copyable, Movable):
    """Single operation in the computation graph."""
    var op: Int  # Operation type: 0=none(input), 1=add, 2=mul, 3=sin
    var inputs: List[Int]  # Indices into tape
    var output: Int
    var partials: List[Float64]  # ∂output/∂input for each input


struct Tape:
    """Reverse-mode autodiff tape. Records operations, plays back gradients."""
    var entries: List[TapeEntry]
    var values: List[Float64]
    var adjoints: List[Float64]

    def __init__(out self):
        self.entries = []
        self.values = []
        self.adjoints = []

    def record_value(mut self, value: Float64) -> Int:
        """Add an input value and return its index."""
        var idx = len(self.values)
        self.values.append(value)
        var entry = TapeEntry(op=0, inputs=[], output=idx, partials=[])
        self.entries.append(entry^)
        return idx

    def record_add(mut self, a: Int, b: Int) -> Int:
        """Record c = a + b. Returns index of c."""
        var c_val = self.values[a] + self.values[b]
        var idx = len(self.values)
        self.values.append(c_val)
        var entry = TapeEntry(op=1, inputs=[a, b], output=idx, partials=[1.0, 1.0])
        self.entries.append(entry^)
        return idx

    def record_mul(mut self, a: Int, b: Int) -> Int:
        """Record c = a * b. Returns index of c. Partials: dc/da=b, dc/db=a."""
        var c_val = self.values[a] * self.values[b]
        var idx = len(self.values)
        self.values.append(c_val)
        var entry = TapeEntry(op=2, inputs=[a, b], output=idx, partials=[self.values[b], self.values[a]])
        self.entries.append(entry^)
        return idx

    def record_sin(mut self, x: Int) -> Int:
        """Record y = sin(x). Returns index of y. Partial: dy/dx = cos(x)."""
        var y_val = sin(self.values[x])
        var idx = len(self.values)
        self.values.append(y_val)
        var entry = TapeEntry(op=3, inputs=[x], output=idx, partials=[cos(self.values[x])])
        self.entries.append(entry^)
        return idx

    def record_linear(
        mut self,
        W_idx: List[Int],
        b_idx: List[Int],
        x_idx: List[Int],
    ) -> List[Int]:
        """Record y = W @ x + b.
        W is flat: [W[0][0], W[1][0], ..., W[0][1], W[1][1], ...] (in_dim x out_dim)
        Returns indices of output values.
        """
        var in_dim = len(x_idx)
        var out_dim = len(b_idx)
        var y_out: List[Int] = []

        for j in range(out_dim):
            var sum_idx: Int = -1
            for i in range(in_dim):
                var w_idx = W_idx[i * out_dim + j]
                var prod = self.record_mul(w_idx, x_idx[i])
                if sum_idx == -1:
                    sum_idx = prod
                else:
                    sum_idx = self.record_add(sum_idx, prod)
            var y_j = self.record_add(sum_idx, b_idx[j])
            y_out.append(y_j)
        return y_out^

    def backward(mut self, loss_idx: Int):
        """Backward pass: accumulate gradients from loss to all inputs."""
        var n = len(self.entries)
        self.adjoints = []
        for _ in range(n):
            self.adjoints.append(0.0)
        self.adjoints[loss_idx] = 1.0

        # Reverse topological order
        for rev in range(n):
            var idx = n - 1 - rev
            var adj = self.adjoints[idx]
            for k in range(len(self.entries[idx].inputs)):
                self.adjoints[self.entries[idx].inputs[k]] += adj * self.entries[idx].partials[k]

    def gradients_for(self, param_indices: List[Int]) -> List[Float64]:
        """After backward(), extract gradients for specified parameter indices."""
        var grads: List[Float64] = []
        for i in range(len(param_indices)):
            grads.append(self.adjoints[param_indices[i]])
        return grads^
