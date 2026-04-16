from std.math import pi, cos, sqrt
from std.collections import List


struct GaussLegendre:
    var order: Int
    var nodes: List[Float64]
    var weights: List[Float64]

    def __init__(out self, order: Int):
        self.order = order
        if order == 2:
            var s = sqrt(1.0 / 3.0)
            self.nodes = [-s, s]
            self.weights = [0.5, 0.5]
        elif order == 3:
            self.nodes = [-sqrt(3.0 / 5.0), 0.0, sqrt(3.0 / 5.0)]
            self.weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        elif order == 4:
            var s1 = sqrt(3.0 / 7.0 + 2.0 / 7.0 * sqrt(5.0 / 6.0))
            var s2 = sqrt(3.0 / 7.0 - 2.0 / 7.0 * sqrt(5.0 / 6.0))
            var w1 = (18.0 - sqrt(30.0)) / 36.0
            var w2 = (18.0 + sqrt(30.0)) / 36.0
            self.nodes = [-s1, -s2, s2, s1]
            self.weights = [w1, w2, w2, w1]
        elif order == 5:
            self.nodes = [
                -0.9061798459386639927976,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386639927976,
            ]
            self.weights = [
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ]
        elif order == 6:
            var s1 = sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0
            var s2 = sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)) / 3.0
            var s3 = sqrt(5.0 / 3.0 - 2.0 / 21.0 * sqrt(5.0 / 2.0))
            var s4 = sqrt(5.0 / 3.0 + 2.0 / 21.0 * sqrt(5.0 / 2.0))
            var w1 = (322.0 + 13.0 * sqrt(70.0)) / 1800.0
            var w2 = (322.0 - 13.0 * sqrt(70.0)) / 1800.0
            var w3 = 128.0 / 225.0
            self.nodes = [-s2, -s4, -s3, s3, s4, s2]
            self.weights = [w1, w3, w2, w2, w3, w1]
        else:
            self.nodes = [0.0]
            self.weights = [1.0]


@align(64)
struct GenerateKnots(Copyable, Movable):
    var n: Int
    var degree: Int
    var method: String
    var center: Float64
    var boundary: Tuple[Float64, Float64]
    var mean: Float64
    var std: Float64

    def __init__(
        out self,
        n: Int,
        degree: Int,
        method: String = "uniform",
        center: Float64 = 0.2,
        boundary: Tuple[Float64, Float64] = (0.0, 1.0),
        mean: Float64 = 50.0,
        std: Float64 = 0.1,
    ):
        self.n = n
        self.degree = degree
        self.method = method
        self.center = center
        self.boundary = boundary
        self.mean = mean
        self.std = std

    def normalized_boundary(self) -> Tuple[Float64, Float64]:
        var lo = self.boundary[0]
        var hi = self.boundary[1]
        if hi <= lo:
            return (0.0, 1.0)
        return (lo, hi)

    def linspace(self, start: Float64, stop: Float64, count: Int) -> List[Float64]:
        var out: List[Float64] = []
        if count <= 0:
            return out^
        if count == 1:
            out.append(start)
            return out^

        var step = (stop - start) / Float64(count - 1)
        for i in range(count):
            out.append(start + Float64(i) * step)
        return out^

    def chebyshev_nodes(self, count: Int, low: Float64, high: Float64) -> List[Float64]:
        var out: List[Float64] = []
        if count <= 0: return out^
        for i in range(count):
            var angle = (2.0 * Float64(i) + 1.0) * pi / (2.0 * Float64(count))
            var node = cos(angle)
            out.append(0.5 * (node + 1.0) * (high - low) + low)
        return out^

    def parabolic_internal_knots(self, count: Int) -> List[Float64]:
        var out: List[Float64] = []
        var u = self.linspace(0.0, 1.0, count)
        var c = self.center
        for i in range(len(u)):
            var ui = u[i]
            if ui <= 0.5:
                var val = 2.0 * ui
                out.append(c * val * val)
            else:
                var val = 2.0 * (1.0 - ui)
                out.append(1.0 - (1.0 - c) * val * val)
        return out^

    def generate_knots(self) -> List[Float64]:
        var p = self.degree
        var b_low = self.boundary[0]
        var b_high = self.boundary[1]
        
        var x_min = self.mean - 4.5 * self.std
        var x_max = self.mean + 4.5 * self.std
        var lb_norm = (x_min - b_low) / (b_high - b_low)
        var ub_norm = (x_max - b_low) / (b_high - b_low)

        var internal_num = self.n - 2 * p
        var knots_raw: List[Float64] = []
        
        if self.method == "non-uniform":
            var c_nodes = self.chebyshev_nodes(13, lb_norm, ub_norm)
            for i in range(len(c_nodes)): knots_raw.append(c_nodes[i])
            
            var p_nodes = self.parabolic_internal_knots(internal_num)
            for i in range(len(p_nodes)): knots_raw.append(p_nodes[i])
        else:
            var u_nodes = self.linspace(0.0, 1.0, internal_num)
            for i in range(len(u_nodes)): knots_raw.append(u_nodes[i])

        knots_raw.append(0.0)
        knots_raw.append(1.0)
        
        for i in range(len(knots_raw)):
            for j in range(i + 1, len(knots_raw)):
                if knots_raw[i] > knots_raw[j]:
                    var tmp = knots_raw[i]
                    knots_raw[i] = knots_raw[j]
                    knots_raw[j] = tmp

        var knots_unique: List[Float64] = []
        if len(knots_raw) > 0:
            knots_unique.append(knots_raw[0])
            for i in range(1, len(knots_raw)):
                if knots_raw[i] > knots_raw[i-1] + 1e-9:
                    knots_unique.append(knots_raw[i])

        var final_knots: List[Float64] = []
        for _ in range(p): final_knots.append(0.0)
        for i in range(len(knots_unique)): final_knots.append(knots_unique[i])
        for _ in range(p): final_knots.append(1.0)

        return final_knots^
