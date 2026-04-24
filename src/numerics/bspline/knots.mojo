from std.math import pi, cos, sqrt, abs
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
            var _s1 = sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0
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
    var cheby_knots: Int

    def __init__(
        out self,
        n: Int,
        degree: Int,
        method: String = "uniform",
        center: Float64 = 0.2,
        boundary: Tuple[Float64, Float64] = (0.0, 1.0),
        mean: Float64 = 50.0,
        std: Float64 = 0.1,
        cheby_num: Int = 13,
    ):
        self.n = n
        self.degree = degree
        self.method = method
        self.center = center
        self.boundary = boundary
        self.mean = mean
        self.std = std
        self.cheby_knots = cheby_num

    def normalize(self, x: List[Float64]) -> List[Float64]:
        var min_val = x[0]
        var max_val = x[0]
        for i in range(len(x)):
            if x[i] < min_val:
                min_val = x[i]
            if x[i] > max_val:
                max_val = x[i]
        var result: List[Float64] = []
        for i in range(len(x)):
            result.append((x[i] - min_val) / (max_val - min_val))
        return result^

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

    def func_parabolic(self, n: Int, boundary: Tuple[Float64, Float64]) -> Tuple[Int, List[Float64]]:
        var low_y = boundary[0]
        var high_y = boundary[1]
        var centor = self.center

        var divide = sqrt((high_y - centor) / (centor - low_y)) + 1.0
        var n_adj = n
        if Float64(n) % divide == 0.0:
            n_adj = n
        else:
            n_adj = Int(Float64(n) / divide) * Int(divide)
        if n_adj < 1:
            n_adj = n
        var factor = 2.0

        var a = abs(2.0 * (low_y - centor) / (factor * factor))

        var upward = sqrt(2.0 * (high_y - centor) / a) + factor

        var x: List[Float64] = []
        var lin_part = self.linspace(0.0, upward, n_adj - 1)
        for i in range(len(lin_part)):
            x.append(lin_part[i])
        x.append(factor)

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i] > x[j]:
                    var tmp = x[i]
                    x[i] = x[j]
                    x[j] = tmp

        var y: List[Float64] = []
        for _ in range(len(x)):
            y.append(0.0)

        for i in range(len(x)):
            y[i] = -0.5 * a * (x[i] - factor) * (x[i] - factor) + centor

        var min_dist = abs(y[0] - centor)
        var center_idx = 0
        for i in range(1, len(y)):
            var d = abs(y[i] - centor)
            if d < min_dist:
                min_dist = d
                center_idx = i

        for i in range(len(x)):
            if i <= center_idx:
                y[i] = -0.5 * a * (x[i] - factor) * (x[i] - factor) + centor
            else:
                y[i] = 0.5 * a * (x[i] - factor) * (x[i] - factor) + centor

        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] > y[j]:
                    var tmp = y[i]
                    y[i] = y[j]
                    y[j] = tmp

        return (n_adj, y^)

    def chebyshev_knots(self, n: Int, a: Float64, b: Float64) -> List[Float64]:
        var result: List[Float64] = []
        for i in range(n):
            var angle = (2.0 * Float64(i) + 1.0) * pi / (2.0 * Float64(n))
            var node = cos(angle)
            result.append(node)
        var out: List[Float64] = []
        out.append(-1.0)
        for i in range(len(result)):
            out.append(result[i])
        out.append(1.0)
        var transformed: List[Float64] = []
        for i in range(len(out)):
            transformed.append((b - a) / 2.0 * out[i] + (a + b) / 2.0)
        return transformed^

    def knots_concat(
        self,
        knots: List[Float64],
        medim_knot: List[Float64],
        left: Float64,
        right: Float64,
    ) -> List[Float64]:
        var left_knots: List[Float64] = []
        var right_knots: List[Float64] = []
        for i in range(len(knots)):
            if abs(knots[i] - left) < 1e-10 or knots[i] < left:
                left_knots.append(knots[i])
            if abs(knots[i] - right) < 1e-10 or knots[i] > right:
                right_knots.append(knots[i])

        var points: List[Float64] = []
        for i in range(len(left_knots)):
            points.append(left_knots[i])
        for i in range(len(medim_knot)):
            points.append(medim_knot[i])
        for i in range(len(right_knots)):
            points.append(right_knots[i])

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] > points[j]:
                    var tmp = points[i]
                    points[i] = points[j]
                    points[j] = tmp

        var rounded: List[Float64] = []
        for i in range(len(points)):
            rounded.append(Float64(Int(points[i] * 1e8)) / 1e8)

        var unique: List[Float64] = []
        if len(rounded) > 0:
            unique.append(rounded[0])
            for i in range(1, len(rounded)):
                if rounded[i] > unique[len(unique) - 1] + 1e-9:
                    unique.append(rounded[i])
        return unique^

    def generate_knots(self) -> List[Float64]:
        var p = self.degree
        var boundary = self.boundary
        var boundary_normal = self.normalize([boundary[0], boundary[1]])
        var mean = self.mean
        var std = self.std
        var x_min = mean - 4.5 * std
        var x_min_normal = (x_min - boundary[0]) / (boundary[1] - boundary[0])
        var x_max = mean + 4.5 * std
        var x_max_normal = (x_max - boundary[0]) / (boundary[1] - boundary[0])
        var n_knots = self.cheby_knots

        var internal_num = self.n - 2 * p

        var internal_knots: List[Float64] = []

        if self.method == "uniform":
            internal_knots = self.linspace(0.0, 1.0, internal_num)
        elif self.method == "non-uniform":
            var x_knots = self.chebyshev_knots(n_knots, x_min, x_max)
            var x_normal: List[Float64] = []
            for i in range(len(x_knots)):
                x_normal.append(
                    (x_knots[i] - boundary[0]) / (boundary[1] - boundary[0])
                )

            var parabolic_result = self.func_parabolic(
                internal_num, (boundary_normal[0], boundary_normal[1])
            )
            var parabolic_knots = parabolic_result[1].copy()

            internal_knots = self.knots_concat(
                parabolic_knots, x_normal, x_min_normal, x_max_normal
            )

        var final_knots: List[Float64] = []
        for _ in range(p):
            final_knots.append(boundary_normal[0])
        for i in range(len(internal_knots)):
            final_knots.append(internal_knots[i])
        for _ in range(p):
            final_knots.append(boundary_normal[1])

        return final_knots^
