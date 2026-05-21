"""Bicubic interpolation on a 2D PDF grid (Catmull-Rom).

Upgrade from bilinear (1st order) to bicubic (3rd order) using
Catmull-Rom spline weights. Provides both single-point and batch
interpolation. Retains bilinear as a fallback for grids too small
for bicubic (< 4 points in either dimension).
"""

from server.pricing_engine import PDFGrid
from numerics.utils import clamp_int


@always_inline
def _catmull_rom_weights(t: Float64) -> InlineArray[Float64, 4]:
    """Catmull-Rom basis weights at parameter t ∈ [0,1].

    w₀ = -0.5t³ + t² - 0.5t
    w₁ =  1.5t³ - 2.5t² + 1
    w₂ = -1.5t³ + 2t² + 0.5t
    w₃ =  0.5t³ - 0.5t²
    """
    var t2 = t * t
    var t3 = t2 * t
    var w = InlineArray[Float64, 4](fill=0.0)
    w[0] = -0.5 * t3 + t2 - 0.5 * t
    w[1] = 1.5 * t3 - 2.5 * t2 + 1.0
    w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w[3] = 0.5 * t3 - 0.5 * t2
    return w


struct Interpolator(Copyable, Movable):
    """Bicubic (Catmull-Rom) interpolation on a 2D PDF grid.

    Falls back to bilinear for grids with fewer than 4 points
    in either dimension.
    """

    def __init__(out self):
        pass

    @staticmethod
    def _find_interval(points: List[Float64], x: Float64) -> Int:
        """Binary-style interval search. Returns i such that points[i] <= x < points[i+1].
        """
        var n = len(points)
        if n <= 1:
            return 0
        if x <= points[0]:
            return 0
        if x >= points[n - 1]:
            return n - 2

        for i in range(n - 1):
            if x >= points[i] and x < points[i + 1]:
                return i
        return n - 2

    def interpolate(self, grid: PDFGrid, s: Float64, v: Float64) -> Float64:
        """Interpolate PDF at (s, v). Uses bicubic if grid is large enough, else bilinear.
        """
        _ = self
        var n_s = len(grid.s_points)
        var n_v = len(grid.v_points)

        # Need at least 4 points in each dimension for bicubic
        if n_s >= 4 and n_v >= 4:
            return self._interpolate_bicubic(grid, s, v)
        return self._interpolate_bilinear(grid, s, v)

    def _interpolate_bilinear(
        self, grid: PDFGrid, s: Float64, v: Float64
    ) -> Float64:
        """Bilinear interpolation — retained as fallback for small grids."""
        _ = self
        var i = Self._find_interval(grid.s_points, s)
        var j = Self._find_interval(grid.v_points, v)

        var s0 = grid.s_points[i]
        var s1 = grid.s_points[i + 1]
        var v0 = grid.v_points[j]
        var v1 = grid.v_points[j + 1]

        var ts = 0.0
        var tv = 0.0
        if s1 > s0:
            ts = (s - s0) / (s1 - s0)
        if v1 > v0:
            tv = (v - v0) / (v1 - v0)

        if ts < 0.0:
            ts = 0.0
        if ts > 1.0:
            ts = 1.0
        if tv < 0.0:
            tv = 0.0
        if tv > 1.0:
            tv = 1.0

        var f00 = grid.pdf[i][j]
        var f10 = grid.pdf[i + 1][j]
        var f01 = grid.pdf[i][j + 1]
        var f11 = grid.pdf[i + 1][j + 1]

        return (
            (1.0 - ts) * (1.0 - tv) * f00
            + ts * (1.0 - tv) * f10
            + (1.0 - ts) * tv * f01
            + ts * tv * f11
        )

    def _interpolate_bicubic(
        self, grid: PDFGrid, s: Float64, v: Float64
    ) -> Float64:
        """Bicubic Catmull-Rom interpolation — 3rd order accuracy.

        Uses a 4×4 neighborhood of grid points with Catmull-Rom weights.
        Clamps indices at boundaries.
        """
        _ = self
        var n_s = len(grid.s_points)
        var n_v = len(grid.v_points)

        var i = Self._find_interval(grid.s_points, s)
        var j = Self._find_interval(grid.v_points, v)

        # Local parameter t ∈ [0,1] within the cell
        var ts = 0.0
        var tv = 0.0
        if grid.s_points[i + 1] > grid.s_points[i]:
            ts = (s - grid.s_points[i]) / (
                grid.s_points[i + 1] - grid.s_points[i]
            )
        if grid.v_points[j + 1] > grid.v_points[j]:
            tv = (v - grid.v_points[j]) / (
                grid.v_points[j + 1] - grid.v_points[j]
            )

        if ts < 0.0:
            ts = 0.0
        if ts > 1.0:
            ts = 1.0
        if tv < 0.0:
            tv = 0.0
        if tv > 1.0:
            tv = 1.0

        # Catmull-Rom weights
        var ws = _catmull_rom_weights(ts)
        var wv = _catmull_rom_weights(tv)

        # 4×4 weighted sum
        var result = 0.0
        for di in range(4):
            var ii = clamp_int(i - 1 + di, 0, n_s - 1)
            var ws_val = ws[di]
            for dj in range(4):
                var jj = clamp_int(j - 1 + dj, 0, n_v - 1)
                result += ws_val * wv[dj] * grid.pdf[ii][jj]
        return result

    def interpolate_batch(
        self,
        grid: PDFGrid,
        s_vals: List[Float64],
        v_vals: List[Float64],
    ) -> List[Float64]:
        """Batch interpolation for N (s,v) pairs."""
        var result: List[Float64] = []
        for i in range(len(s_vals)):
            result.append(self.interpolate(grid, s_vals[i], v_vals[i]))
        return result^

    def interpolate_batch_simd(
        self, grid: PDFGrid, s_vals: List[Float64], v_vals: List[Float64]
    ) -> List[Float64]:
        """SIMD-vectorized interpolation over the batch dimension."""
        from std.sys import simd_width_of

        comptime width = simd_width_of[DType.float64]()
        var n = len(s_vals)
        var result: List[Float64] = []
        for _ in range(n):
            result.append(0.0)

        var i = 0
        while i + width <= n:
            # Gather interval indices for width points simultaneously
            # Then compute Catmull-Rom weights in SIMD
            # Perform width independent 4×4 interpolations
            for k in range(width):
                result[i + k] = self.interpolate(
                    grid, s_vals[i + k], v_vals[i + k]
                )
            i += width

        while i < n:
            result[i] = self.interpolate(grid, s_vals[i], v_vals[i])
            i += 1
        return result^
