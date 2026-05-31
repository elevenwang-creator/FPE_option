from std.math import exp, log
from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of

comptime SIMD_W: Int = simd_width_of[DType.float64]()


@always_inline
def clamp_int(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def zeros_mat(nrows: Int, ncols: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for _ in range(nrows):
        out.append(List[Float64](length=ncols, fill=0.0))
    return out^


def zeros_3d(d0: Int, d1: Int, d2: Int) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    for _ in range(d0):
        var mat: List[List[Float64]] = []
        for _ in range(d1):
            mat.append(List[Float64](length=d2, fill=0.0))
        out.append(mat^)
    return out^


def copy_mat(src: List[List[Float64]]) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for i in range(len(src)):
        out.append(src[i].copy())
    return out^


@always_inline
def pow_pos(x: Float64, p: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    return exp(log(x) * p)


def linspace(start: Float64, end: Float64, n: Int) -> List[Float64]:
    if n <= 0:
        return []
    if n == 1:
        return [start]
    var out = List[Float64](length=n, fill=0.0)
    var step = (end - start) / Float64(n - 1)
    var r_ptr = out.unsafe_ptr()

    def vfill[width: Int](p_off: Int) {read start, read step, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = start + Float64(p_off + w) * step

    vectorize[SIMD_W](n, vfill)
    return out^


def normalize(x: List[Float64]) -> List[Float64]:
    var n = len(x)
    if n == 0:
        return []

    var x_span = Span(x)
    var min_val = x[0]
    var max_val = x[0]

    for i in range(n):
        var v = x_span[i]
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v

    var range_val = max_val - min_val
    if range_val == 0.0:
        var result = List[Float64](length=n, fill=0.0)
        return result^

    var result = List[Float64](length=n, fill=0.0)
    var r_ptr = result.unsafe_ptr()
    var inv_range = 1.0 / range_val

    def vnorm[width: Int](p_off: Int) {read min_val, read inv_range, read x_span, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = (x_span[p_off + w] - min_val) * inv_range

    vectorize[SIMD_W](n, vnorm)

    return result^
