from std.math import exp, log
from std.algorithm.backend.vectorize import vectorize
from std.memory import Span
from std.sys import simd_width_of

comptime SIMD_W: Int = simd_width_of[DType.float64]()


@always_inline
def abs_f64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


@always_inline
def max_f64(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


@always_inline
def min_f64(a: Float64, b: Float64) -> Float64:
    if a < b:
        return a
    return b


@always_inline
def max_int(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


@always_inline
def min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


@always_inline
def clamp_int(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def zeros(n: Int) -> List[Float64]:
    var out: List[Float64] = []
    for _ in range(n):
        out.append(0.0)
    return out^


def zeros_mat(nrows: Int, ncols: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for _ in range(nrows):
        var row: List[Float64] = []
        for _ in range(ncols):
            row.append(0.0)
        out.append(row^)
    return out^


def zeros_3d(d0: Int, d1: Int, d2: Int) -> List[List[List[Float64]]]:
    var out: List[List[List[Float64]]] = []
    for _ in range(d0):
        var mat: List[List[Float64]] = []
        for _ in range(d1):
            var row: List[Float64] = []
            for _ in range(d2):
                row.append(0.0)
            mat.append(row^)
        out.append(mat^)
    return out^


def copy_vec(src: List[Float64]) -> List[Float64]:
    var out: List[Float64] = []
    for i in range(len(src)):
        out.append(src[i])
    return out^


def copy_mat(src: List[List[Float64]]) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    for i in range(len(src)):
        out.append(copy_vec(src[i]))
    return out^


def swap_rows(mut A: List[List[Float64]], i: Int, j: Int):
    assert i >= 0 and i < len(A), "Row index i out of bounds"
    assert j >= 0 and j < len(A), "Row index j out of bounds"
    var ncols = len(A[i])
    assert len(A[j]) == ncols, "Rows must have same length"
    for c in range(ncols):
        var tmp = A[i][c]
        A[i][c] = A[j][c]
        A[j][c] = tmp


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
    var out = List[Float64]()
    for _ in range(n):
        out.append(0.0)
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
        var result: List[Float64] = []
        for _ in range(n):
            result.append(0.0)
        return result^

    var result = List[Float64]()
    for _ in range(n):
        result.append(0.0)
    var r_ptr = result.unsafe_ptr()
    var inv_range = 1.0 / range_val

    def vnorm[width: Int](p_off: Int) {read min_val, read inv_range, read x_span, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = (x_span[p_off + w] - min_val) * inv_range

    vectorize[SIMD_W](n, vnorm)

    return result^
