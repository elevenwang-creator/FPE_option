"""Shared numerical utility functions.

Consolidates duplicated helpers (_abs, _max, _min, _zeros, _copy_vec, etc.)
from 7+ files into a single source of truth. All hot-path functions are
@always_inline to eliminate call overhead in inner loops.
"""


from std.math import exp, log

@always_inline
def abs_f64(x: Float64) -> Float64:
    """Absolute value for Float64. Inline for zero overhead."""
    if x < 0.0:
        return -x
    return x


@always_inline
def max_f64(a: Float64, b: Float64) -> Float64:
    """Max of two Float64 values. Inline for zero overhead."""
    if a > b:
        return a
    return b


@always_inline
def min_f64(a: Float64, b: Float64) -> Float64:
    """Min of two Float64 values. Inline for zero overhead."""
    if a < b:
        return a
    return b


@always_inline
def max_int(a: Int, b: Int) -> Int:
    """Max of two Int values."""
    if a > b:
        return a
    return b


@always_inline
def min_int(a: Int, b: Int) -> Int:
    """Min of two Int values."""
    if a < b:
        return a
    return b


@always_inline
def clamp_int(x: Int, lo: Int, hi: Int) -> Int:
    """Clamp integer to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def zeros(n: Int) -> List[Float64]:
    """Create a zero-initialized vector of length n."""
    var out: List[Float64] = []
    for _ in range(n):
        out.append(0.0)
    return out^


def zeros_mat(nrows: Int, ncols: Int) -> List[List[Float64]]:
    """Create a zero-initialized matrix of size nrows × ncols."""
    var out: List[List[Float64]] = []
    for _ in range(nrows):
        var row: List[Float64] = []
        for _ in range(ncols):
            row.append(0.0)
        out.append(row^)
    return out^


def zeros_3d(d0: Int, d1: Int, d2: Int) -> List[List[List[Float64]]]:
    """Create a zero-initialized 3D tensor of size d0 × d1 × d2."""
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
    """Deep copy a vector."""
    var out: List[Float64] = []
    for i in range(len(src)):
        out.append(src[i])
    return out^


def copy_mat(src: List[List[Float64]]) -> List[List[Float64]]:
    """Deep copy a matrix."""
    var out: List[List[Float64]] = []
    for i in range(len(src)):
        out.append(copy_vec(src[i]))
    return out^


def swap_rows(mut A: List[List[Float64]], i: Int, j: Int):
    """Swap rows i and j in a dense matrix."""
    var ncols = len(A[i])
    for c in range(ncols):
        var tmp = A[i][c]
        A[i][c] = A[j][c]
        A[j][c] = tmp


@always_inline
def pow_pos(x: Float64, p: Float64) -> Float64:
    """- x^p for positive x, 0 otherwise."""
    if x <= 0.0:
        return 0.0
    return exp(log(x) * p)


def linspace(start: Float64, end: Float64, n: Int) -> List[Float64]:
    """Generate n evenly-spaced points in [start, end]."""
    var out: List[Float64] = []
    if n <= 1:
        out.append(start)
        return out^
    var step = (end - start) / Float64(n - 1)
    for i in range(n):
        out.append(start + Float64(i) * step)
    return out^
