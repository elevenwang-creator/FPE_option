"""Shared numerical utility functions.

Consolidates duplicated helpers (_abs, _max, _min, _zeros, _copy_vec, etc.)
from 7+ files into a single source of truth. All hot-path functions are
@always_inline to eliminate call overhead in inner loops.

Performance: All SIMD operations use contiguous UnsafePointer.load/store
instead of scalar element-by-element gathers.
"""

from std.math import exp, log
from std.memory import UnsafePointer, alloc, memcpy, memset_zero
from std.sys import simd_width_of

comptime CACHE_LINE_SIZE: Int = 64
comptime SIMD_WIDTH: Int = simd_width_of[DType.float64]()
comptime MAX_VECTOR_SIZE: Int = 1 << 30


@align(CACHE_LINE_SIZE)
struct FixedSizeVector(Copyable, Movable, Writable):
    var _ptr: UnsafePointer[Float64, MutExternalOrigin]
    var _len: Int
    var _owns: Bool

    @always_inline
    def __init__(out self, n: Int, fill: Float64 = 0.0):
        assert n > 0, "Vector length must be positive"
        assert n <= MAX_VECTOR_SIZE, "Vector length exceeds maximum"
        self._ptr = alloc[Float64](n)
        self._len = n
        self._owns = True
        if fill == 0.0:
            memset_zero(self._ptr, n)
        else:
            for i in range(n):
                self._ptr[i] = fill

    @always_inline
    def __init__(out self, *, copy: Self):
        self._ptr = alloc[Float64](copy._len)
        self._len = copy._len
        self._owns = True
        memcpy(dest=self._ptr, src=copy._ptr, count=self._len)

    @always_inline
    def __init__(out self, *, deinit take: Self):
        self._ptr = take._ptr
        self._len = take._len
        self._owns = take._owns
        take._owns = False
        take._len = 0

    @always_inline
    def __del__(deinit self):
        if self._owns and self._len > 0:
            self._ptr.free()
            self._owns = False

    @always_inline
    def __getitem__(self, i: Int) -> Float64:
        assert i >= 0 and i < self._len, "Index out of bounds"
        return self._ptr[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: Float64):
        assert i >= 0 and i < self._len, "Index out of bounds"
        self._ptr[i] = val

    @always_inline
    def len(self) -> Int:
        return self._len

    @always_inline
    def ptr(self) -> UnsafePointer[Float64, MutExternalOrigin]:
        return self._ptr

    @always_inline
    def zero_out(mut self):
        memset_zero(self._ptr, self._len)

    @always_inline
    def copy_from(mut self, src: List[Float64]):
        assert len(src) >= self._len, "Source list too short for copy_from"
        for i in range(self._len):
            self._ptr[i] = src[i]

    @always_inline
    def copy_from_fixed(mut self, src: Self):
        assert src._len >= self._len, "Source too short for copy_from_fixed"
        memcpy(dest=self._ptr, src=src._ptr, count=self._len)

    @always_inline
    def to_list(self) -> List[Float64]:
        var out: List[Float64] = []
        # Pre-size the list then copy
        for i in range(self._len):
            out.append(self._ptr[i])
        return out^

    @always_inline
    def add_from(mut self, a: Self, b: Self):
        assert self._len == a._len and self._len == b._len
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var sa = (a._ptr + i).load[width=width]()
            var sb = (b._ptr + i).load[width=width]()
            (self._ptr + i).store[width=width](sa + sb)
            i += width
        while i < self._len:
            self._ptr[i] = a._ptr[i] + b._ptr[i]
            i += 1

    @always_inline
    def addassign(mut self, other: Self):
        assert self._len == other._len
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var ss = (self._ptr + i).load[width=width]()
            var so = (other._ptr + i).load[width=width]()
            (self._ptr + i).store[width=width](ss + so)
            i += width
        while i < self._len:
            self._ptr[i] = self._ptr[i] + other._ptr[i]
            i += 1

    @always_inline
    def lin_comb_3(
        mut self,
        c1: Float64,
        v1: Self,
        c2: Float64,
        v2: Self,
        c3: Float64,
        v3: Self,
    ):
        assert (
            self._len == v1._len
            and self._len == v2._len
            and self._len == v3._len
        )
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var s1 = (v1._ptr + i).load[width=width]()
            var s2 = (v2._ptr + i).load[width=width]()
            var s3 = (v3._ptr + i).load[width=width]()
            var sr = c1 * s1 + c2 * s2 + c3 * s3
            (self._ptr + i).store[width=width](sr)
            i += width
        while i < self._len:
            self._ptr[i] = c1 * v1._ptr[i] + c2 * v2._ptr[i] + c3 * v3._ptr[i]
            i += 1

    @always_inline
    def scaled_norm_sq(self, scal: Self) -> Float64:
        assert self._len == scal._len
        comptime width = SIMD_WIDTH
        var total = 0.0
        var i = 0
        while i + width <= self._len:
            var sv = (self._ptr + i).load[width=width]()
            var ss = (scal._ptr + i).load[width=width]()
            var ratio = sv / ss
            total += (ratio * ratio).reduce_add()
            i += width
        while i < self._len:
            var ratio = self._ptr[i] / scal._ptr[i]
            total += ratio * ratio
            i += 1
        return total

    @always_inline
    def addassign_offset(mut self, src: Self, offset: Int):
        assert offset + self._len <= src._len
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var ss = (self._ptr + i).load[width=width]()
            var so = (src._ptr + offset + i).load[width=width]()
            (self._ptr + i).store[width=width](ss + so)
            i += width
        while i < self._len:
            self._ptr[i] = self._ptr[i] + src._ptr[offset + i]
            i += 1

    @always_inline
    def lin_comb_2(mut self, c1: Float64, v1: Self, c2: Float64, v2: Self):
        assert self._len == v1._len and self._len == v2._len
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var s1 = (v1._ptr + i).load[width=width]()
            var s2 = (v2._ptr + i).load[width=width]()
            var sr = c1 * s1 + c2 * s2
            (self._ptr + i).store[width=width](sr)
            i += width
        while i < self._len:
            self._ptr[i] = c1 * v1._ptr[i] + c2 * v2._ptr[i]
            i += 1

    @always_inline
    def update_scal(mut self, atol: Float64, rtol: Float64, y: Self):
            assert self._len == y._len
            var safe_atol = max_f64(atol, 1e-300)
            var safe_rtol = max_f64(rtol, 1e-300)
            comptime width = SIMD_WIDTH
            var i = 0
            while i + width <= self._len:
                    var sy = (y._ptr + i).load[width=width]()
                    var sa = SIMD[DType.float64, width](safe_atol)
                    var sr = SIMD[DType.float64, width](safe_rtol)
                    var result = sa + sr * abs(sy)
                    (self._ptr + i).store[width=width](result)
                    i += width
            while i < self._len:
                    self._ptr[i] = safe_atol + safe_rtol * abs_f64(y._ptr[i])
                    i += 1

    @always_inline
    def update_scal_max(mut self, atol: Float64, rtol: Float64, y1: Self, y2: Self):
            assert self._len == y1._len and self._len == y2._len
            var safe_atol = max_f64(atol, 1e-300)
            var safe_rtol = max_f64(rtol, 1e-300)
            comptime width = SIMD_WIDTH
            var i = 0
            while i + width <= self._len:
                    var sy1 = (y1._ptr + i).load[width=width]()
                    var sy2 = (y2._ptr + i).load[width=width]()
                    var sa = SIMD[DType.float64, width](safe_atol)
                    var sr = SIMD[DType.float64, width](safe_rtol)
                    var result = sa + sr * max(abs(sy1), abs(sy2))
                    (self._ptr + i).store[width=width](result)
                    i += width
            while i < self._len:
                    self._ptr[i] = safe_atol + safe_rtol * max_f64(abs_f64(y1._ptr[i]), abs_f64(y2._ptr[i]))
                    i += 1

    @always_inline
    def sub_scaled(mut self, a: Self, alpha: Float64, b: Self):
        assert self._len == a._len and self._len == b._len
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var sa = (a._ptr + i).load[width=width]()
            var sb = (b._ptr + i).load[width=width]()
            (self._ptr + i).store[width=width](sa - alpha * sb)
            i += width
        while i < self._len:
            self._ptr[i] = a._ptr[i] - alpha * b._ptr[i]
            i += 1

    @always_inline
    def scale_assign(mut self, alpha: Float64):
        """Scale all elements by alpha in-place."""
        comptime width = SIMD_WIDTH
        var i = 0
        while i + width <= self._len:
            var sv = (self._ptr + i).load[width=width]()
            (self._ptr + i).store[width=width](alpha * sv)
            i += width
        while i < self._len:
            self._ptr[i] = alpha * self._ptr[i]
            i += 1


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
    assert n > 0, "Number of points must be positive"
    var out: List[Float64] = []
    if n == 1:
        out.append(start)
        return out^
    var step = (end - start) / Float64(n - 1)
    for i in range(n):
        out.append(start + Float64(i) * step)
    return out^
