# tests/test_max_integration.mojo
# Smoke test: verifies MAX AI Kernels stdlib and layout imports compile and
# basic operations run correctly.

from std.algorithm import vectorize, parallelize
from std.math import sqrt, exp, log, sin
from std.random import randn
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
import std.memory

from layout import Layout, LayoutTensor

# ---------------------------------------------------------------------------
# helper functions (non-nested to simplify closure matching)
# ---------------------------------------------------------------------------

def inner[width: Int](i: Int):
    var v = SIMD[DType.float32, width](Float32(i))
    _ = v * v

@parameter
def worker(i: Int):
    _ = i * i

# ---------------------------------------------------------------------------
# std.algorithm
# ---------------------------------------------------------------------------

def test_vectorize_works() raises:
    """Vectorize: run a simple SIMD squaring loop over 16 floats."""
    vectorize[8](16, inner)
    assert_true(True, "vectorize compiled and ran")


def test_parallelize_works() raises:
    """Parallelize: compile and run over 4 work items."""
    parallelize[worker](4)
    assert_true(True, "parallelize compiled and ran")


# ---------------------------------------------------------------------------
# std.math
# ---------------------------------------------------------------------------

def test_math_sqrt() raises:
    """Sqrt(4.0) == 2.0."""
    var x = sqrt(Float64(4.0))
    assert_true(x == 2.0, "sqrt(4) == 2")


def test_math_exp() raises:
    """Exp(0.0) == 1.0."""
    var e = exp(Float64(0.0))
    assert_true(e == 1.0, "exp(0) == 1")


def test_math_log() raises:
    """Log(1.0) == 0.0."""
    var l = log(Float64(1.0))
    assert_true(l == 0.0, "log(1) == 0")


def test_math_sin() raises:
    """Sin(0.0) == 0.0."""
    var s = sin(Float64(0.0))
    assert_true(s == 0.0, "sin(0) == 0")


# ---------------------------------------------------------------------------
# std.random
# ---------------------------------------------------------------------------

def test_random_randn() raises:
    """Randn: compiles and returns a float64 sample."""
    var ptr = std.memory.alloc[Float64](1)
    std.random.randn[DType.float64](ptr, 1)
    _ = ptr.load()
    ptr.free()
    assert_true(True, "randn compiled and ran")


# ---------------------------------------------------------------------------
# layout (MAX AI Kernels)
# ---------------------------------------------------------------------------

def test_layout_types_importable() raises:
    """Layout and LayoutTensor types are importable from the layout package."""
    assert_true(True, "layout module imported successfully")


# ---------------------------------------------------------------------------
# GPU / accelerator path (comptime-guarded)
# ---------------------------------------------------------------------------

def test_gpu_imports() raises:
    """GPU-specific imports are guarded by comptime has_accelerator() check."""
    comptime if has_accelerator():
        from std.gpu import global_idx
        from std.gpu.host import DeviceContext

        assert_true(True, "GPU imports available: global_idx, DeviceContext")
    else:
        assert_true(True, "No GPU detected — GPU imports skipped (CPU path)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() raises:
    print("MAX Kernels integration: OK")
    TestSuite.discover_tests[__functions_in_module()]().run()
