from bindings._convert import PyKnotsResult, PyGridPointsResult, PyCsrMatrix, PyListResult, PyGrid2DResult, PyPriceResult
from std.python import Python, PythonObject
from std.testing import TestSuite, assert_true, assert_equal

def test_knots_to_python() raises:
    var s_list: List[Float64] = [1.0, 2.0, 3.0]
    var v_list: List[Float64] = [4.0, 5.0]
    var r = PyKnotsResult(s=s_list^, v=v_list^)
    var py = r^.to_python_object()
    var builtins = Python.import_module("builtins")
    assert_equal(Int(py=builtins.len(py["s"])), 3)
    assert_equal(Int(py=builtins.len(py["v"])), 2)

def test_grid_points_to_python() raises:
    var s: List[Float64] = [1.0]
    var v: List[Float64] = [2.0]
    var sw: List[Float64] = [0.5]
    var vw: List[Float64] = [0.5]
    var r = PyGridPointsResult(s=s^, v=v^, s_weights=sw^, v_weights=vw^)
    var py = r^.to_python_object()

def test_csr_to_python() raises:
    var data: List[Float64] = [1.0, 2.0]
    var indices: List[Int32] = [0, 1]
    var indptr: List[Int32] = [0, 1, 2]
    var r = PyCsrMatrix(data=data^, indices=indices^, indptr=indptr^, nrows=2, ncols=2)
    var py = r^.to_python_object()
    assert_equal(Int(py=py["nrows"]), 2)

def test_list_result() raises:
    var data: List[Float64] = [1.0, 2.0, 3.0]
    var r = PyListResult(data=data^)
    var py = r^.to_python_object()
    var builtins = Python.import_module("builtins")
    assert_equal(Int(py=builtins.len(py)), 3)

def test_grid_2d_result() raises:
    var inner: List[Float64] = [1.0, 2.0]
    var outer = List[List[Float64]]()
    outer.append(inner^)
    var r = PyGrid2DResult(data=outer^)
    var py = r^.to_python_object()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
