from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder, PythonTypeBuilder
from server.compute_pipeline import ComputePipeline
from server.option_types import FpeParams
from bindings._params import fpe_params_from_kwargs
from bindings._convert import (
    PyKnotsResult, PyGridPointsResult, PyBasis1DResult, PyCsrMatrix,
    PyListResult, PyGrid2DResult, PyPriceResult, PyGreeksResult,
)
from sparse.csr import CSRMatrix


@fieldwise_init
struct PyComputePipeline(Writable, Movable):
    var inner: ComputePipeline

    @staticmethod
    def py_init(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        var fp = fpe_params_from_kwargs(kwargs)
        self = Self(ComputePipeline(fp^))

    @staticmethod
    def knots(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var knot_tup = self_ptr[].inner.knots()
        var s = knot_tup[0].copy()
        var v = knot_tup[1].copy()
        return PyKnotsResult(s=s^, v=v^).to_python_object()

    @staticmethod
    def grid_points(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var gp_tup = self_ptr[].inner.grid_points()
        var s = gp_tup[0].copy()
        var v = gp_tup[1].copy()
        var sw = gp_tup[2].copy()
        var vw = gp_tup[3].copy()
        return PyGridPointsResult(s=s^, v=v^, s_weights=sw^, v_weights=vw^).to_python_object()

    @staticmethod
    def basis_1d(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var b1_tup = self_ptr[].inner.basis_1d()
        var Bs = b1_tup[0].copy()
        var dBs = b1_tup[1].copy()
        var Bv = b1_tup[2].copy()
        var dBv = b1_tup[3].copy()
        return PyBasis1DResult(
            Bs=_csr_to_py(Bs^), dBs=_csr_to_py(dBs^),
            Bv=_csr_to_py(Bv^), dBv=_csr_to_py(dBv^),
        ).to_python_object()

    @staticmethod
    def basis_2d(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var B_2d = self_ptr[].inner.basis_2d()
        return _csr_to_py(B_2d^).to_python_object()

    @staticmethod
    def initial_condition(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var q0 = self_ptr[].inner.initial_condition()
        return PyListResult(data=q0^).to_python_object()

    @staticmethod
    def solve(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var sol = self_ptr[].inner.solve()
        return PyGrid2DResult(data=sol^).to_python_object()

    @staticmethod
    def pdf(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var pdf_grid = self_ptr[].inner.pdf()
        return PyGrid2DResult(data=pdf_grid^).to_python_object()

    @staticmethod
    def payoff_price(py_self: PythonObject, K_obj: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var builtins = Python.import_module("builtins")
        var strikes: List[Float64] = []
        if builtins.isinstance(K_obj, builtins.list):
            var k_len = Int(py=builtins.len(K_obj))
            for i in range(k_len):
                strikes.append(Float64(py=K_obj[i]))
        elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
            strikes.append(Float64(py=K_obj))
        else:
            raise Error("K must be float, int, or list of floats")
        var prices = self_ptr[].inner.price_at(strikes)
        return PyListResult(data=prices^).to_python_object()

    @staticmethod
    def greeks(py_self: PythonObject, K_obj: PythonObject) raises -> PythonObject:
        var self_ptr = py_self.downcast_value_ptr[Self]()
        var builtins = Python.import_module("builtins")
        var strikes: List[Float64] = []
        if builtins.isinstance(K_obj, builtins.list):
            var k_len = Int(py=builtins.len(K_obj))
            for i in range(k_len):
                strikes.append(Float64(py=K_obj[i]))
        elif builtins.isinstance(K_obj, builtins.float) or builtins.isinstance(K_obj, builtins.int):
            strikes.append(Float64(py=K_obj))
        else:
            raise Error("K must be float, int, or list of floats")
        var g_tup = self_ptr[].inner.greeks(strikes)
        var deltas = g_tup[0].copy()
        var gammas = g_tup[1].copy()
        var vegas = g_tup[2].copy()
        return PyGreeksResult(delta=deltas^, gamma=gammas^, vega=vegas^).to_python_object()

    def write_to(self, mut writer: Some[Writer]):
        t"PyComputePipeline(...)".write_to(writer)


def _csr_to_py(var mat: CSRMatrix) -> PyCsrMatrix:
    var data = List[Float64](capacity=len(mat.data))
    var indices = List[Int32](capacity=len(mat.indices))
    var indptr = List[Int32](capacity=len(mat.indptr))
    for i in range(len(mat.data)):
        data.append(mat.data[i])
    for i in range(len(mat.indices)):
        indices.append(Int32(mat.indices[i]))
    for i in range(len(mat.indptr)):
        indptr.append(Int32(mat.indptr[i]))
    return PyCsrMatrix(data=data^, indices=indices^, indptr=indptr^, nrows=mat.nrows, ncols=mat.ncols)
