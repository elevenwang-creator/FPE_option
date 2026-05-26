from std.python import Python, PythonObject
from std.python.conversions import ConvertibleToPython

@fieldwise_init
struct PyCsrMatrix(ConvertibleToPython, Movable, Copyable):
    var data: List[Float64]
    var indices: List[Int32]
    var indptr: List[Int32]
    var nrows: Int
    var ncols: Int

    def to_python_object(var self) raises -> PythonObject:
        var py_data = Python.list()
        var py_indices = Python.list()
        var py_indptr = Python.list()
        for i in range(len(self.data)):
            _ = py_data.append(PythonObject(self.data[i]))
        for i in range(len(self.indices)):
            _ = py_indices.append(PythonObject(self.indices[i]))
        for i in range(len(self.indptr)):
            _ = py_indptr.append(PythonObject(self.indptr[i]))
        return Python.dict(
            data=py_data, indices=py_indices, indptr=py_indptr,
            nrows=PythonObject(self.nrows), ncols=PythonObject(self.ncols),
        )

@fieldwise_init
struct PyKnotsResult(ConvertibleToPython, Movable, Copyable):
    var s: List[Float64]
    var v: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_s = Python.list()
        var py_v = Python.list()
        for i in range(len(self.s)):
            _ = py_s.append(PythonObject(self.s[i]))
        for i in range(len(self.v)):
            _ = py_v.append(PythonObject(self.v[i]))
        return Python.dict(s=py_s, v=py_v)

@fieldwise_init
struct PyGridPointsResult(ConvertibleToPython, Movable, Copyable):
    var s: List[Float64]
    var v: List[Float64]
    var s_weights: List[Float64]
    var v_weights: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_s = Python.list()
        var py_v = Python.list()
        var py_sw = Python.list()
        var py_vw = Python.list()
        for i in range(len(self.s)):
            _ = py_s.append(PythonObject(self.s[i]))
        for i in range(len(self.v)):
            _ = py_v.append(PythonObject(self.v[i]))
        for i in range(len(self.s_weights)):
            _ = py_sw.append(PythonObject(self.s_weights[i]))
        for i in range(len(self.v_weights)):
            _ = py_vw.append(PythonObject(self.v_weights[i]))
        return Python.dict(s=py_s, v=py_v, s_weights=py_sw, v_weights=py_vw)

@fieldwise_init
struct PyBasis1DResult(ConvertibleToPython, Movable, Copyable):
    var Bs: PyCsrMatrix
    var dBs: PyCsrMatrix
    var Bv: PyCsrMatrix
    var dBv: PyCsrMatrix

    def to_python_object(var self) raises -> PythonObject:
        return Python.dict(
            Bs=self.Bs.copy().to_python_object(), dBs=self.dBs.copy().to_python_object(),
            Bv=self.Bv.copy().to_python_object(), dBv=self.dBv.copy().to_python_object(),
        )

@fieldwise_init
struct PyListResult(ConvertibleToPython, Movable, Copyable):
    var data: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_data = Python.list()
        for i in range(len(self.data)):
            _ = py_data.append(PythonObject(self.data[i]))
        return py_data

@fieldwise_init
struct PyGrid2DResult(ConvertibleToPython, Movable, Copyable):
    var data: List[List[Float64]]

    def to_python_object(var self) raises -> PythonObject:
        var py_outer = Python.list()
        for row in range(len(self.data)):
            var py_row = Python.list()
            for col in range(len(self.data[row])):
                _ = py_row.append(PythonObject(self.data[row][col]))
            _ = py_outer.append(py_row)
        return py_outer

@fieldwise_init
struct PyPriceResult(ConvertibleToPython, Movable, Copyable):
    var prices: List[Float64]
    var deltas: List[Float64]
    var gammas: List[Float64]
    var vegas: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_prices = Python.list()
        var py_deltas = Python.list()
        var py_gammas = Python.list()
        var py_vegas = Python.list()
        for i in range(len(self.prices)):
            _ = py_prices.append(PythonObject(self.prices[i]))
            _ = py_deltas.append(PythonObject(self.deltas[i]))
            _ = py_gammas.append(PythonObject(self.gammas[i]))
            _ = py_vegas.append(PythonObject(self.vegas[i]))
        return Python.dict(
            prices=py_prices, deltas=py_deltas,
            gammas=py_gammas, vegas=py_vegas,
        )

@fieldwise_init
struct PyGreeksResult(ConvertibleToPython, Movable, Copyable):
    var delta: List[Float64]
    var gamma: List[Float64]
    var vega: List[Float64]

    def to_python_object(var self) raises -> PythonObject:
        var py_delta = Python.list()
        var py_gamma = Python.list()
        var py_vega = Python.list()
        for i in range(len(self.delta)):
            _ = py_delta.append(PythonObject(self.delta[i]))
        for i in range(len(self.gamma)):
            _ = py_gamma.append(PythonObject(self.gamma[i]))
        for i in range(len(self.vega)):
            _ = py_vega.append(PythonObject(self.vega[i]))
        return Python.dict(delta=py_delta, gamma=py_gamma, vega=py_vega)
