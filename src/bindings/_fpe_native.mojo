from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from server.option_types import FpeParams
from server.pricing_engine import PricingEngine
from server.compute_pipeline import ComputePipeline
from bindings._params import fpe_params_from_kwargs
from bindings._convert import PyPriceResult, PyGrid2DResult
from bindings._py_pipeline import PyComputePipeline


def py_price(params_obj: PythonObject) raises -> PythonObject:
    var fp = fpe_params_from_kwargs(params_obj)
    var engine = PricingEngine()
    var results = engine.price(fp)

    var prices = List[Float64](capacity=len(results))
    var deltas = List[Float64](capacity=len(results))
    var gammas = List[Float64](capacity=len(results))
    var vegas = List[Float64](capacity=len(results))

    for i in range(len(results)):
        prices.append(results[i].price)
        deltas.append(results[i].delta)
        gammas.append(results[i].gamma)
        vegas.append(results[i].vega)

    return PyPriceResult(
        prices=prices^, deltas=deltas^,
        gammas=gammas^, vegas=vegas^,
    ).to_python_object()


@export
def PyInit__fpe_native() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_fpe_native")
        module.def_function[py_price]("price")
        _ = module.add_type[PyComputePipeline]("Compute")\
            .def_py_init[PyComputePipeline.py_init]()\
            .def_method[PyComputePipeline.knots]("knots")\
            .def_method[PyComputePipeline.grid_points]("grid_points")\
            .def_method[PyComputePipeline.basis_1d]("basis_1d")\
            .def_method[PyComputePipeline.basis_2d]("basis_2d")\
            .def_method[PyComputePipeline.initial_condition]("initial_condition")\
            .def_method[PyComputePipeline.solve]("solve")\
            .def_method[PyComputePipeline.pdf]("pdf")\
            .def_method[PyComputePipeline.payoff_price]("payoff_price")\
            .def_method[PyComputePipeline.greeks]("greeks")
        return module.finalize()
    except e:
        print("Failed to init _fpe_native module: ", e)
        return PythonObject(None)
