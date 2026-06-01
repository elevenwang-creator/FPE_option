from bindings._params import fpe_params_from_kwargs, option_type_from_py
from std.python import Python, PythonObject
from std.testing import TestSuite, assert_true, assert_equal

def test_option_type_from_string() raises:
    assert_equal(option_type_from_py(PythonObject("european_call")), 8)
    assert_equal(option_type_from_py(PythonObject("european_put")), 9)

def test_option_type_from_int() raises:
    assert_equal(option_type_from_py(PythonObject(0)), 0)
    assert_equal(option_type_from_py(PythonObject(9)), 9)

def test_fpe_params_from_kwargs() raises:
    var builtins = Python.import_module("builtins")
    var kwargs = Python.dict(
        kappa=PythonObject(1.2), theta=PythonObject(0.05),
        sigma=PythonObject(0.35), rho=PythonObject(-0.4),
        r=PythonObject(0.05), T=PythonObject(0.5),
        S0=PythonObject(60.0), V0=PythonObject(0.1),
        n_s=PythonObject(16), n_v=PythonObject(16),
        K=PythonObject(60.0),
        option_type=PythonObject("european_call"),
    )
    var fp = fpe_params_from_kwargs(kwargs)
    assert_true(fp.is_valid())
    assert_equal(fp.heston.kappa, 1.2)
    assert_equal(fp.heston.S0, 60.0)
    assert_equal(fp.option_type, 8)

def test_fpe_params_from_kwargs_domain() raises:
    var builtins = Python.import_module("builtins")
    var kwargs = Python.dict(
        S0=PythonObject(60.0), V0=PythonObject(0.1),
        n_s=PythonObject(16), n_v=PythonObject(16),
        K=PythonObject(60.0),
        option_type=PythonObject("european_call"),
        s_min=PythonObject(10.0), s_max=PythonObject(200.0),
    )
    var fp = fpe_params_from_kwargs(kwargs)
    assert_true(fp.is_valid())
    assert_equal(fp.heston.S_min, 10.0)
    assert_equal(fp.heston.S_max, 200.0)

def test_fpe_params_from_kwargs_list_K() raises:
    var builtins = Python.import_module("builtins")
    var k_list = Python.list()
    _ = k_list.append(PythonObject(50.0))
    _ = k_list.append(PythonObject(60.0))
    var kwargs = Python.dict(
        S0=PythonObject(60.0), V0=PythonObject(0.1),
        n_s=PythonObject(16), n_v=PythonObject(16),
        K=k_list, option_type=PythonObject("european_call"),
    )
    var fp = fpe_params_from_kwargs(kwargs)
    assert_equal(len(fp.strikes), 2)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
