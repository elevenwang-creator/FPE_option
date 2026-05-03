from fpe_option import price, HestonParams
from server.option_types import OptionParams, PricingResult
from server.pdf_cache import PDFGrid
from server.pricing_engine import PricingEngine
from server.pricer import PricingRequest
from std.testing import assert_true, assert_false, assert_equal, TestSuite


def test_option_params_validation() raises:
    var valid = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=0.0, option_type=1
    )
    assert_true(valid.is_valid(), "Valid params should pass")

    var invalid_S = OptionParams(
        S=-1.0, K=100.0, V=0.1, barrier=0.0, option_type=1
    )
    assert_false(invalid_S.is_valid(), "Negative S should fail validation")

    var invalid_K = OptionParams(
        S=100.0, K=-1.0, V=0.1, barrier=0.0, option_type=1
    )
    assert_false(invalid_K.is_valid(), "Negative K should fail validation")

    var invalid_barrier = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=50.0, option_type=1
    )
    assert_false(
        invalid_barrier.is_valid(),
        "Barrier below S should fail validation",
    )

    var invalid_type = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=0.0, option_type=5
    )
    assert_false(invalid_type.is_valid(), "Invalid option_type should fail")

    var valid_barrier = OptionParams(
        S=100.0, K=100.0, V=0.1, barrier=150.0, option_type=0
    )
    assert_true(
        valid_barrier.is_valid(),
        "Barrier above S with type=0 should be valid",
    )


def test_heston_params_validation() raises:
    var valid = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=1e-4,
        V_max=1.0,
    )
    assert_true(valid.is_valid(), "Valid Heston params should pass")

    var invalid_kappa = HestonParams(
        kappa=-1.0,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=1e-4,
        V_max=1.0,
    )
    assert_false(invalid_kappa.is_valid(), "Negative kappa should fail")

    var invalid_rho = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=1.5,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=1e-4,
        V_max=1.0,
    )
    assert_false(invalid_rho.is_valid(), "Rho > 1 should fail")


def test_pricing_engine_call_option() raises:
    var pdf: List[List[Float64]] = []
    pdf.append([0.5, 1.0])
    pdf.append([1.0, 2.0])
    pdf.append([0.5, 1.0])
    var s_points: List[Float64] = [80.0, 110.0, 130.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=s_points^,
        v_points=v_points^,
        T=1.0,
        ds_weights=ds^,
        dv_weights=dv^,
    )

    var engine = PricingEngine()
    engine.store_pdf(1, grid^)

    var req = PricingRequest(
        S=110.0, K=100.0, V=0.15, barrier=0.0, payoff_type=1, param_hash=1
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)

    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Call pricing should succeed")
    assert_true(results[0].price > 0.0, "Call price should be positive")


def test_pricing_engine_barrier_up_and_out() raises:
    var pdf: List[List[Float64]] = []
    pdf.append([1.0, 1.0])
    pdf.append([1.0, 1.0])
    pdf.append([1.0, 1.0])
    var s_points: List[Float64] = [80.0, 100.0, 120.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=s_points^,
        v_points=v_points^,
        T=1.0,
        ds_weights=ds^,
        dv_weights=dv^,
    )

    var engine = PricingEngine()
    engine.store_pdf(1, grid^)

    var req = PricingRequest(
        S=100.0, K=100.0, V=0.15, barrier=110.0, payoff_type=0, param_hash=1
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)

    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Barrier pricing should succeed")
    assert_true(results[0].price >= 0.0, "Barrier price should be non-negative")


def test_pricing_engine_put_option() raises:
    var pdf: List[List[Float64]] = []
    pdf.append([1.0, 1.0])
    pdf.append([0.5, 0.5])
    pdf.append([0.1, 0.1])
    var s_points: List[Float64] = [80.0, 100.0, 120.0]
    var v_points: List[Float64] = [0.1, 0.2]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=s_points^,
        v_points=v_points^,
        T=1.0,
        ds_weights=ds^,
        dv_weights=dv^,
    )

    var engine = PricingEngine()
    engine.store_pdf(1, grid^)

    var req = PricingRequest(
        S=100.0, K=110.0, V=0.15, barrier=0.0, payoff_type=3, param_hash=1
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)

    assert_equal(len(results), 1, "Should return one result")
    assert_true(results[0].success, "Put pricing should succeed")
    assert_true(results[0].price > 0.0, "Put price should be positive")


def test_pricing_engine_missing_cache() raises:
    var engine = PricingEngine()

    var req = PricingRequest(
        S=100.0, K=100.0, V=0.1, barrier=0.0, payoff_type=1, param_hash=999
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)

    assert_equal(len(results), 1, "Should return one result even on cache miss")
    assert_false(results[0].success, "Should fail when PDF not cached")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
