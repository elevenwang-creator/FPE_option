from server.pdf_cache import PDFGrid
from server.pricer import PricingRequest
from server.pricing_engine import PricingEngine
from std.testing import TestSuite, assert_true


def test_pricing_engine_cache_miss() raises:
    var engine = PricingEngine()
    var req = PricingRequest(S=100.0, K=105.0, V=0.1, barrier=120.0, payoff_type=0, param_hash=99999)
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)
    assert_true(len(results) == 1)
    assert_true(not results[0].success)


def test_pricing_engine_with_cached_pdf() raises:
    var engine = PricingEngine()

    var n_s = 5
    var n_v = 5
    var pdf: List[List[Float64]] = []
    for _ in range(n_s):
        var row: List[Float64] = []
        for _ in range(n_v):
            row.append(0.04)
        pdf.append(row^)

    var s_pts: List[Float64] = [80.0, 90.0, 100.0, 110.0, 120.0]
    var v_pts: List[Float64] = [0.02, 0.05, 0.1, 0.2, 0.4]
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_pts^, v_points=v_pts^, T=0.5, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    engine.store_pdf(42, grid^)

    var req = PricingRequest(S=100.0, K=95.0, V=0.1, barrier=130.0, payoff_type=1, param_hash=42)
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)
    assert_true(len(results) == 1)
    assert_true(results[0].success)
    assert_true(results[0].price >= 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
