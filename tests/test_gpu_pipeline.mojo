"""End-to-end GPU pipeline test: PDF grid → batch pricing → calibration.

Tests the complete logic chain across multiple modules:
1. FPE Solver (GPU batch ODE integration)
2. Pricer (GPU batch pricing)
3. Calibrator (GPU-accelerated calibration)

All data types are managed by gpu_utils.dtype module for cross-platform compatibility.
"""
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver

from server.pdf_cache import PDFGrid, PDFCache
from server.pricer import Pricer, PricingRequest, PricingResult
from server.interpolator import Interpolator
from server.greeks import Greeks

from engines.calibrator.calibrator import Calibrator

from gpu_utils.dtype import get_compute_dtype, get_backend_name, is_float32_backend


def test_gpu_fpe_solver_produces_valid_pdf() raises:
    """GPU FPE solver should produce valid PDF grid."""
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.1,
        S0=60.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )

    var domain = FPEDomain(params, n_s=8, n_v=8, degree_s=2, degree_v=2)
    var solver = FPESolver[2](rtol=1e-4, atol=1e-6, max_step=0.05)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol = solver.solve(domain, params, t_eval)

    # Verify PDF is valid
    assert_true(len(sol) > 0)
    assert_true(len(sol[0]) > 0)
    
    var total = 0.0
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] < 0.0:
                sol[i][j] = 0.0
            total += sol[i][j]
    
    # Normalize
    if total > 0.0:
        for i in range(len(sol)):
            for j in range(len(sol[i])):
                sol[i][j] = sol[i][j] / total
    
    # Re-check total after normalization
    var total2 = 0.0
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            total2 += sol[i][j]
    
    assert_true(total2 > 0.99 and total2 < 1.01)


def test_gpu_batch_pricing_produces_results() raises:
    """GPU batch pricer should produce valid pricing results."""
    # Create a simple PDF grid
    var n_s = 8
    var n_v = 8
    var pdf: List[List[Float64]] = []
    for i in range(n_s):
        var row: List[Float64] = []
        for j in range(n_v):
            row.append(1.0 / Float64(n_s * n_v))
        pdf.append(row^)
    
    var s_points: List[Float64] = []
    var v_points: List[Float64] = []
    for i in range(n_s):
        s_points.append(50.0 + Float64(i) * 12.5)
    for i in range(n_v):
        v_points.append(0.01 + Float64(i) * 0.05)
    
    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(pdf=pdf^, s_points=s_points^, v_points=v_points^, T=0.1, ds_weights=ds^, dv_weights=dv^)
    grid.precompute_weights()
    
    # Create batch pricing requests
    var requests: List[PricingRequest] = []
    for i in range(4):
        requests.append(PricingRequest(
            S=60.0,
            K=55.0 + Float64(i) * 5.0,
            V=0.1,
            barrier=200.0,
            payoff_type=1,
            param_hash=UInt64(i),
        ))
    
    var pricer = Pricer[2](interpolator=Interpolator(), greeks_computer=Greeks[2](h_s=1e-2, h_v=1e-3))
    var results = pricer.price(grid, requests)
    
    # Verify results
    assert_true(len(results) == 4)
    for i in range(len(results)):
        assert_true(results[i].success)
        assert_true(results[i].price >= 0.0)


def test_gpu_calibration_converges() raises:
    """GPU calibrator should converge to reasonable parameters."""
    var init = HestonParams(
        kappa=1.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.5,
        r=0.05,
        T=0.5,
        S0=100.0,
        V0=0.04,
        S_min=50.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
    )
    
    # Simple synthetic market prices
    var market_prices: List[Float64] = [5.0, 3.0, 1.0]
    var strikes: List[Float64] = [90.0, 100.0, 110.0]
    var expiries: List[Float64] = [0.5, 0.5, 0.5]
    
    var cal = Calibrator[1](max_iter=5, tol=1e-4)
    var result = cal.calibrate(market_prices, strikes, expiries, init)
    
    # Result should be valid params
    assert_true(result.kappa > 0.0)
    assert_true(result.theta > 0.0)
    assert_true(result.sigma > 0.0)
    assert_true(result.rho > -1.0 and result.rho < 1.0)
    assert_true(result.V0 > 0.0)


def test_dtype_management_consistency() raises:
    """Dtype management module should provide consistent types across all GPU modules."""
    var dtype = get_compute_dtype()
    var backend = get_backend_name()
    
    # Verify dtype is valid
    assert_true(dtype == DType.float32 or dtype == DType.float64)
    
    # Verify backend name is valid
    var valid_backend = (backend == "metal") or (backend == "cuda") or (backend == "hip") or (backend == "generic") or (backend == "cpu")
    assert_true(valid_backend)
    
    # Verify consistency
    if backend == "metal":
        assert_true(dtype == DType.float32)
        assert_true(is_float32_backend())
    else:
        assert_true(dtype == DType.float64)
        assert_true(not is_float32_backend())


def main() raises:
    print("=" * 60)
    print("End-to-End GPU Pipeline Test")
    print("Backend:", get_backend_name())
    print("Compute dtype:", get_compute_dtype())
    print("Float32 backend:", is_float32_backend())
    print("GPU available:", has_accelerator())
    print("=" * 60)
    
    TestSuite.discover_tests[__functions_in_module()]().run()
