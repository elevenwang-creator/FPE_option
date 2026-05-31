"""Four-pipeline architecture verification test.

Verifies the complete logic chains:
1. Heston single option pricing → CPU (B=1)
2. Heston batch option pricing → GPU (B>1)
3. Heston calibration → GPU (B>1 through FPESolver[B])
4. NAIS portfolio pricing → GPU (B>1 through Trainer[B])

All data types managed by gpu_utils.dtype module.
"""
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from engines.fpe.solver import FPESolver

from engines.calibrator.calibrator import Calibrator

from engines.nais.nais_net import NaisNet
from engines.nais.trainer import Trainer
from engines.nais.fbsde import FBSDEParams

from gpu_utils.dtype import GPU_DTYPE


def test_pipeline1_single_pricing_cpu() raises:
    """Pipeline 1: Heston single option pricing on CPU (B=1)."""
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.1,
        S0=60.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var domain = FPEDomain(params, n_s=6, n_v=6, degree_s=2, degree_v=2)
    var solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.05)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol = solver.solve(domain, params, t_eval)

    assert_true(len(sol) > 0)
    assert_true(len(sol[0]) > 0)


def test_pipeline2_batch_pricing_gpu() raises:
    """Pipeline 2: Heston batch option pricing on GPU (B=2)."""
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4, r=0.1, T=0.1,
        S0=60.0, V0=0.1, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var domain = FPEDomain(params, n_s=6, n_v=6, degree_s=2, degree_v=2)
    var solver = FPESolver[2](rtol=1e-4, atol=1e-6, max_step=0.05)
    var t_eval: List[Float64] = [0.0, 0.1]
    var sol = solver.solve(domain, params, t_eval)

    assert_true(len(sol) > 0)
    assert_true(len(sol[0]) > 0)


def test_pipeline3_calibration_gpu() raises:
    """Pipeline 3: Heston calibration on GPU (B=2 through FPESolver[2])."""
    var init = HestonParams(
        kappa=1.0, theta=0.04, sigma=0.3, rho=-0.5, r=0.05, T=0.5,
        S0=100.0, V0=0.04, S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var market_prices: List[Float64] = [5.0, 3.0, 1.0]
    var strikes: List[Float64] = [90.0, 100.0, 110.0]
    var expiries: List[Float64] = [0.5, 0.5, 0.5]

    var cal = Calibrator[2](max_iter=3, tol=1e-3)
    var result = cal.calibrate(market_prices, strikes, expiries, init)

    assert_true(result.kappa > 0.0)
    assert_true(result.theta > 0.0)
    assert_true(result.sigma > 0.0)


def test_pipeline4_nais_training_gpu() raises:
    """Pipeline 4: NAIS portfolio pricing on GPU (B=2 through Trainer[2])."""
    var net = NaisNet(in_dim=3, hidden=6, phi_dim=2)
    var trainer = Trainer[2](learning_rate=1e-2, n_iter=3)
    var params = FBSDEParams(
        Xi=[100.0, 0.04], T=0.2, M=4, N=6, D=1,
        H=0.1, eta=1.2, pho=-0.7, r=0.02, epsilon_t=0.09,
    )
    var losses = trainer.train(net, params)
    assert_true(len(losses) == 3)


def test_gpu_dtype_is_float() raises:
    assert_true(GPU_DTYPE == DType.float32 or GPU_DTYPE == DType.float64)


def main() raises:
    print("=" * 60)
    print("Four-Pipeline Architecture Verification")
    print("GPU_DTYPE:", GPU_DTYPE)
    print("GPU available:", has_accelerator())
    print("=" * 60)
    print()
    print("Pipeline 1: Heston Single Pricing (CPU, B=1)")
    print("Pipeline 2: Heston Batch Pricing (GPU, B>1)")
    print("Pipeline 3: Heston Calibration (GPU, B>1)")
    print("Pipeline 4: NAIS Portfolio Pricing (GPU, B>1)")
    print("=" * 60)

    TestSuite.discover_tests[__functions_in_module()]().run()
