"""Facade module: single-call API for FPE option pricing, calibration, and NAIS."""

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.solver import FPESolver
from server.pricing_engine import PricingEngine
from server.pdf_cache import PDFGrid
from server.pricer import PricingRequest, PricingResult
from server.option_types import RoughBergomiParams, NAISModel
from engines.nais.trainer import Trainer
from engines.nais.inferencer import Inferencer
from engines.nais.nais_net import NaisNet
from engines.calibrator.calibrator import Calibrator


def _param_hash(params: HestonParams) -> UInt64:
    var h: UInt64 = 5381
    h = ((h << 5) + h) ^ UInt64(Int(params.kappa * 1e6))
    h = ((h << 5) + h) ^ UInt64(Int(params.theta * 1e7))
    h = ((h << 5) + h) ^ UInt64(Int(params.sigma * 1e8))
    h = ((h << 5) + h) ^ UInt64(Int((params.rho + 1.0) * 1e9))
    h = ((h << 5) + h) ^ UInt64(Int(params.r * 1e10))
    h = ((h << 5) + h) ^ UInt64(Int(params.T * 1e11))
    h = ((h << 5) + h) ^ UInt64(Int(params.S0 * 1e4))
    h = ((h << 5) + h) ^ UInt64(Int(params.V0 * 1e6))
    return h


def _build_pdf_grid(
    heston: HestonParams,
    n_s: Int,
    n_v: Int,
    rtol: Float64,
    atol: Float64,
) raises -> PDFGrid:
    var domain = FPEDomain[3, 3](heston, n_s=n_s, n_v=n_v)
    var solver = FPESolver[1](
        rtol=rtol, atol=atol, max_step=0.1, first_step=0.0
    )
    var t_eval: List[Float64] = [0.0, heston.T]
    var sol = solver.solve(domain, heston, t_eval)

    var n_s_pts = len(domain.s_points)
    var n_v_pts = len(domain.v_points)
    var pdf: List[List[Float64]] = []
    for i in range(n_s_pts):
        var row: List[Float64] = []
        for j in range(n_v_pts):
            row.append(sol[len(sol) - 1][i * n_v_pts + j])
        pdf.append(row^)

    var ds: List[Float64] = []
    var dv: List[Float64] = []
    var grid = PDFGrid(
        pdf=pdf^,
        s_points=domain.s_points.copy(),
        v_points=domain.v_points.copy(),
        T=heston.T,
        ds_weights=ds^,
        dv_weights=dv^,
    )
    grid.precompute_weights()
    return grid^


def price(
    heston: HestonParams,
    K: Float64,
    barrier: Float64 = 0.0,
    payoff_type: Int = 1,
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> PricingResult:
    var grid = _build_pdf_grid(heston, n_s, n_v, rtol, atol)
    var param_hash = _param_hash(heston)
    var engine = PricingEngine()
    engine.store_pdf(param_hash, grid^)
    var req = PricingRequest(
        S=heston.S0,
        K=K,
        V=heston.V0,
        barrier=barrier,
        payoff_type=payoff_type,
        param_hash=param_hash,
    )
    var requests: List[PricingRequest] = [req^]
    var results = engine.price[1](requests)
    return results[0].copy()


def price_batch(
    heston: HestonParams,
    options: List[Tuple[Float64, Float64, Int]],
    n_s: Int = 38,
    n_v: Int = 38,
    rtol: Float64 = 1e-4,
    atol: Float64 = 1e-6,
) raises -> List[PricingResult]:
    var grid = _build_pdf_grid(heston, n_s, n_v, rtol, atol)
    var param_hash = _param_hash(heston)
    var engine = PricingEngine()
    engine.store_pdf(param_hash, grid^)
    var requests: List[PricingRequest] = []
    for i in range(len(options)):
        var req = PricingRequest(
            S=heston.S0,
            K=options[i][0],
            V=heston.V0,
            barrier=options[i][1],
            payoff_type=options[i][2],
            param_hash=param_hash,
        )
        requests.append(req^)
    return engine.price[1](requests)


def calibrate(
    market_prices: List[Float64],
    strikes: List[Float64],
    expiries: List[Float64],
    init: HestonParams,
    max_iter: Int = 50,
    tol: Float64 = 1e-6,
) raises -> HestonParams:
    var calibrator = Calibrator[1](max_iter=max_iter, tol=tol)
    return calibrator.calibrate(market_prices, strikes, expiries, init)


def nais_train(
    var bergomi: RoughBergomiParams,
    iters: Int = 100,
    lr: Float64 = 1e-3,
) raises -> NAISModel:
    var net = NaisNet(in_dim=3, hidden=12, phi_dim=2)
    var fbsde_params = bergomi.to_fbsde_params()
    var trainer = Trainer[1](learning_rate=lr, n_iter=iters)
    var losses = trainer.train(net, fbsde_params)
    return NAISModel(net=net^, params=bergomi^)


def nais_vol_surface(
    model: NAISModel,
    strikes: List[Float64],
    expiries: List[Float64],
) -> List[List[Float64]]:
    var inferencer = Inferencer[1](
        net=model.net.copy(),
        risk_free_rate=model.params.r,
    )
    return inferencer.vol_surface(strikes, expiries)
