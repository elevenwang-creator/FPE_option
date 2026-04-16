"""Heston Model Single Option Pricing via FPE.

Full pricing pipeline matching Python architecture:
1. HestonParams  - set Heston stochastic volatility parameters
2. FPEDomain     - generate B-spline knots (normalized) and GL quadrature grid
3. GalerkinAssembler - assemble mass M and stiffness K matrices (sparse CSR)
4. InitialCondition  - compute initial coefficient vector q0 via ADMM-OSQP
5. FPESolver     - solve ODE  M dq/dt = -K q  with RADAU5 (sparse linear)
6. PDFComputer   - evaluate PDF from q(T): p = Phi @ q(T)
7. Option pricing using marginal distribution with physical coordinates
"""

from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from engines.fpe.solver import FPESolver
from engines.fpe.pdf import PDFComputer
from std.math import exp, sqrt


def marginal_s_distribution(
    pdf: List[List[Float64]],
    v_weights: List[Float64],
) -> List[Float64]:
    var n_s = len(pdf)
    var marginal: List[Float64] = []
    for i in range(n_s):
        var val = 0.0
        for j in range(len(pdf[i])):
            val += pdf[i][j] * v_weights[j]
        marginal.append(val)
    return marginal^


def vanilla_call_price(
    marginal: List[Float64],
    s_points_phys: List[Float64],
    s_weights: List[Float64],
    strike: Float64,
    discount: Float64,
) -> Float64:
    var price = 0.0
    for i in range(len(marginal)):
        var payoff = s_points_phys[i] - strike
        if payoff <= 0.0:
            continue
        price += marginal[i] * payoff * s_weights[i]
    return discount * price


def barrier_call_price(
    marginal: List[Float64],
    s_points_phys: List[Float64],
    s_weights: List[Float64],
    strike: Float64,
    discount: Float64,
    barrier: Float64,
) -> Float64:
    var price = 0.0
    for i in range(len(marginal)):
        if s_points_phys[i] <= barrier:
            continue
        var payoff = s_points_phys[i] - strike
        if payoff <= 0.0:
            continue
        price += marginal[i] * payoff * s_weights[i]
    return discount * price


def main() raises:
    print("=" * 60)
    print("  Heston Model Single Option Pricing via FPE")
    print("=" * 60)

    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )
    params.validate()
    print()
    print("[1/7] Heston parameters:")
    print("  kappa=" + String(params.kappa) + " theta=" + String(params.theta) +
          " sigma=" + String(params.sigma) + " rho=" + String(params.rho))
    print("  r=" + String(params.r) + " T=" + String(params.T) +
          " S0=" + String(params.S0) + " V0=" + String(params.V0))
    print("  Feller condition: 2*kappa*theta/sigma^2 - 1 = " +
          String(params.feller_condition()))

    var n_s = 38
    var n_v = 38
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v, num_insert=251)
    print()
    print("[2/7] Domain constructed:")
    print("  S knots: " + String(len(domain.s_knots)) + "  GL points: " + String(len(domain.s_points)))
    print("  V knots: " + String(len(domain.v_knots)) + "  GL points: " + String(len(domain.v_points)))
    print("  S normalized: [" + String(domain.s_points[0]) + ", " + String(domain.s_points[len(domain.s_points) - 1]) + "]")
    print("  S physical:   [" + String(domain.s_points_phys[0]) + ", " + String(domain.s_points_phys[len(domain.s_points_phys) - 1]) + "]")

    print()
    print("[3/7] Assembling Galerkin matrices (Kronecker structure)...")
    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)
    print("  System size: " + String(M.nrows) + " x " + String(M.ncols))
    print("  M nnz: " + String(M.nnz()) + ", K nnz: " + String(K.nnz()))

    print()
    print("[4/7] Computing initial condition q0 (ADMM-OSQP + D-scaling)...")
    var q0 = InitialCondition[1]().compute(domain, params, sigma0=0.1)
    var q0_sum = 0.0
    for i in range(len(q0)):
        q0_sum += q0[i]
    print("  q0 length: " + String(len(q0)) + ", sum: " + String(q0_sum))

    print()
    print("[5/7] Solving FPE ODE with RADAU5 (sparse linear)...")

    var Kq0 = K.spmv(q0)
    var kq0_norm = 0.0
    for i in range(len(Kq0)):
        kq0_norm += Kq0[i] * Kq0[i]
    kq0_norm = sqrt(kq0_norm)
    var q0_norm = 0.0
    for i in range(len(q0)):
        q0_norm += q0[i] * q0[i]
    q0_norm = sqrt(q0_norm)
    print("  ||K*q0|| = " + String(kq0_norm))
    print("  ||q0|| = " + String(q0_norm))

    var t_eval = [0.0, params.T]

    var fpe_solver = FPESolver[1](rtol=1e-4, atol=1e-6, max_step=0.1, first_step=1e-6)
    var q_t = fpe_solver.solve(domain, params, t_eval)
    print("  ODE steps: " + String(len(q_t)))

    var q_diff = q_t[len(q_t) - 1].copy()
    for i in range(len(q_diff)):
        q_diff[i] = q_diff[i] - q_t[0][i]
    var diff_norm = 0.0
    for i in range(len(q_diff)):
        diff_norm += q_diff[i] * q_diff[i]
    diff_norm = sqrt(diff_norm)
    print("  ||q(T) - q(0)|| = " + String(diff_norm))
    print("  ||q(T) - q(0)|| / ||q0|| = " + String(diff_norm / q0_norm))

    var q_final = q_t[len(q_t) - 1].copy()
    var q_init = q_t[0].copy()
    var pdf_computer = PDFComputer[1]()
    var pdf_grid_init = pdf_computer.compute(domain, q_init)
    var pdf_norm_init = 0.0
    for i in range(len(pdf_grid_init)):
        for j in range(len(pdf_grid_init[i])):
            pdf_norm_init += pdf_grid_init[i][j] * domain.s_weights[i] * domain.v_weights[j]
    print("  PDF integral at t=0: " + String(pdf_norm_init) + " (should be ~1.0)")

    var e_s_init = 0.0
    for i in range(len(pdf_grid_init)):
        var v_sum = 0.0
        for j in range(len(pdf_grid_init[i])):
            v_sum += pdf_grid_init[i][j] * domain.v_weights[j]
        e_s_init += domain.s_points_phys[i] * domain.s_weights[i] * v_sum
    print("  E[S] at t=0: " + String(e_s_init) + " (should be S0=" + String(params.S0) + ")")

    print()
    print("[6/7] Computing PDF from q(T)...")
    var pdf_grid = pdf_computer.compute(domain, q_final)

    var pdf_norm = 0.0
    for i in range(len(pdf_grid)):
        for j in range(len(pdf_grid[i])):
            pdf_norm += pdf_grid[i][j] * domain.s_weights[i] * domain.v_weights[j]
    print("  PDF integral: " + String(pdf_norm) + " (should be ~1.0)")

    var e_s = 0.0
    for i in range(len(pdf_grid)):
        var v_sum = 0.0
        for j in range(len(pdf_grid[i])):
            v_sum += pdf_grid[i][j] * domain.v_weights[j]
        e_s += domain.s_points_phys[i] * domain.s_weights[i] * v_sum
    print("  E[S] under PDF = " + String(e_s) + " (should be S0*exp(rT)=" + String(params.S0 * exp(params.r * params.T)) + ")")

    print()
    print("[7/7] Pricing options...")
    var discount = exp(-params.r * params.T)
    var marginal = marginal_s_distribution(pdf_grid, domain.v_weights)

    var strikes = [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]
    var barrier = 50.0
    print()
    print("  Vanilla Call Prices (r=" + String(params.r) + ", T=" + String(params.T) + "):")
    print("  Strike   Price")
    for idx in range(len(strikes)):
        var K_strike = strikes[idx]
        var call_price = vanilla_call_price(
            marginal, domain.s_points_phys, domain.s_weights, K_strike, discount,
        )
        print("  " + String(K_strike) + "     " + String(call_price))

    print()
    print("  Barrier Down-and-Out Call Prices (barrier=" + String(barrier) + "):")
    print("  Strike   Price")
    for idx in range(len(strikes)):
        var K_strike = strikes[idx]
        var b_call = barrier_call_price(
            marginal, domain.s_points_phys, domain.s_weights, K_strike, discount, barrier,
        )
        print("  " + String(K_strike) + "     " + String(b_call))

    print()
    print("=" * 60)
    print("  Functional test: " + ("PASSED" if pdf_norm > 0.5 else "FAILED"))
    print("=" * 60)
