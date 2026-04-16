from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from std.math import exp, pi


def main() raises:
    print("Test: Gauss-Legendre Quadrature Grid Generation")
    print("=============================================")

    var params = HestonParams(
        kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7,
        r=0.05, T=1.0, S0=100.0, V0=0.04,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=0.5
    )
    params.validate()

    var domain = FPEDomain[3, 3](params, n_s=20, n_v=20, gl_order=5)
    
    print("S domain: " + String(params.S_min) + " to " + String(params.S_max))
    print("V domain: " + String(params.V_min) + " to " + String(params.V_max))
    print("S knots count: " + String(len(domain.s_knots)))
    print("V knots count: " + String(len(domain.v_knots)))
    print("S points (GL quadrature): " + String(len(domain.s_points)))
    print("V points (GL quadrature): " + String(len(domain.v_points)))
    print("S weights count: " + String(len(domain.s_weights)))
    print("V weights count: " + String(len(domain.v_weights)))
    
    var s_sum = 0.0
    for i in range(len(domain.s_weights)):
        s_sum = s_sum + domain.s_weights[i]
    print("S weights sum: " + String(s_sum) + " (should be ~1.0 for normalized)")

    var norm = 0.0
    for i in range(len(domain.s_weights)):
        for j in range(len(domain.v_weights)):
            norm = norm + domain.s_weights[i] * domain.v_weights[j]
    print("Jacobian factor: " + String(domain.jacobian_factor()))
    print("2D integration sum: " + String(norm) + " (should be ~" + String(domain.jacobian_factor()) + ")")
    
    var sigma_s = 0.1 * (params.S_max - params.S_min)
    var sigma_v = 0.1 * (params.V_max - params.V_min)
    if sigma_s <= 0.0:
        sigma_s = 10.0
    if sigma_v <= 0.0:
        sigma_v = 0.05

    var pdf_peak = 0.0
    for i in range(len(domain.s_points)):
        var ds = (domain.s_points[i] - params.S0) / sigma_s
        for j in range(len(domain.v_points)):
            var dv = (domain.v_points[j] - params.V0) / sigma_v
            var val = exp(-0.5 * (ds * ds + dv * dv))
            if val > pdf_peak:
                pdf_peak = val

    print("Gaussian peak value at grid: " + String(pdf_peak) + " (should be > 0)")
    
    var sum_check = 0.0
    for i in range(len(domain.s_points)):
        var ds = (domain.s_points[i] - params.S0) / sigma_s
        for j in range(len(domain.v_points)):
            var dv = (domain.v_points[j] - params.V0) / sigma_v
            var val = exp(-0.5 * (ds * ds + dv * dv))
            sum_check = sum_check + val * domain.s_weights[i] * domain.v_weights[j]

    print("Gaussian integral on GL grid: " + String(sum_check) + " (should be ~1.0)")
    print("")
    print("Test complete.")
