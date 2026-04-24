from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams


def main() raises:
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        T=0.6,
        S0=60.0,
        V0=0.1,
        S_min=50.0,
        S_max=150.0,
        V_min=1e-4,
        V_max=1.0,
    )
    params.validate()
    var n_s = 38
    var n_v = 38
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v, num_insert=251)
    var s_mesh = domain.s_points_phys.copy()
    for idx in range(len(s_mesh)):
        print("  S physical: [" + String(domain.s_points_phys[idx]) + "]")
