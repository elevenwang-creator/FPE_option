from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )

    var n_s = 11
    var n_v = 11
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v)

    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)

    var q0_default = InitialCondition[1]().compute(domain, params)
    var q0_s10 = InitialCondition[1]().compute(domain, params, sigma0=10.0)

    var sum_default = 0.0
    var sum_s10 = 0.0
    for i in range(len(q0_default)):
        sum_default += q0_default[i]
        sum_s10 += q0_s10[i]

    print("q0 default (sigma0=2.0) sum: " + String(sum_default))
    print("q0 sigma0=10.0 sum: " + String(sum_s10))

    with open("dump_q0_default.txt", "w") as f:
        for i in range(len(q0_default)):
            f.write(String(q0_default[i]) + "\n")

    print("Dumped q0_default to dump_q0_default.txt")
