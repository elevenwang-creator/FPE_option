from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from std.math import sqrt, abs


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
    var q0 = InitialCondition[1]().compute(domain, params, sigma0=10.0)

    var n = M.nrows

    with open("dump_M.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(String(n) + " " + String(n) + " " + String(len(M.data)) + "\n")
        for i in range(n):
            for p in range(M.indptr[i], M.indptr[i + 1]):
                f.write(String(i + 1) + " " + String(M.indices[p] + 1) + " " + String(M.data[p]) + "\n")

    with open("dump_K.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(String(n) + " " + String(n) + " " + String(len(K.data)) + "\n")
        for i in range(n):
            for p in range(K.indptr[i], K.indptr[i + 1]):
                f.write(String(i + 1) + " " + String(K.indices[p] + 1) + " " + String(K.data[p]) + "\n")

    with open("dump_q0.txt", "w") as f:
        for i in range(n):
            f.write(String(q0[i]) + "\n")

    with open("dump_domain.txt", "w") as f:
        f.write(String(len(domain.s_points)) + "\n")
        for i in range(len(domain.s_points)):
            f.write(String(domain.s_points[i]) + " " + String(domain.s_weights[i]) + "\n")
        f.write(String(len(domain.v_points)) + "\n")
        for i in range(len(domain.v_points)):
            f.write(String(domain.v_points[i]) + " " + String(domain.v_weights[i]) + "\n")

    print("Dumped M (" + String(len(M.data)) + " nnz), K (" + String(len(K.data)) + " nnz), q0 (" + String(n) + "), domain")
