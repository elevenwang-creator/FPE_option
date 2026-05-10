from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from engines.fpe.initial_cond import InitialCondition
from sparse.diag_scale import diag_scale
from std.math import sqrt


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.1, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=0.0, V_max=1.0,
    )

    var n_s = 38
    var n_v = 38
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v, num_insert=251)

    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    var K = assembler.stiffness_matrix(domain, params)
    var q0 = InitialCondition[1]().compute(domain, params, sigma0=0.1)

    var n = M.nrows
    var m_nnz = M.nnz()
    var k_nnz = K.nnz()

    # D-scaling
    var D = List[Float64]()
    var Dinv = List[Float64]()
    for i in range(n):
        var m_ii = 0.0
        for p in range(M.indptr[i], M.indptr[i + 1]):
            if M.indices[p] == i:
                m_ii = M.data[p]
                break
        if m_ii > 1e-30:
            D.append(sqrt(m_ii))
            Dinv.append(1.0 / sqrt(m_ii))
        else:
            D.append(1.0)
            Dinv.append(1.0)

    var M_s = diag_scale(M, Dinv, Dinv)
    var K_s = diag_scale(K, Dinv, Dinv)

    var q0_s = List[Float64]()
    for i in range(n):
        q0_s.append(D[i] * q0[i])

    # Use file I/O matching existing codebase style
    with open("/tmp/M_s.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(String(n) + " " + String(n) + " " + String(m_nnz) + "\n")
        for i in range(n):
            for p in range(M_s.indptr[i], M_s.indptr[i + 1]):
                var j = M_s.indices[p]
                var v = M_s.data[p]
                f.write(String(i + 1) + " " + String(j + 1) + " " + String(v) + "\n")

    with open("/tmp/K_s.mtx", "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(String(n) + " " + String(n) + " " + String(k_nnz) + "\n")
        for i in range(n):
            for p in range(K_s.indptr[i], K_s.indptr[i + 1]):
                var j = K_s.indices[p]
                var v = K_s.data[p]
                f.write(String(i + 1) + " " + String(j + 1) + " " + String(v) + "\n")

    with open("/tmp/q0_s.txt", "w") as f:
        for i in range(n):
            f.write(String(q0_s[i]) + "\n")

    print("Done n", n, "m_nnz", m_nnz, "k_nnz", k_nnz)
