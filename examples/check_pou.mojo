from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.galerkin import GalerkinAssembler
from std.math import max, min, abs


def main() raises:
    var params = HestonParams(
        kappa=1.2, theta=0.05, sigma=0.35, rho=-0.4,
        r=0.05, T=0.6, S0=60.0, V0=0.1,
        S_min=50.0, S_max=150.0, V_min=1e-4, V_max=1.0,
    )
    var n_s = 11; var n_v = 11
    var domain = FPEDomain[3, 3](params, n_s=n_s, n_v=n_v)
    
    var basis = domain.build_basis()
    var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
    
    var n_pts = len(domain.s_points) * len(domain.v_points)
    var n_basis = Phi.ncols
    
    print("Phi shape: " + String(n_pts) + " x " + String(n_basis))
    
    var pou_min = 1e30; var pou_max = -1e30; var pou_sum = 0.0
    for k in range(n_pts):
        var row_sum = 0.0
        for p in range(Phi.indptr[k], Phi.indptr[k + 1]):
            row_sum += Phi.data[p]
        pou_min = min(pou_min, row_sum)
        pou_max = max(pou_max, row_sum)
        pou_sum += row_sum
    print("Partition of unity: min=" + String(pou_min) + " max=" + String(pou_max) + " mean=" + String(pou_sum / Float64(n_pts)))
    
    var assembler = GalerkinAssembler[1]()
    var M = assembler.mass_matrix(domain)
    
    var M_rowsum: List[Float64] = []
    for j in range(n_basis):
        var rs = 0.0
        for p in range(M.indptr[j], M.indptr[j + 1]):
            rs += M.data[p]
        M_rowsum.append(rs)
    
    var m_vec: List[Float64] = []
    for j in range(n_basis):
        var m_val = 0.0
        for i in range(len(domain.s_points)):
            for k in range(len(domain.v_points)):
                var idx = i * len(domain.v_points) + k
                for p in range(Phi.indptr[idx], Phi.indptr[idx + 1]):
                    if Phi.indices[p] == j:
                        m_val += Phi.data[p] * domain.s_weights[i] * domain.v_weights[k]
        m_vec.append(m_val)
    
    var diff_max = 0.0
    var diff_rel_max = 0.0
    for j in range(n_basis):
        var diff = abs(M_rowsum[j] - m_vec[j])
        diff_max = max(diff_max, diff)
        if abs(M_rowsum[j]) > 1e-14:
            diff_rel_max = max(diff_rel_max, diff / abs(M_rowsum[j]))
    print("M*1 vs m: max_abs_diff=" + String(diff_max) + " max_rel_diff=" + String(diff_rel_max))
    
    var m_sum = 0.0; var M1_sum = 0.0
    for j in range(n_basis):
        m_sum += m_vec[j]
        M1_sum += M_rowsum[j]
    print("sum(m)=" + String(m_sum) + " sum(M*1)=" + String(M1_sum))
