from engines.fpe.domain import FPEDomain


def _reshape_to_grid(flat: List[Float64], n_s: Int, n_v: Int) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    var idx = 0
    for _ in range(n_s):
        var row: List[Float64] = []
        for _ in range(n_v):
            var value = 0.0
            if idx < len(flat):
                value = flat[idx]
            row.append(value)
            idx += 1
        out.append(row^)
    return out^


struct PDFComputer[B: Int]:
    def __init__(out self):
        pass

    def compute(self, domain: FPEDomain, q_t: List[Float64]) -> List[List[Float64]]:
        var basis = domain.build_basis()
        var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
        var pdf_flat_scalar = Phi.spmv(q_t)

        var pdf_flat: List[Float64] = []
        for i in range(len(pdf_flat_scalar)):
            pdf_flat.append(pdf_flat_scalar[i])

        return _reshape_to_grid(pdf_flat, len(domain.s_points), len(domain.v_points))
